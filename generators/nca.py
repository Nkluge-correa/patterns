"""Neural Cellular Automata (NCA) pattern generator.

A stochastic neural cellular automaton evolves a small 2D grid of discrete
cell states under a randomly-initialised local update rule (a tiny CNN with
toroidal padding). Each sample uses a freshly-sampled rule, so every
sequence corresponds to a different dynamical system.

The generator flattens the rollout trajectory frame-by-frame in row-major
order, wrapping each frame in reserved `<grid>` / `</grid>` delimiter
tokens (the first two IDs of the vocabulary). Cell states map to the
remaining vocab IDs so they never collide with the delimiters.

A sample's `target_len` must accommodate at least two full frames
(`2 * (grid_size**2 + 2)` tokens); any leftover slack after packing as
many full frames as possible is padded with random cell-state IDs.

This has been adapted from the reference implementation:
- See https://github.com/danihyunlee/nca-pre-pretraining/blob/main/utils/nca.py

ELI5 version:

Imagine a tiny 8x8 chessboard whose squares can take 8 different "colours"
(states 0..7). A randomly-wired neural network looks at every square and
its 8 neighbours, then rolls dice (biased by the network) to decide what
colour each square turns into next tick. We let the board evolve for a
short warm-up, then start recording snapshots ("frames").

To turn each snapshot into a sequence of token IDs we:

1. Pick the first two IDs from the vocabulary -- `vocab[0]` and `vocab[1]`
   -- and reserve them as the "<grid>" and "</grid>" tags. They never
   appear inside a frame, only at its boundaries.
2. Map each cell state `s` (0..7) to `vocab[s + 2]`. With the default vocab
   `[0, 1, 2, ..., 255]`, state 0 -> token 2, state 1 -> token 3, ...,
   state 7 -> token 9. No patches, no lookup table -- just an offset by 2.
3. Read the snapshot row by row (left-to-right, top-to-bottom) and emit:

       <grid>  c00 c01 ... c07  c10 c11 ... c77  </grid>

   That is 1 + 64 + 1 = 66 tokens per frame.
4. Concatenate as many *complete* frames as fit in `target_len`. Any
   leftover tail (fewer than 66 tokens) is filled with random cell-state
   IDs purely so the sample lands at exactly `target_len`; we deliberately
   never emit a half-frame, because that would leave an orphan delimiter.

So every NCA sample ends up using only IDs `{0, 1}` (delimiters) and
`{2..9}` (cell states), which trivially fits inside the project's
`[0, 256)` vocabulary budget.

Things To Ablate:

    _GRID_SIZE — Directly scales sequence length per frame (grid_size² + 2 tokens). Larger
    grids slow convergence to attractors and produce richer spatial patterns, but blow up
    target_len requirements.
    Worth probing: 4, 8, 16.

    _D_STATE — Controls the "colour palette" of the automaton. At 2 it collapses to a binary
    CA (very repetitive), at 16+ it starts resembling a continuous system.
    Worth probing: 2, 4, 8, 16.

    Vocab size — Currently only affects d_state indirectly (clamped to len(vocab) - 2), so
    it's not an independent axis unless we decouple the clamping. Probably least interesting
    unless we change the mapping.

    _TEMPERATURE — This is the primary dial between ordered (low -> near-deterministic, sharp
    fixed-point attractors) and chaotic (high -> random noise) regimes. The "edge of chaos"
    is around 1.0.
    Worth probing: 0.1, 0.5, 1.0, 2.0, 5.0.

    _IDENTITY_BIAS — Acts as a persistence prior: positive values make cells "sticky" (slow
    oscillators, stable blobs), negative values drive constant churn. Combined with temperature
    it covers most of the dynamical phase diagram. 
    Worth probing: -2, 0, 2, 5.
"""

import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from registry import register

# Use CUDA when available. For the tiny default grid (8×8) the GPU kernel-
# launch overhead may partially offset the gain, but across many thousands of
# samples the network forward-passes still benefit from GPU execution.
# Set to torch.device("cpu") to force CPU regardless.
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hardcoded NCA defaults
# d_state is the number of discrete cell states. The vocab budget per sample
# is `len(vocab) - 2` (the first two IDs are reserved as <grid> / </grid>
# delimiters), and the project caps vocab at 256, so d_state stays small.
# The grid is intentionally tiny so a handful of rollout steps fills typical
# context windows.
_GRID_SIZE = 8
_D_STATE = 8
_IDENTITY_BIAS = 0.0
_TEMPERATURE = 1.0
# Burn-in steps discarded before recording the trajectory, so the sample
# captures the rule's attractor rather than the random initial condition.
_BURN_IN_STEPS = 4

# Per-frame serialization is [OPEN, c_0, c_1, ..., c_{HW-1}, CLOSE], so the
# minimum useful context is at least this many full frames.
_MIN_FRAMES_PER_SAMPLE = 2

# Reserved delimiter slots inside the caller's vocabulary. The vocab is the
# contiguous range [0, vocab_size), so we simply take the first two IDs as
# <grid> / </grid> tokens; cell states map to vocab[2 : 2 + d_state].
_OPEN_IDX = 0
_CLOSE_IDX = 1
_N_RESERVED = 2


class _NCANetwork(nn.Module):
    """Local update rule: pad-wrap -> 3x3 conv -> 1x1 conv -> relu -> 1x1 conv.

    Operates on a single grid (no batch dim) shaped (H, W, d_state).
    """

    def __init__(self, d_state: int):
        super().__init__()
        self.conv1 = nn.Conv2d(d_state, 4, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=1)
        self.conv3 = nn.Conv2d(16, d_state, kernel_size=1)

    def forward(self, x_hwc: torch.Tensor) -> torch.Tensor:
        # x_hwc: (H, W, C) one-hot state -> logits (H, W, C).
        x = x_hwc.permute(2, 0, 1).unsqueeze(0)          # (1, C, H, W)
        x = F.pad(x, (1, 1, 1, 1), mode="circular")      # toroidal padding
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        return x.squeeze(0).permute(1, 2, 0)             # (H, W, C)


def _make_rule(d_state: int, gen: torch.Generator, device: torch.device) -> _NCANetwork:
    """Build a fresh NCA with weights drawn from the seeded generator."""
    # Weights are initialised on CPU (generator is CPU-bound) then moved.
    net = _NCANetwork(d_state=d_state)
    with torch.no_grad():
        for p in net.parameters():
            if p.dim() >= 2:
                # Lecun-style normal: std = 1 / sqrt(fan_in).
                fan_in = p.shape[1] * (p.shape[2] * p.shape[3] if p.dim() == 4 else 1)
                p.copy_(torch.empty_like(p).normal_(generator=gen) / max(1, fan_in) ** 0.5)
            else:
                p.zero_()
    net = net.to(device)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _rollout(
    n_frames: int,
    grid_size: int,
    d_state: int,
    identity_bias: float,
    temperature: float,
    gen: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Run the NCA and return a (n_frames, grid_size, grid_size) int tensor (CPU)."""
    net = _make_rule(d_state, gen, device)

    # Initial state sampled on CPU (generator is CPU-bound), then moved.
    init_logits = torch.empty(d_state).normal_(generator=gen)
    probs0 = torch.softmax(init_logits, dim=-1)
    state = torch.multinomial(
        probs0, num_samples=grid_size * grid_size, replacement=True, generator=gen
    ).reshape(grid_size, grid_size).to(device)

    total_steps = _BURN_IN_STEPS + n_frames
    frames = []
    for t in range(total_steps):
        one_hot = F.one_hot(state, num_classes=d_state).float()            # (H, W, C)
        logits = net(one_hot)                                              # (H, W, C)
        logits = (logits + one_hot * identity_bias) / temperature
        probs = torch.softmax(logits, dim=-1).reshape(-1, d_state)
        # torch.multinomial requires a CPU generator; sample on CPU and move back.
        state = torch.multinomial(
            probs.cpu(), num_samples=1, generator=gen
        ).to(device).reshape(grid_size, grid_size)
        if t >= _BURN_IN_STEPS:
            frames.append(state)

    return torch.stack(frames, dim=0).cpu()  # (n_frames, H, W) on CPU


@register(
    "nca",
    "Neural Cellular Automaton: a small 2D grid of discrete cell states "
    "evolves under a freshly-sampled random local CNN rule. "
    "The rollout trajectory is flattened time-major / row-major "
    "into a 1D token stream. Produces locally-coherent, globally-varying "
    "patterns reminiscent of Lenia / Game-of-Life dynamics.",
)
def gen_nca(vocab: List[int], target_len: int, rng: random.Random) -> List[int]:
    # Reserve the first two vocab IDs as <grid> / </grid> delimiters; cell
    # states use the remainder. d_state is clamped to the remaining vocab.
    if len(vocab) < _N_RESERVED + 2:
        raise ValueError(
            f"NCA requires at least {_N_RESERVED + 2} vocab IDs "
            f"(2 delimiters + >=2 cell states); got {len(vocab)}."
        )
    open_tok = vocab[_OPEN_IDX]
    close_tok = vocab[_CLOSE_IDX]
    state_vocab = vocab[_N_RESERVED:]
    d_state = min(_D_STATE, len(state_vocab))

    grid_size = _GRID_SIZE
    frame_size = grid_size * grid_size + 2  # cells + open + close
    min_target = frame_size * _MIN_FRAMES_PER_SAMPLE
    if target_len < min_target:
        raise ValueError(
            f"NCA requires target_len >= {min_target} "
            f"({_MIN_FRAMES_PER_SAMPLE} frames of {frame_size} tokens at "
            f"grid_size={grid_size}); got {target_len}."
        )
    # Emit only complete frames; any leftover slack is padded so the sample
    # still has exactly `target_len` tokens.
    n_frames = target_len // frame_size
    leftover = target_len - n_frames * frame_size

    # Seed a torch RNG deterministically from the caller's Random instance so
    # generation stays reproducible alongside the rest of the pipeline.
    seed = rng.getrandbits(63)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    frames = _rollout(
        n_frames=n_frames,
        grid_size=grid_size,
        d_state=d_state,
        identity_bias=_IDENTITY_BIAS,
        temperature=_TEMPERATURE,
        gen=gen,
        device=_DEVICE,
    )  # (n_frames, H, W) on CPU

    # Serialize: per-frame [open_tok, row-major cells mapped to state_vocab, close_tok].
    cells = frames.reshape(n_frames, -1).tolist()
    out: List[int] = []
    for frame_cells in cells:
        out.append(open_tok)
        out.extend(state_vocab[s] for s in frame_cells)
        out.append(close_tok)

    # Pad any residual tokens with random cell-state IDs so structure stays
    # clean (no orphan delimiters). With frame_size = grid_size**2 + 2 = 66,
    # all powers of 2 >= 256 leave only a small tail.
    if leftover:
        out.extend(rng.choice(state_vocab) for _ in range(leftover))

    return out
