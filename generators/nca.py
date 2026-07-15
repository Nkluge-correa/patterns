"""Neural Cellular Automata (NCA) pattern generator.

A stochastic neural cellular automaton evolves a small 2D grid of discrete
cell states under a randomly-initialised local update rule (a tiny CNN with
toroidal padding). Each sample uses a freshly-sampled rule, so every
sequence corresponds to a different dynamical system.

This follows the setup of the reference study (Lee et al. 2026): a 12x12
grid with 10 cell states, softmax temperature 1e-3, and 2x2 PATCH
tokenization -- each non-overlapping 2x2 patch of cells maps bijectively to
one token, giving a patch vocabulary of 10**4 tokens. The generator
flattens the rollout trajectory frame-by-frame in row-major *patch* order,
wrapping each frame in reserved `<grid>` / `</grid>` delimiter tokens
(IDs 1 and 2; ID 0 is a dedicated pad token). Patch tokens map to the
remaining vocab IDs so they never collide with the delimiters or the pad.
The full setup therefore needs a vocab of 3 + 10**4 = 10003 IDs.

A sample's `target_len` must accommodate at least two full frames
(`2 * ((grid_size // patch_size)**2 + 2)` tokens); any leftover slack after
packing as many full frames as possible is padded with the dedicated pad
token (ID 0), whose loss should be masked during training.

This has been adapted from the reference implementation:
- See https://github.com/danihyunlee/nca-pre-pretraining/blob/main/utils/nca.py

Things To Ablate:

    _GRID_SIZE — Directly scales sequence length per frame ((grid_size/patch)² + 2 tokens).
    Larger grids slow convergence to attractors and produce richer spatial patterns, but
    blow up target_len requirements. Must be divisible by _PATCH_SIZE.
    Worth probing: 8, 12, 16.

    _D_STATE — Controls the "colour palette" of the automaton. At 2 it collapses to a binary
    CA (very repetitive), at 16+ it starts resembling a continuous system. The patch vocab
    grows as d_state**(patch_size²), so keep an eye on the vocab budget.
    Worth probing: 2, 4, 8, 10, 16.

    _PATCH_SIZE — Side of the square cell patch mapped to a single token. 2 reproduces the
    reference paper (patch vocab d_state⁴); 1 recovers direct per-cell tokens (vocab
    d_state + 3).

    _TEMPERATURE — The softmax temperature applied to the rule's transition logits.
    The raw CNN logits are tiny, so this is the primary dial between ordered (low ->
    near-deterministic, sharp fixed-point attractors) and chaotic (high -> near-uniform
    noise) regimes. NOTE: it is normally set indirectly via _REGIME (see below) rather
    than edited here.
    Worth probing: 1e-3, 0.1, 0.5, 1.0, 2.0.

    _IDENTITY_BIAS — Acts as a persistence prior: positive values make cells "sticky" (slow
    oscillators, stable blobs), negative values drive constant churn. Combined with temperature
    it covers most of the dynamical phase diagram. Normally set via _REGIME.
    Worth probing: -2, 0, 2, 5.

    _REGIME — The headline difficulty knob. Selects a (temperature, identity_bias) preset
    placing the dynamics in a difficulty band labelled by the oracle next-cell loss as a
    fraction of ln(d_state). See _REGIMES for the presets and tools/validate.py to measure.
"""

from __future__ import annotations

import gzip
import random
import sys
import threading
from array import array

import torch
import torch.nn as nn
import torch.nn.functional as F

from registry import register
from utils import PAD_ID

# Use CUDA when available. For the tiny default grid (8×8) the GPU kernel-
# launch overhead may partially offset the gain, but across many thousands of
# samples the network forward-passes still benefit from GPU execution.
# Set to torch.device("cpu") to force CPU regardless.
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hardcoded NCA defaults (reference-paper setup, Lee et al. 2026).
# d_state is the number of discrete cell states. Tokens are 2x2 cell PATCHES,
# so the patch vocabulary is d_state ** (_PATCH_SIZE**2) = 10**4 and the total
# vocab budget per sample is `_N_RESERVED + d_state**4` (= 10003 at defaults).
_GRID_SIZE = 12
_D_STATE = 10
# Side of the square, non-overlapping cell patch mapped to a single token.
# _GRID_SIZE must be divisible by _PATCH_SIZE.
_PATCH_SIZE = 2
_PATCH_CELLS = _PATCH_SIZE**2

# --- Dynamical regime -------------------------------------------------------
# The raw CNN rule (Lecun-init weights, one-hot inputs) emits very small logits,
# so at temperature 1.0 with zero bias the softmax transition is almost uniform
# and the automaton degenerates into a near-RNG with NO learnable signal (the
# next cell state is ~independent of the grid). The presets below rescale the
# transition sharpness via the softmax temperature and an identity
# (self-persistence) bias to place the dynamics in a chosen difficulty band.
#
# The "paper" regime reproduces the reference study's tau = 1e-3 (mild
# stochasticity on top of an essentially deterministic rule). The remaining
# presets keep the earlier difficulty ladder so easier / harder regimes can
# still be sampled by tweaking the temperature. The oracle-loss fractions
# quoted below were measured at grid=8, d_state=8, shared-rule seed=42 via
# tools/validate.py; re-measure after changing _GRID_SIZE / _D_STATE / seed.
#
#   regime          (temperature, identity_bias)   ~oracle loss / ln(d_state)
#   "paper"         (1e-3, 0.0)                     near-deterministic (ref. paper)
#   "learnable_25"  (0.5, 2.0)                      ~25%  (clear local structure)
#   "learnable_50"  (0.2, 0.0)                      ~50%  (hard but learnable)
#   "unlearnable"   (1.0, 0.0)                      ~99%  (control / baseline)
# NOTE: to fully reproduce the reference paper's results, _BURN_IN_STEPS must be 0 and
# _SHARED_RULE must be False (each sample draws a fresh random rule). However, we keep the
# `--max-context-length 4_096` instead of lowering to 1_024 (the paper's 8x8 grid).
_REGIMES = {
    "paper": (1e-3, 0.0),
    "learnable_25": (0.5, 2.0),
    "learnable_50": (0.2, 0.0),
    "unlearnable": (1.0, 0.0),
}

# Selected difficulty. Change this string to tune the task; see _REGIMES.
_REGIME = "paper"
_TEMPERATURE, _IDENTITY_BIAS = _REGIMES[_REGIME]
# When True, a single randomly-initialised NCA network is created on first
# call and reused for every sample. This makes the pattern space much easier
# for downstream models to learn (one dynamical system instead of a
# meta-learning problem over all possible NCAs) while retaining diversity
# through stochastic updates and varying initial states. When False each
# sample draws a fresh random rule.
_SHARED_RULE = False
# Fixed seed for the shared network when _SHARED_RULE is True. Change this
# to sample a different rule family from the NCA distribution.
_SHARED_RULE_SEED = 42
# Burn-in steps discarded before recording the trajectory, so the sample
# captures the rule's attractor rather than the random initial condition.
_BURN_IN_STEPS = 0 if _REGIME == "paper" else 4

# Keep only trajectories in the paper's 50%+ gzip-complexity band. The score
# is computed from the exact serialized uint16 sample consumed by training and
# reported by tools/complexity.py (including delimiters and trailing padding).
_MIN_GZIP_COMPLEXITY = 0.5
_MAX_COMPLEXITY_ATTEMPTS = 1_000
_PAPER_CONTEXT_LENGTH = 4_096

if _REGIME == "paper":
    _SHARED_RULE = False

# Per-frame serialization is [OPEN, p_0, p_1, ..., p_{(G/P)^2 - 1}, CLOSE]
# where p_i are 2x2 patch tokens, so the minimum useful context is at least
# this many full frames.
_MIN_FRAMES_PER_SAMPLE = 2

# Reserved slots inside the caller's vocabulary. The vocab is the contiguous
# range [0, vocab_size). ID 0 (PAD_ID from utils) is a dedicated pad token;
# IDs 1 and 2 are the <grid> / </grid> delimiters; patch tokens map to
# vocab[3 : 3 + d_state**_PATCH_CELLS].
_OPEN_IDX = 1
_CLOSE_IDX = 2
_N_RESERVED = 3

# Module-level cache for the shared-rule networks (populated lazily on first
# call when _SHARED_RULE is True), keyed by d_state since a vocab-degraded
# call needs a rule with matching channel count. The lock guards against
# races when multiple dataloader workers trigger the first call concurrently.
_shared_nets: dict[int, _NCANetwork] = {}
_shared_net_lock: threading.Lock | None = None

# Set once a too-small --vocab-size has been warned about, so the warning
# prints only once per run instead of once per sample.
_vocab_size_warned = False


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
        x = x_hwc.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        x = F.pad(x, (1, 1, 1, 1), mode="circular")  # toroidal padding
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        return x.squeeze(0).permute(1, 2, 0)  # (H, W, C)


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
    net: _NCANetwork | None = None,
) -> torch.Tensor:
    """Run the NCA and return a (n_frames, grid_size, grid_size) int tensor (CPU).

    When *net* is None a fresh rule is sampled (original per-sample
    behaviour); otherwise the supplied network is reused across calls so
    every sample evolves under the same dynamical system.
    """
    if net is None:
        net = _make_rule(d_state, gen, device)

    # Initial state sampled on CPU (generator is CPU-bound), then moved.
    init_logits = torch.empty(d_state).normal_(generator=gen)
    probs0 = torch.softmax(init_logits, dim=-1)
    state = (
        torch.multinomial(
            probs0, num_samples=grid_size * grid_size, replacement=True, generator=gen
        )
        .reshape(grid_size, grid_size)
        .to(device)
    )

    burn_in_steps = 0 if _REGIME == "paper" else _BURN_IN_STEPS
    total_steps = burn_in_steps + n_frames
    frames = []
    for t in range(total_steps):
        one_hot = F.one_hot(state, num_classes=d_state).float()  # (H, W, C)
        logits = net(one_hot)  # (H, W, C)
        logits = (logits + one_hot * identity_bias) / temperature
        probs = torch.softmax(logits, dim=-1).reshape(-1, d_state)
        # torch.multinomial requires a CPU generator; sample on CPU and move back.
        state = (
            torch.multinomial(probs.cpu(), num_samples=1, generator=gen)
            .to(device)
            .reshape(grid_size, grid_size)
        )
        if t >= burn_in_steps:
            frames.append(state)

    return torch.stack(frames, dim=0).cpu()  # (n_frames, H, W) on CPU


@register(
    "nca",
    "Neural Cellular Automaton: a small 2D grid of discrete cell states "
    "evolves under a freshly-sampled random local CNN rule. "
    "The rollout trajectory is flattened time-major / row-major into "
    "non-overlapping 2x2 cell-patch tokens (patch vocab d_state**4). "
    "Produces locally-coherent, globally-varying "
    "patterns reminiscent of Lenia / Game-of-Life dynamics.",
)
def gen_nca(vocab: list[int], target_len: int, rng: random.Random) -> list[int]:
    # NCA manages its own token IDs internally (pad=0, <grid>/</grid> = 1/2,
    # patch tokens = 3..3+d_state**4-1) and ignores --vocab-size entirely --
    # exactly like dyck's fixed bracket IDs. d_state never degrades; if the
    # caller's vocab is smaller than needed we just warn once and keep
    # emitting IDs from NCA's own fixed range.
    global _vocab_size_warned
    if _REGIME == "paper" and target_len != _PAPER_CONTEXT_LENGTH:
        raise ValueError(
            f"The NCA paper regime requires target_len={_PAPER_CONTEXT_LENGTH}; got {target_len}."
        )
    if _GRID_SIZE % _PATCH_SIZE != 0:
        raise ValueError(
            f"_GRID_SIZE ({_GRID_SIZE}) must be divisible by _PATCH_SIZE ({_PATCH_SIZE})."
        )
    d_state = _D_STATE
    required = _N_RESERVED + d_state**_PATCH_CELLS
    if len(vocab) < required and not _vocab_size_warned:
        print(
            f"WARNING: nca ignores --vocab-size for its own vocabulary; it "
            f"always needs {required} IDs (pad + 2 delimiters + "
            f"d_state**{_PATCH_CELLS} patch tokens) to keep d_state={d_state} "
            f"fixed, but only {len(vocab)} were supplied. Proceeding with token "
            f"IDs up to {required - 1}, which exceed the requested vocab size.",
            file=sys.stderr,
        )
        _vocab_size_warned = True
    open_tok = _OPEN_IDX
    close_tok = _CLOSE_IDX
    state_vocab = list(range(_N_RESERVED, required))

    grid_size = _GRID_SIZE
    patches_per_side = grid_size // _PATCH_SIZE
    frame_size = patches_per_side**2 + 2  # patch tokens + open + close
    min_target = frame_size * _MIN_FRAMES_PER_SAMPLE
    if target_len < min_target:
        raise ValueError(
            f"NCA requires target_len >= {min_target} "
            f"({_MIN_FRAMES_PER_SAMPLE} frames of {frame_size} tokens at "
            f"grid_size={grid_size}, patch_size={_PATCH_SIZE}); got {target_len}."
        )
    # Emit only complete frames; any leftover slack is padded so the sample
    # still has exactly `target_len` tokens.
    n_frames = target_len // frame_size
    leftover = target_len - n_frames * frame_size

    # shared-rule path: lazily build one network per d_state and reuse it
    use_shared_rule = _SHARED_RULE and _REGIME != "paper"
    if use_shared_rule:
        global _shared_net_lock
        if _shared_net_lock is None:
            _shared_net_lock = threading.Lock()
        with _shared_net_lock:
            if d_state not in _shared_nets:
                shared_gen = torch.Generator(device="cpu")
                shared_gen.manual_seed(_SHARED_RULE_SEED)
                _shared_nets[d_state] = _make_rule(d_state, shared_gen, _DEVICE)
        net = _shared_nets[d_state]
    else:
        net = None

    p = _PATCH_SIZE
    g = patches_per_side
    weights = d_state ** torch.arange(p * p - 1, -1, -1, dtype=torch.long)

    for _attempt in range(1, _MAX_COMPLEXITY_ATTEMPTS + 1):
        # Seed each candidate deterministically from the caller's Random
        # instance so rejection sampling remains reproducible.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(rng.getrandbits(63))

        frames = _rollout(
            n_frames=n_frames,
            grid_size=grid_size,
            d_state=d_state,
            identity_bias=_IDENTITY_BIAS,
            temperature=_TEMPERATURE,
            gen=gen,
            device=_DEVICE,
            net=net,
        )  # (n_frames, H, W) on CPU

        # Tokenize each frame into non-overlapping cell patches in row-major
        # order, interpreting each patch as one base-d_state integer.
        patch_cells = (
            frames.reshape(n_frames, g, p, g, p)
            .permute(0, 1, 3, 2, 4)
            .reshape(n_frames, g * g, p * p)
        )
        patch_ids = (patch_cells * weights).sum(dim=-1)

        out: list[int] = []
        for frame_patches in patch_ids.tolist():
            out.append(open_tok)
            out.extend(state_vocab[pid] for pid in frame_patches)
            out.append(close_tok)
        if leftover:
            out.extend([PAD_ID] * leftover)

        raw = array("H", out).tobytes()
        complexity = len(gzip.compress(raw, compresslevel=9)) / len(raw)
        if complexity > _MIN_GZIP_COMPLEXITY:
            return out

    shared_hint = " Disable _SHARED_RULE." if use_shared_rule else ""
    raise RuntimeError(
        f"NCA failed to sample a trajectory with gzip complexity > "
        f"{_MIN_GZIP_COMPLEXITY} after {_MAX_COMPLEXITY_ATTEMPTS} attempts."
        f"{shared_hint}"
    )
