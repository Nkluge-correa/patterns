"""Mixer pattern generator.

Fills the context with consecutive non-overlapping segments drawn from
different pattern types, separated by the reserved PAD_ID (0) token.
"""

import random

from registry import register
from utils import PAD_ID, pad_to

_MIXER_EXCLUDE = frozenset({"dyck", "shuffle_dyck", "random", "mixer", "nca", "identity"})
_MIXER_MIN_SEGMENT_LEN = 12

# Relative "hardness" weight per pattern name, used only to bias how the
# leftover length budget (everything beyond each segment's guaranteed
# _MIXER_MIN_SEGMENT_LEN floor) is distributed across segments. Patterns with
# higher weight tend to receive a larger share of the context, which raises a
# sample's resistance to compression (gzip complexity). Weights are derived
# from each pattern's measured standalone gzip complexity.
_MIXER_HARDNESS_WEIGHTS: dict[str, float] = {
    "composite_mirror_repeat": 9,
    "copy": 8,
    "counting_anbncn": 2.5,
    "counting_anbn": 2,
    "permutation_cycle": 1,
    "hierarchical": 1,
    "periodic": 0.8,
    "interleaving": 0.8,
}


def _mixer_copy(vocab: list[int], target_len: int, rng: random.Random) -> list[int]:
    """Mixer-local variant of the "copy" pattern with a continuous duplication ratio."""
    unique_fraction = rng.uniform(1 / 8, 1 / 3)
    block_len = max(1, round(target_len * unique_fraction))
    reps = max(1, -(-target_len // block_len))  # ceil division
    block = [rng.choice(vocab) for _ in range(block_len)]
    return (block * reps)[:target_len]


@register(
    "mixer",
    "Fills the context with consecutive non-overlapping segments from different "
    "pattern types, separated by the reserved PAD_ID (0) token. Every sample "
    "has a unique structure: pattern types and segment lengths are randomised "
    "(each at least 12 tokens when the target length permits it). dyck, "
    "shuffle_dyck, random, nca, and identity are excluded from the pool.",
)
def gen_mixer(vocab: list[int], target_len: int, rng: random.Random) -> list[int]:
    # Import here to avoid a circular import at module load time; PATTERNS is
    # fully populated by the time any generator is actually called.
    from registry import PATTERNS

    # Use PAD_ID (0) as the segment separator. Since no generator emits
    # PAD_ID as content (only as trailing padding), it's a natural delimiter.
    # Generators already skip ID 0, so we can pass vocab directly.

    candidates = [
        (name, fn) for name, (_desc, fn) in PATTERNS.items() if name not in _MIXER_EXCLUDE
    ]
    # Swap in the mixer-local copy variant (wider reps range; see
    # _mixer_copy docstring) so its heavy hardness weight doesn't collapse
    # every sample into one of only two compressibility regimes.
    candidates = [(name, _mixer_copy if name == "copy" else fn) for name, fn in candidates]
    rng.shuffle(candidates)

    if not candidates:
        # Degenerate fallback: no eligible patterns available.
        return [rng.choice(vocab) for _ in range(target_len)]

    # With n segments there are n-1 separator slots. Each segment gets at
    # least _MIXER_MIN_SEGMENT_LEN tokens when target_len permits it:
    #   min*n + (n-1) <= target_len  =>  n <= (target_len + 1) // (min + 1)
    max_segments = max(1, (target_len + 1) // (_MIXER_MIN_SEGMENT_LEN + 1))
    candidates = candidates[:max_segments]
    n = len(candidates)

    # Tokens available for pattern content after reserving n-1 separators.
    content_len = target_len - (n - 1)

    # Random composition of content_len into n parts. For normal lengths, each
    # segment gets at least _MIXER_MIN_SEGMENT_LEN tokens; for tiny direct calls
    # (target_len < min), a single shorter segment is the only valid option.
    min_segment_len = min(_MIXER_MIN_SEGMENT_LEN, content_len // n)
    lengths = [min_segment_len] * n
    extra = content_len - (min_segment_len * n)
    weights = [_MIXER_HARDNESS_WEIGHTS.get(name, 1) for name, _fn in candidates]
    for _ in range(extra):
        lengths[rng.choices(range(n), weights=weights)[0]] += 1

    result: list[int] = []
    for i, ((_name, fn), seg_len) in enumerate(zip(candidates, lengths, strict=False)):
        if i > 0:
            result.append(PAD_ID)
        segment = fn(vocab, seg_len, rng)
        # Sub-generators pad their own slack with the reserved PAD_ID. Strip
        # that trailing pad so the reserved token never lands inside the body
        # of the mixer sample; the single final pad_to below keeps all padding
        # in the tail where its loss is masked.
        while segment and segment[-1] == PAD_ID:
            segment.pop()
        result.extend(segment)

    return pad_to(result, target_len, vocab, rng)
