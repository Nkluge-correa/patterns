"""Mixer pattern generator.

Fills the context with consecutive non-overlapping segments drawn from
different pattern types, separated by a single separator token.
"""

import random
from typing import List

from registry import register
from utils import pad_to

_MIXER_EXCLUDE = frozenset({"dyck", "shuffle_dyck", "random", "mixer", "nca"})
_MIXER_MIN_SEGMENT_LEN = 6


@register(
    "mixer",
    "Fills the context with consecutive non-overlapping segments from different "
    "pattern types, separated by a single separator token. Both the separator "
    "token and the pattern types are chosen fresh each call, so every sample "
    "has a unique structure. Segment lengths are randomised (each at least 6 "
    "tokens when the target length permits it). dyck, shuffle_dyck, random, "
    "and nca are excluded from the pool.",
)
def gen_mixer(vocab: List[int], target_len: int, rng: random.Random) -> List[int]:
    # Import here to avoid a circular import at module load time; PATTERNS is
    # fully populated by the time any generator is actually called.
    from registry import PATTERNS

    # One separator token for the whole sample, drawn fresh each call.
    sep_token = rng.choice(vocab)
    # Patterns must not emit the separator, so give them a reduced vocab.
    pattern_vocab = [t for t in vocab if t != sep_token]
    if not pattern_vocab:
        # Degenerate: vocab has only one token, fall back to full vocab.
        pattern_vocab = vocab

    candidates = [
        (name, fn)
        for name, (_desc, fn) in PATTERNS.items()
        if name not in _MIXER_EXCLUDE
    ]
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
    for _ in range(extra):
        lengths[rng.randrange(n)] += 1

    result: List[int] = []
    for i, ((_name, fn), seg_len) in enumerate(zip(candidates, lengths)):
        if i > 0:
            result.append(sep_token)
        segment = fn(pattern_vocab, seg_len, rng)
        result.extend(segment)

    return pad_to(result, target_len, pattern_vocab, rng)
