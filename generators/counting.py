"""Counting pattern generators.

Patterns based on symbol counting beyond regular languages:
counting_anbn (A^n B^n) and counting_anbncn (A^n B^n C^n).

The context is tiled with consecutive segments whose run length `n` is drawn
fresh PER SEGMENT. Randomising `n` makes this a genuine counting task: because
the switch point is unpredictable, a model must actually count the run it has
seen so far to know when the next symbol flips, rather than memorising a fixed
position. Any trailing slack too small for one more segment is filled with the
reserved, loss-masked pad token.
"""

import random

from registry import register
from utils import pad_to, sample_distinct

# Per-segment run length is drawn uniformly from [_RUN_MIN, _RUN_MAX] (capped
# by the remaining budget). Short runs keep symbol-switch boundaries frequent
# and varied so the counting signal is dense across the context.
_RUN_MIN = 1
_RUN_MAX = 16


@register(
    "counting_anbn",
    "A^n B^n: equal runs of two symbols with a fresh random n per segment, "
    "tiled to fill the context, e.g. AABB AAABBB AB. Randomising n forces the "
    "model to count rather than learn a fixed switch position.",
)
def gen_anbn(vocab: list[int], target_len: int, rng: random.Random) -> list[int]:
    a, b = sample_distinct(vocab, 2, rng)
    out: list[int] = []
    while True:
        remaining = target_len - len(out)
        if remaining < 2 * _RUN_MIN:  # no room for another full segment
            break
        n = rng.randint(_RUN_MIN, max(_RUN_MIN, min(_RUN_MAX, remaining // 2)))
        out.extend([a] * n + [b] * n)
    # Tail too short for a segment is padded with the reserved pad token.
    return pad_to(out, target_len, vocab, rng)


@register(
    "counting_anbncn",
    "A^n B^n C^n: equal runs of three symbols with a fresh random n per "
    "segment, tiled to fill the context, e.g. AABBCC ABC. A context-sensitive "
    "(mildly beyond CFG) counting task; randomising n forces genuine counting.",
)
def gen_anbncn(vocab: list[int], target_len: int, rng: random.Random) -> list[int]:
    a, b, c = sample_distinct(vocab, 3, rng)
    out: list[int] = []
    while True:
        remaining = target_len - len(out)
        if remaining < 3 * _RUN_MIN:  # no room for another full segment
            break
        n = rng.randint(_RUN_MIN, max(_RUN_MIN, min(_RUN_MAX, remaining // 3)))
        out.extend([a] * n + [b] * n + [c] * n)
    # Tail too short for a segment is padded with the reserved pad token.
    return pad_to(out, target_len, vocab, rng)
