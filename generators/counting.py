"""Counting pattern generators.

Patterns based on symbol counting beyond regular languages:
counting_anbn (A^n B^n) and counting_anbncn (A^n B^n C^n).
"""

import random
from typing import List

from registry import register
from utils import pad_to, sample_distinct


@register(
    "counting_anbn",
    "A^n B^n: equal counts of two symbols, e.g. AAABBB.",
)
def gen_anbn(vocab: List[int], target_len: int, rng: random.Random) -> List[int]:
    n = max(1, target_len // 2)
    a, b = sample_distinct(vocab, 2, rng)
    # When target_len is odd, [a]*n + [b]*n is one short; pad with a random
    # token so the structural prefix A^n B^n is preserved exactly.
    return pad_to([a] * n + [b] * n, target_len, vocab, rng)


@register(
    "counting_anbncn",
    "A^n B^n C^n: equal counts of three symbols, e.g. AAABBBCCC. A context-"
    "sensitive (mildly beyond CFG) counting task.",
)
def gen_anbncn(vocab: List[int], target_len: int, rng: random.Random) -> List[int]:
    n = max(1, target_len // 3)
    a, b, c = sample_distinct(vocab, 3, rng)
    return pad_to([a] * n + [b] * n + [c] * n, target_len, vocab, rng)
