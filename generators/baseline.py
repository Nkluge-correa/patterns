"""Baseline / control pattern generators.

Unstructured and degenerate patterns used as controls:
random (uniform noise) and identity (constant single token).
"""

import random
from typing import List

from registry import register


@register(
    "random",
    "Uniformly random token IDs drawn (with replacement) from the filtered "
    "vocabulary. Serves as an unstructured baseline / control for comparing "
    "against the structured patterns.",
)
def gen_random(vocab: List[int], target_len: int, rng: random.Random) -> List[int]:
    return [rng.choice(vocab) for _ in range(target_len)]


@register(
    "identity",
    "Constant repetition of a single token, e.g. AAAAAA. The simplest possible "
    "structure: zero entropy, infinite locality. Useful as a degenerate floor "
    "that any sequence model should fit trivially.",
)
def gen_identity(vocab: List[int], target_len: int, rng: random.Random) -> List[int]:
    tok = rng.choice(vocab)
    return [tok] * target_len
