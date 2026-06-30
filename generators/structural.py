"""Structural pattern generators.

Patterns based on symmetry, repetition, and positional structure:
periodic, copy, interleaving, permutation_cycle, hierarchical,
composite_mirror_repeat.
"""

import random

from registry import register
from utils import pad_to, sample_distinct


@register(
    "periodic",
    "Repeating block of length p, e.g. ABCABCABC.",
)
def gen_periodic(vocab: list[int], target_len: int, rng: random.Random) -> list[int]:
    period = rng.randint(2, max(2, min(6, target_len // 2)))
    block = sample_distinct(vocab, period, rng)
    reps = max(2, target_len // period) + 1  # over-generate, then trim
    return pad_to((block * reps), target_len, vocab, rng)


@register(
    "copy",
    "Duplication of a block, e.g. ABCD ABCD ABCD.",
)
def gen_copy(vocab: list[int], target_len: int, rng: random.Random) -> list[int]:
    reps = rng.choice([2, 3]) if target_len >= 6 else 2
    block_len = max(1, target_len // reps)
    block = [rng.choice(vocab) for _ in range(block_len)]
    return pad_to((block * reps), target_len, vocab, rng)


@register(
    "interleaving",
    "Interleaved tokens: ABABAB or AABBAABB.",
)
def gen_interleaving(vocab: list[int], target_len: int, rng: random.Random) -> list[int]:
    a, b = sample_distinct(vocab, 2, rng)
    style = rng.choice(["alt", "block"])
    out = [a, b] * (target_len // 2 + 1) if style == "alt" else [a, a, b, b] * (target_len // 4 + 1)
    return pad_to(out, target_len, vocab, rng)


@register(
    "permutation_cycle",
    "Cyclic permutations of a base block, e.g. ABCD BCDA CDAB DABC.",
)
def gen_permutation_cycle(vocab: list[int], target_len: int, rng: random.Random) -> list[int]:
    k = rng.randint(2, max(2, min(5, target_len // 2)))
    base = sample_distinct(vocab, k, rng)
    out: list[int] = []
    i = 0
    while len(out) < target_len:
        out.extend(base[i % k :] + base[: i % k])
        i += 1
    return pad_to(out, target_len, vocab, rng)


@register(
    "hierarchical",
    "Local + global structure mixed, e.g. ABAB CCCC ABAB.",
)
def gen_hierarchical(vocab: list[int], target_len: int, rng: random.Random) -> list[int]:
    third = max(2, target_len // 3)
    a, b, c = sample_distinct(vocab, 3, rng)
    block_ab = ([a, b] * ((third // 2) + 1))[:third]
    block_c = [c] * third
    out = block_ab + block_c + block_ab
    return pad_to(out, target_len, vocab, rng)


@register(
    "composite_mirror_repeat",
    "Multi-rule composition: a small symmetric block repeated, e.g. ABCCBA ABCCBA. "
    "Tests combining symmetry and periodicity.",
)
def gen_composite(vocab: list[int], target_len: int, rng: random.Random) -> list[int]:
    half = max(1, target_len // 4)
    seq = [rng.choice(vocab) for _ in range(half)]
    palin = seq + seq[::-1]
    reps = max(2, target_len // max(1, len(palin))) + 1  # over-generate
    return pad_to((palin * reps), target_len, vocab, rng)
