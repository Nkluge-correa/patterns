"""Structural pattern generators.

Patterns based on symmetry, repetition, and positional structure:
periodic, palindrome, copy, reverse, nested, interleaving,
permutation_cycle, hierarchical, noisy_palindrome, composite_mirror_repeat.
"""

import random
from typing import List

from registry import register
from utils import pad_to, sample_distinct


@register(
    "periodic",
    "Repeating block of length p, e.g. ABCABCABC.",
)
def gen_periodic(vocab: List[int], target_len: int, rng: random.Random) -> List[int]:
    period = rng.randint(2, max(2, min(6, target_len // 2)))
    block = sample_distinct(vocab, period, rng)
    reps = max(2, target_len // period) + 1  # over-generate, then trim
    return pad_to((block * reps), target_len, vocab, rng)


@register(
    "palindrome",
    "Mirror symmetry: seq + reverse(seq), e.g. ABCCBA.",
)
def gen_palindrome(vocab: List[int], target_len: int, rng: random.Random) -> List[int]:
    half = max(1, target_len // 2)
    seq = [rng.choice(vocab) for _ in range(half)]
    if target_len % 2:  # odd-length palindrome with a center token
        out = seq + [rng.choice(vocab)] + seq[::-1]
    else:
        out = seq + seq[::-1]
    return pad_to(out, target_len, vocab, rng)


@register(
    "copy",
    "Duplication of a block, e.g. ABCD ABCD ABCD.",
)
def gen_copy(vocab: List[int], target_len: int, rng: random.Random) -> List[int]:
    reps = rng.choice([2, 3]) if target_len >= 6 else 2
    block_len = max(1, target_len // reps)
    block = [rng.choice(vocab) for _ in range(block_len)]
    return pad_to((block * reps), target_len, vocab, rng)


@register(
    "reverse",
    "Source followed by its reverse with a delimiter, e.g. ABCD | DCBA. Like a "
    "palindrome but with an explicit boundary token from the vocab.",
)
def gen_reverse(vocab: List[int], target_len: int, rng: random.Random) -> List[int]:
    # Need at least 3 slots: source token, delimiter, mirrored token.
    # For shorter target_len there is no meaningful reverse structure, so
    # we build the smallest valid form (length 3) and the caller pads if
    # necessary -- but in practice target_len >= 2 is enforced upstream.
    effective = max(target_len, 3)
    half = max(1, (effective - 1) // 2)
    seq = [rng.choice(vocab) for _ in range(half)]
    delim = rng.choice(vocab)
    out = seq + [delim] + seq[::-1]
    return pad_to(out, target_len, vocab, rng)


@register(
    "nested",
    "Recursive palindromic structure from CFG S -> a S a | epsilon, e.g. "
    "ABCDDCBA.",
)
def gen_nested(vocab: List[int], target_len: int, rng: random.Random) -> List[int]:
    depth = max(1, target_len // 2)
    seq = sample_distinct(vocab, depth, rng)
    out = seq + seq[::-1]
    if target_len % 2:
        out.insert(depth, rng.choice(vocab))
    return pad_to(out, target_len, vocab, rng)


@register(
    "interleaving",
    "Interleaved tokens: ABABAB or AABBAABB.",
)
def gen_interleaving(vocab: List[int], target_len: int, rng: random.Random) -> List[int]:
    a, b = sample_distinct(vocab, 2, rng)
    style = rng.choice(["alt", "block"])
    if style == "alt":
        out = [a, b] * (target_len // 2 + 1)
    else:
        out = [a, a, b, b] * (target_len // 4 + 1)
    return pad_to(out, target_len, vocab, rng)


@register(
    "permutation_cycle",
    "Cyclic permutations of a base block, e.g. ABCD BCDA CDAB DABC.",
)
def gen_permutation_cycle(vocab: List[int], target_len: int, rng: random.Random) -> List[int]:
    k = rng.randint(2, max(2, min(5, target_len // 2)))
    base = sample_distinct(vocab, k, rng)
    out: List[int] = []
    i = 0
    while len(out) < target_len:
        out.extend(base[i % k:] + base[: i % k])
        i += 1
    return pad_to(out, target_len, vocab, rng)


@register(
    "hierarchical",
    "Local + global structure mixed, e.g. ABAB CCCC ABAB.",
)
def gen_hierarchical(vocab: List[int], target_len: int, rng: random.Random) -> List[int]:
    third = max(2, target_len // 3)
    a, b, c = sample_distinct(vocab, 3, rng)
    block_ab = ([a, b] * ((third // 2) + 1))[:third]
    block_c = [c] * third
    out = block_ab + block_c + block_ab
    return pad_to(out, target_len, vocab, rng)


@register(
    "noisy_palindrome",
    "Palindrome with a small fraction of random corruptions.",
)
def gen_noisy_palindrome(vocab: List[int], target_len: int, rng: random.Random) -> List[int]:
    out = gen_palindrome(vocab, target_len, rng)
    # Roughly 10% corruption, but only when the sequence is long enough to
    # still recognize the underlying palindrome (>= 10 tokens). For shorter
    # sequences we apply no noise to avoid destroying the structure entirely.
    if len(out) >= 10:
        n_noise = max(1, round(len(out) * 0.1))
        for _ in range(n_noise):
            i = rng.randrange(len(out))
            out[i] = rng.choice(vocab)
    return out


@register(
    "composite_mirror_repeat",
    "Multi-rule composition: a small palindrome repeated, e.g. ABCCBA ABCCBA. "
    "Tests combining symmetry and periodicity.",
)
def gen_composite(vocab: List[int], target_len: int, rng: random.Random) -> List[int]:
    half = max(1, target_len // 4)
    seq = [rng.choice(vocab) for _ in range(half)]
    palin = seq + seq[::-1]
    reps = max(2, target_len // max(1, len(palin))) + 1  # over-generate
    return pad_to((palin * reps), target_len, vocab, rng)
