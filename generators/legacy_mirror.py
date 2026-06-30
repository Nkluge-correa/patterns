"""Legacy mirror-symmetry pattern generators — NOT FOR USE.

These generators (palindrome, reverse, nested, noisy_palindrome) were
removed from the active codebase because mirror-symmetry patterns are
unlearnable by causal transformers (see logs/README.md § 2026-06-30).
"""

import random


def gen_palindrome(vocab: list[int], target_len: int, rng: random.Random) -> list[int]:
    """Legacy: mirror symmetry — seq + reverse(seq), e.g. ABCCBA.

    Unlearnable by causal attention (see § 2026-06-30).  Do not use.
    """
    half = max(1, target_len // 2)
    seq = [rng.choice(vocab) for _ in range(half)]
    out = seq + [rng.choice(vocab)] + seq[::-1] if target_len % 2 else seq + seq[::-1]
    # Legacy pad helper not imported; return exact length
    return out[:target_len]


def gen_reverse(vocab: list[int], target_len: int, rng: random.Random) -> list[int]:
    """Legacy: source + delimiter + reverse(source), e.g. ABCD | DCBA.

    Unlearnable by causal attention (see § 2026-06-30).  Do not use.
    """
    effective = max(target_len, 3)
    half = max(1, (effective - 1) // 2)
    seq = [rng.choice(vocab) for _ in range(half)]
    delim = rng.choice(vocab)
    out = seq + [delim] + seq[::-1]
    return out[:target_len]


def gen_nested(vocab: list[int], target_len: int, rng: random.Random) -> list[int]:
    """Legacy: recursive palindromic structure (S → a S a | ε), e.g. ABCDDCBA.

    Unlearnable by causal attention (see § 2026-06-30).  Do not use.
    """
    depth = max(1, target_len // 2)
    # Legacy used sample_distinct; replicate inline to avoid imports
    pool = list(vocab)
    rng.shuffle(pool)
    seq = pool[:depth]
    out = seq + seq[::-1]
    if target_len % 2:
        out.insert(depth, rng.choice(vocab))
    return out[:target_len]


def gen_noisy_palindrome(vocab: list[int], target_len: int, rng: random.Random) -> list[int]:
    """Legacy: palindrome with ~10% random corruptions.

    Unlearnable by causal attention (see § 2026-06-30).  Do not use.
    """
    out = gen_palindrome(vocab, target_len, rng)
    if len(out) >= 10:
        n_noise = max(1, round(len(out) * 0.1))
        for _ in range(n_noise):
            i = rng.randrange(len(out))
            out[i] = rng.choice(vocab)
    return out
