"""Shared vocabulary and sequence utilities used by all pattern generators."""

import random
from typing import List


def get_vocab(vocab_size: int) -> List[int]:
    """Return the full integer vocabulary as a list of IDs.

    The vocabulary is simply the contiguous range [0, vocab_size), which lets
    generators sample token IDs without requiring a tokenizer.

    Parameters
    ----------
    vocab_size : total number of distinct token IDs.

    Returns
    -------
    list[int] : [0, 1, ..., vocab_size - 1]
    """
    return list(range(vocab_size))


def sample_distinct(vocab: List[int], k: int, rng: random.Random) -> List[int]:
    """Sample `k` distinct IDs (falls back to with-replacement if vocab too small)."""
    if k <= len(vocab):
        return rng.sample(vocab, k)
    return [rng.choice(vocab) for _ in range(k)]


def pad_to(out: List[int], target_len: int,
           vocab: List[int], rng: random.Random) -> List[int]:
    """Ensure `out` has exactly `target_len` tokens.

    Truncates if too long, otherwise pads the tail with uniformly-random IDs
    drawn from `vocab`. Used so every generator returns sequences of the
    exact requested length, even when the underlying structure imposes
    divisibility constraints (e.g. A^n B^n requires an even length).
    """
    if len(out) >= target_len:
        return out[:target_len]
    return out + [rng.choice(vocab) for _ in range(target_len - len(out))]
