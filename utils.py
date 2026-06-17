"""Shared vocabulary and sequence utilities used by all pattern generators."""

import random

# Reserved pad token. ID 0 is never emitted as pattern *content* by any
# generator (with the exception of `mixer`); it is used exclusively to
# fill the trailing slack of a sample (or separate regions in `mixer`)
# so every sequence lands at exactly `target_len`. Its loss MUST be masked
# during training (it carries no learnable signal).
PAD_ID = 0


def get_vocab(vocab_size: int) -> list[int]:
    """Return the full integer vocabulary as a list of IDs.

    The vocabulary is simply the contiguous range [0, vocab_size), which lets
    generators sample token IDs without requiring a tokenizer.

    Parameters
    ----------
    vocab_size : int
        Total number of distinct token IDs.

    Returns
    -------
        list[int] : [0, 1, ..., vocab_size - 1]
        The full vocabulary as a list of token IDs.
    Note
    ----
        The pad ID is always 0.
    """
    return list(range(vocab_size))


def sample_distinct(vocab: list[int], k: int, rng: random.Random) -> list[int]:
    """Sample `k` distinct IDs (falls back to with-replacement if vocab too small)."""
    if k <= len(vocab):
        return rng.sample(vocab, k)
    return [rng.choice(vocab) for _ in range(k)]


def pad_to(out: list[int], target_len: int, vocab: list[int], rng: random.Random) -> list[int]:
    """Ensure `out` has exactly `target_len` tokens.

    Truncates if too long, otherwise pads the tail with the reserved
    `PAD_ID` (0), whose loss must be masked during training. Used so every
    generator returns sequences of the exact requested length, even when the
    underlying structure imposes divisibility constraints (e.g. A^n B^n
    requires an even length). The `vocab` and `rng` arguments are retained for
    backwards-compatible call sites but are no longer used: padding is always
    the reserved pad token so that the slack carries no learnable signal and
    never collides with genuine pattern content.
    """
    if len(out) >= target_len:
        return out[:target_len]
    return out + [PAD_ID] * (target_len - len(out))
