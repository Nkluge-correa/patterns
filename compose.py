"""Sample composition: one whole-context pattern instance per sample.

Every sample is a SINGLE instance of one pattern type that fills the entire
context window. Pattern symbols are drawn fresh per sample (the rule is symbol
invariant), which keeps the task from degenerating into memorizing a fixed
token sequence.
"""

import random
from collections.abc import Callable

from utils import PAD_ID

# Patterns that manage the reserved pad ID (and any other reserved IDs)
# internally and therefore must receive the FULL vocabulary, including ID 0.
# dyck / shuffle_dyck treat vocab[0] as their pad and draw brackets from
# vocab[1:]; nca reserves vocab[0:3] for pad / <grid> / </grid>. Every other
# pattern receives a content vocabulary with the pad ID removed so it can
# never emit the pad token as genuine content.
_FULL_VOCAB_PATTERNS = frozenset({"dyck", "shuffle_dyck", "nca"})


def _content_length(sample: list[int]) -> int:
    """Length of the leading structured region (pad only ever fills the tail)."""
    n = len(sample)
    while n > 0 and sample[n - 1] == PAD_ID:
        n -= 1
    return n


def _compose_sample_once(
    pattern_name: str,
    pattern_fn: Callable,
    vocab: list[int],
    max_context_length: int,
    rng: random.Random,
) -> tuple[list[int], list[dict]]:
    """Build one max-context-length sample from a single pattern instance.

    The generator is invoked once with `target_len == max_context_length`. For
    patterns that manage reserved IDs themselves (dyck / shuffle_dyck / nca)
    the full vocabulary is passed through unchanged; every other pattern gets a
    content vocabulary with the pad ID stripped, so the only occurrences of the
    pad token are the trailing slack appended by `pad_to`.

    Returns
    -------
    sample : list[int]    -- length == max_context_length
    insertions : list[{"start": int, "length": int}]
                          -- a single entry spanning the structured (non-pad)
                             prefix of the sample.
    """
    if pattern_name in _FULL_VOCAB_PATTERNS:
        sample = pattern_fn(vocab, max_context_length, rng)
    else:
        content_vocab = [t for t in vocab if t != PAD_ID]
        if not content_vocab:  # degenerate single-token vocab
            content_vocab = vocab
        sample = pattern_fn(content_vocab, max_context_length, rng)

    insertions = [{"start": 0, "length": _content_length(sample)}]
    return sample, insertions


def compose_sample(
    pattern_name: str,
    pattern_fn: Callable,
    vocab: list[int],
    max_context_length: int,
    rng: random.Random,
) -> tuple[list[int], list[dict]]:
    """Build one max-context-length sample.

    Thin wrapper around `_compose_sample_once`. Patterns are emitted as-is
    (whatever the generator produces).

    If you want to add some sort of filtering or post-processing (e.g., to
    enforce a minimum complexity), this is the place to do it.
    """
    return _compose_sample_once(
        pattern_name=pattern_name,
        pattern_fn=pattern_fn,
        vocab=vocab,
        max_context_length=max_context_length,
        rng=rng,
    )
