"""Sample composition: random background + multiple pattern insertions."""

import random
from typing import Callable, Dict, List, Tuple

# Patterns that fill the entire context with a single call to the
# generator, with no repetition or random background noise spliced
# around them. Dyck patterns must remain a single valid expression;
# `random` is the unstructured baseline and would be artificially
# compressible if a single random block were repeated under
# signal_floor coverage.
_WHOLE_CONTEXT_PATTERNS = frozenset({"dyck", "shuffle_dyck", "random"})


def compose_sample(
    pattern_name: str,
    pattern_fn: Callable,
    vocab: List[int],
    max_context_length: int,
    length_min: int,
    length_max: int,
    rng: random.Random,
    signal_floor: float = 0.5,
) -> Tuple[List[int], List[Dict]]:
    """Build a max-context-length vector with multiple pattern insertions.

    For dyck and shuffle_dyck patterns, the entire sequence is a single valid
    Dyck expression (no random background). For all other patterns, ONE pattern
    instance is generated and then repeated (same exact tokens) at multiple
    non-overlapping positions inside a random-noise background. The number of
    repetitions is chosen so that the pattern occupies at least `signal_floor`
    fraction of the context (default 0.5); this is what makes the pattern
    learnable within a single sample. Different samples will see different
    random pattern instances, but within one sample the pattern is always
    identical.

    Returns
    -------
    sample : list[int]    -- length == max_context_length
    insertions : list[{"start": int, "length": int}]
    """
    # For Dyck patterns, generate the entire sequence as one valid expression.
    if pattern_name in _WHOLE_CONTEXT_PATTERNS:
        sample = pattern_fn(vocab, max_context_length, rng)
        insertions = [{"start": 0, "length": max_context_length}]
        return sample, insertions

    # Generate ONE pattern instance for this sample; we will repeat it.
    plen = rng.randint(length_min, length_max)
    pattern = pattern_fn(vocab, plen, rng)
    plen = len(pattern)  # defensive: trust generator's actual length

    # Decide how many copies to place. We aim for >= signal_floor coverage,
    # but also ensure there is at least one gap token between copies.
    max_copies = max(1, max_context_length // max(1, plen))
    # Number of copies needed for signal_floor coverage (rounded up).
    target_signal_tokens = int(max_context_length * signal_floor + 0.5)
    min_copies_for_signal = -(-target_signal_tokens // plen)  # ceil div
    # Need room for n copies + at least (n-1) gap tokens of 1 each.
    # i.e. n*plen + (n-1) <= max_context_length  =>  n <= (L+1)/(plen+1)
    fits_with_gaps = max(1, (max_context_length + 1) // (plen + 1))
    n_copies = min(max_copies, max(min_copies_for_signal, 1))
    n_copies = min(n_copies, fits_with_gaps)
    n_copies = max(1, n_copies)

    # Distribute the leftover (background) tokens across n_copies + 1 gaps.
    total_signal = n_copies * plen
    total_gap = max_context_length - total_signal
    # n_copies + 1 gap slots (before first, between each pair, after last).
    n_gaps = n_copies + 1
    # Random non-negative integer composition of total_gap into n_gaps parts.
    if n_gaps == 1:
        gaps = [total_gap]
    else:
        # Pick n_gaps-1 cut points uniformly in [0, total_gap].
        cuts = sorted(rng.randint(0, total_gap) for _ in range(n_gaps - 1))
        gaps = [cuts[0]] + [cuts[i] - cuts[i - 1]
                            for i in range(1, n_gaps - 1)] + [total_gap - cuts[-1]]

    # Build the sample: random background filled with repeated pattern copies.
    sample = [rng.choice(vocab) for _ in range(max_context_length)]
    insertions: List[Dict] = []
    cursor = gaps[0]
    for i in range(n_copies):
        sample[cursor:cursor + plen] = pattern
        insertions.append({"start": cursor, "length": plen})
        cursor += plen + gaps[i + 1]

    return sample, insertions
