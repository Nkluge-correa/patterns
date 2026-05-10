"""Dyck language generators.

Patterns based on formal bracket languages:
dyck (Dyck-1, single bracket type) and shuffle_dyck (Dyck-k, typed brackets).
"""

import random
from typing import List

from registry import register
from utils import sample_distinct


@register(
    "dyck",
    "Dyck-1: properly balanced brackets of a single type, e.g. (()()).",
)
def gen_dyck(vocab: List[int], target_len: int, rng: random.Random) -> List[int]:
    # Need just 2 distinct vocab IDs: one for open, one for close
    open_id, close_id = sample_distinct(vocab, 2, rng)

    sequence: List[int] = []
    depth = 0
    while len(sequence) < target_len:
        # Must open if everything is closed or randomly choose to open
        if depth == 0:
            sequence.append(open_id)
            depth += 1
        elif rng.random() < 0.5 and depth < target_len // 2:
            sequence.append(open_id)
            depth += 1
        else:
            sequence.append(close_id)
            depth -= 1

    # Close any remaining open brackets to ensure balanced sequence
    while depth > 0:
        sequence.append(close_id)
        depth -= 1

    return sequence[:target_len]


@register(
    "shuffle_dyck",
    "Typed Dyck language (Dyck-k): k independent bracket types whose open/close "
    "tokens may interleave freely, e.g. ( [ ) { } ].",
)
def gen_shuffle_dyck(vocab: List[int], target_len: int, rng: random.Random,
                     k: int = 3, p_open: float = 0.5, max_depth: int = 4) -> List[int]:
    # Need 2*k distinct vocab IDs: indices 0..k-1 are openers, k..2k-1 closers.
    n_needed = 2 * k
    if len(vocab) < n_needed:
        # Degrade gracefully: shrink k to what the vocab supports.
        k = max(1, len(vocab) // 2)
        n_needed = 2 * k
    bracket_ids = sample_distinct(vocab, n_needed, rng)
    openers, closers = bracket_ids[:k], bracket_ids[k:]

    sequence: List[int] = []
    counts = [0] * k  # open-bracket counts per type
    while len(sequence) < target_len:
        depth = sum(counts)
        # Must open if everything is closed.
        if depth == 0:
            b = rng.randrange(k)
            sequence.append(openers[b])
            counts[b] += 1
            continue
        # Force a close at max depth.
        if depth >= max_depth:
            open_types = [i for i, c in enumerate(counts) if c > 0]
            b = rng.choice(open_types)
            sequence.append(closers[b])
            counts[b] -= 1
            continue
        # Otherwise stochastically open or close.
        if rng.random() < p_open:
            b = rng.randrange(k)
            sequence.append(openers[b])
            counts[b] += 1
        else:
            open_types = [i for i, c in enumerate(counts) if c > 0]
            b = rng.choice(open_types)
            sequence.append(closers[b])
            counts[b] -= 1
    return sequence[:target_len]
