"""Dyck language generators.

Patterns based on formal bracket languages:
dyck (Dyck-1, single bracket type) and shuffle_dyck (nested Dyck-k, k typed
bracket types with strict top-of-stack matching).
"""

import random

from registry import register
from utils import PAD_ID, sample_distinct

# When True, dyck uses fixed bracket IDs (open=1, close=2) and shuffle_dyck
# uses 1..2k, while ID 0 is always reserved as the pad token.  When False
# (original behaviour), fresh random distinct bracket IDs are drawn per
# sample from vocab[1:] (ID 0 stays the pad token regardless).
_SHARED_IDS = False


@register(
    "dyck",
    "Dyck-1: properly balanced brackets of a single type, e.g. (()()). "
    "ID 0 is a reserved pad token used for the tail.",
)
def gen_dyck(vocab: list[int], target_len: int, rng: random.Random) -> list[int]:
    # Brackets are drawn from vocab[1:]; PAD_ID (0) is the reserved pad token.
    if _SHARED_IDS:
        open_id, close_id = vocab[1], vocab[2]
    else:
        open_id, close_id = sample_distinct(vocab[1:], 2, rng)

    sequence: list[int] = []
    depth = 0
    # Greedily build the longest balanced word that fits in target_len.
    # Invariant: we always keep enough budget to close every open bracket,
    # i.e. (target_len - len(sequence)) >= depth at all times.
    while len(sequence) < target_len:
        budget = target_len - len(sequence)
        if depth == 0:
            # Only an opener is valid; it needs room for its matching close.
            if budget >= 2:
                sequence.append(open_id)
                depth += 1
            else:
                break  # a lone slot cannot host a balanced token -> pad it
        elif budget >= depth + 2:
            # Enough room to open (and still close everything) -> free choice.
            if rng.random() < 0.5:
                sequence.append(open_id)
                depth += 1
            else:
                sequence.append(close_id)
                depth -= 1
        else:
            # budget in {depth, depth+1}: must close to finish in time.
            sequence.append(close_id)
            depth -= 1

    # Pad the parity remainder (0 or 1 token) so len == target_len exactly.
    sequence.extend([PAD_ID] * (target_len - len(sequence)))
    return sequence


@register(
    "shuffle_dyck",
    "Nested Dyck-k: k bracket types with strict hierarchical matching -- a "
    "closer must match the most recently opened (top-of-stack) bracket type, "
    "e.g. ( [ { } ] ). ID 0 is a reserved pad token used for the tail.",
)
def gen_shuffle_dyck(
    vocab: list[int],
    target_len: int,
    rng: random.Random,
    k: int = 3,
    p_open: float = 0.5,
    max_depth: int = 4,
) -> list[int]:
    # Brackets are drawn from vocab[1:]; PAD_ID (0) is the reserved pad token.
    # We need 2*k distinct bracket IDs plus the pad, i.e. 2*k + 1 IDs total.
    n_needed = 2 * k + 1
    if len(vocab) < n_needed:
        # Degrade gracefully: shrink k to what the vocab supports (minus pad).
        k = max(1, (len(vocab) - 1) // 2)
    if _SHARED_IDS:
        openers = vocab[1 : k + 1]
        closers = vocab[k + 1 : 2 * k + 1]
    else:
        bracket_ids = sample_distinct(vocab[1:], 2 * k, rng)
        openers, closers = bracket_ids[:k], bracket_ids[k:]

    sequence: list[int] = []
    stack: list[int] = []  # open bracket *types*, most-recent on top
    # Greedily build the longest balanced word that fits in target_len.
    # Invariant: (target_len - len(sequence)) >= len(stack) at all times so
    # every open bracket can always be closed within the budget.
    #
    # Unlike a shuffle language, a closer must match the type on top of the
    # stack. This makes the closer *type* a deterministic function of the
    # context (the open brackets seen so far), creating a genuine
    # hierarchical dependency the model has to track to predict it.
    while len(sequence) < target_len:
        budget = target_len - len(sequence)
        depth = len(stack)
        if depth == 0:
            # Only an opener is valid; it needs room for its matching close.
            if budget >= 2:
                b = rng.randrange(k)
                sequence.append(openers[b])
                stack.append(b)
            else:
                break  # a lone slot cannot host a balanced token -> pad it
            continue
        # We may open only if there is room to close the new bracket too and
        # we are under the depth cap; otherwise we are forced to close.
        can_open = budget >= depth + 2 and depth < max_depth
        if can_open and rng.random() < p_open:
            b = rng.randrange(k)
            sequence.append(openers[b])
            stack.append(b)
        else:
            # Must close the most recently opened type (top of stack).
            b = stack.pop()
            sequence.append(closers[b])

    # Pad the parity remainder (0 or 1 token) so len == target_len exactly.
    sequence.extend([PAD_ID] * (target_len - len(sequence)))
    return sequence
