"""Dyck language generators.

Patterns based on formal bracket languages:
dyck (Dyck-1, single bracket type) and shuffle_dyck (nested Dyck-k, k typed
bracket types with strict top-of-stack matching).
"""

import random
import sys

from registry import register
from utils import PAD_ID, sample_distinct

# When True, dyck uses fixed bracket IDs (open=1, close=2) and shuffle_dyck
# uses 1..2k, while ID 0 is always reserved as the pad token.  When False
# (original behaviour), fresh random distinct bracket IDs are drawn per
# sample from vocab[1:] (ID 0 stays the pad token regardless).
_SHARED_IDS = False

# The default number of bracket types for shuffle_dyck.
K = 8

# Set once a too-small --vocab-size has been warned about for each pattern,
# so the warning prints only once per run instead of once per sample.
_dyck_vocab_warned = False
_shuffle_dyck_vocab_warned = False


@register(
    "dyck",
    "Dyck-1: properly balanced brackets of a single type, e.g. (()()). "
    "ID 0 is a reserved pad token used for the tail.",
)
def gen_dyck(vocab: list[int], target_len: int, rng: random.Random) -> list[int]:
    # dyck manages its own bracket IDs internally and ignores --vocab-size
    # entirely when _SHARED_IDS is set: open=1, close=2, regardless of the
    # caller's vocab. If the supplied vocab is smaller than needed we just
    # warn once and keep using the fixed IDs anyway.
    global _dyck_vocab_warned
    if _SHARED_IDS:
        required = 3
        if len(vocab) < required and not _dyck_vocab_warned:
            print(
                f"WARNING: dyck ignores --vocab-size for its own vocabulary; it "
                f"always needs {required} IDs (pad + 2 bracket IDs), but only "
                f"{len(vocab)} were supplied. Proceeding with fixed bracket IDs "
                f"1 and 2, which exceed the requested vocab size.",
                file=sys.stderr,
            )
            _dyck_vocab_warned = True
        open_id, close_id = 1, 2
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
    f"Nested Dyck-k: k bracket types (default k={K}) with strict hierarchical "
    "matching -- a closer must match the most recently opened (top-of-stack) "
    "bracket type, e.g. ( [ { } ] ). Open/close is a fair coin (p_open=0.5) "
    "with no depth cap, yielding a harmonic distribution over depths "
    "(Hu et al. 2025). ID 0 is a reserved pad token used for the tail.",
)
def gen_shuffle_dyck(
    vocab: list[int],
    target_len: int,
    rng: random.Random,
    k: int = K,
    p_open: float = 0.5,
    max_depth: int | None = None,
) -> list[int]:
    # shuffle_dyck manages its own bracket IDs internally and ignores
    # --vocab-size entirely when _SHARED_IDS is set: openers=1..k,
    # closers=k+1..2k, regardless of the caller's vocab. If the supplied
    # vocab is smaller than needed we just warn once and keep using the
    # fixed IDs anyway (k is never degraded).
    global _shuffle_dyck_vocab_warned
    n_needed = 2 * k + 1
    if _SHARED_IDS:
        if len(vocab) < n_needed and not _shuffle_dyck_vocab_warned:
            print(
                f"WARNING: shuffle_dyck ignores --vocab-size for its own "
                f"vocabulary; it always needs {n_needed} IDs (pad + 2*k bracket "
                f"IDs, k={k}), but only {len(vocab)} were supplied. Proceeding "
                f"with fixed bracket IDs 1..{2 * k}, which exceed the requested "
                f"vocab size.",
                file=sys.stderr,
            )
            _shuffle_dyck_vocab_warned = True
        openers = list(range(1, k + 1))
        closers = list(range(k + 1, 2 * k + 1))
    else:
        if len(vocab) < n_needed:
            # Degrade gracefully: shrink k to what the vocab supports (minus pad).
            k = max(1, (len(vocab) - 1) // 2)
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
    #
    # With p_open=0.5 and max_depth=None the depth performs a fair random
    # walk reflected at 0, which yields a harmonic distribution over depths
    # -- the corpus construction used by Hu et al. (2025).
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
        # we are under the (optional) depth cap; otherwise we must close.
        can_open = budget >= depth + 2 and (max_depth is None or depth < max_depth)
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
