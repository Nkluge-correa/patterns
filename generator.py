"""
Generates synthetic structured token sequences using a HuggingFace
`AutoTokenizer` vocabulary. The patterns are inspired by formal
language theory for sequence models.

Available patterns:
    - periodic: Repeating block (e.g. ABCABCABC).
    - palindrome: Mirror symmetry (e.g. ABCCBA).
    - copy: Block duplication (e.g. ABCD ABCD ABCD).
    - reverse: Sequence + reverse with delimiter (e.g. ABCD | DCBA).
    - counting_anbn: Equal counts of two symbols (e.g. AAABBB).
    - counting_anbncn: Equal counts of three symbols (e.g. AAABBBCCC).
    - nested: Recursive palindromic structure (e.g. ABCDDCBA).
    - interleaving: Interleaved patterns (e.g. ABABAB or AABBAABB).
    - permutation_cycle: Cyclic permutations (e.g. ABCD BCDA CDAB DABC).
    - hierarchical: Local + global structure (e.g. ABAB CCCC ABAB).
    - noisy_palindrome: Palindrome with ~10% random corruption.
    - dyck: Single bracket type (e.g. (()())).
    - shuffle_dyck: Multiple interleaving bracket types (e.g. ( [ ) { } ]).
    - random: Uniformly random tokens.
    - identity: Constant repetition of single token (e.g. AAAAAA).
    - composite_mirror_repeat: Repeated palindrome (e.g. ABCCBA ABCCBA).

Every emitted sample has exactly `--max-context-length` token IDs:

    1. The sample is initialized with uniformly-random IDs from the
       filtered vocab (the "background noise").
    2. Multiple instances of a SINGLE pattern type are then spliced into
       the background at non-overlapping positions, separated by
       variable-length random gaps. The number of instances is whatever
       greedily fits; their individual lengths are drawn from
       [length_min, length_max].
    3. Each sample contains one pattern type only.

Output:
    * `--debug`: prints one composed sample per pattern (truncated for
      readability) and exits.
    * otherwise: streams one JSON record per line to one or more sharded
      `.jsonl` files. Each shard is capped at `--max-tokens-per-shard`
      tokens (default 100_000_000) so very large datasets do not blow up
      memory or single-file size. The base name supplied via `--output`
      gets a `.NNNN.jsonl` suffix per shard (e.g. `patterns.0000.jsonl`,
      `patterns.0001.jsonl`, ...). Each line has:
        - `input_ids` : list[int]  (the full max-context-length vector)
        - `metadata`  : {pattern_type, vocab_size, max_context_length,
                         range, n_insertions, insertions:[{start,length}]}

Usage:
    python generator.py \
        --tokenizer gpt2 \
        --mode tokens \
        --token-filter all \
        --max-context-length 64 \
        --length-min 2 --length-max 16 \
        --samples-per-pattern 10 \
        --output patterns.jsonl \
        --debug
"""

import argparse
import gzip
import json
import os
import random
import sys
import time
from typing import Callable, Dict, List, Tuple

from transformers import AutoTokenizer


# Vocabulary handling
def get_filtered_vocab(
    tokenizer,
    multi_char_only: bool = False,
    single_char_only: bool = False,
) -> List[int]:
    """Return token IDs from the tokenizer vocab, excluding special tokens.

    Parameters
    ----------
    multi_char_only : keep only IDs that decode to >= 2 visible characters.
    single_char_only : keep only IDs that decode to exactly 1 visible character.

    Returns
    -------
    list[int] : token IDs that pass the filters
    """
    special_ids = set(tokenizer.all_special_ids or [])
    vocab = tokenizer.get_vocab()  # token_string -> id
    kept: List[int] = []
    for _, tid in vocab.items():
        if tid in special_ids:
            continue
        # Use `decode` so that BPE/byte-level prefixes (e.g. "Ġ", or "_") are
        # resolved to the actual user-visible string.
        decoded = tokenizer.decode([tid], skip_special_tokens=True).strip()
        if not decoded:
            continue
        if multi_char_only and len(decoded) < 2:
            continue
        if single_char_only and len(decoded) != 1:
            continue
        kept.append(tid)
    return kept


def sample_distinct(vocab: List[int], k: int, rng: random.Random) -> List[int]:
    """Sample `k` distinct IDs (falls back to with-replacement if vocab too small)."""
    if k <= len(vocab):
        return rng.sample(vocab, k)
    return [rng.choice(vocab) for _ in range(k)]


def _pad_to(out: List[int], target_len: int,
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


# Pattern library
# Each generator has the signature: (vocab, target_len, rng) -> list[int]
# The returned list length is approximately `target_len` (rounding may
# occur for patterns whose internal structure imposes parity constraints,
# e.g. A^n B^n requires an even total length).
PATTERNS: Dict[str, Tuple[str, Callable]] = {}


def _register(name: str, description: str):
    def deco(fn):
        PATTERNS[name] = (description, fn)
        return fn
    return deco


@_register(
    "periodic",
    "Repeating block of length p, e.g. ABCABCABC.",
)
def gen_periodic(vocab, target_len, rng):
    period = rng.randint(2, max(2, min(6, target_len // 2)))
    block = sample_distinct(vocab, period, rng)
    reps = max(2, target_len // period) + 1  # over-generate, then trim
    return _pad_to((block * reps), target_len, vocab, rng)


@_register(
    "palindrome",
    "Mirror symmetry: seq + reverse(seq), e.g. ABCCBA.",
)
def gen_palindrome(vocab, target_len, rng):
    half = max(1, target_len // 2)
    seq = [rng.choice(vocab) for _ in range(half)]
    if target_len % 2:  # odd-length palindrome with a center token
        out = seq + [rng.choice(vocab)] + seq[::-1]
    else:
        out = seq + seq[::-1]
    return _pad_to(out, target_len, vocab, rng)


@_register(
    "copy",
    "Duplication of a block, e.g. ABCD ABCD ABCD.",
)
def gen_copy(vocab, target_len, rng):
    reps = rng.choice([2, 3]) if target_len >= 6 else 2
    block_len = max(1, target_len // reps)
    block = [rng.choice(vocab) for _ in range(block_len)]
    return _pad_to((block * reps), target_len, vocab, rng)


@_register(
    "reverse",
    "Source followed by its reverse with a delimiter, e.g. ABCD | DCBA. Like a "
    "palindrome but with an explicit boundary token from the vocab.",
)
def gen_reverse(vocab, target_len, rng):
    # Need at least 3 slots: source token, delimiter, mirrored token.
    # For shorter target_len there is no meaningful reverse structure, so
    # we build the smallest valid form (length 3) and the caller pads if
    # necessary -- but in practice target_len >= 2 is enforced upstream.
    effective = max(target_len, 3)
    half = max(1, (effective - 1) // 2)
    seq = [rng.choice(vocab) for _ in range(half)]
    delim = rng.choice(vocab)
    out = seq + [delim] + seq[::-1]
    return _pad_to(out, target_len, vocab, rng)


@_register(
    "counting_anbn",
    "A^n B^n: equal counts of two symbols, e.g. AAABBB.",
)
def gen_anbn(vocab, target_len, rng):
    n = max(1, target_len // 2)
    a, b = sample_distinct(vocab, 2, rng)
    # When target_len is odd, [a]*n + [b]*n is one short; pad with a random
    # token so the structural prefix A^n B^n is preserved exactly.
    return _pad_to([a] * n + [b] * n, target_len, vocab, rng)


@_register(
    "counting_anbncn",
    "A^n B^n C^n: equal counts of three symbols, e.g. AAABBBCCC. A context-"
    "sensitive (mildly beyond CFG) counting task.",
)
def gen_anbncn(vocab, target_len, rng):
    n = max(1, target_len // 3)
    a, b, c = sample_distinct(vocab, 3, rng)
    return _pad_to([a] * n + [b] * n + [c] * n, target_len, vocab, rng)


@_register(
    "nested",
    "Recursive palindromic structure from CFG S -> a S a | epsilon, e.g. "
    "ABCDDCBA.",
)
def gen_nested(vocab, target_len, rng):
    depth = max(1, target_len // 2)
    seq = sample_distinct(vocab, depth, rng)
    out = seq + seq[::-1]
    if target_len % 2:
        out.insert(depth, rng.choice(vocab))
    return _pad_to(out, target_len, vocab, rng)


@_register(
    "interleaving",
    "Interleaved tokens: ABABAB or AABBAABB.",
)
def gen_interleaving(vocab, target_len, rng):
    a, b = sample_distinct(vocab, 2, rng)
    style = rng.choice(["alt", "block"])
    if style == "alt":
        out = [a, b] * (target_len // 2 + 1)
    else:
        out = [a, a, b, b] * (target_len // 4 + 1)
    return _pad_to(out, target_len, vocab, rng)


@_register(
    "permutation_cycle",
    "Cyclic permutations of a base block, e.g. ABCD BCDA CDAB DABC.",
)
def gen_permutation_cycle(vocab, target_len, rng):
    k = rng.randint(2, max(2, min(5, target_len // 2)))
    base = sample_distinct(vocab, k, rng)
    out: List[int] = []
    i = 0
    while len(out) < target_len:
        out.extend(base[i % k:] + base[: i % k])
        i += 1
    return _pad_to(out, target_len, vocab, rng)


@_register(
    "hierarchical",
    "Local + global structure mixed, e.g. ABAB CCCC ABAB.",
)
def gen_hierarchical(vocab, target_len, rng):
    third = max(2, target_len // 3)
    a, b, c = sample_distinct(vocab, 3, rng)
    block_ab = ([a, b] * ((third // 2) + 1))[:third]
    block_c = [c] * third
    out = block_ab + block_c + block_ab
    return _pad_to(out, target_len, vocab, rng)


@_register(
    "noisy_palindrome",
    "Palindrome with a small fraction of random corruptions.",
)
def gen_noisy_palindrome(vocab, target_len, rng):
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


@_register(
    "dyck",
    "Dyck-1: properly balanced brackets of a single type, e.g. (()()).",
)
def gen_dyck(vocab, target_len, rng):
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


@_register(
    "shuffle_dyck",
    "Typed Dyck language (Dyck-k): k independent bracket types whose open/close "
    "tokens may interleave freely, e.g. ( [ ) { } ].",
)
def gen_shuffle_dyck(vocab, target_len, rng,
                     k: int = 3, p_open: float = 0.5, max_depth: int = 8):
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


@_register(
    "random",
    "Uniformly random token IDs drawn (with replacement) from the filtered "
    "vocabulary. Serves as an unstructured baseline / control for comparing "
    "against the structured patterns.",
)
def gen_random(vocab, target_len, rng):
    return [rng.choice(vocab) for _ in range(target_len)]


@_register(
    "identity",
    "Constant repetition of a single token, e.g. AAAAAA. The simplest possible "
    "structure: zero entropy, infinite locality. Useful as a degenerate floor "
    "that any sequence model should fit trivially.",
)
def gen_identity(vocab, target_len, rng):
    tok = rng.choice(vocab)
    return [tok] * target_len


@_register(
    "composite_mirror_repeat",
    "Multi-rule composition: a small palindrome repeated, e.g. ABCCBA ABCCBA. "
    "Tests combining symmetry and periodicity.",
)
def gen_composite(vocab, target_len, rng):
    half = max(1, target_len // 4)
    seq = [rng.choice(vocab) for _ in range(half)]
    palin = seq + seq[::-1]
    reps = max(2, target_len // max(1, len(palin))) + 1  # over-generate
    return _pad_to((palin * reps), target_len, vocab, rng)


# Sample composition: random background + multiple pattern insertions
def compose_sample(pattern_name: str,
                   pattern_fn: Callable,
                   vocab: List[int],
                   max_context_length: int,
                   length_min: int,
                   length_max: int,
                   rng: random.Random,
                   signal_floor: float = 0.5) -> Tuple[List[int], List[Dict]]:
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
    if pattern_name in ("dyck", "shuffle_dyck"):
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



def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tokenizer",
        required=True,
        help="HuggingFace tokenizer name or local path (e.g. 'gpt2').",
    )
    ap.add_argument(
        "--mode",
        choices=["ids", "tokens"],
        default="ids",
        help="Display mode used in --debug. The JSONL output always stores "
             "integer token IDs (per spec).",
    )
    ap.add_argument(
        "--token-filter",
        choices=["all", "multi", "single"],
        default="all",
        help="Restrict the vocab to tokens whose decoded form is multi-char, "
             "single-char, or any (default).",
    )
    ap.add_argument("--max-context-length", type=int, default=32)
    ap.add_argument("--length-min", type=int, default=2)
    ap.add_argument("--length-max", type=int, default=16)
    ap.add_argument("--samples-per-pattern", type=int, default=100)
    ap.add_argument(
        "--output",
        default="patterns.jsonl",
        help="Base output path. A shard index is inserted before the .jsonl "
             "extension, e.g. 'patterns.jsonl' -> 'patterns.0000.jsonl'.",
    )
    ap.add_argument(
        "--max-tokens-per-shard",
        type=int,
        default=100_000_000,
        help="Maximum total token IDs per output shard before rolling over "
             "to a new file.",
    )
    ap.add_argument(
        "--gzip",
        action="store_true",
        help="Compress each shard with gzip on the fly. Roughly 3-5x smaller "
             "output. Adds '.gz' to each shard's filename.",
    )
    ap.add_argument(
        "--lean-metadata",
        action="store_true",
        help="Drop the verbose 'insertions' list from per-record metadata to "
             "shrink output (still keeps n_insertions, pattern_type, etc.).",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Print a progress line every N samples written (0 = disabled).",
    )
    ap.add_argument(
        "--signal-floor",
        type=float,
        default=0.5,
        help="Fraction of each sample's context that must be covered by the "
             "repeated pattern (the 'signal'). Default 0.5. Allowed range: "
             "[0.10, 0.90]. Values < 0.5 or > 0.8 emit a warning. Does not "
             "apply to dyck / shuffle_dyck (always 100%).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Print one random sample for every registered pattern and exit.",
    )
    ap.add_argument(
        "--only-random",
        action="store_true",
        help="Generate only the unstructured 'random' pattern (skip all "
             "structured patterns). Useful for building a baseline corpus.",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    # Validate length range against the context budget.
    # length_max may equal max_context_length (a single pattern fills the
    # whole sample); it just may not exceed it.
    if not (2 <= args.length_min <= args.length_max <= args.max_context_length):
        raise SystemExit(
            f"Invalid range: require 2 <= length_min ({args.length_min}) <= "
            f"length_max ({args.length_max}) <= max_context_length "
            f"({args.max_context_length})."
        )

    # Validate signal-floor: hard bounds [0.10, 0.90], warn outside [0.5, 0.8].
    if not (0.10 <= args.signal_floor <= 0.90):
        raise SystemExit(
            f"Invalid --signal-floor ({args.signal_floor}): must be in "
            f"[0.10, 0.90]."
        )
    if args.signal_floor < 0.5:
        print(f"WARNING: --signal-floor={args.signal_floor} is below 0.5; "
              "patterns may be hard to learn (low signal-to-noise ratio).",
              file=sys.stderr)
    elif args.signal_floor > 0.8:
        print(f"WARNING: --signal-floor={args.signal_floor} is above 0.8; "
              "samples will be dominated by the pattern with very little "
              "background noise.", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_ids = get_filtered_vocab(
        tokenizer,
        multi_char_only=(args.token_filter == "multi"),
        single_char_only=(args.token_filter == "single"),
    )
    if len(vocab_ids) < 4:
        raise SystemExit(
            f"Filtered vocabulary too small ({len(vocab_ids)} tokens); "
            "loosen --token-filter."
        )

    rng = random.Random(args.seed)

    # Restrict the active pattern set if --only-random was passed. The
    # 'random' generator returns iid samples, so composing it just yields
    # an unstructured background of the requested length.
    active_patterns = (
        {"random": PATTERNS["random"]} if args.only_random else PATTERNS
    )

    def display(ids: List[int]):
        """Render a sample for human inspection in --debug mode."""
        if args.mode == "tokens":
            return tokenizer.decode(ids)
        return ids

    # DEBUG: one composed sample per pattern, print and exit
    if args.debug:
        debug_log_path = "debug.log"
        with open(debug_log_path, "w", encoding="utf-8") as debug_file:
            def debug_print(msg: str = ""):
                print(msg)
                debug_file.write(msg + "\n")

            debug_print(f"# tokenizer        : {args.tokenizer}")
            debug_print(f"# filtered vocab   : {len(vocab_ids)} tokens "
                        f"(filter={args.token_filter})")
            debug_print(f"# length range     : [{args.length_min}, {args.length_max}]")
            debug_print(f"# max context len  : {args.max_context_length}")
            for name, (desc, fn) in active_patterns.items():
                sample, insertions = compose_sample(
                    name, fn, vocab_ids, args.max_context_length,
                    args.length_min, args.length_max, rng,
                    signal_floor=args.signal_floor,
                )
                debug_print(f"\n[{name}]  ({desc})")
                debug_print(f"  total length   = {len(sample)}")
                debug_print(f"  n_insertions   = {len(insertions)}")
                # Within a sample every insertion is the same instance,
                # so printing the first one is sufficient.
                if insertions:
                    s = insertions[0]["start"]
                    e = s + insertions[0]["length"]
                    debug_print(f"  pattern        = {display(sample[s:e])}")
                debug_print(f"  full sample    = {display(sample)}")
        return

    # Stream samples to disk. Open one shard at a time; when its token
    # budget is exhausted, close it and roll to the next. We write each
    # record immediately (no in-memory accumulation) to keep memory flat
    # for very large datasets.
    base, ext = os.path.splitext(args.output)
    if not ext:
        ext = ".jsonl"
    gz_suffix = ".gz" if args.gzip else ""

    def shard_path(idx: int) -> str:
        return f"{base}.{idx:04d}{ext}{gz_suffix}"

    def open_shard(path: str):
        # 1 MiB write buffer to amortize syscall overhead on big runs.
        if args.gzip:
            return gzip.open(path, "wt", encoding="utf-8", compresslevel=4)
        return open(path, "w", encoding="utf-8", buffering=1024 * 1024)

    # Up-front cost estimate so the user can abort before filling the disk.
    total_samples = args.samples_per_pattern * len(active_patterns)
    total_tokens = total_samples * args.max_context_length
    # Rough bytes/token estimate for JSONL output (digits + comma);
    # gzip typically compresses this ~3-4x for integer text.
    bytes_per_token = 6 if not args.gzip else 2
    est_bytes = total_tokens * bytes_per_token
    est_shards = max(1, -(-total_tokens // args.max_tokens_per_shard))
    print(f"Plan: {total_samples:,} samples x {args.max_context_length} "
          f"tokens = {total_tokens:,} tokens")
    print(f"Estimated output: ~{est_bytes / 1e9:.1f} GB across "
          f"~{est_shards} shard(s){' (gzip)' if args.gzip else ''}.")
    if est_bytes > 50 * 1e9:
        print("WARNING: estimated output exceeds 50 GB. Consider reducing "
              "--samples-per-pattern, enabling --gzip, or using "
              "--lean-metadata. Press Ctrl-C within 5s to abort.",
              file=sys.stderr)
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            raise SystemExit("Aborted by user.")

    n_written = 0
    shard_idx = 0
    shard_tokens = 0
    shard_records = 0
    f = open_shard(shard_path(shard_idx))
    shard_paths = [shard_path(shard_idx)]
    t0 = time.time()
    try:
        for name, (_desc, fn) in active_patterns.items():
            for _ in range(args.samples_per_pattern):
                sample, insertions = compose_sample(
                    name, fn, vocab_ids, args.max_context_length,
                    args.length_min, args.length_max, rng,
                    signal_floor=args.signal_floor,
                )
                # Roll to a new shard if adding this sample would exceed
                # the per-shard token budget (and the current shard is
                # non-empty -- never produce an empty shard).
                if (shard_records > 0 and
                        shard_tokens + len(sample) > args.max_tokens_per_shard):
                    f.close()
                    print(f"  shard {shard_path(shard_idx)}: "
                          f"{shard_records} records, {shard_tokens} tokens")
                    shard_idx += 1
                    shard_tokens = 0
                    shard_records = 0
                    f = open_shard(shard_path(shard_idx))
                    shard_paths.append(shard_path(shard_idx))

                meta = {
                    "pattern_type": name,
                    "vocab_size": len(vocab_ids),
                    "max_context_length": args.max_context_length,
                    "range": [args.length_min, args.length_max],
                    "n_insertions": len(insertions),
                }
                if not args.lean_metadata:
                    meta["insertions"] = insertions
                record = {"input_ids": sample, "metadata": meta}
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
                shard_tokens += len(sample)
                shard_records += 1
                n_written += 1

                if (args.progress_every and
                        n_written % args.progress_every == 0):
                    elapsed = time.time() - t0
                    rate = n_written / elapsed if elapsed > 0 else 0.0
                    pct = 100.0 * n_written / max(1, total_samples)
                    print(f"  progress: {n_written:,}/{total_samples:,} "
                          f"({pct:.1f}%) at {rate:,.0f} samples/s, "
                          f"shard={shard_idx} "
                          f"shard_tokens={shard_tokens:,}")
    finally:
        f.close()
        print(f"  shard {shard_path(shard_idx)}: "
              f"{shard_records} records, {shard_tokens} tokens")

    print(f"Wrote {n_written} samples across {len(active_patterns)} patterns "
          f"to {len(shard_paths)} shard(s).")


if __name__ == "__main__":
    main()
