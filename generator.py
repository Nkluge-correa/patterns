"""
Generates synthetic structured token sequences using a simple integer
vocabulary. The patterns are inspired by formal language theory for
sequence models.

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
    - nca: 2D grid evolved by a random Neural Cellular Automaton rule.
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
    4. For dyck and shuffle_dyck patterns, the entire sample is a single valid
       Dyck expression (no random background).
    5. For mixer, the context is filled with consecutive segments from different 
       pattern types (excluding dyck, shuffle_dyck, and random).
    6. For nca, the entire context is a single NCA rollout flattened into a
       1D token stream (padded to max_context_length if needed).

Output:
    * `--debug`: prints one composed sample per pattern (truncated for
      readability) and exits.
    * otherwise: streams one JSON record per line to per-pattern sharded
      `.jsonl` files inside a subdirectory named after the pattern. Each
      shard is capped at `--max-tokens-per-shard` tokens (default
      100_000_000). The base name supplied via `--output` gets a `.NNNN.jsonl`
      suffix per shard (e.g. `periodic/patterns.0000.jsonl`). Each line has:
        - `input_ids` : list[int]  (the full max-context-length vector)
        - `metadata`  : {pattern_type, vocab_size, max_context_length,
                         range, n_insertions, insertions:[{start,length}]}
          (omitted when --no-metadata is set)

Usage:
    python generator.py \\
        --patterns all \\
        --vocab-size 256 \\
        --max-context-length 1024 \\
        --length-min 2 --length-max 32 \\
        --samples-per-pattern 1000 \\
        --output patterns.jsonl \\
        --output-dir ./data \\
        --max-tokens-per-shard 100000000 \\
        --no-metadata \\
        --signal-floor 0.8 \\
        --min-complexity 0.2 \\
        --max-attempts 100 \\
        --seed 42

Use `--debug` to print one sample per pattern and exit.
Use `--gzip` to compress output shards on the fly (adds .gz suffix).

Note on vocab size:

To fully utilize the token ID space without wasted bits, the vocab size should be a power of 2. 
This ensures that each token ID can be represented in a fixed number of bytes with no unused values. 
Here's a quick reference:

| dtype    | vocab size | bytes/token | why                                         |
|----------|------------|-------------|---------------------------------------------|
| `uint8`  | 256        | 1           | fills all 8 bits; random -> ~1.0 complexity |
| `uint16` | 65536      | 2           | fills all 16 bits exactly                   |
| `uint32` | 4294967296 | 4           | fills all 32 bits — impractical             |

For practical use, **256** is the sweet spot: 1 byte per token, no wasted bits, gzip operates directly on 
the token stream with no encoding artifact, and the vocab is large enough for all patterns.

"""

import argparse
import gzip
import json
import os
import random
import sys
import time
from typing import List

import generators  # noqa: F401 — registers all built-in patterns as a side effect
from compose import compose_sample
from registry import PATTERNS
from utils import get_vocab


def main(args):

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

    if args.vocab_size < 6:
        raise SystemExit(
            f"--vocab-size must be at least 6 (got {args.vocab_size}); "
            "shuffle_dyck requires 2*k=6 distinct token IDs."
        )
    vocab_ids = get_vocab(args.vocab_size)

    rng = random.Random(args.seed)

    # Resolve the active pattern set from --patterns.
    if args.patterns == ["all"]:
        active_patterns = PATTERNS
    else:
        unknown = [p for p in args.patterns if p not in PATTERNS]
        if unknown:
            raise SystemExit(
                f"Unknown pattern(s): {', '.join(unknown)}. "
                f"Available: {', '.join(PATTERNS)}"
            )
        active_patterns = {p: PATTERNS[p] for p in args.patterns}

    def display(ids: List[int]):
        return ids

    # DEBUG: one composed sample per pattern, print and exit
    if args.debug:
        debug_log_path = "debug.log"
        with open(debug_log_path, "w", encoding="utf-8") as debug_file:
            def debug_print(msg: str = ""):
                print(msg)
                debug_file.write(msg + "\n")

            debug_print(f"# vocab size       : {args.vocab_size}")
            debug_print(f"# length range     : [{args.length_min}, {args.length_max}]")
            debug_print(f"# max context len  : {args.max_context_length}")
            for name, (desc, fn) in active_patterns.items():
                sample, insertions = compose_sample(
                    name, fn, vocab_ids, args.max_context_length,
                    args.length_min, args.length_max, rng,
                    signal_floor=args.signal_floor,
                    min_complexity=args.min_complexity,
                    max_attempts=args.max_attempts,
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

    def shard_path(pattern_name: str, idx: int) -> str:
        out_dir = os.path.join(args.output_dir, pattern_name)
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.basename(base)
        return os.path.join(out_dir, f"{stem}.{idx:04d}{ext}{gz_suffix}")

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
              "--lean-metadata. Press Ctrl-C within 60s to abort.",
              file=sys.stderr)
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            raise SystemExit("Aborted by user.")

    n_written = 0
    all_shard_paths: List[str] = []
    t0 = time.time()

    for name, (_desc, fn) in active_patterns.items():
        pattern_samples = 0
        pattern_tokens = 0
        shard_idx = 0
        shard_tokens = 0
        shard_records = 0
        current_path = shard_path(name, shard_idx)
        all_shard_paths.append(current_path)
        f = open_shard(current_path)
        try:
            for _ in range(args.samples_per_pattern):
                sample, insertions = compose_sample(
                    name, fn, vocab_ids, args.max_context_length,
                    args.length_min, args.length_max, rng,
                    signal_floor=args.signal_floor,
                    min_complexity=args.min_complexity,
                    max_attempts=args.max_attempts,
                )
                # Roll to a new shard if adding this sample would exceed
                # the per-shard token budget (and the current shard is
                # non-empty -- never produce an empty shard).
                if (shard_records > 0 and
                        shard_tokens + len(sample) > args.max_tokens_per_shard):
                    f.close()
                    print(f"  shard {current_path}: "
                          f"{shard_records} records, {shard_tokens} tokens")
                    shard_idx += 1
                    shard_tokens = 0
                    shard_records = 0
                    current_path = shard_path(name, shard_idx)
                    all_shard_paths.append(current_path)
                    f = open_shard(current_path)

                if args.no_metadata:
                    record = {"input_ids": sample}
                else:
                    meta = {
                        "pattern_type": name,
                        "vocab_size": args.vocab_size,
                        "max_context_length": args.max_context_length,
                        "range": [args.length_min, args.length_max],
                        "n_insertions": len(insertions),
                        "insertions": insertions,
                    }
                    record = {"input_ids": sample, "metadata": meta}
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
                shard_tokens += len(sample)
                shard_records += 1
                n_written += 1
                pattern_samples += 1
                pattern_tokens += len(sample)

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
            print(f"  shard {current_path}: "
                  f"{shard_records} records, {shard_tokens} tokens")

        # Write per-pattern .metadata YAML file.
        n_shards = shard_idx + 1
        columns = ["input_ids"]
        if not args.no_metadata:
            columns.append("metadata")
        metadata_yaml = (
            f"samples: {pattern_samples}\n"
            f"tokens: {pattern_tokens}\n"
            f"tokens_per_chunk: {args.max_tokens_per_shard}\n"
            f"chunks: {n_shards}\n"
            f"block_size: {args.max_context_length}\n"
            f"columns: {columns}\n"
        )
        meta_path = os.path.join(args.output_dir, name, ".metadata")
        with open(meta_path, "w", encoding="utf-8") as meta_f:
            meta_f.write(metadata_yaml)
        print(f"  metadata: {meta_path}")

    print(f"Wrote {n_written} samples across {len(active_patterns)} patterns "
          f"to {len(all_shard_paths)} shard(s).")


if __name__ == "__main__":
    
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    ap.add_argument(
        "--vocab-size",
        type=int,
        default=256,
        metavar="N",
        help="Number of distinct token IDs. The vocabulary is simply "
             "range(0, N). Must be at least 6 (shuffle_dyck needs 2*k=6 "
             "distinct IDs).",
    )
    ap.add_argument("--max-context-length", type=int, default=32)
    ap.add_argument("--length-min", type=int, default=2)
    ap.add_argument("--length-max", type=int, default=16)
    ap.add_argument("--samples-per-pattern", type=int, default=100)
    ap.add_argument(
        "--output",
        default="patterns.jsonl",
        help="Base output filename. A shard index is inserted before the .jsonl "
             "extension, e.g. 'patterns.jsonl' -> 'patterns.0000.jsonl'. "
             "The directory component is ignored when --output-dir is set.",
    )
    ap.add_argument(
        "--output-dir",
        default="./data",
        metavar="DIR",
        help="Root directory for all output shards. Each pattern writes into "
             "a subdirectory named after the pattern (e.g. DIR/periodic/). "
             "Defaults to the directory component of --output.",
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
        "--no-metadata",
        action="store_true",
        help="Omit metadata entirely from each record. When set, every sample "
             "is stored as {\"input_ids\": list[int]} only, ready for direct "
             "training use.",
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
             "apply to dyck / shuffle_dyck (always 100%%).",
    )
    ap.add_argument(
        "--min-complexity",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="Reject samples whose gzip complexity (compressed / original bytes) "
             "is below this threshold and regenerate until it is met. "
             "Range (0, 1]. Higher values select for less compressible / more "
             "random-looking samples; lower values keep more regular, structured "
             "samples. Default: disabled (no filtering).",
    )
    ap.add_argument(
        "--max-attempts",
        type=int,
        default=100,
        metavar="N",
        help="Maximum number of regeneration attempts per sample when "
             "--min-complexity is set. Raises an error if the threshold "
             "is never met within this budget. Default: 100.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Print one random sample for every registered pattern and exit.",
    )
    ap.add_argument(
        "--patterns",
        nargs="+",
        default=["all"],
        metavar="PATTERN",
        help="One or more pattern names to generate, or 'all' for every "
             "registered pattern (default). Example: --patterns periodic "
             "palindrome dyck.",
    )
    args = ap.parse_args()
    
    main(args)
