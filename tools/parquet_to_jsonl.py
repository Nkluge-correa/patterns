"""Convert Parquet files to JSONL format.

Reads one or more directories of Parquet files and writes the records
as line-delimited JSON (.jsonl), optionally sharded and/or gzip-compressed.

Usage:
    # Convert a single directory of parquet shards to one JSONL file
    python tools/parquet_to_jsonl.py \\
        --paths ./data/fineweb-edu/tokenized \\
        --output ./data/fineweb-edu/tokenized.jsonl

    # Convert with per-directory output, gzip compressed
    python tools/parquet_to_jsonl.py \\
        --paths ./data/patterns ./data/fineweb-edu/packed \\
        --output-dir ./data/converted \\
        --gzip

    # Shard the output (max 100k records per file)
    python tools/parquet_to_jsonl.py \\
        --paths ./data/fineweb-edu/packed \\
        --output ./data/packed.jsonl \\
        --max-records-per-shard 100000

    # Select only specific columns
    python tools/parquet_to_jsonl.py \\
        --paths ./data/fineweb-edu/packed \\
        --output ./data/packed_input_ids.jsonl \\
        --columns input_ids

    # Flatten array-type columns: write each element as its own record
    python tools/parquet_to_jsonl.py \\
        --paths ./data/fineweb-edu/packed \\
        --output ./data/packed_flattened.jsonl \\
        --flatten input_ids

    # Read a single parquet file directly
    python tools/parquet_to_jsonl.py \\
        --paths ./data/fineweb-edu/text/000_00000.parquet \\
        --output ./data/text.jsonl

Output:
    - By default, writes sharded .jsonl files with 100,000 records per shard
      (or .jsonl.gz if --gzip is set).
    - With --max-records-per-shard 0, writes a single unsplit file.
    - With --output-dir, creates one file (or shard set) per input directory
      inside the output directory.
"""

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import List, Optional
import numpy as np


# File I/O helpers
def _collect_parquet(paths: List[str]) -> List[Path]:
    """Resolve a list of paths to concrete .parquet files.

    - If a path points to a single .parquet file, include it directly.
    - If a path points to a directory, collect all .parquet files inside.
    """
    files: List[Path] = []
    for raw in paths:
        p = Path(raw).resolve()
        if not p.exists():
            raise SystemExit(f"Path does not exist: {p}")
        if p.is_file():
            if p.suffix == ".parquet":
                files.append(p)
            else:
                print(f"WARNING: skipping non-parquet file: {p}", file=sys.stderr)
        elif p.is_dir():
            candidates = sorted(
                f for f in p.iterdir()
                if f.is_file() and f.suffix == ".parquet"
            )
            if not candidates:
                print(f"WARNING: no .parquet files in {p}", file=sys.stderr)
            files.extend(candidates)
    return files


def _open_jsonl_writer(path: Path, gzip_enabled: bool):
    """Return a context manager for writing .jsonl (optionally gzipped)."""
    if gzip_enabled:
        return gzip.open(path, "wt", encoding="utf-8", compresslevel=4)
    return open(path, "w", encoding="utf-8", buffering=1024 * 1024)



# Parquet reader (lazy — uses datasets library if available, else pyarrow)
def _iter_records_pyarrow(files: List[Path], columns: Optional[List[str]]):
    """Stream records from Parquet files using pyarrow directly.

    Yields dicts with Python-native types (lists for arrow lists, scalars
    otherwise).
    """
    import pyarrow.parquet as pq

    read_kw = {"columns": columns} if columns else {}

    for path in files:
        # Read file in row-group batches to keep memory low.
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx, **read_kw)
            col_names = table.column_names
            for i in range(table.num_rows):
                record = {}
                for name in col_names:
                    col = table.column(name)
                    val = col[i].as_py()
                    record[name] = val
                yield record


def _iter_records_datasets(files: List[Path], columns: Optional[List[str]]):
    """Stream records from Parquet files using HuggingFace datasets.

    Falls back to iterating the dataset in streaming mode, which is
    memory-efficient for large datasets.
    """
    from datasets import load_dataset

    read_kw = {"columns": columns} if columns else {}

    ds = load_dataset(
        "parquet",
        data_files=[str(p) for p in files],
        split="train",
        streaming=True,
    )
    if read_kw:
        ds = ds.select_columns(**read_kw)

    for example in ds:
        yield example


def _iter_records(files: List[Path], columns: Optional[List[str]]):
    """Stream records from Parquet files with automatic backend selection.

    Prefers HuggingFace ``datasets`` (streaming) for memory efficiency,
    falling back to pyarrow if datasets is unavailable.
    """
    try:
        import datasets  # noqa: F401
        yield from _iter_records_datasets(files, columns)
    except ImportError:
        try:
            import pyarrow  # noqa: F401
            yield from _iter_records_pyarrow(files, columns)
        except ImportError:
            raise SystemExit(
                "No Parquet reader found. Install either:\n"
                "  pip install datasets\n"
                "  pip install pyarrow"
            )



# Flattening helpers
def _flatten_record(record: dict, flatten_col: str) -> List[dict]:
    """If *record[flatten_col]* is a list, yield one sub-record per element.

    Each sub-record has the original columns, except that *flatten_col*
    contains a single scalar value instead of a list.
    """
    val = record.get(flatten_col)
    if val is None or not isinstance(val, (list, np.ndarray)):
        # Nothing to flatten — yield the record as-is (single item).
        return [record]

    sub_records = []
    for item in val:
        sub = dict(record)
        sub[flatten_col] = item
        sub_records.append(sub)
    return sub_records



# Serialisation helpers
def _make_serialisable(record: dict) -> dict:
    """Convert numpy types to native Python types for JSON serialisation."""
    out = {}
    for k, v in record.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (list, tuple)):
            out[k] = [_make_scalar_serialisable(x) for x in v]
        else:
            out[k] = v
    return out


def _make_scalar_serialisable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    elif isinstance(v, (np.floating,)):
        return float(v)
    elif isinstance(v, np.ndarray):
        return v.tolist()
    return v



# Main conversion
def convert_files(
    files: List[Path],
    output_path: Path,
    *,
    columns: Optional[List[str]],
    flatten: Optional[str],
    max_records_per_shard: int,
    gzip_enabled: bool,
    progress_every: int,
    quiet: bool,
) -> int:
    """Read parquet files and write JSONL records.

    Returns the total number of records written.
    """
    stream = _iter_records(files, columns=columns)

    shard_idx = 0
    shard_records = 0
    total_records = 0
    t0 = time.time()

    # Determine output path pattern.
    if max_records_per_shard > 0:
        # Sharded: insert shard index before extension.
        stem = output_path.stem
        suffix = output_path.suffix  # .jsonl
        gz_suffix = ".gz" if gzip_enabled else ""
        shard_template = output_path.with_name(f"{stem}.{{:04d}}{suffix}{gz_suffix}")
    else:
        gz_suffix = ".gz" if gzip_enabled else ""
        shard_path = output_path.with_suffix(output_path.suffix + gz_suffix) if gzip_enabled else output_path

    def _open_next_shard():
        nonlocal shard_idx, shard_records
        if max_records_per_shard > 0:
            path = Path(str(shard_template).format(shard_idx))
        else:
            path = shard_path
        print(f"  writing {path}" + ("" if quiet else ""), file=sys.stderr if quiet else sys.stdout)
        return _open_jsonl_writer(path, gzip_enabled)

    f = _open_next_shard()
    try:
        for record in stream:
            # Flatten if requested.
            if flatten:
                sub_records = _flatten_record(record, flatten)
            else:
                sub_records = [record]

            for sub in sub_records:
                sub = _make_serialisable(sub)

                # Roll to next shard if needed.
                if (max_records_per_shard > 0 and
                        shard_records >= max_records_per_shard):
                    f.close()
                    shard_idx += 1
                    shard_records = 0
                    f = _open_next_shard()

                f.write(json.dumps(sub, separators=(",", ":")) + "\n")
                shard_records += 1
                total_records += 1

                if progress_every and total_records % progress_every == 0:
                    elapsed = time.time() - t0
                    rate = total_records / elapsed if elapsed > 0 else 0.0
                    print(
                        f"  progress: {total_records:,} records "
                        f"({rate:,.0f} rec/s)",
                        file=sys.stderr if quiet else sys.stdout,
                    )
    finally:
        f.close()

    elapsed = time.time() - t0
    if not quiet:
        print(f"  done: {total_records:,} records in {elapsed:.1f}s "
              f"({total_records / elapsed:,.0f} rec/s)")

    return total_records


def main(args):
    # Resolve input files.
    files = _collect_parquet(args.paths)
    if not files:
        raise SystemExit("No .parquet files found.")

    if not args.quiet:
        print(f"Found {len(files)} parquet file(s):")
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f} ({size_mb:.1f} MB)")

    max_records = args.max_records_per_shard if args.max_records_per_shard > 0 else 0

    if args.output_dir:
        # Per-directory output: each input directory gets its own JSONL file.
        # Group files by their parent directory.
        from collections import defaultdict
        dirs: dict = defaultdict(list)
        for f in files:
            # Find the original input path that is an ancestor of this file.
            for raw in args.paths:
                rp = Path(raw).resolve()
                if rp == f.parent or rp == f:
                    dirs[str(rp)].append(f)
                    break
            else:
                # Fallback: use grandparent.
                dirs[str(f.parent)].append(f)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        total_all = 0
        for dir_label, dir_files in dirs.items():
            dir_name = Path(dir_label).name
            output_path = out_dir / f"{dir_name}.jsonl"
            if not args.quiet:
                print(f"\n--- {dir_name} ({len(dir_files)} shard(s)) ---")
            n = convert_files(
                dir_files,
                output_path,
                columns=args.columns,
                flatten=args.flatten,
                max_records_per_shard=max_records,
                gzip_enabled=args.gzip,
                progress_every=args.progress_every,
                quiet=args.quiet,
            )
            total_all += n

        if not args.quiet:
            print(f"\nTotal: {total_all:,} records written to {out_dir}")
    else:
        # Single output.
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        convert_files(
            files,
            output_path,
            columns=args.columns,
            flatten=args.flatten,
            max_records_per_shard=max_records,
            gzip_enabled=args.gzip,
            progress_every=args.progress_every,
            quiet=args.quiet,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--paths",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more paths to .parquet files or directories containing "
             ".parquet files.",
    )
    ap.add_argument(
        "--output",
        default="output.jsonl",
        metavar="FILE",
        help="Output JSONL file path. Ignored when --output-dir is set "
             "(default: output.jsonl).",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="If set, writes one JSONL file (or shard set) per input "
             "directory into this directory, named after each directory. "
             "Overrides --output.",
    )
    ap.add_argument(
        "--columns",
        nargs="+",
        default=None,
        metavar="COL",
        help="Read only these columns from the parquet files (default: all).",
    )
    ap.add_argument(
        "--flatten",
        default=None,
        metavar="COL",
        help="Flatten an array-type column: each element becomes its own "
             "record, with the column holding a scalar value.",
    )
    ap.add_argument(
        "--max-records-per-shard",
        type=int,
        default=100_000,
        metavar="N",
        help="Split output into shards of at most N records each "
             "(default: 100,000).",
    )
    ap.add_argument(
        "--gzip",
        action="store_true",
        help="Compress output with gzip (adds .gz suffix).",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        metavar="N",
        help="Print progress every N records (0 = disabled).",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output (only errors printed).",
    )
    args = ap.parse_args()

    main(args)
