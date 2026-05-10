"""Measure gzip-based complexity metrics for pattern dataset JSONL files.

Computes, for each dataset directory:

  global_gzip_complexity
      compressed / uncompressed bytes over the full token stream
      (all samples streamed incrementally into one gzip compressor).

  mean_per_sample_complexity
      average of (compressed / uncompressed) computed independently per
      sample over up to --per-sample-limit records.

Results are written as a .complexity.yaml file inside each analyzed
directory, or in their common parent when multiple directories are
processed at once.

Usage:
  python tools/complexity.py \
    --paths ./data \
    --output-dir ./data/results/ \
    --vocab-size 256 \
    --per-sample-limit 1000 \
    --compresslevel 9 \
    --plot
"""

import argparse
import gzip as gzip_mod
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

METADATA_FILENAME = ".complexity.yaml"


# Dtype selection
def _dtype_for_vocab(vocab_size: int):
    if vocab_size <= 256:
        return np.uint8
    if vocab_size <= 65_536:
        return np.uint16
    return np.uint32


# File helpers
def _open_jsonl(path: Path):
    """Open a .jsonl or .jsonl.gz file for line-by-line reading."""
    if path.suffix == ".gz":
        return gzip_mod.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _collect_jsonl(directory: Path) -> List[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and (
            p.name.endswith(".jsonl") or p.name.endswith(".jsonl.gz")
        )
    )


def _iter_records(paths: List[Path]):
    """Yield parsed record dicts from a list of JSONL files."""
    for path in paths:
        with _open_jsonl(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _sniff_vocab_size(paths: List[Path]) -> Optional[int]:
    """Return vocab_size from the first record that carries metadata."""
    for record in _iter_records(paths):
        meta = record.get("metadata")
        if meta and "vocab_size" in meta:
            return int(meta["vocab_size"])
    return None

# Metric computation
def global_gzip_complexity(
    paths: List[Path],
    dtype,
    batch_size: int = 10_000,
    compresslevel: int = 9,
) -> Tuple[int, int, int]:
    """Stream all token arrays into a temporary gzip file.

    Returns (n_samples, total_uncompressed_bytes, total_compressed_bytes).
    """
    total_uncompressed = 0
    n_samples = 0
    chunk = bytearray()

    with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with open(tmp_path, "wb") as raw_out:
            with gzip_mod.GzipFile(
                fileobj=raw_out, mode="wb", compresslevel=compresslevel
            ) as gz:
                for i, record in enumerate(_iter_records(paths), 1):
                    tokens = np.asarray(record["input_ids"], dtype=dtype)
                    b = tokens.tobytes()
                    total_uncompressed += len(b)
                    n_samples += 1
                    chunk.extend(b)
                    if i % batch_size == 0:
                        gz.write(chunk)
                        chunk.clear()
                if chunk:
                    gz.write(chunk)

        total_compressed = os.path.getsize(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return n_samples, total_uncompressed, total_compressed


def per_sample_complexity(
    paths: List[Path],
    dtype,
    max_samples: Optional[int],
    compresslevel: int = 9,
) -> Tuple[List[float], int]:
    """Compute per-sample gzip complexity for up to max_samples records.

    Returns (complexities, n_evaluated) where complexities is a list of
    (compressed_bytes / original_bytes) values, one per sample.
    """
    complexities: List[float] = []
    for record in _iter_records(paths):
        tokens = np.asarray(record["input_ids"], dtype=dtype)
        raw = tokens.tobytes()
        compressed = gzip_mod.compress(raw, compresslevel=compresslevel)
        complexities.append(len(compressed) / len(raw))
        if max_samples is not None and len(complexities) >= max_samples:
            break

    return complexities, len(complexities)


# Per-directory analysis
def analyze_directory(
    directory: Path,
    vocab_size: Optional[int],
    per_sample_limit: Optional[int],
    compresslevel: int,
    verbose: bool,
    store_sample_complexities: bool = False,
) -> Optional[dict]:
    files = _collect_jsonl(directory)
    if not files:
        if verbose:
            print(f"  [skip] {directory}: no .jsonl files found")
        return None

    if verbose:
        print(f"  {directory.name}: {len(files)} shard(s)")

    effective_vocab = vocab_size or _sniff_vocab_size(files)
    if effective_vocab is None:
        print(
            f"  WARNING: could not determine vocab_size for {directory}; "
            "defaulting to uint32.",
            file=sys.stderr,
        )
        effective_vocab = 2 ** 32

    dtype = _dtype_for_vocab(effective_vocab)
    dtype_name = np.dtype(dtype).name

    t0 = time.time()

    n_samples, n_uncompressed, n_compressed = global_gzip_complexity(
        files, dtype, compresslevel=compresslevel
    )
    sample_complexities, n_evaluated = per_sample_complexity(
        files, dtype, max_samples=per_sample_limit, compresslevel=compresslevel
    )
    mean_per_sample = float(np.mean(sample_complexities)) if sample_complexities else float("nan")

    elapsed = time.time() - t0
    n_tokens = n_uncompressed // np.dtype(dtype).itemsize

    result = {
        "directory": str(directory),
        "n_shards": len(files),
        "n_samples": n_samples,
        "n_tokens": n_tokens,
        "vocab_size": effective_vocab,
        "dtype": dtype_name,
        "compresslevel": compresslevel,
        "global": {
            "original_bytes": n_uncompressed,
            "compressed_bytes": n_compressed,
            "gzip_complexity": n_compressed / n_uncompressed,
            "compression_ratio": n_uncompressed / n_compressed,
            "space_saving": 1.0 - (n_compressed / n_uncompressed),
        },
        "per_sample": {
            "n_evaluated": n_evaluated,
            "mean_gzip_complexity": mean_per_sample,
        },
        "sample_complexities": sample_complexities if store_sample_complexities else None,
        "elapsed_seconds": elapsed,
    }

    if verbose:
        g = result["global"]
        ps = result["per_sample"]
        print(f"    tokens              : {n_tokens:,}")
        print(
            f"    global complexity   : {g['gzip_complexity']:.4f}  "
            f"(ratio {g['compression_ratio']:.2f}x, "
            f"saving {g['space_saving'] * 100:.1f}%)"
        )
        print(
            f"    per-sample (n={n_evaluated:,}): "
            f"{ps['mean_gzip_complexity']:.4f}"
        )
        print(f"    elapsed             : {elapsed:.1f}s")

    return result


# YAML output
def write_metadata(
    output_dir: Path, results: List[dict], timestamp: str
) -> Path:
    out_path = output_dir / METADATA_FILENAME
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# gzip complexity metadata\n")
        f.write(f"# generated: {timestamp}\n")
        f.write(f"n_datasets: {len(results)}\n")
        f.write("datasets:\n")
        for r in results:
            name = Path(r["directory"]).name
            f.write(f"  {name}:\n")
            for k in ("n_shards", "n_samples", "n_tokens", "vocab_size",
                      "dtype", "compresslevel"):
                f.write(f"    {k}: {r[k]}\n")
            f.write("    global:\n")
            for k, v in r["global"].items():
                if isinstance(v, float):
                    f.write(f"      {k}: {v:.6f}\n")
                else:
                    f.write(f"      {k}: {v}\n")
            f.write("    per_sample:\n")
            f.write(f"      n_evaluated: {r['per_sample']['n_evaluated']}\n")
            mps = r["per_sample"]["mean_gzip_complexity"]
            f.write(f"      mean_gzip_complexity: {mps:.6f}\n")
            f.write(f"    elapsed_seconds: {r['elapsed_seconds']:.2f}\n")
    return out_path


# Histogram plot
def _adaptive_bins(vals: List[float], cap: int = 60) -> int:
    """Choose bin count using numpy's 'auto' heuristic, capped at *cap*."""
    edges = np.histogram_bin_edges(vals, bins="auto")
    return min(len(edges) - 1, cap)


def plot_histogram(r: dict, output_path: Path) -> Optional[Path]:
    """Plot a histogram for a single dataset result and save to *output_path*."""
    import matplotlib.pyplot as plt

    vals = r.get("sample_complexities")
    if not vals:
        print(
            f"WARNING: no per-sample data to plot for "
            f"{Path(r['directory']).name}.",
            file=sys.stderr,
        )
        return None

    name = Path(r["directory"]).name
    mean = r["per_sample"]["mean_gzip_complexity"]
    n_bins = _adaptive_bins(vals)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=n_bins, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(mean, color="crimson", linewidth=1.5, linestyle="--",
               label=f"mean={mean:.3f}")
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel(r"complexity  $=$ compressed / original", fontsize=9)
    ax.set_ylabel("samples", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    fig.suptitle(
        "Per-sample gzip complexity\n"
        r"complexity $=$ compressed$\,/\,$original  ·  "
        r"compression ratio $=$ original$\,/\,$compressed",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(args):

    if not (1 <= args.compresslevel <= 9):
        raise SystemExit("--compresslevel must be between 1 and 9.")

    per_sample_limit = args.per_sample_limit if args.per_sample_limit > 0 else None
    verbose = not args.quiet
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Resolve input paths: if a directory contains .jsonl files directly,
    # treat it as a single dataset; otherwise descend one level.
    directories: List[Path] = []
    for raw in args.paths:
        p = Path(raw).resolve()
        if not p.exists():
            raise SystemExit(f"Path does not exist: {p}")
        if p.is_file():
            raise SystemExit(f"Expected a directory, got a file: {p}")
        if _collect_jsonl(p):
            directories.append(p)
        else:
            subdirs = sorted(d for d in p.iterdir() if d.is_dir())
            if not subdirs:
                print(
                    f"WARNING: no .jsonl files or subdirectories in {p}",
                    file=sys.stderr,
                )
            directories.extend(subdirs)

    if not directories:
        raise SystemExit("No dataset directories found.")

    print(f"Analyzing {len(directories)} dataset(s) ...")

    all_results: List[dict] = []
    for d in directories:
        result = analyze_directory(
            d,
            vocab_size=args.vocab_size,
            per_sample_limit=per_sample_limit,
            compresslevel=args.compresslevel,
            verbose=verbose,
            store_sample_complexities=args.plot,
        )
        if result is not None:
            all_results.append(result)

    if not all_results:
        raise SystemExit("No data found in any of the specified paths.")

    if args.output_dir:
        # Combined yaml + individual plots, all in output_dir.
        meta_dir = Path(args.output_dir)
        meta_dir.mkdir(parents=True, exist_ok=True)

        out_path = write_metadata(meta_dir, all_results, timestamp)
        print(f"\nMetadata written to {out_path}")

        if args.plot:
            for r in all_results:
                name = Path(r["directory"]).name
                plot_path = plot_histogram(r, meta_dir / f"{name}_complexity.png")
                if plot_path:
                    print(f"Histogram saved to  {plot_path}")
    else:
        # Per-directory yaml + plot, written into each dataset directory.
        for r in all_results:
            d = Path(r["directory"])
            out_path = write_metadata(d, [r], timestamp)
            print(f"\nMetadata written to {out_path}")

            if args.plot:
                plot_path = plot_histogram(r, d / "complexity_histogram.png")
                if plot_path:
                    print(f"Histogram saved to  {plot_path}")


if __name__ == "__main__":

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--paths",
        nargs="+",
        metavar="PATH",
        help="One or more dataset directories (or a root directory whose "
             "subdirectories each contain .jsonl shards).",
    )
    ap.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        metavar="N",
        help="Override vocab size used to select the integer dtype. "
             "Auto-detected from record metadata when not set.",
    )
    ap.add_argument(
        "--per-sample-limit",
        type=int,
        default=1000,
        metavar="N",
        help="Max number of samples to evaluate for per-sample complexity "
             "(default: 1000). Pass 0 to evaluate all samples (slow for "
             "large datasets).",
    )
    ap.add_argument(
        "--compresslevel",
        type=int,
        default=9,
        metavar="1-9",
        help="Gzip compression level (default: 9).",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Write the .complexity.yaml file to this directory instead of "
             "the common parent of the analyzed directories.",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Save a per-sample complexity histogram (PNG) next to the "
             "metadata file. Requires matplotlib.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-directory progress output.",
    )
    args = ap.parse_args()

    main(args)
