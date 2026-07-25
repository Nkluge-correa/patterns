#!/usr/bin/env python3
"""Compare structural complexity metrics across multiple model runs."""

import json
import math
import os
import sys

import pandas as pd

# Where the complexity reports are stored
REPORTS_DIR = "/home/nicholas/Documents/patterns/logs/reports"

# Hardcoded list of report folders to compare (relative to REPORTS_DIR)
FOLDERS = [
    "c4",
    "fineweb_edu",
    "codeparrot",
    "open_web_math",
    "composite_mirror_repeat",
    "copy",
    "counting_anbn",
    "counting_anbncn",
    "hierarchical",
    "identity",
    "interleaving",
    "periodic",
    "permutation_cycle",
    "random",
    "mixer",
    "nca_paper",
    "nca_learnable_25",
    "nca_learnable_50",
    "shuffle_dyck_8",
    "shuffle_dyck_16",
    "shuffle_dyck_32",
    "shuffle_dyck_64",
]

# Metrics to extract from the JSON report
METRICS = [
    "final_val_loss_nats",
    "val_ppl",
    "S_bits_per_train_token",
    "H_bits_per_test_token",
    "total_info_bits_per_token",
    "structural_fraction",
    "gzip_complexity",
    "oracle_loss_nats",
    "language_entropy_nats",
]

# Human-readable names for table header
METRIC_NAMES = {
    "final_val_loss_nats": "Val Loss",
    "val_ppl": "Val PPL",
    "S_bits_per_train_token": "S (bits/tok)",
    "H_bits_per_test_token": "H (bits/tok)",
    "total_info_bits_per_token": "Total (b/t)",
    "structural_fraction": "Struct Frac",
    "gzip_complexity": "GZip Compl.",
    "oracle_loss_nats": "Oracle Loss",
    "language_entropy_nats": "Lang Entropy",
}


def load_report(folder_path: str) -> dict | None:
    """Load and parse the 670m.json report from the given folder path."""
    json_path = os.path.join(folder_path, "670m.json")
    if not os.path.isfile(json_path):
        print(f"  [WARN] 670m.json not found in: {folder_path}", file=sys.stderr)
        return None
    with open(json_path) as f:
        return json.load(f)


def extract_metrics(data: dict) -> dict:
    """Extract the relevant metrics from the parsed report JSON."""
    metrics = {}
    for key in METRICS:
        if key == "val_ppl":
            # Compute perplexity from final validation loss
            loss = data.get("final_val_loss_nats")
            metrics[key] = math.exp(loss) if loss is not None else None
        else:
            metrics[key] = data.get(key)
    return metrics


def format_value(value) -> str:
    """Format a metric value for table display."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        # Use more precision for small values like structural_fraction
        if abs(value) < 0.01:
            return f"{value:.6f}"
        return f"{value:.4f}"
    return str(value)


def main():
    model_names = []
    all_metrics = []

    print(f"Scanning {len(FOLDERS)} report folders in: {REPORTS_DIR}\n")

    for folder in FOLDERS:
        full_path = os.path.join(REPORTS_DIR, folder)
        model_name = folder

        data = load_report(full_path)
        if data is None:
            continue

        metrics = extract_metrics(data)
        model_names.append(model_name)
        all_metrics.append(metrics)

    if not model_names:
        print("No report data found.", file=sys.stderr)
        sys.exit(1)

    # Build DataFrame
    df = pd.DataFrame(all_metrics, index=model_names)
    df.index.name = "Dataset/Pattern"

    # Rename columns and order per METRICS
    df = df.rename(columns=METRIC_NAMES)
    display_order = [METRIC_NAMES[m] for m in METRICS if METRIC_NAMES[m] in df.columns]
    df = df[display_order]

    # Sort by GZip Compl. (desc), then Struct Frac (desc); N/A to end
    sort_by = []
    gzip_col = METRIC_NAMES["gzip_complexity"]
    struct_col = METRIC_NAMES["structural_fraction"]
    if gzip_col in df.columns:
        sort_by.append(gzip_col)
    if struct_col in df.columns:
        sort_by.append(struct_col)
    if sort_by:
        df = df.sort_values(sort_by, ascending=False, na_position="last")

    # Format all columns for display
    for col in df.columns:
        df[col] = df[col].apply(format_value)

    print(df.to_markdown(index=True))

    # Save as CSV
    csv_dir = os.path.join(os.path.dirname(REPORTS_DIR), "measurements")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "complexity.csv")
    df.to_csv(csv_path, index=True)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
