"""Compare eval results across multiple model runs."""

import json
import math
import os
import sys

import yaml

# Where the logs are stored
LOGS_DIR = "/home/nicholas/Documents/patterns/logs/runs"

# Hardcoded list of run folders to compare (relative to LOGS_DIR)
FOLDERS = [
    "c4/670m",
    "shuffle_dyck_8_c4/670m",
    "shuffle_dyck_16_c4/670m",
    "shuffle_dyck_32_c4/670m",
    "shuffle_dyck_64_c4/670m",
    "nca_paper_c4/670m",
    "nca_learnable_25_c4/670m",
    "nca_learnable_50_c4/670m",
    "mixer_c4/670m",
    "composite_mirror_repeat_c4/670m",
    "copy_c4/670m",
    "counting_anbncn_c4/670m",
]

# Metrics to extract from evals.yaml
EVAL_METRICS = [
    "arc_easy_acc_norm",
    "blimp_acc",
    "hellaswag_acc_norm",
    "lambada_openai_acc",
    "winogrande_acc",
]

# Extra columns (computed or from validation.jsonl)
EXTRA_METRICS = [
    "val_loss",
    "val_ppl",
]

# Combined ordered list for table printing
METRICS = EVAL_METRICS + EXTRA_METRICS

# Human-readable names for table header
METRIC_NAMES = {
    "arc_easy_acc_norm": "ARC-Easy",
    "blimp_acc": "BLiMP",
    "hellaswag_acc_norm": "HellaSwag",
    "lambada_openai_acc": "LAMBADA",
    "winogrande_acc": "Winogrande",
    "val_loss": "Val Loss",
    "val_ppl": "Val PPL",
}


def load_evals(folder_path: str) -> dict | None:
    """Load and parse evals.yaml from the given folder path."""
    yaml_path = os.path.join(folder_path, "evals.yaml")
    if not os.path.isfile(yaml_path):
        print(f"  [WARN] evals.yaml not found in: {folder_path}", file=sys.stderr)
        return None
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def load_validation_loss(folder_path: str) -> float | None:
    """Load the final validation loss from the last line of validation.jsonl."""
    jsonl_path = os.path.join(folder_path, "validation.jsonl")
    if not os.path.isfile(jsonl_path):
        print(f"  [WARN] validation.jsonl not found in: {folder_path}", file=sys.stderr)
        return None
    with open(jsonl_path) as f:
        last_line = None
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    if last_line is None:
        return None
    record = json.loads(last_line)
    return record.get("loss", None)


def extract_metrics(data: dict, val_loss: float | None) -> dict:
    """Extract the relevant metrics from the parsed YAML results and validation."""
    results = data.get("results", {})
    metrics = {key: results.get(key, None) for key in EVAL_METRICS}
    metrics["val_loss"] = val_loss
    metrics["val_ppl"] = math.exp(val_loss) if val_loss is not None else None
    return metrics


def format_value(value) -> str:
    """Format a metric value for table display."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_table(model_names: list[str], all_metrics: list[dict]):
    """Print a formatted comparison table."""
    # Column widths: model name + one per metric
    col_widths = {"model": max(len("Model"), max(len(n) for n in model_names))}

    for metric in METRICS:
        header = METRIC_NAMES.get(metric, metric)
        max_val_len = len(header)
        for m in all_metrics:
            val_str = format_value(m.get(metric))
            max_val_len = max(max_val_len, len(val_str))
        col_widths[metric] = max_val_len

    header_cells = ["Model"] + [METRIC_NAMES.get(m, m) for m in METRICS]
    header_row = "  " + header_cells[0].ljust(col_widths["model"])
    for i, metric in enumerate(METRICS):
        header_row += "  " + header_cells[i + 1].rjust(col_widths[metric])
    print(header_row)

    sep = "  " + "-" * col_widths["model"]
    for metric in METRICS:
        sep += "  " + "-" * col_widths[metric]
    print(sep)

    for name, metrics in zip(model_names, all_metrics, strict=False):
        row = "  " + name.ljust(col_widths["model"])
        for metric in METRICS:
            val = format_value(metrics.get(metric))
            row += "  " + val.rjust(col_widths[metric])
        print(row)

    print()


def main():
    model_names = []
    all_metrics = []

    print(f"Scanning {len(FOLDERS)} folders in: {LOGS_DIR}\n")

    for folder in FOLDERS:
        full_path = os.path.join(LOGS_DIR, folder)
        model_name = folder.rstrip("/").rsplit("/", 1)[0]

        data = load_evals(full_path)
        if data is None:
            continue

        val_loss = load_validation_loss(full_path)
        metrics = extract_metrics(data, val_loss)
        model_names.append(model_name)
        all_metrics.append(metrics)

    if not model_names:
        print("No eval data found.", file=sys.stderr)
        sys.exit(1)

    print_table(model_names, all_metrics)


if __name__ == "__main__":
    main()
