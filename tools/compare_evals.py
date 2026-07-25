"""Compare eval results across multiple model runs."""

import json
import math
import os
import sys

import pandas as pd
import yaml

# Where the logs are stored
LOGS_DIR = "/home/nicholas/Documents/patterns/logs/runs"

# Window size for exponential moving average smoothing of training loss
EMA_WINDOW = 100

# Hardcoded list of run folders to compare (relative to LOGS_DIR)
FOLDERS = [
    "c4/670m",
    "c4_c4/670m",
    "shuffle_dyck_8_c4/670m",
    "shuffle_dyck_16_c4/670m",
    "shuffle_dyck_32_c4/670m",
    "shuffle_dyck_64_c4/670m",
    "nca_paper_c4/670m",
    "nca_learnable_25_c4/670m",
    "nca_learnable_50_c4/670m",
    "composite_mirror_repeat_c4/670m",
    "copy_c4/670m",
    "counting_anbn_c4/670m",
    "counting_anbncn_c4/670m",
    "hierarchical_c4/670m",
    "identity_c4/670m",
    "interleaving_c4/670m",
    "mixer_c4/670m",
    "periodic_c4/670m",
    "permutation_cycle_c4/670m",
    "random_c4/670m",
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
    "ppl_improv",
    "speedup",
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
    "ppl_improv": "PPL Impr%",
    "speedup": "Speedup",
}

# Direction: +1 = higher is better, -1 = lower is better
METRIC_DIRECTION = {
    "arc_easy_acc_norm": 1,
    "blimp_acc": 1,
    "hellaswag_acc_norm": 1,
    "lambada_openai_acc": 1,
    "winogrande_acc": 1,
    "val_loss": -1,
    "val_ppl": -1,
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
    metrics["ppl_improv"] = None  # computed after baseline is known
    metrics["speedup"] = None  # computed after baseline is known
    return metrics


def load_training_losses(folder_path: str) -> list[float] | None:
    """Load all training loss values from training.jsonl."""
    jsonl_path = os.path.join(folder_path, "training.jsonl")
    if not os.path.isfile(jsonl_path):
        print(f"  [WARN] training.jsonl not found in: {folder_path}", file=sys.stderr)
        return None
    losses = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                losses.append(record.get("loss"))
    return losses


def ema_smooth(losses: list[float], window: int = EMA_WINDOW) -> list[float]:
    """Apply exponential moving average smoothing to a loss curve.

    Uses alpha = 2 / (window + 1), the standard EMA formula.
    """
    if not losses:
        return []
    alpha = 2.0 / (window + 1)
    smoothed = [losses[0]]
    for loss in losses[1:]:
        smoothed.append(alpha * loss + (1 - alpha) * smoothed[-1])
    return smoothed


def find_convergence_step(smoothed: list[float], target_loss: float) -> int | None:
    """Find the first step (1-indexed) where smoothed loss <= target_loss.

    Returns None if the curve never reaches the target.
    """
    for i, loss in enumerate(smoothed):
        if loss <= target_loss:
            return i + 1  # 1-indexed step
    return None


def compute_ppl_improvement(ppl: float | None, baseline_ppl: float | None) -> float | None:
    """Percentage improvement of perplexity vs baseline.

    Positive means the run is better (lower perplexity) than baseline.
    Formula: (baseline - run) / baseline * 100
    """
    if ppl is None or baseline_ppl is None or baseline_ppl == 0:
        return None
    return (baseline_ppl - ppl) / baseline_ppl * 100.0


def compute_speedup(
    folder_path: str,
    baseline_smoothed_losses: list[float],
    baseline_steps: int,
) -> float | None:
    """Compute convergence speedup vs baseline.

    Returns the ratio: baseline_steps / pattern_steps_to_reach_baseline_final_loss.
    A value > 1.0 means the pattern converged faster.
    Returns None if the pattern never reaches the baseline's final loss.
    """
    losses = load_training_losses(folder_path)
    if losses is None or not losses:
        return None

    # Baseline's final smoothed loss is the target
    target_loss = baseline_smoothed_losses[-1]

    smoothed = ema_smooth(losses)
    pattern_step = find_convergence_step(smoothed, target_loss)

    if pattern_step is None:
        return None

    return baseline_steps / pattern_step


def format_value(value, metric: str = "") -> str:
    """Format a metric value for table display."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        if metric == "ppl_improv":
            # Show signed percentage, e.g. "+3.52%" or "-1.23%"
            return f"{value:+.2f}%"
        if metric == "speedup":
            # Show multiplier, e.g. "1.25×"
            return f"{value:.2f}×"
        return f"{value:.4f}"
    return str(value)


def compare_vs_baseline(value, baseline, direction: int) -> str:
    """Return a comparison emoji: 🟢 if better, 🔴 if worse, '' if equal or N/A."""
    if value is None or baseline is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(baseline, float) and math.isnan(baseline):
        return ""
    if value == baseline:
        return ""
    # direction=+1: higher is better; direction=-1: lower is better
    better = (value > baseline) if direction == 1 else (value < baseline)
    return "🟢" if better else "🔴"


def main():
    model_names = []
    all_metrics = []

    print(f"Scanning {len(FOLDERS)} folders in: {LOGS_DIR}\n")

    # --- First pass: collect eval metrics and val loss ---
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

    # --- Identify baseline (c4) and compute its smoothed training loss ---
    baseline_idx = None
    for i, name in enumerate(model_names):
        if name == "c4":
            baseline_idx = i
            break

    baseline_ppl = None
    baseline_smoothed_losses = None
    baseline_steps = 0

    if baseline_idx is not None:
        baseline_metrics = all_metrics[baseline_idx]
        baseline_ppl = baseline_metrics.get("val_ppl")

        # Load and smooth baseline training loss for speedup computation
        baseline_folder = os.path.join(LOGS_DIR, FOLDERS[baseline_idx])
        baseline_losses = load_training_losses(baseline_folder)
        if baseline_losses:
            baseline_smoothed_losses = ema_smooth(baseline_losses)
            baseline_steps = len(baseline_losses)

    # --- Second pass: compute ppl_improv and speedup for each run ---
    for _i, folder in enumerate(FOLDERS):
        model_name = folder.rstrip("/").rsplit("/", 1)[0]
        if model_name not in model_names:
            continue  # folder was skipped due to missing evals

        # Find the metrics dict for this model
        metrics_idx = model_names.index(model_name)
        metrics = all_metrics[metrics_idx]

        # PPL improvement vs baseline
        if model_name != "c4":
            metrics["ppl_improv"] = compute_ppl_improvement(metrics.get("val_ppl"), baseline_ppl)

        # Convergence speedup vs baseline
        if model_name != "c4" and baseline_smoothed_losses is not None:
            full_path = os.path.join(LOGS_DIR, folder)
            metrics["speedup"] = compute_speedup(
                full_path, baseline_smoothed_losses, baseline_steps
            )

    # --- Extract c4 baseline metrics for comparison ---
    baseline_metrics = None
    for name, m in zip(model_names, all_metrics, strict=False):
        if name == "c4":
            baseline_metrics = m
            break

    # Build DataFrame
    df = pd.DataFrame(all_metrics, index=model_names)
    df.index.name = "Model"

    # Rename columns and order per METRICS
    df = df.rename(columns=METRIC_NAMES)
    display_order = [METRIC_NAMES[m] for m in METRICS if METRIC_NAMES[m] in df.columns]
    df = df[display_order]

    # Sort by Val Loss ascending; N/A to end
    if "Val Loss" in df.columns:
        df = df.sort_values("Val Loss", ascending=True, na_position="last")

    # Build display DataFrame with formatted values and emoji indicators
    reverse_names = {v: k for k, v in METRIC_NAMES.items()}
    display_df = pd.DataFrame(index=df.index)

    for col in df.columns:
        orig_key = reverse_names.get(col, "")
        display_col = []
        for idx in df.index:
            val = df.loc[idx, col]
            val_str = format_value(val, orig_key)

            # Emoji vs baseline (skip baseline row and derived metrics)
            if baseline_metrics and idx != "c4" and orig_key not in ("ppl_improv", "speedup"):
                bl_val = baseline_metrics.get(orig_key)
                direction = METRIC_DIRECTION.get(orig_key, 0)
                emoji = compare_vs_baseline(val, bl_val, direction)
                if emoji:
                    val_str = f"{val_str} {emoji}"

            display_col.append(val_str)
        display_df[col] = display_col

    print(display_df.to_markdown(index=True))

    # Save as CSV (strip emoji indicators from cell values)
    csv_dir = os.path.join(os.path.dirname(LOGS_DIR), "measurements")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "evals.csv")
    csv_df = display_df.apply(lambda col: col.str.replace(r"[🟢🔴]", "", regex=True).str.strip())
    csv_df.to_csv(csv_path, index=True)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
