"""Epiplexity estimation tool.

Computes the prequential epiplexity S_T and time-bounded entropy H_T
following Finzi et al. (2026):

    S_T ≈ Σ_i ( log 1/P_i(Z_i) - log 1/P_M(Z_i) )
    H_T ≈ E[ log 1/P_M(X) ]

Usage:
Accurate (with loss and validation curve files):
    python tools/epiplexity.py \\
        --model-params    670e6 \\
        --train-tokens    5.2e9 \\
        --loss-curve      training.jsonl \\
        --val-curve       validation.jsonl

Approximate (without a loss curve — uses endpoints only):
    python tools/epiplexity.py \\
        --model-params    670e6 \\
        --train-tokens    5.2e9 \\
        --initial-loss    11.08 \\
        --final-train-loss 2.63 \\
        --final-val-loss  2.6954

With reference metrics:
    python tools/epiplexity.py \\
        --model-params    670e6 \\
        --train-tokens    5.2e9 \\
        --loss-curve      training.jsonl \\
        --val-curve       validation.jsonl \\
        --gzip-complexity 0.3766 \\
        --language-entropy 1.9 \\
        --run-name        fineweb-edu-670m

For synthetic patterns, use --oracle-loss (known generating process):
    python tools/epiplexity.py \\
        --model-params    10e6 \\
        --train-tokens    100e6 \\
        --final-val-loss  0.70 \\
        --loss-curve      training.jsonl \\
        --gzip-complexity 0.15 \\
        --oracle-loss     0.659 \\
        --run-name        dyck
"""

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

LN2 = math.log(2)


def _parse_big_int(value: str) -> int:
    """Parse a potentially large integer from scientific or plain notation.

    Accepts forms like ``670127616``, ``670e6``, ``5.2e9``, ``100e6``.
    """
    return int(float(value))


@dataclass
class EpiplexityResult:
    """Computed epiplexity / time-bounded entropy for a run."""

    run_name: str = ""

    # Inputs
    model_params: int = 0
    total_train_tokens: int = 0
    total_train_flops: float = 0.0
    initial_train_loss: float = 0.0
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0

    # Prequential S_T
    S_nats_total: float = 0.0
    S_bits_per_train_token: float = 0.0

    # Time-bounded entropy H_T
    H_bits_per_test_token: float = 0.0

    # Derived
    total_info_bits_per_token: float = 0.0
    structural_fraction: float = 0.0

    # Reference data
    gzip_complexity: float | None = None
    oracle_loss_nats: float | None = None
    uniform_baseline_nats: float | None = None

    # Language / data entropy floor
    # For natural language: the estimated Shannon entropy of the language
    #   (e.g. ~1.8–2.2 nats/token for English with subword tokenizers).
    # For synthetic patterns, use oracle_loss_nats instead, since it's exact.
    # When set, the report shows how close the model is to this floor.
    language_entropy_nats: float | None = None

    # Misc
    approximation_used: bool = False
    n_loss_points: int = 0
    loss_area_nats: float = 0.0
    warnings: list[str] = field(default_factory=list)


def load_loss_curve(
    path: Path,
    loss_key: str = "loss",
) -> list[float]:
    """Load per-step training losses from a JSONL file.

    Each line must be a JSON object with a *loss_key* field containing
    the per-token cross-entropy value in nats.

    Returns a list of per-token cross-entropy values in nats.
    """
    losses: list[float] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        losses.append(float(rec[loss_key]))
    return losses


def load_val_loss(
    path: Path,
    loss_key: str = "loss",
    step_key: str = "step",
) -> float:
    """Extract the final validation loss from a JSONL file.

    Each line must be a JSON object.  The loss from the record with the
    highest *step_key* value is returned.
    """
    best_step = -1
    best_loss = 0.0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        step = int(rec[step_key])
        if step > best_step:
            best_step = step
            best_loss = float(rec[loss_key])
    return best_loss


def compute_epiplexity(
    run_name: str,
    model_params: int,
    train_tokens: int,
    final_val_loss: float,
    *,
    loss_curve: Sequence[float] | None = None,
    initial_loss: float | None = None,
    final_train_loss: float | None = None,
    gzip_complexity: float | None = None,
    oracle_loss_nats: float | None = None,
    uniform_baseline_nats: float | None = None,
    language_entropy_nats: float | None = None,
) -> EpiplexityResult:
    """Compute prequential epiplexity S_T and time-bounded entropy H_T.

    Parameters
    ----------
    language_entropy_nats :
        The Shannon entropy floor of the data source, in nats/token.
        For NATURAL LANGUAGE this is the estimated entropy of the language
        itself (~1.8-2.2 nats/token for English with subword tokenizers).
        For SYNTHETIC patterns, use oracle_loss_nats instead (since the
        generating process is known).  These are mutually exclusive concepts:
        language entropy is estimated; oracle loss is exact.
    """

    result = EpiplexityResult(
        run_name=run_name,
        model_params=model_params,
        total_train_tokens=train_tokens,
        total_train_flops=6.0 * model_params * train_tokens,
        final_val_loss=final_val_loss,
        gzip_complexity=gzip_complexity,
        oracle_loss_nats=oracle_loss_nats,
        uniform_baseline_nats=uniform_baseline_nats,
        language_entropy_nats=language_entropy_nats,
    )

    result.H_bits_per_test_token = final_val_loss / LN2

    # S_T from loss curve
    if loss_curve is not None and len(loss_curve) > 0:
        result.n_loss_points = len(loss_curve)

        # Final training loss: use the last value in the curve
        ftl = loss_curve[-1]
        result.final_train_loss = ftl

        # Initial loss
        result.initial_train_loss = loss_curve[0]

        # Tokens per step
        tokens_per_step = train_tokens / len(loss_curve)

        total_excess = 0.0
        n_below = 0
        for loss in loss_curve:
            excess = loss - ftl
            if excess > 0:
                total_excess += excess
            elif excess < -1e-9:
                n_below += 1

        if n_below > 0:
            result.warnings.append(
                f"{n_below} steps have loss BELOW final training loss "
                f"({ftl:.4f}); the model may still be improving."
            )

        result.S_nats_total = total_excess * tokens_per_step
        result.S_bits_per_train_token = (total_excess / len(loss_curve)) / LN2
        result.loss_area_nats = sum(loss_curve)

    elif initial_loss is not None and final_train_loss is not None:
        # Approximate S_T from endpoints only
        result.approximation_used = True
        result.initial_train_loss = initial_loss
        result.final_train_loss = final_train_loss

        # Assume linear decay. This *underestimates* S_T because real loss
        # curves are convex (steep early drop, slow tail).  The true S_T is
        # between the linear approximation and ~1.5× the linear approximation.
        excess_per_token = 0.5 * (initial_loss - final_train_loss)
        result.S_nats_total = excess_per_token * train_tokens
        result.S_bits_per_train_token = excess_per_token / LN2

        result.warnings.append(
            "S_T approximated from endpoints only (linear-decay assumption). "
            "For convex loss curves (steep early drop → long tail) this "
            "OVERESTIMATES S_T, potentially by 5-10x.  Provide --loss-curve "
            "for an accurate measurement."
        )

    else:
        result.warnings.append(
            "Provide EITHER --loss-curve OR both --initial-loss and "
            "--final-train-loss to compute S_T."
        )
        result.initial_train_loss = initial_loss or 0.0
        result.final_train_loss = final_train_loss or 0.0

    # Derived
    result.total_info_bits_per_token = result.S_bits_per_train_token + result.H_bits_per_test_token
    if result.total_info_bits_per_token > 0:
        result.structural_fraction = (
            result.S_bits_per_train_token / result.total_info_bits_per_token
        )
    else:
        result.structural_fraction = 0.0

    return result


def _fmt(n: float, suffix: str = "") -> str:
    """Human-readable number with SI prefixes."""
    if abs(n) < 1e-9:
        return f"0{suffix}"
    for threshold, prefix in [
        (1e18, "E"),
        (1e15, "P"),
        (1e12, "T"),
        (1e9, "B"),
        (1e6, "M"),
        (1e3, "K"),
    ]:
        if abs(n) >= threshold:
            return f"{n / threshold:.2f}{prefix}{suffix}"
    if abs(n) >= 100:
        return f"{n:.2f}{suffix}"
    if abs(n) >= 1:
        return f"{n:.4f}{suffix}"
    return f"{n:.6f}{suffix}"


def render_report(result: EpiplexityResult, json_output: bool = False) -> str:
    """Render results as Markdown or JSON."""

    if json_output:
        return json.dumps(
            {
                "run_name": result.run_name,
                "model_params": result.model_params,
                "total_train_tokens": result.total_train_tokens,
                "total_train_flops": result.total_train_flops,
                "initial_train_loss_nats": result.initial_train_loss,
                "final_train_loss_nats": result.final_train_loss,
                "final_val_loss_nats": result.final_val_loss,
                "S_nats_total": result.S_nats_total,
                "S_bits": result.S_nats_total / LN2,
                "S_bits_per_train_token": result.S_bits_per_train_token,
                "H_bits_per_test_token": result.H_bits_per_test_token,
                "total_info_bits_per_token": result.total_info_bits_per_token,
                "structural_fraction": result.structural_fraction,
                "gzip_complexity": result.gzip_complexity,
                "oracle_loss_nats": result.oracle_loss_nats,
                "language_entropy_nats": result.language_entropy_nats,
                "approximation_used": result.approximation_used,
                "warnings": result.warnings,
            },
            indent=2,
        )

    # Markdown
    L: list[str] = []
    L.append(f"# Epiplexity Report: `{result.run_name}`")
    L.append("")

    if result.warnings:
        L.append("## \u26a0 Warnings")
        for w in result.warnings:
            L.append(f"- {w}")
        L.append("")

    # Inputs
    L.append("## Inputs")
    L.append("")
    L.append("| Parameter | Value |")
    L.append("|---|---|")
    L.append(f"| Model parameters | {result.model_params:,} |")
    L.append(f"| Total training tokens | {_fmt(result.total_train_tokens)} |")
    L.append(f"| Total training FLOPs | {_fmt(result.total_train_flops, ' FLOPs')} |")
    if result.n_loss_points:
        L.append(f"| Loss-curve points | {result.n_loss_points:,} |")
    L.append(f"| Initial training loss | {result.initial_train_loss:.4f} nats/token |")
    L.append(f"| Final training loss | {result.final_train_loss:.4f} nats/token |")
    L.append(f"| Final validation loss | {result.final_val_loss:.4f} nats/token |")
    L.append("")

    # Epiplexity
    L.append("## Epiplexity Decomposition")
    L.append("")
    L.append("| Quantity | Symbol | Per-token (bits) | Total |")
    L.append("|---|---|---|---|")
    L.append(
        f"| **Epiplexity** (structural info) | $S_T$ | "
        f"**{result.S_bits_per_train_token:.4f}** | "
        f"{_fmt(result.S_nats_total / LN2, ' bits')} |"
    )
    L.append(
        f"| **Time-bounded entropy** (random info) | $H_T$ | "
        f"**{result.H_bits_per_test_token:.4f}** | "
        f"depends on test-set size |"
    )
    L.append(
        f"| **Total information** | $S_T + H_T$ | **{result.total_info_bits_per_token:.4f}** | — |"
    )
    L.append("")

    # Ratios
    L.append("## Derived Ratios")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Structural fraction $S_T / (S_T+H_T)$ | **{result.structural_fraction:.2%}** |")
    if result.uniform_baseline_nats is not None:
        L.append(
            f"| $H_T$ as fraction of uniform baseline "
            f"($\\ln(V)$ = {result.uniform_baseline_nats:.4f}) | "
            f"{result.final_val_loss / result.uniform_baseline_nats:.2%} |"
        )
    if result.language_entropy_nats is not None:
        gap = result.final_val_loss - result.language_entropy_nats
        L.append(
            f"| Language / data entropy floor | "
            f"{result.language_entropy_nats:.4f} nats "
            f"({result.language_entropy_nats / LN2:.4f} bits) |"
        )
        L.append(
            f"| $H_T$ as fraction of entropy floor | "
            f"{result.final_val_loss / result.language_entropy_nats:.2%} |"
        )
        L.append(f"| Gap to entropy floor | {gap:.4f} nats ({gap / LN2:.4f} bits) |")
    L.append(f"| $S_T$ per 1B training tokens | {result.S_bits_per_train_token * 1e9:,.0f} bits |")
    L.append("")

    # Reference
    if result.gzip_complexity is not None or result.oracle_loss_nats is not None:
        L.append("## Reference Metrics")
        L.append("")
        L.append("| Metric | Value |")
        L.append("|---|---|")
        if result.gzip_complexity is not None:
            L.append(f"| Gzip complexity | {result.gzip_complexity:.4f} |")
        if result.oracle_loss_nats is not None:
            L.append(
                f"| Oracle next-token loss | {result.oracle_loss_nats:.4f} nats "
                f"({result.oracle_loss_nats / LN2:.4f} bits) |"
            )
        L.append("")

    # Interpretation
    L.append("## Interpretation")
    L.append("")
    sf = result.structural_fraction
    if sf < 0.01:
        L.append(
            "**Near-zero structural fraction** — the data contains almost no "
            "learnable structure.  Characteristic of random or trivially "
            "predictable data."
        )
    elif sf < 0.05:
        L.append(
            "**Low structural fraction** — only a modest amount of reusable "
            "structure was extracted.  Pre-pretraining transfer may be limited."
        )
    elif sf < 0.15:
        L.append(
            "**Moderate structural fraction** — the data contains meaningful "
            "learnable structure.  The model likely built non-trivial internal "
            "circuits that *may* transfer to downstream tasks."
        )
    else:
        L.append(
            "**High structural fraction** — the data is rich in learnable "
            "structure.  Strong candidate for pre-pretraining."
        )

    if result.approximation_used:
        L.append(
            "\n*Note: S_T was approximated from endpoints.  Provide "
            "`--loss-curve` for an accurate value.*"
        )
    L.append("")

    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required identifiers
    parser.add_argument(
        "--run-name", default="unnamed", help="Label for the report (default: 'unnamed')."
    )

    # Required numeric inputs
    parser.add_argument(
        "--model-params",
        type=_parse_big_int,
        required=True,
        help="Number of trainable parameters (e.g. 670127616 or 670e6).",
    )
    parser.add_argument(
        "--train-tokens",
        type=_parse_big_int,
        required=True,
        help="Total training tokens seen (batch_tokens * steps; e.g. 5242880000 or 5.2e9).",
    )
    parser.add_argument(
        "--final-val-loss",
        type=float,
        help="Per-token validation cross-entropy of the final "
        "checkpoint, in NATS (e.g. 2.6954).  If omitted, "
        "provide --val-curve to extract it automatically.",
    )

    # Loss-curve group (at least one of these)
    curve = parser.add_argument_group("Loss curve (accurate S_T)")
    curve.add_argument(
        "--loss-curve",
        type=Path,
        help="JSONL file with per-step training losses (field name controlled by --loss-key).",
    )
    curve.add_argument(
        "--loss-key",
        default="loss",
        help="JSON field name for loss in the loss-curve file (default: 'loss').",
    )
    curve.add_argument(
        "--val-curve",
        type=Path,
        help="JSONL file with per-checkpoint validation losses.  "
        "The loss from the record with the highest 'step' "
        "is used as --final-val-loss.",
    )

    # Endpoint group (fallback when --loss-curve is absent)
    ep = parser.add_argument_group("Endpoints (approximate S_T)")
    ep.add_argument(
        "--initial-loss", type=float, help="Per-token training CE at step 1, in NATS (e.g. 11.08)."
    )
    ep.add_argument(
        "--final-train-loss",
        type=float,
        help="Per-token training CE at the final step, in NATS (e.g. 2.63).",
    )

    # Reference metrics (optional)
    ref = parser.add_argument_group("Reference metrics (optional)")
    ref.add_argument(
        "--gzip-complexity",
        type=float,
        help="Global gzip complexity ratio "
        "(compressed / original bytes).  Use the *global* "
        "value, not per-sample — the global measurement "
        "compresses the full concatenated token stream the "
        "way the model sees it.",
    )
    ref.add_argument(
        "--oracle-loss",
        type=float,
        help="Oracle next-token loss in NATS — the irreducible "
        "entropy of the generating process.  Only meaningful "
        "for SYNTHETIC patterns where you control the "
        "generator (use --language-entropy for natural "
        "language data).",
    )
    ref.add_argument(
        "--uniform-baseline",
        type=float,
        help="ln(vocab_size) — the loss of a uniform model (e.g. ln(49152) ≈ 10.80).",
    )
    ref.add_argument(
        "--language-entropy",
        type=float,
        help="Shannon entropy floor of the data source, in "
        "nats/token.  For NATURAL LANGUAGE this is the "
        "estimated entropy of the language itself "
        "(~1.8–2.2 nats/token for English with subword "
        "tokenizers; derived from ~0.6–1.3 bits/char × "
        "~4 chars/token, converted with ×ln(2)).  For "
        "synthetic patterns, prefer --oracle-loss instead.",
    )

    # Output
    parser.add_argument("--json", action="store_true", help="Output JSON instead of Markdown.")
    parser.add_argument("--output", "-o", type=Path, help="Write report to FILE instead of stdout.")

    args = parser.parse_args()

    # Validate: need at least one S_T source
    if args.loss_curve is None and (args.initial_loss is None or args.final_train_loss is None):
        parser.error(
            "Provide EITHER --loss-curve (for accurate S_T) "
            "OR both --initial-loss and --final-train-loss "
            "(for approximate S_T)."
        )

    # Resolve final validation loss
    final_val_loss = args.final_val_loss
    if final_val_loss is None:
        if args.val_curve:
            if not args.val_curve.exists():
                parser.error(f"Validation file not found: {args.val_curve}")
            final_val_loss = load_val_loss(args.val_curve)
        else:
            parser.error(
                "Provide EITHER --final-val-loss OR --val-curve "
                "(to extract it automatically from a validation log)."
            )

    # Load loss curve if provided
    loss_curve = None
    if args.loss_curve:
        if not args.loss_curve.exists():
            parser.error(f"Loss-curve file not found: {args.loss_curve}")
        loss_curve = load_loss_curve(args.loss_curve, loss_key=args.loss_key)

    # Compute
    result = compute_epiplexity(
        run_name=args.run_name,
        model_params=args.model_params,
        train_tokens=args.train_tokens,
        final_val_loss=final_val_loss,
        loss_curve=loss_curve,
        initial_loss=args.initial_loss,
        final_train_loss=args.final_train_loss,
        gzip_complexity=args.gzip_complexity,
        oracle_loss_nats=args.oracle_loss,
        uniform_baseline_nats=args.uniform_baseline,
        language_entropy_nats=args.language_entropy,
    )

    # Render
    report = render_report(result, json_output=args.json)

    if args.output:
        args.output.write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
