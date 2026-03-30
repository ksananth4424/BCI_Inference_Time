"""Analyze Phase-2 intervention outputs with paired significance tests.

This script computes:
- Accuracy deltas (baseline vs corrected)
- Flip-to-correct / flip-to-wrong counts and rates
- McNemar exact and continuity-corrected statistics

Usage:
    python scripts/analyze_phase2_results.py \
      --csv results/e1_with_gt_injection/results_E1_gqa_replace_all.csv \
      --label E1 \
      --csv results/e3_with_filter_only/results_E3_gqa_remove_single.csv \
      --label E3
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any

import pandas as pd


def _to_bool_series(series: pd.Series) -> pd.Series:
    """Convert mixed-type accuracy columns to bool safely."""
    if pd.api.types.is_bool_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def mcnemar_exact_p_value(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value via Binomial(n=b+c, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0

    k = min(b, c)

    def pmf(i: int) -> float:
        return comb(n, i) * (0.5 ** n)

    p_left = sum(pmf(i) for i in range(0, k + 1))
    p_two_sided = min(1.0, 2.0 * p_left)
    return p_two_sided


def mcnemar_chi_square_cc(b: int, c: int) -> float:
    """McNemar chi-square with continuity correction."""
    n = b + c
    if n == 0:
        return 0.0
    return ((abs(b - c) - 1) ** 2) / n


def analyze_csv(csv_path: Path, label: str) -> dict[str, Any]:
    df = pd.read_csv(csv_path)

    required = {"baseline_correct", "corrected_correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing required columns: {sorted(missing)}")

    baseline = _to_bool_series(df["baseline_correct"])
    corrected = _to_bool_series(df["corrected_correct"])

    n = len(df)
    b = int((baseline & ~corrected).sum())  # flip-to-wrong
    c = int((~baseline & corrected).sum())  # flip-to-correct

    baseline_acc = float(baseline.mean()) if n else 0.0
    corrected_acc = float(corrected.mean()) if n else 0.0

    result = {
        "label": label,
        "csv": str(csv_path),
        "n_samples": n,
        "baseline_correct": int(baseline.sum()),
        "corrected_correct": int(corrected.sum()),
        "baseline_accuracy": baseline_acc,
        "corrected_accuracy": corrected_acc,
        "delta_accuracy_pp": (corrected_acc - baseline_acc) * 100.0,
        "flip_to_correct": c,
        "flip_to_wrong": b,
        "flip_to_correct_rate": (c / n) if n else 0.0,
        "flip_to_wrong_rate": (b / n) if n else 0.0,
        "net_gain": c - b,
        "net_gain_rate": ((c - b) / n) if n else 0.0,
        "mcnemar": {
            "b_baseline_only": b,
            "c_corrected_only": c,
            "exact_p_value": mcnemar_exact_p_value(b, c),
            "chi_square_cc": mcnemar_chi_square_cc(b, c),
        },
    }

    return result


def format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def print_summary(results: list[dict[str, Any]]) -> None:
    print("=" * 72)
    print("PHASE 2 INTERVENTION ANALYSIS")
    print("=" * 72)
    for r in results:
        print(f"\n[{r['label']}]")
        print(f"  Samples:              {r['n_samples']}")
        print(
            "  Accuracy:             "
            f"{format_pct(r['baseline_accuracy'])} -> {format_pct(r['corrected_accuracy'])} "
            f"(delta {r['delta_accuracy_pp']:+.1f} pp)"
        )
        print(
            "  Flip dynamics:        "
            f"to-correct={r['flip_to_correct']} ({format_pct(r['flip_to_correct_rate'])}), "
            f"to-wrong={r['flip_to_wrong']} ({format_pct(r['flip_to_wrong_rate'])})"
        )
        print(
            "  Net gain:             "
            f"{r['net_gain']} samples ({format_pct(r['net_gain_rate'])})"
        )
        print(
            "  McNemar (exact):      "
            f"p={r['mcnemar']['exact_p_value']:.3e}, "
            f"chi2_cc={r['mcnemar']['chi_square_cc']:.3f}"
        )

    if len(results) >= 2:
        print("\n" + "-" * 72)
        first, second = results[0], results[1]
        print(
            "Intervention contrast:  "
            f"{first['label']} vs {second['label']} | "
            f"delta(net_gain_rate)={(first['net_gain_rate'] - second['net_gain_rate']) * 100.0:+.1f} pp"
        )

    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase-2 result CSV files")
    parser.add_argument(
        "--csv",
        action="append",
        required=True,
        help="Path to a results CSV (repeat for multiple)",
    )
    parser.add_argument(
        "--label",
        action="append",
        required=True,
        help="Label for each CSV, same count/order as --csv",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save full analysis JSON",
    )
    args = parser.parse_args()

    if len(args.csv) != len(args.label):
        raise ValueError("Number of --csv and --label arguments must match")

    results: list[dict[str, Any]] = []
    for csv_path, label in zip(args.csv, args.label):
        results.append(analyze_csv(Path(csv_path), label))

    print_summary(results)

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump({"results": results}, f, indent=2)
        print(f"Saved JSON: {args.save_json}")


if __name__ == "__main__":
    main()
