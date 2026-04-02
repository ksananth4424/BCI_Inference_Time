"""Build a unified comparison table for competitor baselines and BCI outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_competitor_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sample_id", "method", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in competitor CSV: {sorted(missing)}")
    df["correct"] = df["correct"].astype(int)
    return df


def load_bci_results(path: Path, method_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sample_id", "baseline_correct", "corrected_correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in BCI CSV {path}: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "sample_id": df["sample_id"],
            "method": method_name,
            "correct": df["corrected_correct"].astype(int),
        }
    )
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, group in df.groupby("method"):
        n = int(len(group))
        n_correct = int(group["correct"].sum())
        rows.append(
            {
                "method": method,
                "n": n,
                "n_correct": n_correct,
                "accuracy": n_correct / n if n else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze full competitor matrix")
    parser.add_argument("--competitor-csv", type=Path, required=True)
    parser.add_argument("--e1-csv", type=Path, required=True)
    parser.add_argument("--e3-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    comp = load_competitor_results(args.competitor_csv)
    e1 = load_bci_results(args.e1_csv, "bci_e1")
    e3 = load_bci_results(args.e3_csv, "bci_e3")

    merged = pd.concat([comp[["sample_id", "method", "correct"]], e1, e3], ignore_index=True)
    table = summarize(merged)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "competitor_matrix_summary.csv"
    out_json = args.output_dir / "competitor_matrix_summary.json"
    table.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(table.to_dict(orient="records"), f, indent=2)

    print("=" * 72)
    print("UNIFIED COMPETITOR MATRIX")
    print("=" * 72)
    for _, row in table.iterrows():
        print(f"{row['method']:15s}: {row['accuracy']:.2%} ({int(row['n_correct'])}/{int(row['n'])})")
    print(f"CSV:  {out_csv}")
    print(f"JSON: {out_json}")
    print("=" * 72)


if __name__ == "__main__":
    main()
