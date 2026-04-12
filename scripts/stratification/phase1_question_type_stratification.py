"""Phase 1 for Task 4: Question-type stratified gains (E1 vs E12b).

Outputs are written to:
  results/priorityA_task4_error_stratification/

This phase creates:
  - sample metadata aligned to the experiment split
  - per-sample rows for E1 and E12b with question types
  - question-type summaries (structural + semantic)
  - E1 vs E12b comparison tables by question type
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bci_src.data.benchmarks.registry import get_benchmark_adapter


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def mcnemar_exact_p_value(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)

    def pmf(i: int) -> float:
        return comb(n, i) * (0.5 ** n)

    p_left = sum(pmf(i) for i in range(0, k + 1))
    return min(1.0, 2.0 * p_left)


def load_sample_metadata(n_samples: int, seed: int) -> pd.DataFrame:
    adapter = get_benchmark_adapter("gqa")
    samples = adapter.load_samples(n_samples, seed=seed)

    rows = []
    for s in samples:
        rows.append(
            {
                "sample_id": str(s.sample_id),
                "image_id": str(s.image_id),
                "question": s.question,
                "ground_truth": s.answer,
                "question_type_structural": (s.question_type_structural or "unknown"),
                "question_type_semantic": (s.question_type_semantic or "unknown"),
            }
        )

    return pd.DataFrame(rows)


def load_method_rows(csv_path: Path, method: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"sample_id", "baseline_correct", "corrected_correct", "flip_to_correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")

    out = pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str),
            "method": method,
            "baseline_correct": _to_bool(df["baseline_correct"]).astype(int),
            "corrected_correct": _to_bool(df["corrected_correct"]).astype(int),
            "flip_to_correct": pd.to_numeric(df["flip_to_correct"], errors="coerce").fillna(0).astype(int),
            "intervened": pd.to_numeric(df["intervened"], errors="coerce") if "intervened" in df.columns else pd.Series([pd.NA] * len(df)),
            "used_fallback": pd.to_numeric(df["used_fallback"], errors="coerce") if "used_fallback" in df.columns else pd.Series([pd.NA] * len(df)),
        }
    )

    out["flip_to_wrong"] = ((out["baseline_correct"] == 1) & (out["corrected_correct"] == 0)).astype(int)
    return out


def summarize_by_type(df: pd.DataFrame, type_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (method, qtype), g in df.groupby(["method", type_col], dropna=False):
        n = len(g)
        baseline = g["baseline_correct"].astype(bool)
        corrected = g["corrected_correct"].astype(bool)

        b = int((baseline & ~corrected).sum())  # baseline-only
        c = int((~baseline & corrected).sum())  # corrected-only

        row = {
            "method": method,
            "question_type": str(qtype),
            "n": int(n),
            "baseline_accuracy": float(baseline.mean()) if n else 0.0,
            "corrected_accuracy": float(corrected.mean()) if n else 0.0,
            "delta_accuracy_pp": float((corrected.mean() - baseline.mean()) * 100.0) if n else 0.0,
            "flip_to_correct": int(c),
            "flip_to_wrong": int(b),
            "flip_to_correct_rate": float(c / n) if n else 0.0,
            "flip_to_wrong_rate": float(b / n) if n else 0.0,
            "net_gain": int(c - b),
            "mcnemar_exact_p": float(mcnemar_exact_p_value(b, c)),
            "intervened_rate": float(pd.to_numeric(g["intervened"], errors="coerce").mean()) if "intervened" in g.columns else float("nan"),
            "fallback_rate": float(pd.to_numeric(g["used_fallback"], errors="coerce").mean()) if "used_fallback" in g.columns else float("nan"),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Method-first readability, then biggest gains.
    return out.sort_values(["method", "delta_accuracy_pp", "n"], ascending=[True, False, False]).reset_index(drop=True)


def build_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    piv = summary_df.pivot(index="question_type", columns="method")
    if ("E1" not in piv.columns.get_level_values(1)) or ("E12b" not in piv.columns.get_level_values(1)):
        return pd.DataFrame()

    out = pd.DataFrame(index=piv.index)
    out["n_e1"] = piv[("n", "E1")]
    out["n_e12b"] = piv[("n", "E12b")]
    out["e1_corrected_acc"] = piv[("corrected_accuracy", "E1")]
    out["e12b_corrected_acc"] = piv[("corrected_accuracy", "E12b")]
    out["e1_delta_pp"] = piv[("delta_accuracy_pp", "E1")]
    out["e12b_delta_pp"] = piv[("delta_accuracy_pp", "E12b")]
    out["delta_pp_gap_e1_minus_e12b"] = out["e1_delta_pp"] - out["e12b_delta_pp"]
    out["corrected_acc_gap_e1_minus_e12b"] = out["e1_corrected_acc"] - out["e12b_corrected_acc"]

    out = out.reset_index().rename(columns={"question_type": "question_type"})
    return out.sort_values("delta_pp_gap_e1_minus_e12b", ascending=False).reset_index(drop=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Task4 Phase1: question-type stratification")
    parser.add_argument("--results-root", type=Path, default=repo_root / "results")
    parser.add_argument("--output-subdir", type=str, default="priorityA_task4_error_stratification")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--e1-csv",
        type=Path,
        default=repo_root / "results/e1_n2000/results_E1_gqa_replace_all_n2000.csv",
    )
    parser.add_argument(
        "--e12b-csv",
        type=Path,
        default=repo_root / "results/e12b_n2000/results_E12b_gqa_confidence_gated_e1_tuned.csv",
    )
    args = parser.parse_args()

    out_dir = args.results_root.resolve() / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_sample_metadata(args.n_samples, args.seed)
    e1 = load_method_rows(args.e1_csv.resolve(), "E1")
    e12b = load_method_rows(args.e12b_csv.resolve(), "E12b")

    combined = pd.concat([e1, e12b], ignore_index=True)
    long_df = combined.merge(meta, on="sample_id", how="inner")

    # Phase output 1: sample metadata for replay alignment (phase 2 input)
    meta_out = out_dir / "phase1_sample_metadata.csv"
    meta.to_csv(meta_out, index=False)

    # Phase output 2: per-sample rows for both methods
    long_out = out_dir / "phase1_question_type_long.csv"
    long_df.to_csv(long_out, index=False)

    # Summaries
    structural = summarize_by_type(long_df, "question_type_structural")
    semantic = summarize_by_type(long_df, "question_type_semantic")

    structural_out = out_dir / "phase1_question_type_structural_summary.csv"
    semantic_out = out_dir / "phase1_question_type_semantic_summary.csv"
    structural.to_csv(structural_out, index=False)
    semantic.to_csv(semantic_out, index=False)

    struct_comp = build_comparison(structural)
    sem_comp = build_comparison(semantic)

    struct_comp_out = out_dir / "phase1_question_type_structural_comparison_e1_vs_e12b.csv"
    sem_comp_out = out_dir / "phase1_question_type_semantic_comparison_e1_vs_e12b.csv"
    struct_comp.to_csv(struct_comp_out, index=False)
    sem_comp.to_csv(sem_comp_out, index=False)

    # JSON mirrors
    with open(out_dir / "phase1_question_type_structural_summary.json", "w") as f:
        json.dump(structural.to_dict(orient="records"), f, indent=2)
    with open(out_dir / "phase1_question_type_semantic_summary.json", "w") as f:
        json.dump(semantic.to_dict(orient="records"), f, indent=2)
    with open(out_dir / "phase1_question_type_structural_comparison_e1_vs_e12b.json", "w") as f:
        json.dump(struct_comp.to_dict(orient="records"), f, indent=2)
    with open(out_dir / "phase1_question_type_semantic_comparison_e1_vs_e12b.json", "w") as f:
        json.dump(sem_comp.to_dict(orient="records"), f, indent=2)

    print("=" * 80)
    print("TASK4 PHASE1 COMPLETE")
    print("=" * 80)
    print(f"Sample metadata          : {meta_out}")
    print(f"Per-sample long table    : {long_out}")
    print(f"Structural summary       : {structural_out}")
    print(f"Semantic summary         : {semantic_out}")
    print(f"Structural comparison    : {struct_comp_out}")
    print(f"Semantic comparison      : {sem_comp_out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
