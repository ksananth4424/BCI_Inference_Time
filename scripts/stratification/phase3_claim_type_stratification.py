"""Phase 3 for Task 4: Claim-type stratified gains (E1 vs E12b).

Consumes:
  - phase1_question_type_long.csv (method outcomes)
  - phase2_claim_profile.csv (claim-type profile)

Produces claim-type and dominant-contradiction stratified analyses.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any

import pandas as pd

CLAIM_TYPES = [
    "object_existence",
    "attribute",
    "spatial",
    "counting",
    "action",
    "text_ocr",
]


def mcnemar_exact_p_value(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0

    k = min(b, c)

    def pmf(i: int) -> float:
        return comb(n, i) * (0.5 ** n)

    p_left = sum(pmf(i) for i in range(0, k + 1))
    return min(1.0, 2.0 * p_left)


def summarize_subset(df: pd.DataFrame, label: str) -> dict[str, Any]:
    n = len(df)
    if n == 0:
        return {
            "stratum": label,
            "n": 0,
            "baseline_accuracy": 0.0,
            "corrected_accuracy": 0.0,
            "delta_accuracy_pp": 0.0,
            "flip_to_correct": 0,
            "flip_to_wrong": 0,
            "flip_to_correct_rate": 0.0,
            "flip_to_wrong_rate": 0.0,
            "net_gain": 0,
            "mcnemar_exact_p": 1.0,
        }

    baseline = df["baseline_correct"].astype(int).astype(bool)
    corrected = df["corrected_correct"].astype(int).astype(bool)
    b = int((baseline & ~corrected).sum())
    c = int((~baseline & corrected).sum())

    return {
        "stratum": label,
        "n": int(n),
        "baseline_accuracy": float(baseline.mean()),
        "corrected_accuracy": float(corrected.mean()),
        "delta_accuracy_pp": float((corrected.mean() - baseline.mean()) * 100.0),
        "flip_to_correct": int(c),
        "flip_to_wrong": int(b),
        "flip_to_correct_rate": float(c / n),
        "flip_to_wrong_rate": float(b / n),
        "net_gain": int(c - b),
        "mcnemar_exact_p": float(mcnemar_exact_p_value(b, c)),
    }


def build_method_claim_presence_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, gm in df.groupby("method"):
        for t in CLAIM_TYPES:
            flag_col = f"has_contradicted_{t}"
            g = gm[gm[flag_col] == 1]
            r = summarize_subset(g, t)
            r["method"] = method
            r["stratification"] = "presence_of_contradicted_type"
            rows.append(r)

    out = pd.DataFrame(rows)
    return out.sort_values(["method", "delta_accuracy_pp"], ascending=[True, False]).reset_index(drop=True)


def build_method_dominant_type_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, gm in df.groupby("method"):
        for dom, gd in gm.groupby("dominant_contradicted_type"):
            r = summarize_subset(gd, str(dom))
            r["method"] = method
            r["stratification"] = "dominant_contradicted_type"
            rows.append(r)

    out = pd.DataFrame(rows)
    return out.sort_values(["method", "n"], ascending=[True, False]).reset_index(drop=True)


def build_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    piv = summary_df.pivot(index="stratum", columns="method")
    if ("E1" not in piv.columns.get_level_values(1)) or ("E12b" not in piv.columns.get_level_values(1)):
        return pd.DataFrame()

    out = pd.DataFrame(index=piv.index)
    out["n_e1"] = piv[("n", "E1")]
    out["n_e12b"] = piv[("n", "E12b")]
    out["e1_corrected_accuracy"] = piv[("corrected_accuracy", "E1")]
    out["e12b_corrected_accuracy"] = piv[("corrected_accuracy", "E12b")]
    out["e1_delta_pp"] = piv[("delta_accuracy_pp", "E1")]
    out["e12b_delta_pp"] = piv[("delta_accuracy_pp", "E12b")]
    out["delta_pp_gap_e1_minus_e12b"] = out["e1_delta_pp"] - out["e12b_delta_pp"]
    out["corrected_acc_gap_e1_minus_e12b"] = out["e1_corrected_accuracy"] - out["e12b_corrected_accuracy"]
    out = out.reset_index().rename(columns={"stratum": "stratum"})
    return out.sort_values("delta_pp_gap_e1_minus_e12b", ascending=False).reset_index(drop=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Task4 Phase3: claim-type stratification")
    parser.add_argument("--results-root", type=Path, default=repo_root / "results")
    parser.add_argument("--output-subdir", type=str, default="priorityA_task4_error_stratification")
    parser.add_argument("--phase1-long", type=Path, default=None)
    parser.add_argument("--phase2-claim-profile", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.results_root.resolve() / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    phase1_long = args.phase1_long.resolve() if args.phase1_long else (out_dir / "phase1_question_type_long.csv")
    phase2_claim = (
        args.phase2_claim_profile.resolve() if args.phase2_claim_profile else (out_dir / "phase2_claim_profile.csv")
    )

    if not phase1_long.exists():
        raise FileNotFoundError(f"Missing phase1 long file: {phase1_long}")
    if not phase2_claim.exists():
        raise FileNotFoundError(f"Missing phase2 claim profile file: {phase2_claim}")

    outcomes = pd.read_csv(phase1_long)
    claims = pd.read_csv(phase2_claim)

    # Build binary presence columns for contradicted claim types.
    for t in CLAIM_TYPES:
        c = f"claims_contradicted_{t}"
        if c not in claims.columns:
            claims[c] = 0
        claims[f"has_contradicted_{t}"] = (pd.to_numeric(claims[c], errors="coerce").fillna(0) > 0).astype(int)

    keep_claim_cols = ["sample_id", "dominant_contradicted_type"] + [f"has_contradicted_{t}" for t in CLAIM_TYPES]
    merged = outcomes.merge(claims[keep_claim_cols], on="sample_id", how="inner")

    # Presence-based stratification by claim type
    presence_summary = build_method_claim_presence_table(merged)
    presence_comp = build_comparison(presence_summary)

    # Dominant contradicted type stratification
    dominant_summary = build_method_dominant_type_table(merged)
    dominant_comp = build_comparison(dominant_summary)

    # Save outputs
    presence_csv = out_dir / "phase3_claim_type_presence_summary.csv"
    presence_json = out_dir / "phase3_claim_type_presence_summary.json"
    presence_comp_csv = out_dir / "phase3_claim_type_presence_comparison_e1_vs_e12b.csv"
    presence_comp_json = out_dir / "phase3_claim_type_presence_comparison_e1_vs_e12b.json"

    dominant_csv = out_dir / "phase3_claim_type_dominant_summary.csv"
    dominant_json = out_dir / "phase3_claim_type_dominant_summary.json"
    dominant_comp_csv = out_dir / "phase3_claim_type_dominant_comparison_e1_vs_e12b.csv"
    dominant_comp_json = out_dir / "phase3_claim_type_dominant_comparison_e1_vs_e12b.json"

    merged_csv = out_dir / "phase3_merged_outcomes_claims.csv"

    merged.to_csv(merged_csv, index=False)
    presence_summary.to_csv(presence_csv, index=False)
    presence_comp.to_csv(presence_comp_csv, index=False)
    dominant_summary.to_csv(dominant_csv, index=False)
    dominant_comp.to_csv(dominant_comp_csv, index=False)

    with open(presence_json, "w") as f:
        json.dump(presence_summary.to_dict(orient="records"), f, indent=2)
    with open(presence_comp_json, "w") as f:
        json.dump(presence_comp.to_dict(orient="records"), f, indent=2)
    with open(dominant_json, "w") as f:
        json.dump(dominant_summary.to_dict(orient="records"), f, indent=2)
    with open(dominant_comp_json, "w") as f:
        json.dump(dominant_comp.to_dict(orient="records"), f, indent=2)

    print("=" * 80)
    print("TASK4 PHASE3 COMPLETE")
    print("=" * 80)
    print(f"Merged outcomes+claims      : {merged_csv}")
    print(f"Presence summary            : {presence_csv}")
    print(f"Presence comparison         : {presence_comp_csv}")
    print(f"Dominant summary            : {dominant_csv}")
    print(f"Dominant comparison         : {dominant_comp_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
