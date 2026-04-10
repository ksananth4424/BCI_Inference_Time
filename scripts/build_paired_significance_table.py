"""Build paired significance tables across methods using McNemar exact tests.

This script creates two outputs:
1) n=500 all-method paired table:
   - baseline (vanilla), E1, E3, E12, E12b, self_correct, cove_lite, resample_lite
2) n=2000 subset paired table:
   - baseline (from each run), E1, E12b

It reads existing result CSVs, aligns by sample_id, computes pairwise wins/losses,
McNemar exact p-values, and Holm-Bonferroni adjusted p-values.

Usage:
    python scripts/build_paired_significance_table.py

Optional:
    python scripts/build_paired_significance_table.py --output-dir results
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


METHOD_LABELS = {
    "baseline": "Baseline",
    "self_correct": "Self Correct",
    "cove_lite": "Cove Lite",
    "resample_lite": "Resample Lite",
}


def format_method_label(name: str) -> str:
    """Convert internal method keys to reader-friendly labels."""
    if name in METHOD_LABELS:
        return METHOD_LABELS[name]
    if name.startswith("E"):
        return name
    return name.replace("_", " ").title()


# -----------------------------------------------------------------------------
# Statistical helpers
# -----------------------------------------------------------------------------

def mcnemar_exact_p_value(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value via Binomial(n=b+c, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0

    k = min(b, c)

    def pmf(i: int) -> float:
        return comb(n, i) * (0.5 ** n)

    p_left = sum(pmf(i) for i in range(0, k + 1))
    return min(1.0, 2.0 * p_left)


def mcnemar_chi_square_cc(b: int, c: int) -> float:
    """McNemar chi-square with continuity correction."""
    n = b + c
    if n == 0:
        return 0.0
    return ((abs(b - c) - 1) ** 2) / n


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Return Holm-Bonferroni adjusted p-values in original order."""
    m = len(p_values)
    if m == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted_sorted = [0.0] * m

    for rank, (_, p) in enumerate(indexed, start=1):
        adjusted_sorted[rank - 1] = min(1.0, p * (m - rank + 1))

    # Enforce monotonicity in sorted order.
    for i in range(1, m):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])

    out = [1.0] * m
    for (orig_idx, _), adj in zip(indexed, adjusted_sorted):
        out[orig_idx] = adj

    return out


# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------

def _to_bool_series(series: pd.Series) -> pd.Series:
    """Convert mixed-type correctness columns to bool safely."""
    if pd.api.types.is_bool_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def load_bci_correctness(csv_path: Path, method_name: str) -> pd.DataFrame:
    """Load a BCI CSV and return canonical (sample_id, method, correct) rows."""
    df = pd.read_csv(csv_path)

    required = {"sample_id", "corrected_correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str),
            "method": method_name,
            "correct": _to_bool_series(df["corrected_correct"]).astype(int),
        }
    )
    return out


def load_baseline_from_bci(csv_path: Path, method_name: str = "baseline") -> pd.DataFrame:
    """Load baseline correctness from a BCI CSV."""
    df = pd.read_csv(csv_path)

    required = {"sample_id", "baseline_correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str),
            "method": method_name,
            "correct": _to_bool_series(df["baseline_correct"]).astype(int),
        }
    )
    return out


def load_competitor_long(csv_path: Path) -> pd.DataFrame:
    """Load competitor long-form CSV with (sample_id, method, correct)."""
    df = pd.read_csv(csv_path)

    required = {"sample_id", "method", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str),
            "method": df["method"].astype(str),
            "correct": _to_bool_series(df["correct"]).astype(int),
        }
    )
    return out


# -----------------------------------------------------------------------------
# Pairwise computation
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PairwiseResult:
    method_a: str
    method_b: str
    n_paired: int
    a_correct: int
    b_correct: int
    a_accuracy: float
    b_accuracy: float
    delta_accuracy_pp_a_minus_b: float
    a_only_correct: int
    b_only_correct: int
    exact_p_value: float
    chi_square_cc: float


def compute_pairwise(method_a: str, method_b: str, wide: pd.DataFrame) -> PairwiseResult:
    """Compute paired McNemar stats for one method pair on aligned rows."""
    pair = wide[[method_a, method_b]].dropna().astype(int)

    a = pair[method_a].astype(bool)
    b = pair[method_b].astype(bool)

    n = len(pair)
    a_correct = int(a.sum())
    b_correct = int(b.sum())

    # McNemar convention:
    # b = A-only-correct, c = B-only-correct
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())

    exact_p = mcnemar_exact_p_value(a_only, b_only)
    chi_cc = mcnemar_chi_square_cc(a_only, b_only)

    a_acc = float(a.mean()) if n else 0.0
    b_acc = float(b.mean()) if n else 0.0

    return PairwiseResult(
        method_a=method_a,
        method_b=method_b,
        n_paired=n,
        a_correct=a_correct,
        b_correct=b_correct,
        a_accuracy=a_acc,
        b_accuracy=b_acc,
        delta_accuracy_pp_a_minus_b=(a_acc - b_acc) * 100.0,
        a_only_correct=a_only,
        b_only_correct=b_only,
        exact_p_value=exact_p,
        chi_square_cc=chi_cc,
    )


def build_pairwise_table(df_long: pd.DataFrame, method_order: List[str]) -> pd.DataFrame:
    """Build full pairwise table for provided methods."""
    # Keep only requested methods and one row per (sample_id, method)
    filtered = df_long[df_long["method"].isin(method_order)].copy()
    filtered = (
        filtered.sort_values(["sample_id", "method"]) 
        .drop_duplicates(subset=["sample_id", "method"], keep="last")
    )

    wide = filtered.pivot(index="sample_id", columns="method", values="correct")

    pairwise_rows: List[Dict[str, Any]] = []
    raw_p_values: List[float] = []

    for i in range(len(method_order)):
        for j in range(i + 1, len(method_order)):
            a = method_order[i]
            b = method_order[j]
            if a not in wide.columns or b not in wide.columns:
                continue

            res = compute_pairwise(a, b, wide)
            row = {
                "method_a": res.method_a,
                "method_b": res.method_b,
                "n_paired": res.n_paired,
                "a_correct": res.a_correct,
                "b_correct": res.b_correct,
                "a_accuracy": res.a_accuracy,
                "b_accuracy": res.b_accuracy,
                "delta_accuracy_pp_a_minus_b": res.delta_accuracy_pp_a_minus_b,
                "a_only_correct": res.a_only_correct,
                "b_only_correct": res.b_only_correct,
                "mcnemar_exact_p": res.exact_p_value,
                "mcnemar_chi_square_cc": res.chi_square_cc,
                "winner": (
                    res.method_a
                    if res.delta_accuracy_pp_a_minus_b > 0
                    else (res.method_b if res.delta_accuracy_pp_a_minus_b < 0 else "tie")
                ),
            }
            pairwise_rows.append(row)
            raw_p_values.append(res.exact_p_value)

    adjusted = holm_bonferroni(raw_p_values)
    for row, adj in zip(pairwise_rows, adjusted):
        row["mcnemar_exact_p_holm"] = adj

    out_unique = pd.DataFrame(pairwise_rows)
    if out_unique.empty:
        return out_unique

    # Expand from unique unordered pairs to method_a-wise ordered pairs.
    # This yields (n_methods * (n_methods - 1)) rows, grouped by method_a.
    mirrored = out_unique.copy()
    mirrored["method_a"] = out_unique["method_b"]
    mirrored["method_b"] = out_unique["method_a"]
    mirrored["a_correct"] = out_unique["b_correct"]
    mirrored["b_correct"] = out_unique["a_correct"]
    mirrored["a_accuracy"] = out_unique["b_accuracy"]
    mirrored["b_accuracy"] = out_unique["a_accuracy"]
    mirrored["delta_accuracy_pp_a_minus_b"] = -out_unique["delta_accuracy_pp_a_minus_b"]
    mirrored["a_only_correct"] = out_unique["b_only_correct"]
    mirrored["b_only_correct"] = out_unique["a_only_correct"]
    mirrored["winner"] = out_unique["winner"].map(
        lambda w: "tie" if w == "tie" else None
    )

    # Recompute winner labels for mirrored direction.
    mirrored.loc[mirrored["delta_accuracy_pp_a_minus_b"] > 0, "winner"] = mirrored.loc[
        mirrored["delta_accuracy_pp_a_minus_b"] > 0, "method_a"
    ]
    mirrored.loc[mirrored["delta_accuracy_pp_a_minus_b"] < 0, "winner"] = mirrored.loc[
        mirrored["delta_accuracy_pp_a_minus_b"] < 0, "method_b"
    ]

    out = pd.concat([out_unique, mirrored], ignore_index=True)

    # Keep a deterministic, easy-to-read order that follows method_order.
    # This is typically easier for paper/table review than p-value sorting.
    method_rank = {m: i for i, m in enumerate(method_order)}
    out["_rank_a"] = out["method_a"].map(method_rank)
    out["_rank_b"] = out["method_b"].map(method_rank)

    out = out.sort_values(["_rank_a", "_rank_b"]).drop(columns=["_rank_a", "_rank_b"]).reset_index(drop=True)

    # Reader-friendly method labels.
    out["method_a"] = out["method_a"].map(format_method_label)
    out["method_b"] = out["method_b"].map(format_method_label)
    out["winner"] = out["winner"].map(lambda x: format_method_label(x) if x != "tie" else "Tie")

    # Column order optimized for readability.
    preferred_cols = [
        "method_a",
        "method_b",
        "winner",
        "n_paired",
        "a_correct",
        "b_correct",
        "a_accuracy",
        "b_accuracy",
        "delta_accuracy_pp_a_minus_b",
        "a_only_correct",
        "b_only_correct",
        "mcnemar_exact_p",
        "mcnemar_exact_p_holm",
        "mcnemar_chi_square_cc",
    ]
    cols = [c for c in preferred_cols if c in out.columns] + [c for c in out.columns if c not in preferred_cols]
    return out[cols]


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------

def build_n500_all_methods(repo_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Construct n=500 all-method long table and pairwise stats."""
    # BCI methods from dedicated phase2 runs
    e1_csv = repo_root / "results/e1_with_gt_injection/results_E1_gqa_replace_all.csv"
    e3_csv = repo_root / "results/e3_with_filter_only/results_E3_gqa_remove_single.csv"
    e12_csv = repo_root / "results/e12_main/results_E12_gqa_confidence_gated_e1.csv"
    e12b_csv = repo_root / "results/e12b_main/results_E12b_gqa_confidence_gated_e1_tuned.csv"

    competitors_csv = repo_root / "results/competitors_main/competitor_results_gqa_500.csv"

    baseline_df = load_baseline_from_bci(e1_csv, method_name="baseline")
    e1_df = load_bci_correctness(e1_csv, method_name="E1")
    e3_df = load_bci_correctness(e3_csv, method_name="E3")
    e12_df = load_bci_correctness(e12_csv, method_name="E12")
    e12b_df = load_bci_correctness(e12b_csv, method_name="E12b")

    comp_df = load_competitor_long(competitors_csv)

    # Normalize competitor method names
    comp_df = comp_df.replace(
        {
            "method": {
                "vanilla": "baseline",
                "self_correct": "self_correct",
                "cove_lite": "cove_lite",
                "resample_lite": "resample_lite",
            }
        }
    )

    combined = pd.concat(
        [baseline_df, e1_df, e3_df, e12_df, e12b_df, comp_df],
        ignore_index=True,
    )

    method_order = [
        "baseline",
        "E1",
        "E3",
        "E12",
        "E12b",
        "self_correct",
        "cove_lite",
        "resample_lite",
    ]

    pairwise = build_pairwise_table(combined, method_order)

    meta = {
        "dataset": "gqa",
        "split": "val_balanced",
        "n_target": 500,
        "methods": method_order,
        "sources": {
            "E1": str(e1_csv),
            "E3": str(e3_csv),
            "E12": str(e12_csv),
            "E12b": str(e12b_csv),
            "competitors": str(competitors_csv),
            "baseline_source": str(e1_csv),
        },
    }

    return pairwise, meta


def build_n2000_subset(repo_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Construct n=2000 subset pairwise table (baseline, E1, E12b)."""
    e1_csv = repo_root / "results/e1_n2000/results_E1_gqa_replace_all_n2000.csv"
    e12b_csv = repo_root / "results/e12b_n2000/results_E12b_gqa_confidence_gated_e1_tuned.csv"

    e1_df = pd.read_csv(e1_csv)
    e12b_df = pd.read_csv(e12b_csv)

    # Baseline taken from each run's baseline_correct on matched sample IDs.
    # We'll define a canonical baseline from E1 run and then align by sample_id.
    baseline_df = pd.DataFrame(
        {
            "sample_id": e1_df["sample_id"].astype(str),
            "method": "baseline",
            "correct": _to_bool_series(e1_df["baseline_correct"]).astype(int),
        }
    )

    e1_long = pd.DataFrame(
        {
            "sample_id": e1_df["sample_id"].astype(str),
            "method": "E1",
            "correct": _to_bool_series(e1_df["corrected_correct"]).astype(int),
        }
    )

    e12b_long = pd.DataFrame(
        {
            "sample_id": e12b_df["sample_id"].astype(str),
            "method": "E12b",
            "correct": _to_bool_series(e12b_df["corrected_correct"]).astype(int),
        }
    )

    combined = pd.concat([baseline_df, e1_long, e12b_long], ignore_index=True)
    method_order = ["baseline", "E1", "E12b"]
    pairwise = build_pairwise_table(combined, method_order)

    meta = {
        "dataset": "gqa",
        "split": "val_balanced",
        "n_target": 2000,
        "methods": method_order,
        "sources": {
            "E1": str(e1_csv),
            "E12b": str(e12b_csv),
            "baseline_source": str(e1_csv),
        },
        "note": "E1 and E12b are aligned by sample_id intersection for pairwise tests.",
    }

    return pairwise, meta


def save_outputs(
    out_dir: Path,
    name_prefix: str,
    table: pd.DataFrame,
    metadata: Dict[str, Any],
) -> Tuple[Path, Path]:
    """Save CSV and JSON outputs for one table."""
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{name_prefix}.csv"
    json_path = out_dir / f"{name_prefix}.json"

    table.to_csv(csv_path, index=False)

    payload = {
        "metadata": metadata,
        "row_count": int(len(table)),
        "table": table.to_dict(orient="records"),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    return csv_path, json_path


def save_pdf_report(
    pdf_path: Path,
    n500_table: pd.DataFrame,
    n2000_table: pd.DataFrame,
) -> Path:
    """Save both paired tables into a multi-page PDF report."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    def _delta_color(v: float) -> tuple:
        # Green for positive, red for negative, light gray near zero.
        if v > 0:
            strength = min(1.0, abs(v) / 25.0)
            return (0.85 - 0.35 * strength, 1.0, 0.85 - 0.35 * strength)
        if v < 0:
            strength = min(1.0, abs(v) / 25.0)
            return (1.0, 0.85 - 0.35 * strength, 0.85 - 0.35 * strength)
        return (0.95, 0.95, 0.95)

    def _p_color(p: float) -> tuple:
        # Highlight significance thresholds.
        if p < 0.01:
            return (1.0, 0.9, 0.6)
        if p < 0.05:
            return (1.0, 0.95, 0.75)
        return (1.0, 1.0, 1.0)

    def _add_table_page(pdf: PdfPages, title: str, table: pd.DataFrame) -> None:
        raw_df = table.copy()
        display_df = raw_df.copy()

        for col in [
            "a_accuracy",
            "b_accuracy",
            "delta_accuracy_pp_a_minus_b",
            "mcnemar_exact_p",
            "mcnemar_exact_p_holm",
            "mcnemar_chi_square_cc",
        ]:
            if col in display_df.columns:
                if col in {"a_accuracy", "b_accuracy"}:
                    display_df[col] = display_df[col].map(lambda x: f"{x:.3f}")
                elif col in {"mcnemar_exact_p", "mcnemar_exact_p_holm"}:
                    display_df[col] = display_df[col].map(lambda x: f"{x:.3e}")
                else:
                    display_df[col] = display_df[col].map(lambda x: f"{x:.3f}")

        rows_per_page = 18
        n_rows = len(display_df)
        n_pages = max(1, (n_rows + rows_per_page - 1) // rows_per_page)

        for page_idx in range(n_pages):
            start = page_idx * rows_per_page
            end = min((page_idx + 1) * rows_per_page, n_rows)
            chunk = display_df.iloc[start:end]
            chunk_raw = raw_df.iloc[start:end]

            fig, ax = plt.subplots(figsize=(16, 9))
            ax.axis("off")
            page_title = f"{title} (rows {start + 1}-{end} of {n_rows})"
            ax.set_title(page_title, fontsize=12, pad=12)

            table_artist = ax.table(
                cellText=chunk.values,
                colLabels=chunk.columns,
                cellLoc="center",
                loc="center",
            )
            table_artist.auto_set_font_size(False)
            table_artist.set_fontsize(8)
            table_artist.scale(1.0, 1.2)

            # Header styling
            for col_idx in range(len(chunk.columns)):
                cell = table_artist[(0, col_idx)]
                cell.set_facecolor((0.9, 0.92, 0.96))
                cell.get_text().set_weight("bold")

            # Cell color coding for readability
            delta_col = chunk.columns.get_loc("delta_accuracy_pp_a_minus_b") if "delta_accuracy_pp_a_minus_b" in chunk.columns else None
            p_col = chunk.columns.get_loc("mcnemar_exact_p_holm") if "mcnemar_exact_p_holm" in chunk.columns else None

            for row_idx in range(len(chunk)):
                if delta_col is not None:
                    delta_val = float(chunk_raw.iloc[row_idx]["delta_accuracy_pp_a_minus_b"])
                    table_artist[(row_idx + 1, delta_col)].set_facecolor(_delta_color(delta_val))
                if p_col is not None:
                    p_val = float(chunk_raw.iloc[row_idx]["mcnemar_exact_p_holm"])
                    table_artist[(row_idx + 1, p_col)].set_facecolor(_p_color(p_val))

                # zebra striping for row tracking
                if row_idx % 2 == 1:
                    for col_idx in range(len(chunk.columns)):
                        if col_idx in {delta_col, p_col}:
                            continue
                        table_artist[(row_idx + 1, col_idx)].set_facecolor((0.985, 0.985, 0.985))

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    with PdfPages(pdf_path) as pdf:
        _add_table_page(pdf, "Paired significance table: n=500 all methods", n500_table)
        _add_table_page(pdf, "Paired significance table: n=2000 subset", n2000_table)

    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build paired significance tables across major methods",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from script location)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs (default: <repo-root>/results)",
    )
    parser.add_argument(
        "--task-subdir",
        type=str,
        default="priorityA_task3_paired_significance",
        help="Subfolder name under output-dir for this task's artifacts",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_base_dir = (args.output_dir.resolve() if args.output_dir else (repo_root / "results"))
    output_dir = output_base_dir / args.task_subdir

    n500_table, n500_meta = build_n500_all_methods(repo_root)
    n2000_table, n2000_meta = build_n2000_subset(repo_root)

    n500_csv, n500_json = save_outputs(
        output_dir,
        "paired_significance_n500_all_methods",
        n500_table,
        n500_meta,
    )
    n2000_csv, n2000_json = save_outputs(
        output_dir,
        "paired_significance_n2000_subset",
        n2000_table,
        n2000_meta,
    )
    pdf_path = save_pdf_report(
        output_dir / "paired_significance_tables.pdf",
        n500_table,
        n2000_table,
    )

    print("=" * 80)
    print("PAIRED SIGNIFICANCE TABLES WRITTEN")
    print("=" * 80)
    print(f"n=500 CSV : {n500_csv}")
    print(f"n=500 JSON: {n500_json}")
    print(f"n=2000 CSV : {n2000_csv}")
    print(f"n=2000 JSON: {n2000_json}")
    print(f"PDF report : {pdf_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
