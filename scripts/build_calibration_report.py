"""Priority B Task 7: Calibration and abstention analysis.

Creates calibration/reliability bins and abstention tradeoff analysis using
`intervention_score` as the risk proxy.

This script only reads existing result CSVs and writes fresh artifacts.
No existing scripts are modified.

Default outputs:
  results/priorityB_task7_calibration/
    - calibration_bins.csv
    - calibration_bins.json
    - abstention_sweep.csv
    - abstention_sweep.json
    - task7_calibration_report.pdf

Usage:
    python scripts/build_calibration_report.py
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def parse_seed_from_path(path: Path) -> int | None:
    m = re.search(r"seed(\d+)", str(path).lower())
    if not m:
        return None
    return int(m.group(1))


def parse_run_label(path: Path) -> str:
    p = str(path)
    if "seed123" in p:
        return "seed123"
    if "seed456" in p:
        return "seed456"
    if "n2000" in p:
        return "seed42"
    return path.parent.name


@dataclass
class MethodConfig:
    name: str
    patterns: list[str]
    color: str


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def discover_method_files(results_root: Path, cfg: MethodConfig) -> list[Path]:
    found: list[Path] = []
    for pattern in cfg.patterns:
        found.extend(sorted(results_root.glob(pattern)))

    # unique while preserving order
    unique: list[Path] = []
    seen = set()
    for p in found:
        if p.exists() and p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def load_method_runs(results_root: Path, cfg: MethodConfig) -> pd.DataFrame:
    files = discover_method_files(results_root, cfg)
    rows: list[pd.DataFrame] = []

    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        required = {"sample_id", "intervention_score", "baseline_correct", "corrected_correct"}
        if not required.issubset(df.columns):
            continue

        out = pd.DataFrame(
            {
                "sample_id": df["sample_id"].astype(str),
                "intervention_score": pd.to_numeric(df["intervention_score"], errors="coerce"),
                "baseline_correct": to_bool_series(df["baseline_correct"]).astype(int),
                "corrected_correct": to_bool_series(df["corrected_correct"]).astype(int),
                "intervened": pd.to_numeric(df["intervened"], errors="coerce") if "intervened" in df.columns else np.nan,
                "used_fallback": pd.to_numeric(df["used_fallback"], errors="coerce") if "used_fallback" in df.columns else np.nan,
            }
        )
        out["method"] = cfg.name
        out["run_label"] = parse_run_label(p)
        out["seed"] = parse_seed_from_path(p)
        out["source_csv"] = str(p)

        # Derived targets
        out["corrected_error"] = 1 - out["corrected_correct"]
        out["baseline_error"] = 1 - out["baseline_correct"]

        out = out.dropna(subset=["intervention_score"]).copy()
        out = out[(out["intervention_score"] >= 0) & (out["intervention_score"] <= 1)].copy()
        rows.append(out)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


# -----------------------------------------------------------------------------
# Calibration metrics
# -----------------------------------------------------------------------------

def make_equal_width_bins(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binned = df.copy()
    binned["bin_id"] = pd.cut(
        binned["intervention_score"],
        bins=bins,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )
    return binned


def calibration_table(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    b = make_equal_width_bins(df, n_bins=n_bins)

    grp = (
        b.groupby(["method", "run_label", "bin_id"], dropna=False)
        .agg(
            n=("sample_id", "count"),
            mean_score=("intervention_score", "mean"),
            corrected_error_rate=("corrected_error", "mean"),
            corrected_accuracy=("corrected_correct", "mean"),
            baseline_accuracy=("baseline_correct", "mean"),
            intervened_rate=("intervened", "mean"),
            fallback_rate=("used_fallback", "mean"),
        )
        .reset_index()
    )

    grp["delta_accuracy_pp"] = (grp["corrected_accuracy"] - grp["baseline_accuracy"]) * 100.0
    grp["abs_calibration_gap"] = (grp["mean_score"] - grp["corrected_error_rate"]).abs()

    # Add bin boundaries for plotting/tables.
    grp["bin_left"] = grp["bin_id"].map(lambda i: i / n_bins if pd.notna(i) else np.nan)
    grp["bin_right"] = grp["bin_id"].map(lambda i: (i + 1) / n_bins if pd.notna(i) else np.nan)
    grp["bin_label"] = grp.apply(
        lambda r: f"[{r['bin_left']:.1f}, {r['bin_right']:.1f})" if pd.notna(r["bin_left"]) else "NA",
        axis=1,
    )

    return grp.sort_values(["method", "run_label", "bin_id"]).reset_index(drop=True)


def ece_from_bins(calib_df: pd.DataFrame) -> pd.DataFrame:
    if calib_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (method, run_label), g in calib_df.groupby(["method", "run_label"]):
        n_total = g["n"].sum()
        if n_total <= 0:
            continue
        ece = float(((g["n"] / n_total) * g["abs_calibration_gap"]).sum())
        rows.append(
            {
                "method": method,
                "run_label": run_label,
                "n": int(n_total),
                "ece": ece,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    agg = (
        out.groupby("method", dropna=False)
        .agg(ece_mean=("ece", "mean"), ece_std=("ece", "std"), runs=("run_label", "count"))
        .reset_index()
    )
    return agg.sort_values("method").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Abstention sweep
# -----------------------------------------------------------------------------

def abstention_sweep(df: pd.DataFrame, n_thresholds: int = 41) -> pd.DataFrame:
    """Abstain on high-risk samples where intervention_score > threshold."""
    if df.empty:
        return pd.DataFrame()

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    rows: list[dict[str, Any]] = []

    for (method, run_label), g in df.groupby(["method", "run_label"]):
        n_total = len(g)
        if n_total == 0:
            continue

        for t in thresholds:
            kept = g[g["intervention_score"] <= t]
            n_kept = len(kept)
            coverage = n_kept / n_total

            if n_kept == 0:
                corrected_acc = np.nan
                corrected_err = np.nan
                baseline_acc = np.nan
                delta_pp = np.nan
                intervened_rate = np.nan
            else:
                corrected_acc = float(kept["corrected_correct"].mean())
                corrected_err = 1.0 - corrected_acc
                baseline_acc = float(kept["baseline_correct"].mean())
                delta_pp = (corrected_acc - baseline_acc) * 100.0
                intervened_rate = float(kept["intervened"].mean()) if "intervened" in kept.columns else np.nan

            rows.append(
                {
                    "method": method,
                    "run_label": run_label,
                    "threshold": float(t),
                    "n_total": int(n_total),
                    "n_kept": int(n_kept),
                    "coverage": float(coverage),
                    "corrected_accuracy_kept": corrected_acc,
                    "corrected_error_kept": corrected_err,
                    "baseline_accuracy_kept": baseline_acc,
                    "delta_accuracy_pp_kept": delta_pp,
                    "intervened_rate_kept": intervened_rate,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["method", "run_label", "threshold"]).reset_index(drop=True)


def summarize_abstention(sweep_df: pd.DataFrame) -> pd.DataFrame:
    if sweep_df.empty:
        return pd.DataFrame()

    # Summarize threshold operating points for each method.
    target_coverages = [0.6, 0.7, 0.8, 0.9]
    rows: list[dict[str, Any]] = []

    for method, g in sweep_df.groupby("method"):
        # Average across runs first.
        avg = (
            g.groupby("threshold", dropna=False)
            .agg(
                coverage=("coverage", "mean"),
                corrected_accuracy_kept=("corrected_accuracy_kept", "mean"),
                corrected_error_kept=("corrected_error_kept", "mean"),
                delta_accuracy_pp_kept=("delta_accuracy_pp_kept", "mean"),
            )
            .reset_index()
        )

        # AURC-like area over coverage-error curve.
        curve = avg.dropna(subset=["coverage", "corrected_error_kept"]).sort_values("coverage")
        if len(curve) >= 2:
            x = curve["coverage"].to_numpy(dtype=float)
            y = curve["corrected_error_kept"].to_numpy(dtype=float)
            if hasattr(np, "trapezoid"):
                aurc = float(np.trapezoid(y, x))
            else:
                # Fallback for older/newer NumPy variants without trapezoid helper.
                aurc = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))
        else:
            aurc = np.nan

        rows.append(
            {
                "method": method,
                "operating_point": "AURC_like",
                "coverage": np.nan,
                "threshold": np.nan,
                "corrected_accuracy_kept": np.nan,
                "corrected_error_kept": np.nan,
                "delta_accuracy_pp_kept": np.nan,
                "value": aurc,
            }
        )

        for c in target_coverages:
            idx = (avg["coverage"] - c).abs().idxmin()
            r = avg.loc[idx]
            rows.append(
                {
                    "method": method,
                    "operating_point": f"coverage~{c:.1f}",
                    "coverage": float(r["coverage"]),
                    "threshold": float(r["threshold"]),
                    "corrected_accuracy_kept": float(r["corrected_accuracy_kept"]),
                    "corrected_error_kept": float(r["corrected_error_kept"]),
                    "delta_accuracy_pp_kept": float(r["delta_accuracy_pp_kept"]),
                    "value": np.nan,
                }
            )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def save_pdf(
    out_pdf: Path,
    raw_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    ece_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    abst_sum_df: pd.DataFrame,
    method_colors: dict[str, str],
) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_pdf) as pdf:
        # Page 1: headline summary
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.axis("off")
        ax.set_title("Task 7: Calibration + Abstention Summary", fontsize=18, weight="bold", pad=16)

        lines = []
        for method, g in raw_df.groupby("method"):
            n = len(g)
            bacc = float(g["baseline_correct"].mean())
            cacc = float(g["corrected_correct"].mean())
            gain = (cacc - bacc) * 100.0
            lines.append(f"{method}: n={n}, baseline={bacc:.2%}, corrected={cacc:.2%}, gain={gain:+.2f} pp")

        ece_lines = []
        if not ece_df.empty:
            for _, r in ece_df.iterrows():
                ece_lines.append(f"{r['method']}: ECE={r['ece_mean']:.4f} ± {0.0 if pd.isna(r['ece_std']) else r['ece_std']:.4f}")

        summary_text = (
            "Calibration objective: align intervention_score with observed corrected error.\n"
            "Abstention policy evaluated: keep samples with score <= threshold (abstain on high-risk).\n\n"
            "Method-level accuracy summary:\n- " + "\n- ".join(lines) + "\n\n"
            "ECE summary (lower is better):\n- " + ("\n- ".join(ece_lines) if ece_lines else "No ECE data")
        )

        ax.text(
            0.02,
            0.95,
            summary_text,
            va="top",
            ha="left",
            fontsize=11,
            family="DejaVu Sans",
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f7f9fc", "edgecolor": "#d0d7e2"},
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: reliability + support context (bin counts)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
        fig.suptitle("Score Behavior: Alignment and Bin Support", fontsize=14, weight="bold")

        ax = axes[0]
        ax.plot([0, 1], [0, 1], linestyle="--", color="#666", linewidth=1.2, label="ideal alignment")

        rel = (
            calib_df.groupby(["method", "bin_id"], dropna=False)
            .agg(
                n=("n", "sum"),
                mean_score=("mean_score", "mean"),
                corrected_error_rate=("corrected_error_rate", "mean"),
                delta_accuracy_pp=("delta_accuracy_pp", "mean"),
            )
            .reset_index()
            .dropna(subset=["mean_score", "corrected_error_rate"])
        )

        for method, g in rel.groupby("method"):
            c = method_colors.get(method, "#333333")
            size = np.clip(g["n"].to_numpy(), 20, None)
            size = 35 + 0.45 * np.sqrt(size)
            ax.plot(g["mean_score"], g["corrected_error_rate"], color=c, linewidth=2.4, label=method)
            ax.scatter(
                g["mean_score"],
                g["corrected_error_rate"],
                s=size,
                color=c,
                alpha=0.8,
                edgecolor="white",
                linewidth=0.7,
            )

        ax.set_title("Reliability-style alignment")
        ax.set_xlabel("Mean intervention score (bin)")
        ax.set_ylabel("Observed corrected error rate")
        ax.grid(alpha=0.25)
        ax.legend(frameon=True)

        ax2 = axes[1]
        pivot_n = rel.pivot_table(index="bin_id", columns="method", values="n", aggfunc="sum").fillna(0)
        bin_ids = pivot_n.index.to_numpy(dtype=float)
        width = 0.35
        methods = list(pivot_n.columns)
        for i, method in enumerate(methods):
            c = method_colors.get(method, "#333333")
            x = bin_ids + (i - (len(methods) - 1) / 2) * width
            ax2.bar(x, pivot_n[method].to_numpy(), width=width, color=c, alpha=0.8, label=method)

        ax2.set_title("Bin support (sample count)")
        ax2.set_xlabel("bin id")
        ax2.set_ylabel("samples")
        ax2.grid(axis="y", alpha=0.2)
        ax2.legend(frameon=True)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: score distributions by correctness
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
        fig.suptitle("Intervention Score Distributions", fontsize=14, weight="bold")

        methods = sorted(raw_df["method"].unique().tolist())
        for i, method in enumerate(methods[:2]):
            ax = axes[i]
            g = raw_df[raw_df["method"] == method]
            good = g[g["corrected_correct"] == 1]["intervention_score"]
            bad = g[g["corrected_correct"] == 0]["intervention_score"]

            ax.hist(good, bins=20, alpha=0.65, color="#4CAF50", label="corrected correct", density=True)
            ax.hist(bad, bins=20, alpha=0.65, color="#EF5350", label="corrected wrong", density=True)
            ax.set_title(method)
            ax.set_xlabel("intervention_score")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)

        if len(methods) < 2:
            axes[1].axis("off")

        axes[0].set_ylabel("density")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 4: coverage-risk and coverage-accuracy curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
        fig.suptitle("Abstention Tradeoff Curves", fontsize=14, weight="bold")

        avg_sweep = (
            sweep_df.groupby(["method", "threshold"], dropna=False)
            .agg(
                coverage=("coverage", "mean"),
                corrected_error_kept=("corrected_error_kept", "mean"),
                corrected_accuracy_kept=("corrected_accuracy_kept", "mean"),
                delta_accuracy_pp_kept=("delta_accuracy_pp_kept", "mean"),
            )
            .reset_index()
        )

        for method, g in avg_sweep.groupby("method"):
            c = method_colors.get(method, "#333333")
            g = g.sort_values("coverage")
            axes[0].plot(g["coverage"], g["corrected_error_kept"], color=c, linewidth=2.2, label=method)
            axes[1].plot(g["coverage"], g["corrected_accuracy_kept"], color=c, linewidth=2.2, label=method)

        axes[0].set_title("Coverage vs selective error (lower better)")
        axes[0].set_xlabel("coverage")
        axes[0].set_ylabel("corrected error on kept set")
        axes[0].grid(alpha=0.25)

        axes[1].set_title("Coverage vs selective accuracy (higher better)")
        axes[1].set_xlabel("coverage")
        axes[1].set_ylabel("corrected accuracy on kept set")
        axes[1].grid(alpha=0.25)

        axes[1].legend(frameon=True)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 5: Advantage-focused view (coverage vs delta gain)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
        fig.suptitle("Method Advantage View (what improves with abstention)", fontsize=14, weight="bold")

        for method, g in avg_sweep.groupby("method"):
            c = method_colors.get(method, "#333333")
            g = g.sort_values("coverage")
            axes[0].plot(g["coverage"], g["delta_accuracy_pp_kept"], color=c, linewidth=2.5, label=method)
            axes[0].scatter(g["coverage"], g["delta_accuracy_pp_kept"], color=c, s=18, alpha=0.7)

        axes[0].axhline(0, color="#555", linewidth=1.0, linestyle="--")
        axes[0].set_title("Coverage vs gain over baseline")
        axes[0].set_xlabel("coverage")
        axes[0].set_ylabel("delta accuracy (pp)")
        axes[0].grid(alpha=0.25)
        axes[0].legend(frameon=True)

        # Dominance scatter at target coverages
        targets = [0.6, 0.7, 0.8, 0.9]
        marker_map = {0.6: "o", 0.7: "s", 0.8: "^", 0.9: "D"}
        for method, g in avg_sweep.groupby("method"):
            c = method_colors.get(method, "#333333")
            for t in targets:
                row = g.iloc[(g["coverage"] - t).abs().argsort()[:1]].iloc[0]
                axes[1].scatter(
                    row["corrected_error_kept"],
                    row["delta_accuracy_pp_kept"],
                    color=c,
                    s=90,
                    marker=marker_map[t],
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.7,
                )

        axes[1].axhline(0, color="#555", linewidth=1.0, linestyle="--")
        axes[1].set_title("Dominance: lower error, higher gain is better")
        axes[1].set_xlabel("selective corrected error")
        axes[1].set_ylabel("delta accuracy (pp)")
        axes[1].grid(alpha=0.25)

        # Build compact legend for method colors and coverage markers
        method_handles = [
            plt.Line2D([0], [0], marker="o", color="w", label=m, markerfacecolor=method_colors.get(m, "#333"), markersize=8)
            for m in sorted(avg_sweep["method"].unique())
        ]
        cov_handles = [
            plt.Line2D([0], [0], marker=marker_map[t], color="#444", label=f"coverage~{t:.1f}", linestyle="None", markersize=7)
            for t in targets
        ]
        axes[1].legend(handles=method_handles + cov_handles, frameon=True, loc="best", fontsize=8)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 6: threshold heatmap for fast reading
        pivot_rows = []
        for method, g in avg_sweep.groupby("method"):
            for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                idx = (g["threshold"] - t).abs().idxmin()
                r = g.loc[idx]
                pivot_rows.append(
                    {
                        "method": method,
                        "threshold": float(r["threshold"]),
                        "coverage": float(r["coverage"]),
                        "acc": float(r["corrected_accuracy_kept"]),
                        "err": float(r["corrected_error_kept"]),
                    }
                )

        hdf = pd.DataFrame(pivot_rows)
        if not hdf.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            # Score for heatmap: accuracy weighted by coverage
            hdf["score"] = hdf["acc"] * hdf["coverage"]
            mat = hdf.pivot(index="method", columns="threshold", values="score")
            im = ax.imshow(mat.values, aspect="auto", cmap="YlGnBu")
            ax.set_title("Threshold grid (score = accuracy × coverage)", fontsize=13, weight="bold")
            ax.set_xticks(range(len(mat.columns)))
            ax.set_xticklabels([f"{c:.1f}" for c in mat.columns])
            ax.set_yticks(range(len(mat.index)))
            ax.set_yticklabels(mat.index)
            ax.set_xlabel("threshold")
            ax.set_ylabel("method")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("score")

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Page 7: operating point table
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("off")
        ax.set_title("Operating Points (coverage targets)", fontsize=13, weight="bold", pad=10)

        if abst_sum_df.empty:
            ax.text(0.5, 0.5, "No abstention summary available", ha="center", va="center")
        else:
            tdf = abst_sum_df.copy()
            for col in ["coverage", "threshold", "corrected_accuracy_kept", "corrected_error_kept"]:
                if col in tdf.columns:
                    tdf[col] = pd.to_numeric(tdf[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
            if "delta_accuracy_pp_kept" in tdf.columns:
                tdf["delta_accuracy_pp_kept"] = pd.to_numeric(tdf["delta_accuracy_pp_kept"], errors="coerce").map(
                    lambda x: "" if pd.isna(x) else f"{x:.2f}"
                )
            if "value" in tdf.columns:
                tdf["value"] = pd.to_numeric(tdf["value"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.4f}")

            cols = [
                "method",
                "operating_point",
                "coverage",
                "threshold",
                "corrected_accuracy_kept",
                "corrected_error_kept",
                "delta_accuracy_pp_kept",
                "value",
            ]
            cols = [c for c in cols if c in tdf.columns]
            tdf = tdf[cols]

            tab = ax.table(cellText=tdf.values, colLabels=tdf.columns, cellLoc="center", loc="center")
            tab.auto_set_font_size(False)
            tab.set_fontsize(8.5)
            tab.scale(1.0, 1.3)
            for j in range(len(tdf.columns)):
                cell = tab[(0, j)]
                cell.set_facecolor((0.9, 0.92, 0.96))
                cell.get_text().set_weight("bold")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Build Task-7 calibration/abstention report")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repo_root / "results",
        help="Results root (default: <repo-root>/results)",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="priorityB_task7_calibration",
        help="Output subfolder under results root",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of equal-width reliability bins (default: 10)",
    )
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    out_dir = results_root / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    method_cfgs = [
        MethodConfig(
            name="E12b",
            patterns=[
                "e12b_n2000/results_E12b_gqa_confidence_gated_e1_tuned.csv",
                "e12b_n2000_seed*/results_E12b_gqa_confidence_gated_e1_tuned_n2000_seed*.csv",
            ],
            color="#ff7f0e",
        ),
        MethodConfig(
            name="E1",
            patterns=[
                "e1_n2000/results_E1_gqa_replace_all_n2000.csv",
                "e1_n2000_seed*/results_E1_gqa_replace_all_n2000_seed*.csv",
            ],
            color="#1f77b4",
        ),
    ]

    dfs = [load_method_runs(results_root, cfg) for cfg in method_cfgs]
    raw_df = pd.concat([d for d in dfs if not d.empty], ignore_index=True) if any(not d.empty for d in dfs) else pd.DataFrame()

    if raw_df.empty:
        # Write empty artifacts but keep script deterministic.
        (out_dir / "calibration_bins.csv").write_text("")
        (out_dir / "abstention_sweep.csv").write_text("")
        with open(out_dir / "calibration_bins.json", "w") as f:
            json.dump({"warning": "No valid input CSVs found."}, f, indent=2)
        with open(out_dir / "abstention_sweep.json", "w") as f:
            json.dump({"warning": "No valid input CSVs found."}, f, indent=2)
        print("No valid inputs found. Wrote empty placeholders.")
        return

    calib_df = calibration_table(raw_df, n_bins=args.n_bins)
    ece_df = ece_from_bins(calib_df)

    sweep_df = abstention_sweep(raw_df)
    abst_sum_df = summarize_abstention(sweep_df)

    calib_csv = out_dir / "calibration_bins.csv"
    calib_json = out_dir / "calibration_bins.json"
    sweep_csv = out_dir / "abstention_sweep.csv"
    sweep_json = out_dir / "abstention_sweep.json"
    ece_csv = out_dir / "calibration_ece_summary.csv"
    ece_json = out_dir / "calibration_ece_summary.json"
    op_csv = out_dir / "abstention_operating_points.csv"
    op_json = out_dir / "abstention_operating_points.json"
    pdf_path = out_dir / "task7_calibration_report.pdf"

    calib_df.to_csv(calib_csv, index=False)
    sweep_df.to_csv(sweep_csv, index=False)
    ece_df.to_csv(ece_csv, index=False)
    abst_sum_df.to_csv(op_csv, index=False)

    with open(calib_json, "w") as f:
        json.dump(calib_df.to_dict(orient="records"), f, indent=2)
    with open(sweep_json, "w") as f:
        json.dump(sweep_df.to_dict(orient="records"), f, indent=2)
    with open(ece_json, "w") as f:
        json.dump(ece_df.to_dict(orient="records"), f, indent=2)
    with open(op_json, "w") as f:
        json.dump(abst_sum_df.to_dict(orient="records"), f, indent=2)

    colors = {cfg.name: cfg.color for cfg in method_cfgs}
    save_pdf(pdf_path, raw_df, calib_df, ece_df, sweep_df, abst_sum_df, colors)

    print("=" * 80)
    print("PRIORITY B TASK 7 ARTIFACTS WRITTEN")
    print("=" * 80)
    print(f"Calibration bins CSV : {calib_csv}")
    print(f"Calibration bins JSON: {calib_json}")
    print(f"Abstention sweep CSV : {sweep_csv}")
    print(f"Abstention sweep JSON: {sweep_json}")
    print(f"ECE summary CSV      : {ece_csv}")
    print(f"ECE summary JSON     : {ece_json}")
    print(f"Operating points CSV : {op_csv}")
    print(f"Operating points JSON: {op_json}")
    print(f"PDF report           : {pdf_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
