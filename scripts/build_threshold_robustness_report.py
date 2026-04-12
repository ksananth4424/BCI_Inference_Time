"""Priority B Task 6: E12b threshold robustness sweep.

This script can:
1) Run threshold sweeps for confidence-gated intervention (E12b) via
   scripts/run_phase2_experiment.py, and
2) Analyze all sweep outputs into CSV/JSON/PDF artifacts.

Default outputs:
  results/priorityB_task6_threshold_robustness/
    - threshold_metrics_long.csv
    - threshold_metrics_long.json
    - threshold_metrics_aggregate.csv
    - threshold_metrics_aggregate.json
    - threshold_vs_baseline_significance.csv
    - threshold_vs_anchor_significance.csv
    - monotonicity_checks.csv
    - task6_threshold_robustness_report.pdf

Usage examples:
  # Run + analyze for two seed configs
  python scripts/build_threshold_robustness_report.py \
    --config configs/phase2/e12b_gqa_confidence_gated_e1_tuned_n2000_seed123.yaml \
    --config configs/phase2/e12b_gqa_confidence_gated_e1_tuned_n2000_seed456.yaml \
    --thresholds 0.10,0.14,0.18,0.22,0.26 \
    --device cuda:0

  # Analyze-only from existing sweep outputs
  python scripts/build_threshold_robustness_report.py --analyze-only
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
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


def format_thr_folder(threshold: float) -> str:
    # e.g., 0.18 -> thr_0p180
    return f"thr_{threshold:.3f}".replace(".", "p")


def parse_thr_from_folder(name: str) -> float | None:
    m = re.match(r"thr_(\d+)p(\d+)", name)
    if not m:
        return None
    whole = m.group(1)
    frac = m.group(2)
    return float(f"{whole}.{frac}")


def parse_seed(text: str) -> int | None:
    m = re.search(r"seed(\d+)", text.lower())
    if m:
        return int(m.group(1))
    return None


def mcnemar_exact_p_value(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0

    k = min(b, c)

    def pmf(i: int) -> float:
        return math.comb(n, i) * (0.5 ** n)

    p_left = sum(pmf(i) for i in range(0, k + 1))
    return min(1.0, 2.0 * p_left)


def safe_std(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) <= 1:
        return 0.0
    return float(x.std(ddof=1))


@dataclass
class RunSpec:
    config_path: Path
    config_label: str
    threshold: float
    run_dir: Path


# -----------------------------------------------------------------------------
# Sweep execution
# -----------------------------------------------------------------------------

def latest_results_csv(run_dir: Path) -> Path | None:
    cands = sorted(run_dir.glob("results_*.csv"))
    return cands[-1] if cands else None


def run_single_threshold(
    repo_root: Path,
    python_exe: str,
    exp_id: str,
    spec: RunSpec,
    device: str | None,
    n_samples: int | None,
    force_rerun: bool,
) -> dict[str, Any]:
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    existing = latest_results_csv(spec.run_dir)

    if existing is not None and not force_rerun:
        return {
            "status": "skipped_existing",
            "config_label": spec.config_label,
            "config_path": str(spec.config_path),
            "threshold": spec.threshold,
            "run_dir": str(spec.run_dir),
            "results_csv": str(existing),
            "returncode": 0,
        }

    cmd = [
        python_exe,
        str(repo_root / "scripts" / "run_phase2_experiment.py"),
        "--exp",
        exp_id,
        "--config",
        str(spec.config_path),
        "--output-dir",
        str(spec.run_dir),
        "--enable-confidence-gate",
        "--min-intervention-score",
        f"{spec.threshold:.6f}",
    ]
    if device:
        cmd += ["--device", device]
    if n_samples is not None:
        cmd += ["--n-samples", str(n_samples)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    out_csv = latest_results_csv(spec.run_dir)

    log = {
        "status": "ok" if proc.returncode == 0 else "failed",
        "config_label": spec.config_label,
        "config_path": str(spec.config_path),
        "threshold": spec.threshold,
        "run_dir": str(spec.run_dir),
        "results_csv": str(out_csv) if out_csv else None,
        "returncode": proc.returncode,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-30:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-30:]),
    }
    return log


# -----------------------------------------------------------------------------
# Loading + metrics
# -----------------------------------------------------------------------------

def discover_existing_runs(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(runs_root.glob("*/thr_*/results_*.csv")):
        thr = parse_thr_from_folder(csv_path.parent.name)
        config_label = csv_path.parent.parent.name
        if thr is None:
            continue
        rows.append(
            {
                "status": "existing",
                "config_label": config_label,
                "config_path": None,
                "threshold": thr,
                "run_dir": str(csv_path.parent),
                "results_csv": str(csv_path),
                "returncode": 0,
            }
        )
    return rows


def load_result_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sample_id", "baseline_correct", "corrected_correct", "intervened", "intervention_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    out = pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str),
            "baseline_correct": to_bool_series(df["baseline_correct"]).astype(int),
            "corrected_correct": to_bool_series(df["corrected_correct"]).astype(int),
            "intervened": pd.to_numeric(df["intervened"], errors="coerce").fillna(0).astype(float),
            "intervention_score": pd.to_numeric(df["intervention_score"], errors="coerce"),
            "used_fallback": pd.to_numeric(df.get("used_fallback", 0), errors="coerce").fillna(0).astype(float),
            "flip_to_correct": pd.to_numeric(df.get("flip_to_correct", 0), errors="coerce").fillna(0).astype(float),
        }
    )
    return out


def metrics_from_df(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    if n == 0:
        return {
            "n": 0,
            "baseline_accuracy": 0.0,
            "corrected_accuracy": 0.0,
            "delta_accuracy_pp": 0.0,
            "intervened_rate": 0.0,
            "flip_to_correct_rate": 0.0,
            "flip_to_wrong_rate": 0.0,
            "net_gain_rate": 0.0,
            "mean_intervention_score": 0.0,
            "fallback_rate": 0.0,
        }

    baseline = df["baseline_correct"].astype(int)
    corrected = df["corrected_correct"].astype(int)

    flip_to_correct = int(((baseline == 0) & (corrected == 1)).sum())
    flip_to_wrong = int(((baseline == 1) & (corrected == 0)).sum())

    baseline_acc = float(baseline.mean())
    corrected_acc = float(corrected.mean())

    return {
        "n": int(n),
        "baseline_accuracy": baseline_acc,
        "corrected_accuracy": corrected_acc,
        "delta_accuracy_pp": (corrected_acc - baseline_acc) * 100.0,
        "intervened_rate": float(pd.to_numeric(df["intervened"], errors="coerce").fillna(0).mean()),
        "flip_to_correct_rate": flip_to_correct / n,
        "flip_to_wrong_rate": flip_to_wrong / n,
        "net_gain_rate": (flip_to_correct - flip_to_wrong) / n,
        "mean_intervention_score": float(pd.to_numeric(df["intervention_score"], errors="coerce").dropna().mean()),
        "fallback_rate": float(pd.to_numeric(df["used_fallback"], errors="coerce").fillna(0).mean()),
    }


def compute_long_metrics(run_records: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[tuple[str, float], pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    sample_tables: dict[tuple[str, float], pd.DataFrame] = {}

    for rec in run_records:
        csv_str = rec.get("results_csv")
        if not csv_str:
            continue
        csv_path = Path(csv_str)
        if not csv_path.exists():
            continue

        try:
            df = load_result_csv(csv_path)
        except Exception:
            continue

        config_label = str(rec["config_label"])
        threshold = float(rec["threshold"])
        m = metrics_from_df(df)

        rows.append(
            {
                "config_label": config_label,
                "seed": parse_seed(config_label),
                "threshold": threshold,
                "results_csv": str(csv_path),
                **m,
            }
        )

        sample_tables[(config_label, threshold)] = df[["sample_id", "baseline_correct", "corrected_correct"]].copy()

    long_df = pd.DataFrame(rows)
    if not long_df.empty:
        long_df = long_df.sort_values(["config_label", "threshold"]).reset_index(drop=True)
    return long_df, sample_tables


def compute_aggregate(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    metrics = [
        "baseline_accuracy",
        "corrected_accuracy",
        "delta_accuracy_pp",
        "intervened_rate",
        "flip_to_correct_rate",
        "flip_to_wrong_rate",
        "net_gain_rate",
        "fallback_rate",
    ]

    agg_rows: list[dict[str, Any]] = []
    for thr, g in long_df.groupby("threshold"):
        row: dict[str, Any] = {
            "threshold": float(thr),
            "n_runs": int(len(g)),
            "n_mean": float(pd.to_numeric(g["n"], errors="coerce").mean()),
        }
        for m in metrics:
            row[f"{m}_mean"] = float(pd.to_numeric(g[m], errors="coerce").mean())
            row[f"{m}_std"] = safe_std(g[m])
        agg_rows.append(row)

    out = pd.DataFrame(agg_rows)
    return out.sort_values("threshold").reset_index(drop=True)


def paired_vs_baseline(long_df: pd.DataFrame, sample_tables: dict[tuple[str, float], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, r in long_df.iterrows():
        config_label = str(r["config_label"])
        threshold = float(r["threshold"])
        df = sample_tables.get((config_label, threshold))
        if df is None or df.empty:
            continue

        baseline = df["baseline_correct"].astype(int)
        corrected = df["corrected_correct"].astype(int)

        b = int(((baseline == 1) & (corrected == 0)).sum())
        c = int(((baseline == 0) & (corrected == 1)).sum())

        rows.append(
            {
                "config_label": config_label,
                "seed": parse_seed(config_label),
                "threshold": threshold,
                "b_baseline_only": b,
                "c_corrected_only": c,
                "exact_p_value": mcnemar_exact_p_value(b, c),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["config_label", "threshold"]).reset_index(drop=True)
    return out


def paired_vs_anchor(
    long_df: pd.DataFrame,
    sample_tables: dict[tuple[str, float], pd.DataFrame],
    anchor_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for config_label, g in long_df.groupby("config_label"):
        thrs = sorted(float(x) for x in g["threshold"].unique())
        if not thrs:
            continue

        anchor = min(thrs, key=lambda t: abs(t - anchor_threshold))
        anchor_df = sample_tables.get((str(config_label), float(anchor)))
        if anchor_df is None or anchor_df.empty:
            continue

        anchor_y = anchor_df[["sample_id", "corrected_correct"]].rename(columns={"corrected_correct": "anchor_corrected"})

        for thr in thrs:
            cur_df = sample_tables.get((str(config_label), float(thr)))
            if cur_df is None or cur_df.empty:
                continue

            merged = anchor_y.merge(
                cur_df[["sample_id", "corrected_correct"]].rename(columns={"corrected_correct": "current_corrected"}),
                on="sample_id",
                how="inner",
            )
            if merged.empty:
                continue

            a = merged["anchor_corrected"].astype(int)
            ccur = merged["current_corrected"].astype(int)

            b = int(((a == 1) & (ccur == 0)).sum())  # anchor-only correct
            c = int(((a == 0) & (ccur == 1)).sum())  # current-only correct

            rows.append(
                {
                    "config_label": str(config_label),
                    "seed": parse_seed(str(config_label)),
                    "anchor_threshold": float(anchor),
                    "threshold": float(thr),
                    "b_anchor_only": b,
                    "c_current_only": c,
                    "exact_p_value": mcnemar_exact_p_value(b, c),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["config_label", "threshold"]).reset_index(drop=True)
    return out


def monotonicity_checks(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config_label, g in long_df.groupby("config_label"):
        g = g.sort_values("threshold")
        rates = pd.to_numeric(g["intervened_rate"], errors="coerce").fillna(0).to_numpy()
        thrs = pd.to_numeric(g["threshold"], errors="coerce").to_numpy()

        diffs = np.diff(rates)
        violations = diffs > 1e-12  # should be non-increasing as threshold increases
        n_viol = int(violations.sum())

        row = {
            "config_label": str(config_label),
            "seed": parse_seed(str(config_label)),
            "n_thresholds": int(len(thrs)),
            "intervened_rate_monotone_nonincreasing": bool(n_viol == 0),
            "n_monotonicity_violations": n_viol,
            "max_positive_jump": float(diffs[violations].max()) if n_viol > 0 else 0.0,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("config_label").reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def save_report_pdf(out_pdf: Path, long_df: pd.DataFrame, agg_df: pd.DataFrame) -> None:
    with PdfPages(out_pdf) as pdf:
        # Page 1: corrected accuracy + delta vs threshold
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if not agg_df.empty:
            x = agg_df["threshold"].to_numpy()
            y = agg_df["corrected_accuracy_mean"].to_numpy()
            yerr = agg_df["corrected_accuracy_std"].to_numpy()
            axes[0].errorbar(x, y, yerr=yerr, marker="o", linewidth=2, capsize=3, color="#1f77b4")
            axes[0].plot(x, agg_df["baseline_accuracy_mean"].to_numpy(), linestyle="--", color="#888888", label="baseline")
            axes[0].set_title("Corrected accuracy vs threshold")
            axes[0].set_xlabel("min_intervention_score")
            axes[0].set_ylabel("accuracy")
            axes[0].grid(alpha=0.25)
            axes[0].legend()

            y2 = agg_df["delta_accuracy_pp_mean"].to_numpy()
            y2err = agg_df["delta_accuracy_pp_std"].to_numpy()
            axes[1].errorbar(x, y2, yerr=y2err, marker="o", linewidth=2, capsize=3, color="#2ca02c")
            axes[1].axhline(0.0, color="#999999", linestyle="--", linewidth=1)
            axes[1].set_title("Delta accuracy (pp) vs threshold")
            axes[1].set_xlabel("min_intervention_score")
            axes[1].set_ylabel("delta accuracy (pp)")
            axes[1].grid(alpha=0.25)
        else:
            for ax in axes:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.axis("off")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: intervention coverage + net gain
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        if not agg_df.empty:
            x = agg_df["threshold"].to_numpy()

            axes[0].errorbar(
                x,
                agg_df["intervened_rate_mean"].to_numpy(),
                yerr=agg_df["intervened_rate_std"].to_numpy(),
                marker="o",
                linewidth=2,
                capsize=3,
                color="#d62728",
            )
            axes[0].set_title("Intervened rate vs threshold")
            axes[0].set_xlabel("min_intervention_score")
            axes[0].set_ylabel("intervened rate")
            axes[0].grid(alpha=0.25)

            axes[1].errorbar(
                x,
                agg_df["net_gain_rate_mean"].to_numpy() * 100.0,
                yerr=agg_df["net_gain_rate_std"].to_numpy() * 100.0,
                marker="o",
                linewidth=2,
                capsize=3,
                color="#9467bd",
            )
            axes[1].axhline(0.0, color="#999999", linestyle="--", linewidth=1)
            axes[1].set_title("Net gain rate vs threshold")
            axes[1].set_xlabel("min_intervention_score")
            axes[1].set_ylabel("net gain rate (pp)")
            axes[1].grid(alpha=0.25)
        else:
            for ax in axes:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.axis("off")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: per-config corrected-accuracy trajectories
        fig, ax = plt.subplots(figsize=(12, 6))
        if not long_df.empty:
            for config_label, g in long_df.groupby("config_label"):
                gg = g.sort_values("threshold")
                ax.plot(
                    gg["threshold"].to_numpy(),
                    gg["corrected_accuracy"].to_numpy(),
                    marker="o",
                    linewidth=1.5,
                    alpha=0.85,
                    label=str(config_label),
                )
            ax.set_title("Per-run corrected accuracy trajectories")
            ax.set_xlabel("min_intervention_score")
            ax.set_ylabel("corrected accuracy")
            ax.grid(alpha=0.25)
            if long_df["config_label"].nunique() <= 12:
                ax.legend(loc="best", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def parse_thresholds(raw: str) -> list[float]:
    vals: list[float] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        vals.append(float(s))
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("No thresholds parsed")
    for v in vals:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Threshold out of range [0,1]: {v}")
    return vals


def choose_recommended_threshold(agg_df: pd.DataFrame, anchor_threshold: float) -> float | None:
    if agg_df.empty:
        return None

    # Primary: maximize corrected accuracy mean
    best_acc = float(agg_df["corrected_accuracy_mean"].max())
    cand = agg_df[agg_df["corrected_accuracy_mean"] == best_acc].copy()

    # Tie-break 1: maximize net gain rate
    best_gain = float(cand["net_gain_rate_mean"].max())
    cand = cand[cand["net_gain_rate_mean"] == best_gain].copy()

    # Tie-break 2: closest to anchor threshold
    cand["dist"] = (cand["threshold"] - anchor_threshold).abs()
    cand = cand.sort_values(["dist", "threshold"]).reset_index(drop=True)
    return float(cand.loc[0, "threshold"])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Priority B6: E12b threshold robustness sweep + report")
    parser.add_argument("--config", action="append", default=[], help="Path to phase2 YAML config (repeatable)")
    parser.add_argument("--thresholds", type=str, default="0.10,0.14,0.18,0.22,0.26")
    parser.add_argument("--anchor-threshold", type=float, default=0.18)
    parser.add_argument("--exp", type=str, default="E12b")
    parser.add_argument("--results-root", type=Path, default=repo_root / "results")
    parser.add_argument("--output-subdir", type=str, default="priorityB_task6_threshold_robustness")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)

    out_dir = args.results_root.resolve() / args.output_subdir
    runs_root = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    run_records: list[dict[str, Any]] = []

    if not args.analyze_only:
        if not args.config:
            raise ValueError("At least one --config is required unless --analyze-only is set")

        specs: list[RunSpec] = []
        for cfg in args.config:
            cfg_path = Path(cfg).resolve()
            if not cfg_path.exists():
                raise FileNotFoundError(f"Config not found: {cfg_path}")

            cfg_label = cfg_path.stem
            for thr in thresholds:
                specs.append(
                    RunSpec(
                        config_path=cfg_path,
                        config_label=cfg_label,
                        threshold=thr,
                        run_dir=runs_root / cfg_label / format_thr_folder(thr),
                    )
                )

        print("=" * 90)
        print("TASK B6: RUNNING THRESHOLD SWEEP")
        print("=" * 90)
        print(f"Configs     : {len(args.config)}")
        print(f"Thresholds  : {thresholds}")
        print(f"Total runs  : {len(specs)}")
        print(f"Output root : {out_dir}")
        print("=" * 90)

        for i, spec in enumerate(specs, start=1):
            print(f"[{i:02d}/{len(specs):02d}] {spec.config_label} @ threshold={spec.threshold:.3f}")
            rec = run_single_threshold(
                repo_root=repo_root,
                python_exe=args.python_exe,
                exp_id=args.exp,
                spec=spec,
                device=args.device,
                n_samples=args.n_samples,
                force_rerun=bool(args.force_rerun),
            )
            run_records.append(rec)
            print(f"  -> status={rec['status']}, returncode={rec['returncode']}")

    # include discovered runs so analyze-only and mixed sessions both work.
    discovered = discover_existing_runs(runs_root)

    # prefer explicit run records when same (config, threshold) exists
    merged_map: dict[tuple[str, float], dict[str, Any]] = {}
    for rec in discovered + run_records:
        key = (str(rec.get("config_label")), float(rec.get("threshold")))
        merged_map[key] = rec
    merged_records = list(merged_map.values())

    if not merged_records:
        raise RuntimeError(f"No run artifacts found under: {runs_root}")

    long_df, sample_tables = compute_long_metrics(merged_records)
    agg_df = compute_aggregate(long_df)
    sig_baseline_df = paired_vs_baseline(long_df, sample_tables)
    sig_anchor_df = paired_vs_anchor(long_df, sample_tables, anchor_threshold=float(args.anchor_threshold))
    mono_df = monotonicity_checks(long_df)

    # Save artifacts
    long_csv = out_dir / "threshold_metrics_long.csv"
    long_json = out_dir / "threshold_metrics_long.json"
    agg_csv = out_dir / "threshold_metrics_aggregate.csv"
    agg_json = out_dir / "threshold_metrics_aggregate.json"
    sig_b_csv = out_dir / "threshold_vs_baseline_significance.csv"
    sig_a_csv = out_dir / "threshold_vs_anchor_significance.csv"
    mono_csv = out_dir / "monotonicity_checks.csv"
    run_log_json = out_dir / "sweep_run_log.json"

    long_df.to_csv(long_csv, index=False)
    agg_df.to_csv(agg_csv, index=False)
    sig_baseline_df.to_csv(sig_b_csv, index=False)
    sig_anchor_df.to_csv(sig_a_csv, index=False)
    mono_df.to_csv(mono_csv, index=False)

    dump_json(long_json, long_df.to_dict(orient="records"))
    dump_json(agg_json, agg_df.to_dict(orient="records"))
    dump_json(run_log_json, merged_records)

    report_pdf = out_dir / "task6_threshold_robustness_report.pdf"
    save_report_pdf(report_pdf, long_df, agg_df)

    rec_thr = choose_recommended_threshold(agg_df, anchor_threshold=float(args.anchor_threshold))

    headline = {
        "recommended_threshold": rec_thr,
        "anchor_threshold": float(args.anchor_threshold),
        "n_runs_analyzed": int(len(long_df)),
        "thresholds": sorted(long_df["threshold"].unique().tolist()) if not long_df.empty else [],
    }
    dump_json(out_dir / "task6_headline_summary.json", headline)

    print("=" * 90)
    print("TASK B6 COMPLETE")
    print("=" * 90)
    print(f"Long metrics CSV             : {long_csv}")
    print(f"Aggregate metrics CSV        : {agg_csv}")
    print(f"Vs-baseline significance CSV : {sig_b_csv}")
    print(f"Vs-anchor significance CSV   : {sig_a_csv}")
    print(f"Monotonicity checks CSV      : {mono_csv}")
    print(f"Report PDF                   : {report_pdf}")
    print(f"Headline JSON                : {out_dir / 'task6_headline_summary.json'}")
    if rec_thr is not None:
        print(f"Recommended threshold        : {rec_thr:.3f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
