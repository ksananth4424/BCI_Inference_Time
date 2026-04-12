"""Build Priority-B Task-5 cost-efficiency report from existing artifacts.

This script does NOT require reruns. It reads existing run manifests/summaries/CSVs
from results/, computes runtime-efficiency metrics, and writes report artifacts.

Outputs (default):
  results/priorityB_task5_cost_efficiency/
    - cost_efficiency_per_run.csv
    - cost_efficiency_per_run.json
    - cost_efficiency_aggregated.csv
    - cost_efficiency_aggregated.json
    - cost_efficiency_report.pdf

Usage:
    python scripts/build_cost_efficiency_report.py
    python scripts/build_cost_efficiency_report.py --results-root results
"""

from __future__ import annotations

import argparse
import json
import math
import re
from textwrap import fill
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from pandas.errors import EmptyDataError, ParserError


def safe_div(a: float, b: float) -> float:
    if b == 0:
        return float("nan")
    return a / b


def detect_method(experiment_id: str) -> str:
    e = experiment_id.lower()
    if e.startswith("e12b"):
        return "E12b"
    if e.startswith("e12"):
        return "E12"
    if e.startswith("e1"):
        return "E1"
    if e.startswith("e3"):
        return "E3"
    return "Other"


def parse_seed(experiment_id: str) -> int | None:
    m = re.search(r"seed(\d+)", experiment_id.lower())
    if not m:
        return None
    return int(m.group(1))


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def find_summary_path(manifest_path: Path, experiment_id: str) -> Path | None:
    candidate = manifest_path.parent / f"summary_{experiment_id}.json"
    if candidate.exists():
        return candidate

    summaries = list(manifest_path.parent.glob("summary_*.json"))
    if len(summaries) == 1:
        return summaries[0]
    return None


def find_results_csv_path(manifest_path: Path, experiment_id: str) -> Path | None:
    candidate = manifest_path.parent / f"results_{experiment_id}.csv"
    if candidate.exists():
        return candidate

    csvs = list(manifest_path.parent.glob("results_*.csv"))
    if len(csvs) == 1:
        return csvs[0]
    return None


def extract_optional_csv_metrics(csv_path: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "intervened_rate": float("nan"),
        "fallback_rate": float("nan"),
        "flip_to_wrong": float("nan"),
        "flip_to_wrong_rate": float("nan"),
        "avg_tokens_per_sample": float("nan"),
        "gain_per_1k_tokens_pp": float("nan"),
    }

    if csv_path is None or not csv_path.exists():
        return out

    try:
        df = pd.read_csv(csv_path)
    except (EmptyDataError, ParserError, UnicodeDecodeError, OSError):
        # Some legacy/partial runs may leave empty or non-CSV artifacts.
        # Optional metrics should not fail the entire report build.
        return out

    if df.empty:
        return out

    if "intervened" in df.columns:
        out["intervened_rate"] = float(df["intervened"].mean())

    if "used_fallback" in df.columns:
        out["fallback_rate"] = float(df["used_fallback"].mean())

    if "baseline_correct" in df.columns and "corrected_correct" in df.columns:
        baseline = df["baseline_correct"].astype(bool)
        corrected = df["corrected_correct"].astype(bool)
        flip_to_wrong = int((baseline & ~corrected).sum())
        out["flip_to_wrong"] = flip_to_wrong
        out["flip_to_wrong_rate"] = float(flip_to_wrong / len(df))

    token_cols = ["tokens_total", "total_tokens", "n_tokens_total"]
    available = [c for c in token_cols if c in df.columns]
    if available:
        tok_col = available[0]
        out["avg_tokens_per_sample"] = float(df[tok_col].mean())

    return out


def discover_phase2_runs(results_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for manifest_path in sorted(results_root.rglob("run_*_manifest.json")):
        try:
            manifest = load_json(manifest_path)
        except Exception:
            continue

        experiment_id = str(manifest.get("experiment_id", ""))
        summary = manifest.get("summary", {}) or {}

        required = {"n_samples", "n_baseline_correct", "n_corrected_correct", "n_flipped"}
        if not required.issubset(summary.keys()):
            continue

        n_samples = int(summary["n_samples"])
        n_baseline_correct = int(summary["n_baseline_correct"])
        n_corrected_correct = int(summary["n_corrected_correct"])
        n_flipped = int(summary["n_flipped"])
        duration_s = float(manifest.get("duration_seconds", float("nan")))

        baseline_acc = safe_div(n_baseline_correct, n_samples)
        corrected_acc = safe_div(n_corrected_correct, n_samples)
        delta_pp = (corrected_acc - baseline_acc) * 100.0

        run_h = duration_s / 3600.0 if not math.isnan(duration_s) else float("nan")
        runtime_per_sample_s = safe_div(duration_s, n_samples)
        samples_per_hour = safe_div(n_samples, run_h) if not math.isnan(run_h) else float("nan")
        flips_per_gpu_hour = safe_div(n_flipped, run_h) if not math.isnan(run_h) else float("nan")
        gain_pp_per_gpu_hour = safe_div(delta_pp, run_h) if not math.isnan(run_h) else float("nan")

        summary_path = find_summary_path(manifest_path, experiment_id)
        csv_path = find_results_csv_path(manifest_path, experiment_id)
        optional = extract_optional_csv_metrics(csv_path)

        rec = {
            "experiment_id": experiment_id,
            "method": detect_method(experiment_id),
            "seed": parse_seed(experiment_id),
            "manifest_path": str(manifest_path),
            "summary_path": str(summary_path) if summary_path else "",
            "results_csv_path": str(csv_path) if csv_path else "",
            "n_samples": n_samples,
            "n_baseline_correct": n_baseline_correct,
            "n_corrected_correct": n_corrected_correct,
            "n_flipped": n_flipped,
            "baseline_accuracy": baseline_acc,
            "corrected_accuracy": corrected_acc,
            "delta_accuracy_pp": delta_pp,
            "duration_seconds": duration_s,
            "run_hours": run_h,
            "runtime_per_sample_s": runtime_per_sample_s,
            "samples_per_hour": samples_per_hour,
            "flip_to_correct_per_gpu_hour": flips_per_gpu_hour,
            "gain_pp_per_gpu_hour": gain_pp_per_gpu_hour,
            **optional,
        }

        avg_toks = rec.get("avg_tokens_per_sample", float("nan"))
        if not math.isnan(avg_toks) and avg_toks > 0:
            rec["gain_per_1k_tokens_pp"] = safe_div(delta_pp, avg_toks / 1000.0)

        records.append(rec)

    return records


def aggregate_by_method(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    numeric_cols = [
        "n_samples",
        "baseline_accuracy",
        "corrected_accuracy",
        "delta_accuracy_pp",
        "duration_seconds",
        "run_hours",
        "runtime_per_sample_s",
        "samples_per_hour",
        "flip_to_correct_per_gpu_hour",
        "gain_pp_per_gpu_hour",
        "intervened_rate",
        "fallback_rate",
        "flip_to_wrong_rate",
        "avg_tokens_per_sample",
        "gain_per_1k_tokens_pp",
    ]

    existing_numeric = [c for c in numeric_cols if c in df.columns]
    grouped = df.groupby(["method", "n_samples"], dropna=False)

    rows: list[dict[str, Any]] = []
    for (method, n_samples), g in grouped:
        row: dict[str, Any] = {
            "method": method,
            "n_samples": int(n_samples),
            "n_runs": int(len(g)),
            "seeds": ",".join(str(int(s)) for s in sorted(g["seed"].dropna().unique())) if g["seed"].notna().any() else "",
        }
        for col in existing_numeric:
            s = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}_mean"] = float(s.mean())
            row[f"{col}_std"] = float(s.std(ddof=0))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    method_order = {"E1": 0, "E12b": 1, "E12": 2, "E3": 3, "Other": 9}
    out["_rank"] = out["method"].map(method_order).fillna(99)
    out = out.sort_values(["n_samples", "_rank", "method"]).drop(columns=["_rank"]).reset_index(drop=True)
    return out


def _fmt(v: Any, kind: str = "float") -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass

    if kind == "pct":
        return f"{float(v) * 100:.2f}%"
    if kind == "pp":
        return f"{float(v):.3f}"
    if kind == "int":
        return f"{int(v)}"
    if kind == "s":
        return f"{float(v):.3f}s"
    if kind == "rate":
        return f"{float(v):.2f}"
    return f"{float(v):.6g}"


def save_pdf(report_path: Path, per_run: pd.DataFrame, agg: pd.DataFrame) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def _color_delta(v: float) -> tuple[float, float, float]:
        if pd.isna(v):
            return (1.0, 1.0, 1.0)
        if v > 0:
            strength = min(1.0, abs(v) / 25.0)
            return (0.86 - 0.35 * strength, 1.0, 0.86 - 0.35 * strength)
        if v < 0:
            strength = min(1.0, abs(v) / 25.0)
            return (1.0, 0.86 - 0.35 * strength, 0.86 - 0.35 * strength)
        return (0.95, 0.95, 0.95)

    def _normalize(value: float, min_v: float, max_v: float) -> float:
        if pd.isna(value) or pd.isna(min_v) or pd.isna(max_v) or max_v <= min_v:
            return 0.5
        return max(0.0, min(1.0, (value - min_v) / (max_v - min_v)))

    def _color_runtime(v: float, min_v: float, max_v: float) -> tuple[float, float, float]:
        # Lower runtime is better (greener), higher runtime is worse (redder).
        x = _normalize(v, min_v, max_v)
        r = 0.85 + 0.15 * x
        g = 0.95 - 0.35 * x
        b = 0.85 - 0.15 * x
        return (r, g, b)

    with PdfPages(report_path) as pdf:
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.axis("off")
        ax.set_title("Priority B Task 5: Cost-Efficiency Summary (Aggregated)", fontsize=14, pad=14)

        if agg.empty:
            ax.text(0.5, 0.5, "No eligible runs found", ha="center", va="center", fontsize=12)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        else:
            runtime_min = pd.to_numeric(agg.get("runtime_per_sample_s_mean"), errors="coerce").min()
            runtime_max = pd.to_numeric(agg.get("runtime_per_sample_s_mean"), errors="coerce").max()

            view = pd.DataFrame(
                {
                    "Method": agg["method"],
                    "n": agg["n_samples"].map(lambda x: _fmt(x, "int")),
                    "Runs": agg["n_runs"].map(lambda x: _fmt(x, "int")),
                    "Seeds": agg["seeds"],
                    "Baseline\nAcc (mean)": agg["baseline_accuracy_mean"].map(lambda x: _fmt(x, "pct")),
                    "Corrected\nAcc (mean)": agg["corrected_accuracy_mean"].map(lambda x: _fmt(x, "pct")),
                    "Delta\npp (mean)": agg["delta_accuracy_pp_mean"].map(lambda x: _fmt(x, "pp")),
                    "Runtime\nper sample": agg["runtime_per_sample_s_mean"].map(lambda x: _fmt(x, "s")),
                    "Samples\nper hour": agg["samples_per_hour_mean"].map(lambda x: _fmt(x, "rate")),
                    "Flips\nper hour": agg["flip_to_correct_per_gpu_hour_mean"].map(lambda x: _fmt(x, "rate")),
                    "Gain\npp/hour": agg["gain_pp_per_gpu_hour_mean"].map(lambda x: _fmt(x, "rate")),
                }
            )

            tab = ax.table(cellText=view.values, colLabels=view.columns, cellLoc="center", loc="center")
            tab.auto_set_font_size(False)
            tab.set_fontsize(9)
            tab.scale(1.0, 1.6)

            for j in range(len(view.columns)):
                cell = tab[(0, j)]
                cell.set_facecolor((0.9, 0.92, 0.96))
                cell.get_text().set_weight("bold")

            # Body styling: zebra rows + semantic color encoding
            delta_col = list(view.columns).index("Delta\npp (mean)")
            runtime_col = list(view.columns).index("Runtime\nper sample")
            gain_col = list(view.columns).index("Gain\npp/hour")

            for i in range(len(view)):
                if i % 2 == 1:
                    for j in range(len(view.columns)):
                        tab[(i + 1, j)].set_facecolor((0.985, 0.985, 0.985))

                delta_val = pd.to_numeric(agg.iloc[i]["delta_accuracy_pp_mean"], errors="coerce")
                tab[(i + 1, delta_col)].set_facecolor(_color_delta(delta_val))

                runtime_val = pd.to_numeric(agg.iloc[i]["runtime_per_sample_s_mean"], errors="coerce")
                tab[(i + 1, runtime_col)].set_facecolor(_color_runtime(runtime_val, runtime_min, runtime_max))

                gain_val = pd.to_numeric(agg.iloc[i]["gain_pp_per_gpu_hour_mean"], errors="coerce")
                tab[(i + 1, gain_col)].set_facecolor(_color_delta(gain_val))

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Visualization page: bars for key metrics (n=2000 preferred if available)
            chart_df = agg.copy()
            if "n_samples" in chart_df.columns and (chart_df["n_samples"] == 2000).any():
                chart_df = chart_df[chart_df["n_samples"] == 2000].copy()

            if not chart_df.empty:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle("Cost-Efficiency Visual Summary", fontsize=14)

                x = chart_df["method"].tolist()
                delta = pd.to_numeric(chart_df["delta_accuracy_pp_mean"], errors="coerce").fillna(0.0)
                runtime = pd.to_numeric(chart_df["runtime_per_sample_s_mean"], errors="coerce").fillna(0.0)

                colors_delta = ["#4caf50" if v >= 0 else "#ef5350" for v in delta]
                axes[0].bar(x, delta, color=colors_delta)
                axes[0].axhline(0, color="black", linewidth=0.8)
                axes[0].set_title("Mean Accuracy Gain (pp)")
                axes[0].set_ylabel("pp")
                axes[0].tick_params(axis="x", rotation=25)

                colors_runtime = ["#42a5f5" for _ in x]
                axes[1].bar(x, runtime, color=colors_runtime)
                axes[1].set_title("Mean Runtime per Sample")
                axes[1].set_ylabel("seconds")
                axes[1].tick_params(axis="x", rotation=25)

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                # Additional page 1: Pareto-style scatter (runtime vs gain)
                fig, ax = plt.subplots(figsize=(11, 7))
                fig.suptitle("Pareto View: Runtime vs Accuracy Gain", fontsize=14)

                x = pd.to_numeric(chart_df["runtime_per_sample_s_mean"], errors="coerce").fillna(0.0)
                y = pd.to_numeric(chart_df["delta_accuracy_pp_mean"], errors="coerce").fillna(0.0)
                n_runs = pd.to_numeric(chart_df["n_runs"], errors="coerce").fillna(1.0)

                sizes = 180 + 120 * n_runs
                colors = plt.cm.tab10(range(len(chart_df)))

                for i, (_, row) in enumerate(chart_df.reset_index(drop=True).iterrows()):
                    ax.scatter(x.iloc[i], y.iloc[i], s=sizes.iloc[i], alpha=0.75, color=colors[i], edgecolor="black")
                    ax.annotate(
                        str(row["method"]),
                        (x.iloc[i], y.iloc[i]),
                        textcoords="offset points",
                        xytext=(6, 6),
                        fontsize=10,
                    )

                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_xlabel("Runtime per sample (seconds)")
                ax.set_ylabel("Accuracy gain (pp)")
                ax.grid(alpha=0.25)

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                # Additional page 2: efficiency score bars
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle("Efficiency Metrics by Method", fontsize=14)

                order = chart_df.sort_values("gain_pp_per_gpu_hour_mean", ascending=True)
                methods = order["method"].astype(str).tolist()

                gain_h = pd.to_numeric(order["gain_pp_per_gpu_hour_mean"], errors="coerce").fillna(0.0)
                flips_h = pd.to_numeric(order["flip_to_correct_per_gpu_hour_mean"], errors="coerce").fillna(0.0)

                gain_colors = ["#4caf50" if v >= 0 else "#ef5350" for v in gain_h]
                axes[0].barh(methods, gain_h, color=gain_colors)
                axes[0].axvline(0, color="black", linewidth=0.8)
                axes[0].set_title("Gain pp per GPU hour")
                axes[0].set_xlabel("pp / GPU-hour")

                axes[1].barh(methods, flips_h, color="#42a5f5")
                axes[1].set_title("Flip-to-correct per GPU hour")
                axes[1].set_xlabel("flips / GPU-hour")

                for ax in axes:
                    ax.grid(axis="x", alpha=0.25)

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                # Additional page 3: normalized heatmap for quick ranking
                metrics = [
                    ("delta_accuracy_pp_mean", "Delta pp ↑"),
                    ("gain_pp_per_gpu_hour_mean", "Gain pp/hour ↑"),
                    ("flip_to_correct_per_gpu_hour_mean", "Flips/hour ↑"),
                    ("runtime_per_sample_s_mean", "Runtime/sample ↓"),
                ]
                heat = []
                labels = []
                for col, lab in metrics:
                    s = pd.to_numeric(chart_df[col], errors="coerce")
                    if s.isna().all():
                        norm = pd.Series([0.5] * len(chart_df))
                    else:
                        smin, smax = s.min(), s.max()
                        if smax <= smin:
                            norm = pd.Series([0.5] * len(chart_df))
                        else:
                            norm = (s - smin) / (smax - smin)
                    if "↓" in lab:
                        norm = 1.0 - norm
                    heat.append(norm.fillna(0.5).tolist())
                    labels.append(lab)

                fig, ax = plt.subplots(figsize=(10, 6))
                hm = ax.imshow(list(zip(*heat)), cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)
                ax.set_title("Normalized Performance Heatmap (higher is better)", fontsize=13)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=20, ha="right")
                ax.set_yticks(range(len(chart_df)))
                ax.set_yticklabels(chart_df["method"].astype(str).tolist())
                cbar = fig.colorbar(hm, ax=ax)
                cbar.set_label("normalized score")

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        if per_run.empty:
            return

        # Per-run dispersion page (actual run variability)
        fig, ax = plt.subplots(figsize=(11, 7))
        fig.suptitle("Per-Run Dispersion: Runtime vs Accuracy Gain", fontsize=14)

        run_df = per_run.copy()
        run_df["runtime_per_sample_s"] = pd.to_numeric(run_df.get("runtime_per_sample_s"), errors="coerce")
        run_df["delta_accuracy_pp"] = pd.to_numeric(run_df.get("delta_accuracy_pp"), errors="coerce")

        palette = {
            "E1": "#1f77b4",
            "E12b": "#ff7f0e",
            "E12": "#2ca02c",
            "E3": "#d62728",
            "Other": "#7f7f7f",
        }

        for method, g in run_df.groupby("method"):
            ax.scatter(
                g["runtime_per_sample_s"],
                g["delta_accuracy_pp"],
                label=str(method),
                alpha=0.75,
                s=80,
                color=palette.get(str(method), "#7f7f7f"),
                edgecolor="black",
            )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Runtime per sample (seconds)")
        ax.set_ylabel("Accuracy gain (pp)")
        ax.grid(alpha=0.25)
        ax.legend(title="Method", loc="best")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        cols = [
            "experiment_id",
            "method",
            "seed",
            "n_samples",
            "baseline_accuracy",
            "corrected_accuracy",
            "delta_accuracy_pp",
            "runtime_per_sample_s",
            "samples_per_hour",
            "flip_to_correct_per_gpu_hour",
            "gain_pp_per_gpu_hour",
            "intervened_rate",
            "fallback_rate",
        ]
        cols = [c for c in cols if c in per_run.columns]
        display = per_run[cols].copy()

        fmt_pct_cols = {"baseline_accuracy", "corrected_accuracy", "intervened_rate", "fallback_rate"}
        fmt_pp_cols = {"delta_accuracy_pp"}
        fmt_s_cols = {"runtime_per_sample_s"}
        fmt_rate_cols = {"samples_per_hour", "flip_to_correct_per_gpu_hour", "gain_pp_per_gpu_hour"}

        for c in display.columns:
            if c in fmt_pct_cols:
                display[c] = display[c].map(lambda x: _fmt(x, "pct"))
            elif c in fmt_pp_cols:
                display[c] = display[c].map(lambda x: _fmt(x, "pp"))
            elif c in fmt_s_cols:
                display[c] = display[c].map(lambda x: _fmt(x, "s"))
            elif c in fmt_rate_cols:
                display[c] = display[c].map(lambda x: _fmt(x, "rate"))
            elif c in {"seed", "n_samples"}:
                display[c] = display[c].map(lambda x: "" if pd.isna(x) else _fmt(x, "int"))

        rows_per_page = 22
        n_rows = len(display)
        n_pages = max(1, (n_rows + rows_per_page - 1) // rows_per_page)

        for i in range(n_pages):
            s = i * rows_per_page
            e = min((i + 1) * rows_per_page, n_rows)
            chunk = display.iloc[s:e]

            # Two-line friendly headers for compact readability.
            pretty_cols = {
                "experiment_id": "Experiment\nID",
                "method": "Method",
                "seed": "Seed",
                "n_samples": "n",
                "baseline_accuracy": "Baseline\nAcc",
                "corrected_accuracy": "Corrected\nAcc",
                "delta_accuracy_pp": "Delta\npp",
                "runtime_per_sample_s": "Runtime\n/sample",
                "samples_per_hour": "Samples\n/hour",
                "flip_to_correct_per_gpu_hour": "Flips\n/hour",
                "gain_pp_per_gpu_hour": "Gain\npp/hour",
                "intervened_rate": "Intervened\nrate",
                "fallback_rate": "Fallback\nrate",
            }
            chunk = chunk.rename(columns=pretty_cols)

            if "Experiment\nID" in chunk.columns:
                chunk["Experiment\nID"] = chunk["Experiment\nID"].map(lambda x: fill(str(x), width=28))

            fig, ax = plt.subplots(figsize=(16, 9))
            ax.axis("off")
            ax.set_title(f"Cost-Efficiency Per Run (rows {s+1}-{e} of {n_rows})", fontsize=12, pad=12)
            tab = ax.table(cellText=chunk.values, colLabels=chunk.columns, cellLoc="center", loc="center")
            tab.auto_set_font_size(False)
            tab.set_fontsize(8)
            tab.scale(1.0, 1.35)

            for j in range(len(chunk.columns)):
                cell = tab[(0, j)]
                cell.set_facecolor((0.9, 0.92, 0.96))
                cell.get_text().set_weight("bold")

            # zebra striping for body
            for r in range(len(chunk)):
                if r % 2 == 1:
                    for c in range(len(chunk.columns)):
                        tab[(r + 1, c)].set_facecolor((0.985, 0.985, 0.985))

            # Semantic color encoding for key columns
            if "Delta\npp" in chunk.columns:
                c_idx = list(chunk.columns).index("Delta\npp")
                raw_delta = pd.to_numeric(display.iloc[s:e]["delta_accuracy_pp"], errors="coerce")
                for r, v in enumerate(raw_delta):
                    tab[(r + 1, c_idx)].set_facecolor(_color_delta(v))

            if "Runtime\n/sample" in chunk.columns:
                c_idx = list(chunk.columns).index("Runtime\n/sample")
                raw_runtime = pd.to_numeric(display["runtime_per_sample_s"], errors="coerce")
                min_v, max_v = raw_runtime.min(), raw_runtime.max()
                page_runtime = pd.to_numeric(display.iloc[s:e]["runtime_per_sample_s"], errors="coerce")
                for r, v in enumerate(page_runtime):
                    tab[(r + 1, c_idx)].set_facecolor(_color_runtime(v, min_v, max_v))

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_results_root = repo_root / "results"

    parser = argparse.ArgumentParser(description="Build cost-efficiency report from existing artifacts")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_results_root,
        help="Root directory containing run folders/manifests (default: <repo-root>/results)",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="priorityB_task5_cost_efficiency",
        help="Output subfolder under results root",
    )
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    out_dir = results_root / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    records = discover_phase2_runs(results_root)
    per_run = pd.DataFrame(records)

    if not per_run.empty:
        method_order = {"E1": 0, "E12b": 1, "E12": 2, "E3": 3, "Other": 9}
        per_run["_rank"] = per_run["method"].map(method_order).fillna(99)
        per_run = per_run.sort_values(["n_samples", "_rank", "method", "seed", "experiment_id"]).drop(columns=["_rank"]).reset_index(drop=True)

    agg = aggregate_by_method(per_run)

    per_run_csv = out_dir / "cost_efficiency_per_run.csv"
    per_run_json = out_dir / "cost_efficiency_per_run.json"
    agg_csv = out_dir / "cost_efficiency_aggregated.csv"
    agg_json = out_dir / "cost_efficiency_aggregated.json"
    report_pdf = out_dir / "cost_efficiency_report.pdf"

    per_run.to_csv(per_run_csv, index=False)
    agg.to_csv(agg_csv, index=False)

    with open(per_run_json, "w") as f:
        json.dump(per_run.to_dict(orient="records"), f, indent=2)
    with open(agg_json, "w") as f:
        json.dump(agg.to_dict(orient="records"), f, indent=2)

    save_pdf(report_pdf, per_run, agg)

    print("=" * 80)
    print("PRIORITY B TASK 5: COST-EFFICIENCY ARTIFACTS WRITTEN")
    print("=" * 80)
    print(f"Per-run CSV   : {per_run_csv}")
    print(f"Per-run JSON  : {per_run_json}")
    print(f"Aggregate CSV : {agg_csv}")
    print(f"Aggregate JSON: {agg_json}")
    print(f"PDF report    : {report_pdf}")
    print("=" * 80)


if __name__ == "__main__":
    main()
