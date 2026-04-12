"""Phase 4 for Task 4: Build polished report from Phase 1 + Phase 3 outputs.

Generates a PDF with intuitive visuals highlighting E1 and E12b advantages,
plus compact CSV/JSON headline tables.
"""

from __future__ import annotations

import argparse
import json
from textwrap import fill
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


COLOR_E1 = "#2563eb"      # blue
COLOR_E12B = "#f59e0b"    # amber
COLOR_POS = "#22c55e"     # green
COLOR_NEG = "#ef4444"     # red


def wrap_label(text: str, width: int = 14) -> str:
    return fill(str(text), width=width)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_fmt(v, kind: str = "float") -> str:
    if pd.isna(v):
        return ""
    if kind == "pct":
        return f"{float(v) * 100:.1f}%"
    if kind == "pp":
        return f"{float(v):.2f}"
    if kind == "int":
        return f"{int(v)}"
    return f"{float(v):.4g}"


def save_pdf(
    pdf_path: Path,
    struct_comp: pd.DataFrame,
    sem_comp: pd.DataFrame,
    claim_presence_comp: pd.DataFrame,
    dominant_comp: pd.DataFrame,
) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        # ------------------------------------------------------------------
        # Page 1: Executive summary
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12.5, 8.5))
        ax.axis("off")

        top_struct = struct_comp.head(5) if not struct_comp.empty else pd.DataFrame()
        top_claim = claim_presence_comp.head(5) if not claim_presence_comp.empty else pd.DataFrame()

        lines = ["Task 4 Report: Error-type Stratified Gains (E1 vs E12b)"]
        if not top_struct.empty:
            lines.append("\nTop structural-type advantages (delta pp gap, E1 - E12b):")
            for _, r in top_struct.iterrows():
                lines.append(f"- {r['question_type']}: {float(r['delta_pp_gap_e1_minus_e12b']):+.2f} pp")
        if not top_claim.empty:
            lines.append("\nTop claim-type presence advantages (delta pp gap, E1 - E12b):")
            for _, r in top_claim.iterrows():
                lines.append(f"- {r['stratum']}: {float(r['delta_pp_gap_e1_minus_e12b']):+.2f} pp")

        ax.text(
            0.04,
            0.95,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.7", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
        )

        ax.set_title("Priority A Task 4 — Stratified Analysis", fontsize=18, weight="bold", pad=20)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ------------------------------------------------------------------
        # Page 2: Structural question-type comparison (top by support)
        # ------------------------------------------------------------------
        if not struct_comp.empty:
            fig, ax = plt.subplots(figsize=(13, 7.8))

            plot_df = struct_comp.copy()
            plot_df["n_support"] = plot_df[["n_e1", "n_e12b"]].min(axis=1)
            plot_df = plot_df.sort_values("n_support", ascending=False).head(12).copy()
            plot_df = plot_df.sort_values("delta_pp_gap_e1_minus_e12b", ascending=True)

            y = np.arange(len(plot_df))
            ax.barh(y - 0.18, plot_df["e1_delta_pp"], height=0.34, color=COLOR_E1, label="E1 delta pp")
            ax.barh(y + 0.18, plot_df["e12b_delta_pp"], height=0.34, color=COLOR_E12B, label="E12b delta pp")

            # gap markers
            for i, (_, r) in enumerate(plot_df.iterrows()):
                gap = float(r["delta_pp_gap_e1_minus_e12b"])
                ax.text(
                    max(float(r["e1_delta_pp"]), float(r["e12b_delta_pp"])) + 0.4,
                    i,
                    f"gap {gap:+.1f}",
                    va="center",
                    fontsize=8,
                    color=COLOR_POS if gap >= 0 else COLOR_NEG,
                )

            ax.set_yticks(y)
            ax.set_yticklabels([wrap_label(v, 16) for v in plot_df["question_type"]])
            ax.set_xlabel("Accuracy gain over baseline (pp)")
            ax.set_title("Structural question types (top support) — E1 vs E12b", fontsize=14, weight="bold")
            ax.grid(axis="x", alpha=0.25)
            ax.legend(frameon=True)
            fig.subplots_adjust(left=0.27, right=0.95, top=0.9, bottom=0.12)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ------------------------------------------------------------------
        # Page 3: Semantic question-type gap heatmap
        # ------------------------------------------------------------------
        if not sem_comp.empty:
            fig, ax = plt.subplots(figsize=(12.5, 7.6))
            sem = sem_comp.copy().sort_values("delta_pp_gap_e1_minus_e12b", ascending=False).head(16)
            vals = sem[["delta_pp_gap_e1_minus_e12b"]].to_numpy()
            im = ax.imshow(vals, cmap="RdYlGn", aspect="auto")
            ax.set_title("Semantic type advantage heatmap (E1 - E12b, delta pp)", fontsize=14, weight="bold")
            ax.set_xticks([0])
            ax.set_xticklabels(["delta pp\ngap"])
            ax.set_yticks(np.arange(len(sem)))
            ax.set_yticklabels([wrap_label(v, 18) for v in sem["question_type"]])
            for i, v in enumerate(sem["delta_pp_gap_e1_minus_e12b"].to_list()):
                ax.text(0, i, f"{v:+.2f}", ha="center", va="center", color="black", fontsize=8)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("E1 advantage (pp)")
            fig.subplots_adjust(left=0.33, right=0.93, top=0.9, bottom=0.12)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ------------------------------------------------------------------
        # Page 4: Claim-type presence comparison (core mechanistic page)
        # ------------------------------------------------------------------
        if not claim_presence_comp.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
            fig.suptitle("Claim-type presence strata: method comparison", fontsize=15, weight="bold")

            cp = claim_presence_comp.copy().sort_values("delta_pp_gap_e1_minus_e12b", ascending=False)

            # Left: gap bars
            y = np.arange(len(cp))
            colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in cp["delta_pp_gap_e1_minus_e12b"]]
            axes[0].barh(y, cp["delta_pp_gap_e1_minus_e12b"], color=colors)
            axes[0].axvline(0, color="#555", linewidth=1)
            axes[0].set_yticks(y)
            axes[0].set_yticklabels([wrap_label(v, 14) for v in cp["stratum"]])
            axes[0].set_xlabel("delta pp gap (E1 - E12b)")
            axes[0].set_title("Advantage gap")
            axes[0].grid(axis="x", alpha=0.2)

            # Right: absolute deltas by method
            x = np.arange(len(cp))
            w = 0.38
            axes[1].bar(x - w / 2, cp["e1_delta_pp"], width=w, color=COLOR_E1, label="E1")
            axes[1].bar(x + w / 2, cp["e12b_delta_pp"], width=w, color=COLOR_E12B, label="E12b")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([wrap_label(v, 10) for v in cp["stratum"]], rotation=25, ha="right")
            axes[1].set_ylabel("delta pp")
            axes[1].set_title("Absolute gain per stratum")
            axes[1].grid(axis="y", alpha=0.2)
            axes[1].legend(frameon=True)

            fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.27, wspace=0.32)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ------------------------------------------------------------------
        # Page 5: Dominant claim-type table (two-row headers)
        # ------------------------------------------------------------------
        if not dominant_comp.empty:
            fig, ax = plt.subplots(figsize=(13.2, 7.5))
            ax.axis("off")
            ax.set_title("Dominant contradicted type comparison", fontsize=14, weight="bold", pad=12)

            tbl = dominant_comp.copy()
            cols = [
                "stratum",
                "n_e1",
                "n_e12b",
                "e1_delta_pp",
                "e12b_delta_pp",
                "delta_pp_gap_e1_minus_e12b",
                "corrected_acc_gap_e1_minus_e12b",
            ]
            cols = [c for c in cols if c in tbl.columns]
            tbl = tbl[cols].head(12).copy()

            rename = {
                "stratum": "Dominant\nType",
                "n_e1": "n\nE1",
                "n_e12b": "n\nE12b",
                "e1_delta_pp": "E1\nDelta pp",
                "e12b_delta_pp": "E12b\nDelta pp",
                "delta_pp_gap_e1_minus_e12b": "Gap\n(E1-E12b)",
                "corrected_acc_gap_e1_minus_e12b": "Corrected Acc\nGap",
            }
            tbl = tbl.rename(columns=rename)

            # format numerics
            for c in tbl.columns:
                if "n\n" in c:
                    tbl[c] = tbl[c].map(lambda x: _safe_fmt(x, "int"))
                elif "Delta" in c or "Gap" in c:
                    tbl[c] = pd.to_numeric(tbl[c], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:+.2f}")

            table = ax.table(cellText=tbl.values, colLabels=tbl.columns, cellLoc="center", loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.0, 1.4)

            # style header + zebra + gap color
            for j in range(len(tbl.columns)):
                h = table[(0, j)]
                h.set_facecolor((0.90, 0.93, 0.97))
                h.get_text().set_weight("bold")

            gap_col = list(tbl.columns).index("Gap\n(E1-E12b)") if "Gap\n(E1-E12b)" in tbl.columns else None
            for i in range(len(tbl)):
                if i % 2 == 1:
                    for j in range(len(tbl.columns)):
                        table[(i + 1, j)].set_facecolor((0.985, 0.985, 0.985))
                if gap_col is not None:
                    raw = float(dominant_comp.iloc[i]["delta_pp_gap_e1_minus_e12b"]) if i < len(dominant_comp) else 0.0
                    table[(i + 1, gap_col)].set_facecolor((0.86, 1.0, 0.86) if raw >= 0 else (1.0, 0.86, 0.86))

            fig.subplots_adjust(left=0.04, right=0.96, top=0.90, bottom=0.06)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Task4 Phase4: polished report")
    parser.add_argument("--results-root", type=Path, default=repo_root / "results")
    parser.add_argument("--output-subdir", type=str, default="priorityA_task4_error_stratification")
    args = parser.parse_args()

    out_dir = args.results_root.resolve() / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    struct_comp = load_csv(out_dir / "phase1_question_type_structural_comparison_e1_vs_e12b.csv")
    sem_comp = load_csv(out_dir / "phase1_question_type_semantic_comparison_e1_vs_e12b.csv")
    claim_presence_comp = load_csv(out_dir / "phase3_claim_type_presence_comparison_e1_vs_e12b.csv")
    dominant_comp = load_csv(out_dir / "phase3_claim_type_dominant_comparison_e1_vs_e12b.csv")

    pdf_path = out_dir / "task4_error_stratification_report.pdf"
    save_pdf(pdf_path, struct_comp, sem_comp, claim_presence_comp, dominant_comp)

    # compact headline dump
    headline = {
        "top_structural_advantages": struct_comp.head(5).to_dict(orient="records") if not struct_comp.empty else [],
        "top_semantic_advantages": sem_comp.head(5).to_dict(orient="records") if not sem_comp.empty else [],
        "top_claim_presence_advantages": claim_presence_comp.head(6).to_dict(orient="records") if not claim_presence_comp.empty else [],
    }
    with open(out_dir / "phase4_headline_summary.json", "w") as f:
        json.dump(headline, f, indent=2)

    print("=" * 80)
    print("TASK4 PHASE4 COMPLETE")
    print("=" * 80)
    print(f"Report PDF             : {pdf_path}")
    print(f"Headline summary JSON  : {out_dir / 'phase4_headline_summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
