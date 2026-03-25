"""
BCI — Analysis & Visualization
Generates figures and tables for the paper.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bci.config import FIGURES_DIR, RESULTS_DIR


def plot_error_breakdown(breakdown: dict, save_path: Path | None = None) -> None:
    """
    Plot the error type breakdown — key figure for H1 validation.
    Shows: correct | premise error | reasoning error
    """
    save_path = save_path or FIGURES_DIR / "error_breakdown.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Correct", "Premise Error", "Reasoning Error"]
    counts = [
        breakdown["correct"],
        breakdown["premise_errors"],
        breakdown["reasoning_errors"],
    ]
    total = breakdown["total_samples"]
    percentages = [c / total * 100 for c in counts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    bars = ax1.bar(labels, percentages, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Percentage of Samples (%)", fontsize=12)
    ax1.set_title("Error Type Distribution", fontsize=14, fontweight="bold")
    for bar, pct, count in zip(bars, percentages, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{pct:.1f}%\n(n={count})",
            ha="center", va="bottom", fontsize=10,
        )
    ax1.set_ylim(0, max(percentages) + 15)

    # Among incorrect only
    incorrect = breakdown["incorrect"]
    if incorrect > 0:
        inc_labels = ["Premise Error", "Reasoning Error"]
        inc_counts = [breakdown["premise_errors"], breakdown["reasoning_errors"]]
        inc_pcts = [c / incorrect * 100 for c in inc_counts]

        wedges, texts, autotexts = ax2.pie(
            inc_pcts,
            labels=inc_labels,
            colors=["#e74c3c", "#3498db"],
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 11},
        )
        ax2.set_title(
            f"Among Incorrect Answers (n={incorrect})",
            fontsize=14, fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_premise_correction(results: list[dict], save_path: Path | None = None) -> None:
    """Plot Experiment 2: premise correction flip rate."""
    save_path = save_path or FIGURES_DIR / "premise_correction.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(results)
    flipped = sum(1 for r in results if r["flipped_to_correct"])
    not_flipped = total - flipped

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Corrected\n(flipped to correct)", "Not Corrected"],
        [flipped, not_flipped],
        color=["#2ecc71", "#e74c3c"],
        edgecolor="black", linewidth=0.5,
    )
    for bar, count in zip(bars, [flipped, not_flipped]):
        pct = count / total * 100 if total > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{pct:.1f}%\n(n={count})",
            ha="center", va="bottom", fontsize=11,
        )

    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title(
        f"Premise Correction: Answer Flip Rate\n"
        f"Recovery Rate: {flipped/total*100:.1f}%" if total > 0 else "No data",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylim(0, max(flipped, not_flipped) + max(flipped, not_flipped) * 0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_belief_minimality(results: list[dict], save_path: Path | None = None) -> None:
    """
    Plot Experiment 4: belief minimality.
    Shows % failures fixed vs # beliefs removed.
    """
    save_path = save_path or FIGURES_DIR / "belief_minimality.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate: for each number of removals, what fraction flipped?
    # Since we remove one at a time, we check single-removal fix rate
    total_cases = 0
    fixed_by_one = 0

    for r in results:
        if not r.get("removal_results"):
            continue
        total_cases += 1
        any_fixed = any(rr["flipped_to_correct"] for rr in r["removal_results"])
        if any_fixed:
            fixed_by_one += 1

    fig, ax = plt.subplots(figsize=(6, 5))

    if total_cases > 0:
        fix_rate = fixed_by_one / total_cases * 100
        bars = ax.bar(
            ["Fixed by Removing\n1 False Belief", "Not Fixed"],
            [fixed_by_one, total_cases - fixed_by_one],
            color=["#2ecc71", "#95a5a6"],
            edgecolor="black", linewidth=0.5,
        )
        for bar, count in zip(bars, [fixed_by_one, total_cases - fixed_by_one]):
            pct = count / total_cases * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{pct:.1f}%\n(n={count})",
                ha="center", va="bottom", fontsize=11,
            )
        ax.set_title(
            f"Belief Minimality: Removing 1 False Belief\n"
            f"Fix Rate: {fix_rate:.1f}%",
            fontsize=14, fontweight="bold",
        )
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    ax.set_ylabel("Number of Cases", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_random_vs_targeted(
    targeted_results: list[dict],
    random_results: list[dict],
    save_path: Path | None = None,
) -> None:
    """
    Plot placebo control: targeted vs random belief removal.
    """
    save_path = save_path or FIGURES_DIR / "random_vs_targeted.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Targeted fix rate (from belief minimality or premise correction)
    targeted_total = len(targeted_results)
    targeted_flipped = sum(
        1 for r in targeted_results if r.get("flipped_to_correct", False)
    )
    targeted_rate = targeted_flipped / targeted_total * 100 if targeted_total > 0 else 0

    # Random fix rate
    random_rates = [r.get("flip_rate", 0) for r in random_results]
    random_rate = np.mean(random_rates) * 100 if random_rates else 0
    random_std = np.std(random_rates) * 100 if random_rates else 0

    fig, ax = plt.subplots(figsize=(6, 5))
    methods = ["Targeted\n(Remove Contradicted)", "Random\n(Placebo Control)"]
    rates = [targeted_rate, random_rate]
    errors = [0, random_std]
    colors = ["#2ecc71", "#e74c3c"]

    bars = ax.bar(methods, rates, yerr=errors, color=colors,
                  edgecolor="black", linewidth=0.5, capsize=5)
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{rate:.1f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_ylabel("Answer Flip Rate (%)", fontsize=12)
    ax.set_title(
        "Targeted vs Random Belief Removal",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylim(0, max(rates) + 20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_claim_coverage(coverages: list[dict], save_path: Path | None = None) -> None:
    """Plot claim coverage distribution."""
    save_path = save_path or FIGURES_DIR / "claim_coverage.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    valid = [c["coverage"] for c in coverages if c["coverage"] is not None]

    if not valid:
        print("No valid coverage data to plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(valid, bins=20, color="#3498db", edgecolor="black", alpha=0.8)
    ax.axvline(np.mean(valid), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {np.mean(valid):.2f}")
    ax.axvline(np.median(valid), color="orange", linestyle="--", linewidth=2,
               label=f"Median: {np.median(valid):.2f}")
    ax.set_xlabel("Claim Coverage (fraction of relevant facts surfaced)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Claim Coverage Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def generate_summary_table(
    breakdown: dict,
    correction_results: list[dict] | None = None,
    minimality_results: list[dict] | None = None,
) -> str:
    """Generate a summary table (markdown format) for the paper."""
    lines = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Samples | {breakdown['total_samples']} |",
        f"| Accuracy | {breakdown['accuracy']*100:.1f}% |",
        f"| Premise Errors (of incorrect) | {breakdown['premise_error_rate_of_incorrect']*100:.1f}% |",
        f"| Reasoning Errors (of incorrect) | {breakdown['reasoning_error_rate_of_incorrect']*100:.1f}% |",
    ]

    if correction_results:
        total = len(correction_results)
        flipped = sum(1 for r in correction_results if r["flipped_to_correct"])
        lines.append(
            f"| Answer Flip Rate (Exp 2) | {flipped/total*100:.1f}% |"
            if total > 0
            else "| Answer Flip Rate (Exp 2) | N/A |"
        )

    if minimality_results:
        total = len(minimality_results)
        fixed = sum(
            1 for r in minimality_results
            if any(rr["flipped_to_correct"] for rr in r.get("removal_results", []))
        )
        lines.append(
            f"| Fixed by 1 Belief Removal (Exp 4) | {fixed/total*100:.1f}% |"
            if total > 0
            else "| Fixed by 1 Belief Removal (Exp 4) | N/A |"
        )

    return "\n".join(lines)
