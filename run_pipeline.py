"""
BCI — Main Pipeline
Phase 1: Premise Error Analysis

Usage:
    # Step 1: Download GQA scene graphs
    python run_pipeline.py download

    # Step 2: Setup data (sample questions + download images from HF)
    python run_pipeline.py setup

    # Step 3: Run VLM inference (Experiment 1)
    python run_pipeline.py experiment1

    # Step 4: Analyze results
    python run_pipeline.py analyze

    # Step 5: Run premise correction (Experiment 2)
    python run_pipeline.py experiment2

    # Step 6: Run all experiments
    python run_pipeline.py all
"""
import argparse
import json
import sys
from pathlib import Path

from config import (
    GQA_DIR,
    NUM_SAMPLES,
    RESULTS_DIR,
    FIGURES_DIR,
)


def step_download():
    """Download GQA scene graphs from Stanford."""
    from data_loader import download_scene_graphs
    download_scene_graphs()


def step_setup():
    """
    Load questions from HuggingFace, sample those with scene graphs,
    and download only the needed images.
    """
    from data_loader import setup_gqa_data

    samples, sg, image_dir = setup_gqa_data()

    # Save the sample list for reference
    sample_path = RESULTS_DIR / "sampled_questions.json"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    samples_to_save = [
        {k: v for k, v in s.items() if k != "scene_graph"} for s in samples
    ]
    with open(sample_path, "w") as f:
        json.dump(samples_to_save, f, indent=2)
    print(f"Saved {len(samples)} sampled questions to {sample_path}")


def step_experiment1():
    """
    Experiment 1: Run VLM on sampled GQA questions.
    Generates baseline + belief externalization responses.
    """
    from data_loader import load_scene_graphs, load_gqa_from_hf, sample_questions
    from vlm_inference import run_experiment_1

    sg = load_scene_graphs("val")
    questions = load_gqa_from_hf()
    samples = sample_questions(questions, sg)

    image_dir = GQA_DIR / "images"
    if not image_dir.exists():
        print(f"Error: Image directory not found at {image_dir}")
        print("Run 'python run_pipeline.py setup' first.")
        sys.exit(1)

    # Check how many images we have
    available = {p.stem for p in image_dir.glob("*.jpg")}
    samples_with_images = [s for s in samples if s["image_id"] in available]
    print(f"Have images for {len(samples_with_images)}/{len(samples)} samples.")

    if len(samples_with_images) < 50:
        print("Error: Too few images. Need at least 50.")
        sys.exit(1)

    run_experiment_1(samples_with_images, image_dir)


def step_analyze():
    """
    Analyze Experiment 1 results:
    - Extract claims
    - Verify against scene graphs
    - Classify errors
    - Generate plots
    """
    from data_loader import load_scene_graphs
    from claim_extraction import extract_and_classify
    from claim_verification import verify_all_claims, compute_verification_summary, CONTRADICTED
    from error_classification import (
        classify_error,
        compute_error_breakdown,
        find_recoverable_errors,
    )
    from experiments import compute_claim_coverage
    from analysis import (
        plot_error_breakdown,
        plot_claim_coverage,
        generate_summary_table,
    )

    # Load VLM outputs
    results_path = RESULTS_DIR / "experiment1_vlm_outputs.json"
    if not results_path.exists():
        print(f"Error: Results not found at {results_path}")
        print("Run 'python run_pipeline.py experiment1' first.")
        sys.exit(1)

    with open(results_path) as f:
        vlm_results = json.load(f)

    print(f"Loaded {len(vlm_results)} VLM results.")

    # Load scene graphs
    sg = load_scene_graphs("val")

    # Step 1: Extract and classify claims
    print("\n--- Claim Extraction ---")
    all_extractions = []
    for result in vlm_results:
        extraction = extract_and_classify(result)
        all_extractions.append(extraction)

    total_baseline_claims = sum(e["num_baseline_claims"] for e in all_extractions)
    total_belief_claims = sum(e["num_belief_claims"] for e in all_extractions)
    print(f"Total baseline claims: {total_baseline_claims}")
    print(f"Total belief claims: {total_belief_claims}")
    print(f"Avg baseline claims/question: {total_baseline_claims/len(all_extractions):.1f}")
    print(f"Avg belief claims/question: {total_belief_claims/len(all_extractions):.1f}")

    # Step 2: Verify claims against scene graphs
    print("\n--- Claim Verification ---")
    all_verified = []
    for result, extraction in zip(vlm_results, all_extractions):
        img_id = result["image_id"]
        if img_id not in sg:
            continue
        scene_graph = sg[img_id]

        # Verify belief externalization claims (primary)
        verified = verify_all_claims(extraction["belief_claims"], scene_graph)
        all_verified.append(verified)

        summary = compute_verification_summary(verified)
        result["verification_summary"] = summary
        result["verified_claims"] = verified

    # Step 3: Classify errors
    print("\n--- Error Classification ---")
    classifications = []
    for result in vlm_results:
        if "verified_claims" not in result:
            continue
        classification = classify_error(result, result["verified_claims"])
        # Add question type info
        classification["question_type_structural"] = result.get(
            "question_type_structural", "unknown"
        )
        classifications.append(classification)

    # Step 4: Compute breakdown
    breakdown = compute_error_breakdown(classifications)

    print("\n" + "=" * 60)
    print("EXPERIMENT 1 RESULTS — ERROR BREAKDOWN")
    print("=" * 60)
    print(f"Total samples:   {breakdown['total_samples']}")
    print(f"Correct:         {breakdown['correct']} ({breakdown['accuracy']*100:.1f}%)")
    print(f"Incorrect:       {breakdown['incorrect']}")
    print(f"")
    print(f"Among INCORRECT answers:")
    print(f"  Premise errors:   {breakdown['premise_errors']} "
          f"({breakdown['premise_error_rate_of_incorrect']*100:.1f}%)")
    print(f"  Reasoning errors: {breakdown['reasoning_errors']} "
          f"({breakdown['reasoning_error_rate_of_incorrect']*100:.1f}%)")
    print("=" * 60)

    # Hypothesis check
    per = breakdown["premise_error_rate_of_incorrect"]
    if per > 0.35:
        print(f"\n✓ STRONG SIGNAL: Premise error rate ({per*100:.1f}%) > 35%")
        print("  Hypothesis H1 is well-supported. Proceed to Phase 2.")
    elif per > 0.25:
        print(f"\n~ MODERATE SIGNAL: Premise error rate ({per*100:.1f}%) in 25-35%")
        print("  Hypothesis H1 has moderate support. Nuanced framing needed.")
    else:
        print(f"\n✗ WEAK SIGNAL: Premise error rate ({per*100:.1f}%) < 25%")
        print("  Hypothesis H1 is not well-supported. Reconsider approach.")

    # Step 5: Claim coverage
    print("\n--- Claim Coverage ---")
    coverages = []
    for result, extraction in zip(vlm_results, all_extractions):
        img_id = result["image_id"]
        if img_id not in sg:
            continue
        cov = compute_claim_coverage(
            extraction["belief_claims"],
            sg[img_id],
            result["question"],
        )
        coverages.append(cov)

    valid_cov = [c["coverage"] for c in coverages if c["coverage"] is not None]
    if valid_cov:
        print(f"Mean coverage: {sum(valid_cov)/len(valid_cov):.2f}")
        print(f"Median coverage: {sorted(valid_cov)[len(valid_cov)//2]:.2f}")

    # Step 6: Generate plots
    print("\n--- Generating Figures ---")
    plot_error_breakdown(breakdown)
    plot_claim_coverage(coverages)

    # Step 7: Save analysis
    analysis_output = {
        "breakdown": breakdown,
        "classifications": classifications,
        "coverages": coverages,
    }
    analysis_path = RESULTS_DIR / "experiment1_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis_output, f, indent=2, default=str)
    print(f"\nFull analysis saved to {analysis_path}")

    # Generate summary table
    table = generate_summary_table(breakdown)
    print(f"\n{table}")

    # Save recoverable errors for Experiment 2
    recoverable = find_recoverable_errors(classifications)
    # Enrich with image_id and beliefs
    for rec in recoverable:
        for result in vlm_results:
            if result["question_id"] == rec["question_id"]:
                rec["image_id"] = result["image_id"]
                rec["beliefs"] = result.get("belief_externalization", {}).get("beliefs", [])
                break

    rec_path = RESULTS_DIR / "recoverable_errors.json"
    with open(rec_path, "w") as f:
        json.dump(recoverable, f, indent=2, default=str)
    print(f"\n{len(recoverable)} recoverable errors saved to {rec_path}")


def step_experiment2():
    """Experiment 2: Premise correction test."""
    from data_loader import load_scene_graphs
    from experiments import run_premise_correction
    from analysis import plot_premise_correction

    # Load recoverable errors
    rec_path = RESULTS_DIR / "recoverable_errors.json"
    if not rec_path.exists():
        print("Error: Run 'python run_pipeline.py analyze' first.")
        sys.exit(1)

    with open(rec_path) as f:
        recoverable = json.load(f)

    sg = load_scene_graphs("val")
    image_dir = GQA_DIR / "images"

    results = run_premise_correction(recoverable, sg, image_dir)
    plot_premise_correction(results)


def step_experiment4():
    """Experiment 4: Belief minimality."""
    from data_loader import load_scene_graphs
    from experiments import run_belief_minimality
    from analysis import plot_belief_minimality

    rec_path = RESULTS_DIR / "recoverable_errors.json"
    if not rec_path.exists():
        print("Error: Run 'python run_pipeline.py analyze' first.")
        sys.exit(1)

    with open(rec_path) as f:
        recoverable = json.load(f)

    sg = load_scene_graphs("val")
    image_dir = GQA_DIR / "images"

    results = run_belief_minimality(recoverable, sg, image_dir)
    plot_belief_minimality(results)


def step_random_ablation():
    """Placebo control: random belief removal."""
    from experiments import run_random_ablation
    from analysis import plot_random_vs_targeted

    rec_path = RESULTS_DIR / "recoverable_errors.json"
    exp2_path = RESULTS_DIR / "experiment2_premise_correction.json"

    if not rec_path.exists() or not exp2_path.exists():
        print("Error: Run experiments 1-2 first.")
        sys.exit(1)

    with open(rec_path) as f:
        recoverable = json.load(f)
    with open(exp2_path) as f:
        correction_results = json.load(f)

    image_dir = GQA_DIR / "images"
    random_results = run_random_ablation(recoverable, image_dir)
    plot_random_vs_targeted(correction_results, random_results)


def step_all():
    """Run the complete Phase 1 pipeline."""
    print("=" * 60)
    print("BCI Phase 1: Complete Pipeline")
    print("=" * 60)

    print("\n[1/6] Downloading GQA scene graphs...")
    step_download()

    print("\n[2/6] Setting up data (HF questions + images)...")
    step_setup()

    print("\n[3/6] Running Experiment 1 (VLM inference)...")
    step_experiment1()

    print("\n[4/6] Analyzing results...")
    step_analyze()

    print("\n[5/6] Running Experiment 2 (premise correction)...")
    step_experiment2()

    print("\n[6/6] Running Experiment 4 (belief minimality)...")
    step_experiment4()

    print("\n" + "=" * 60)
    print("Phase 1 complete. Check results/ for outputs.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BCI Phase 1 Pipeline")
    parser.add_argument(
        "step",
        choices=[
            "download", "setup", "experiment1",
            "analyze", "experiment2", "experiment4",
            "random_ablation", "all",
        ],
        help="Pipeline step to run",
    )
    args = parser.parse_args()

    step_map = {
        "download": step_download,
        "setup": step_setup,
        "experiment1": step_experiment1,
        "analyze": step_analyze,
        "experiment2": step_experiment2,
        "experiment4": step_experiment4,
        "random_ablation": step_random_ablation,
        "all": step_all,
    }

    step_map[args.step]()
