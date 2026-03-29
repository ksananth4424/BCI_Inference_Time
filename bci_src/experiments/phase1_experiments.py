"""
BCI — Premise Correction Experiment (Experiment 2)
Tests H1 directly: if we replace false premises with ground truth,
does the VLM produce correct answers?

Also includes:
- Experiment 3: Claim coverage analysis
- Experiment 4: Belief minimality study
- Random belief ablation (placebo control)
"""
import json
import random
from pathlib import Path

from PIL import Image

from bci.config import RANDOM_SEED, RESULTS_DIR
from bci.data.data_loader import scene_graph_to_facts
from bci.models.vlm_inference import VLMInference, CONSTRAINED_REASONING_PROMPT


def generate_ground_truth_beliefs(scene_graph: dict, question: str) -> list[str]:
    """
    Generate natural language beliefs from ground truth scene graph
    that are relevant to the question.
    """
    facts = scene_graph_to_facts(scene_graph)
    beliefs = []

    question_lower = question.lower()

    for fact in facts:
        # Prioritize facts relevant to the question
        subject = fact["subject"].lower()
        obj = (fact.get("object") or "").lower()

        relevance = (
            subject in question_lower
            or obj in question_lower
            or any(w in question_lower for w in subject.split())
            or any(w in question_lower for w in obj.split() if w)
        )

        if fact["type"] == "object" and relevance:
            beliefs.append(f"There is a {fact['subject']} in the image")
        elif fact["type"] == "attribute" and relevance:
            beliefs.append(f"The {fact['subject']} is {fact['object']}")
        elif fact["type"] == "relation" and relevance:
            beliefs.append(
                f"The {fact['subject']} is {fact['predicate']} the {fact['object']}"
            )

    # If no relevant facts found, include all object facts
    if not beliefs:
        for fact in facts:
            if fact["type"] == "object":
                beliefs.append(f"There is a {fact['subject']} in the image")
            elif fact["type"] == "attribute":
                beliefs.append(f"The {fact['subject']} is {fact['object']}")

    return beliefs[:15]  # Cap to avoid overly long prompts


def run_premise_correction(
    recoverable_errors: list[dict],
    scene_graphs: dict,
    image_dir: Path,
    vlm: VLMInference | None = None,
    output_path: Path | None = None,
) -> list[dict]:
    """
    Experiment 2: Replace false premises with ground truth and re-run.

    For each incorrectly-answered question with false premises:
    1. Generate ground truth beliefs from scene graph
    2. Re-prompt VLM with corrected beliefs
    3. Check if answer flips to correct

    Returns list of result dicts with flip status.
    """
    from tqdm import tqdm

    if vlm is None:
        vlm = VLMInference()

    output_path = output_path or RESULTS_DIR / "experiment2_premise_correction.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for item in tqdm(recoverable_errors, desc="Premise Correction"):
        img_path = image_dir / f"{item['image_id']}.jpg"
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        sg = scene_graphs[item["image_id"]]

        # Generate ground truth beliefs
        gt_beliefs = generate_ground_truth_beliefs(sg, item["question"])

        if not gt_beliefs:
            continue

        # Re-run with corrected beliefs
        corrected = vlm.constrained_reasoning(
            image, item["question"], gt_beliefs
        )

        from bci.analysis.error_classification import answers_match

        corrected_answer = corrected.get("answer", "")
        flipped = answers_match(corrected_answer, item["ground_truth"])

        result = {
            "question_id": item["question_id"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "original_answer": item["baseline_answer"],
            "corrected_answer": corrected_answer,
            "gt_beliefs": gt_beliefs,
            "flipped_to_correct": flipped,
            "n_contradicted_claims": item["n_claims_contradicted"],
            "contradicted_claims": [
                c["claim"] for c in item.get("contradicted_claims", [])
            ],
        }
        results.append(result)

        # Save incrementally
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    total = len(results)
    flipped = sum(1 for r in results if r["flipped_to_correct"])
    print(f"\nPremise Correction Results:")
    print(f"  Total recoverable errors: {total}")
    print(f"  Flipped to correct: {flipped} ({flipped/total*100:.1f}%)" if total > 0 else "  No results")

    return results


def run_belief_minimality(
    recoverable_errors: list[dict],
    scene_graphs: dict,
    image_dir: Path,
    vlm: VLMInference | None = None,
    output_path: Path | None = None,
) -> list[dict]:
    """
    Experiment 4: Belief minimality study.
    Remove one contradicted belief at a time and measure answer flips.
    """
    from tqdm import tqdm

    if vlm is None:
        vlm = VLMInference()

    output_path = output_path or RESULTS_DIR / "experiment4_belief_minimality.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for item in tqdm(recoverable_errors, desc="Belief Minimality"):
        if item["n_claims_contradicted"] == 0:
            continue

        img_path = image_dir / f"{item['image_id']}.jpg"
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        sg = scene_graphs[item["image_id"]]

        # Get all beliefs from belief externalization
        all_beliefs = item.get("beliefs", [])
        contradicted = item.get("contradicted_claims", [])

        if not all_beliefs or not contradicted:
            continue

        from bci.analysis.error_classification import answers_match

        # Try removing one contradicted belief at a time
        removal_results = []
        for i, bad_claim in enumerate(contradicted):
            bad_text = bad_claim.get("claim", bad_claim) if isinstance(bad_claim, dict) else bad_claim
            # Keep all beliefs except this one
            remaining = [b for b in all_beliefs if b != bad_text]

            if not remaining:
                continue

            corrected = vlm.constrained_reasoning(
                image, item["question"], remaining
            )
            corrected_answer = corrected.get("answer", "")
            flipped = answers_match(corrected_answer, item["ground_truth"])

            removal_results.append({
                "removed_belief": bad_text,
                "remaining_count": len(remaining),
                "corrected_answer": corrected_answer,
                "flipped_to_correct": flipped,
            })

        results.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "original_answer": item["baseline_answer"],
            "n_beliefs_total": len(all_beliefs),
            "n_contradicted": len(contradicted),
            "removal_results": removal_results,
        })

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


def run_random_ablation(
    error_cases: list[dict],
    image_dir: Path,
    vlm: VLMInference | None = None,
    n_trials: int = 3,
    output_path: Path | None = None,
) -> list[dict]:
    """
    Placebo control: Remove random beliefs instead of contradicted ones.
    If random removal helps as much as targeted removal,
    the verifier adds no value.
    """
    from tqdm import tqdm

    if vlm is None:
        vlm = VLMInference()

    random.seed(RANDOM_SEED)
    output_path = output_path or RESULTS_DIR / "experiment_random_ablation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from bci.analysis.error_classification import answers_match

    results = []

    for item in tqdm(error_cases, desc="Random Ablation"):
        all_beliefs = item.get("beliefs", [])
        n_contradicted = item.get("n_claims_contradicted", 1)

        if not all_beliefs or len(all_beliefs) <= 1:
            continue

        img_path = image_dir / f"{item['image_id']}.jpg"
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")

        # Remove same number of beliefs as contradicted, but randomly
        n_remove = min(n_contradicted, len(all_beliefs) - 1)
        trial_results = []

        for trial in range(n_trials):
            to_remove = random.sample(all_beliefs, n_remove)
            remaining = [b for b in all_beliefs if b not in to_remove]

            corrected = vlm.constrained_reasoning(
                image, item["question"], remaining
            )
            corrected_answer = corrected.get("answer", "")
            flipped = answers_match(corrected_answer, item["ground_truth"])

            trial_results.append({
                "trial": trial,
                "removed": to_remove,
                "flipped_to_correct": flipped,
            })

        results.append({
            "question_id": item["question_id"],
            "n_removed": n_remove,
            "n_trials": n_trials,
            "flip_rate": sum(1 for t in trial_results if t["flipped_to_correct"]) / n_trials,
            "trial_results": trial_results,
        })

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


def compute_claim_coverage(
    belief_claims: list[dict],
    scene_graph: dict,
    question: str,
) -> dict:
    """
    Experiment 3: Measure claim coverage.
    How many relevant scene graph facts are surfaced by the VLM?
    """
    gt_facts = scene_graph_to_facts(scene_graph)

    # Filter to question-relevant facts
    question_lower = question.lower()
    relevant_facts = []
    for fact in gt_facts:
        subj = fact["subject"].lower()
        obj = (fact.get("object") or "").lower()
        if (
            subj in question_lower
            or obj in question_lower
            or any(w in question_lower for w in subj.split())
        ):
            relevant_facts.append(fact)

    if not relevant_facts:
        return {"coverage": None, "relevant_facts": 0, "surfaced": 0}

    # Check how many relevant facts are covered by claims
    surfaced = 0
    for fact in relevant_facts:
        fact_text = f"{fact['subject']} {fact['predicate']} {fact.get('object', '')}"
        for claim in belief_claims:
            claim_text = claim.get("claim", "").lower()
            if (
                fact["subject"].lower() in claim_text
                or (fact.get("object", "") and fact["object"].lower() in claim_text)
            ):
                surfaced += 1
                break

    return {
        "coverage": surfaced / len(relevant_facts),
        "relevant_facts": len(relevant_facts),
        "surfaced": surfaced,
    }
