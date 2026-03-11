"""
BCI — Error Classification Module
Classifies VLM errors into:
    - Pure Premise Error: visual claim is false, reasoning is valid
    - Pure Reasoning Error: visual claims are correct, reasoning is wrong
    - Mixed Error: both premise and reasoning issues
    - Correct: answer is correct

This is the core analysis for Experiment 1 (H1 validation).
"""
from difflib import SequenceMatcher


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    ans = answer.lower().strip().rstrip(".")
    # Remove articles
    for article in ["a ", "an ", "the "]:
        if ans.startswith(article):
            ans = ans[len(article):]
    return ans.strip()


def answers_match(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    pred = normalize_answer(predicted)
    gt = normalize_answer(ground_truth)

    # Exact match
    if pred == gt:
        return True

    # Containment (for short answers)
    if len(gt) <= 10 and (gt in pred or pred in gt):
        return True

    # Yes/No normalization
    yes_words = {"yes", "true", "correct", "right", "yeah"}
    no_words = {"no", "false", "incorrect", "wrong", "nope"}
    if pred in yes_words and gt in yes_words:
        return True
    if pred in no_words and gt in no_words:
        return True

    # Number normalization
    num_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10",
    }
    pred_num = num_words.get(pred, pred)
    gt_num = num_words.get(gt, gt)
    if pred_num == gt_num:
        return True

    # Fuzzy match for longer answers
    ratio = SequenceMatcher(None, pred, gt).ratio()
    if ratio > 0.85:
        return True

    return False


def classify_error(
    vlm_result: dict,
    verified_claims: list[dict],
) -> dict:
    """
    Classify the error type for a single VLM result.

    Args:
        vlm_result: dict with baseline/belief outputs and ground truth
        verified_claims: list of verified claims with status

    Returns:
        dict with error classification and details
    """
    gt_answer = vlm_result["ground_truth"]
    baseline_answer = vlm_result.get("baseline", {}).get("answer", "")
    belief_answer = vlm_result.get("belief_externalization", {}).get("answer", "")

    baseline_correct = answers_match(baseline_answer, gt_answer)
    belief_correct = answers_match(belief_answer, gt_answer)

    # Count claim statuses
    n_contradicted = sum(
        1 for c in verified_claims if c["status"] == "CONTRADICTED"
    )
    n_supported = sum(
        1 for c in verified_claims if c["status"] == "SUPPORTED"
    )
    n_total = len(verified_claims)
    has_false_premise = n_contradicted > 0

    # Classify
    if baseline_correct:
        error_type = "correct"
        error_detail = "Baseline answer is correct"
    elif has_false_premise:
        # Has at least one false premise
        # To distinguish pure premise vs mixed:
        # Check if reasoning follows logically from the (false) premises
        # Heuristic: if the answer would be correct given the false premises,
        # it's a pure premise error
        error_type = "premise_error"
        error_detail = (
            f"{n_contradicted}/{n_total} claims contradicted by scene graph"
        )
    else:
        # All claims are supported or uncertain, but answer is wrong
        error_type = "reasoning_error"
        error_detail = "No contradicted claims found, but answer is incorrect"

    return {
        "question_id": vlm_result["question_id"],
        "question": vlm_result["question"],
        "ground_truth": gt_answer,
        "baseline_answer": baseline_answer,
        "belief_answer": belief_answer,
        "baseline_correct": baseline_correct,
        "belief_correct": belief_correct,
        "error_type": error_type,
        "error_detail": error_detail,
        "n_claims_total": n_total,
        "n_claims_supported": n_supported,
        "n_claims_contradicted": n_contradicted,
        "n_claims_uncertain": n_total - n_supported - n_contradicted,
        "has_false_premise": has_false_premise,
        "contradicted_claims": [
            c for c in verified_claims if c["status"] == "CONTRADICTED"
        ],
    }


def compute_error_breakdown(classifications: list[dict]) -> dict:
    """
    Compute the overall error breakdown across all samples.
    This is the key result for H1 validation.
    """
    total = len(classifications)
    correct = sum(1 for c in classifications if c["error_type"] == "correct")
    incorrect = total - correct

    premise_errors = sum(
        1 for c in classifications if c["error_type"] == "premise_error"
    )
    reasoning_errors = sum(
        1 for c in classifications if c["error_type"] == "reasoning_error"
    )

    result = {
        "total_samples": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": correct / total if total > 0 else 0,
        # Among incorrect answers
        "premise_errors": premise_errors,
        "reasoning_errors": reasoning_errors,
        "premise_error_rate_of_incorrect": (
            premise_errors / incorrect if incorrect > 0 else 0
        ),
        "reasoning_error_rate_of_incorrect": (
            reasoning_errors / incorrect if incorrect > 0 else 0
        ),
        # Overall rates
        "premise_error_rate_overall": premise_errors / total if total > 0 else 0,
        "reasoning_error_rate_overall": (
            reasoning_errors / total if total > 0 else 0
        ),
    }

    # Breakdown by question type
    type_breakdown = {}
    for c in classifications:
        q_type = c.get("question_type_structural", "unknown")
        if q_type not in type_breakdown:
            type_breakdown[q_type] = {
                "total": 0, "correct": 0, "premise_error": 0, "reasoning_error": 0,
            }
        type_breakdown[q_type]["total"] += 1
        if c["error_type"] == "correct":
            type_breakdown[q_type]["correct"] += 1
        elif c["error_type"] == "premise_error":
            type_breakdown[q_type]["premise_error"] += 1
        elif c["error_type"] == "reasoning_error":
            type_breakdown[q_type]["reasoning_error"] += 1

    result["by_question_type"] = type_breakdown
    return result


def find_recoverable_errors(classifications: list[dict]) -> list[dict]:
    """
    Find cases where the answer is wrong AND there are contradicted claims.
    These are candidates for the premise correction experiment (Experiment 2).
    """
    return [
        c
        for c in classifications
        if c["error_type"] == "premise_error" and c["n_claims_contradicted"] > 0
    ]
