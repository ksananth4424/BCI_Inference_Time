from bci.analysis.error_classification import answers_match, classify_error


def test_answers_match_yes_no_variants():
    assert answers_match("yes", "Yes")
    assert answers_match("no", "No")


def test_answers_match_number_word_equivalence():
    assert answers_match("three", "3")
    assert answers_match("10", "ten")


def test_classify_error_premise_error_when_contradicted_claim_exists():
    vlm_result = {
        "question_id": "q1",
        "question": "How many apples?",
        "ground_truth": "2",
        "baseline": {"answer": "3"},
        "belief_externalization": {"answer": "3"},
    }
    verified_claims = [
        {"claim": "There are 3 apples", "status": "CONTRADICTED"},
    ]
    out = classify_error(vlm_result, verified_claims)
    assert out["error_type"] == "premise_error"


def test_classify_error_reasoning_error_when_no_contradictions():
    vlm_result = {
        "question_id": "q2",
        "question": "Is the cup red?",
        "ground_truth": "yes",
        "baseline": {"answer": "no"},
        "belief_externalization": {"answer": "no"},
    }
    verified_claims = [
        {"claim": "The cup is red", "status": "SUPPORTED"},
        {"claim": "There is one cup", "status": "SUPPORTED"},
    ]
    out = classify_error(vlm_result, verified_claims)
    assert out["error_type"] == "reasoning_error"
