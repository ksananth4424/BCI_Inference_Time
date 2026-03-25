from bci.verification.claim_extraction import classify_claim_type, extract_claims_from_text


def test_extract_claims_from_numbered_list():
    text = """
    1. The car is red
    2. The car is left of the tree
    3. There are 2 people
    """
    claims = extract_claims_from_text(text)
    assert len(claims) == 3


def test_claim_type_classification_spatial():
    assert classify_claim_type("The cup is left of the plate") == "spatial"


def test_claim_type_classification_counting():
    assert classify_claim_type("There are 3 apples on the table") == "counting"


def test_claim_type_classification_attribute():
    assert classify_claim_type("The car is red") == "attribute"
