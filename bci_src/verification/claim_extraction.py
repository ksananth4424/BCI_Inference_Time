"""
BCI — Claim Extraction Module
Extracts atomic visual claims from VLM responses using an LLM.
Also supports rule-based extraction from structured responses.
"""
import re


def extract_claims_from_text(text: str) -> list[str]:
    """
    Extract atomic visual claims from free-form VLM text.
    Uses rule-based heuristics for structured outputs.
    Falls back to sentence-level splitting for unstructured outputs.
    """
    claims = []

    # 1. Try numbered list extraction (1. claim, 2. claim, etc.)
    numbered = re.findall(r"^\s*\d+[\.\)]\s*(.+)$", text, re.MULTILINE)
    if numbered:
        return [c.strip() for c in numbered if len(c.strip()) > 5]

    # 2. Try bullet point extraction (- claim, * claim)
    bullets = re.findall(r"^\s*[-\*]\s+(.+)$", text, re.MULTILINE)
    if bullets:
        return [c.strip() for c in bullets if len(c.strip()) > 5]

    # 3. Sentence-level splitting from observation/description sections
    # Look for observational sentences
    sentences = re.split(r"[.!]\s+", text)
    for sent in sentences:
        sent = sent.strip().rstrip(".")
        if len(sent) < 10:
            continue
        # Filter for visual/perceptual statements
        if is_visual_claim(sent):
            claims.append(sent)

    return claims


def is_visual_claim(sentence: str) -> bool:
    """
    Heuristic: does this sentence describe a visual observation?
    Returns True if the sentence likely contains a visual claim.
    """
    lower = sentence.lower()

    # Visual indicators
    visual_keywords = [
        "there is", "there are", "i see", "i can see", "shows",
        "appears", "visible", "image", "picture", "photo",
        "left", "right", "above", "below", "next to", "behind",
        "in front", "on top", "under", "between", "near",
        "red", "blue", "green", "yellow", "white", "black",
        "brown", "orange", "pink", "purple", "gray", "grey",
        "large", "small", "tall", "short", "big", "tiny",
        "round", "square", "circular", "rectangular",
        "holding", "wearing", "sitting", "standing", "lying",
        "eating", "riding", "carrying", "looking",
        "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "several", "many", "few",
        "no ", "none", "empty",
    ]

    # Non-visual indicators (reasoning, not perception)
    reasoning_keywords = [
        "therefore", "because", "since", "so the answer",
        "this means", "we can conclude", "in conclusion",
        "the answer is", "the question asks",
    ]

    has_visual = any(kw in lower for kw in visual_keywords)
    has_reasoning = any(kw in lower for kw in reasoning_keywords)

    return has_visual and not has_reasoning


def extract_claims_from_beliefs(belief_list: list[str]) -> list[dict]:
    """
    Process already-extracted beliefs from the belief externalization prompt.
    Classifies each belief by type.
    """
    classified = []
    for belief in belief_list:
        claim_type = classify_claim_type(belief)
        classified.append({
            "claim": belief,
            "type": claim_type,
        })
    return classified


def classify_claim_type(claim: str) -> str:
    """
    Classify a visual claim into categories for verification routing.

    Categories:
        - object_existence: "There is a cat"
        - attribute: "The car is red"
        - spatial: "The cup is left of the plate"
        - counting: "There are 3 apples"
        - action: "The man is running"
        - text_ocr: "The sign reads STOP"
    """
    lower = claim.lower()

    # Counting patterns
    counting_patterns = [
        r"\b\d+\b.*\b(objects?|items?|things?|people|persons?)\b",
        r"\bthere are \d+\b",
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\b.*(objects?|items?)",
        r"\b\d+\s+\w+s?\b",  # "3 apples"
        r"\bhow many\b",
        r"\bnumber of\b",
        r"\bcount\b",
    ]
    if any(re.search(p, lower) for p in counting_patterns):
        return "counting"

    # Spatial patterns
    spatial_keywords = [
        "left of", "right of", "above", "below", "on top of",
        "under", "beneath", "behind", "in front of", "next to",
        "beside", "between", "near", "far from", "adjacent",
        "to the left", "to the right",
    ]
    if any(kw in lower for kw in spatial_keywords):
        return "spatial"

    # OCR / text patterns
    if any(kw in lower for kw in ["reads", "says", "text", "written", "sign", "label", "letter"]):
        return "text_ocr"

    # Action patterns
    action_keywords = [
        "running", "walking", "sitting", "standing", "eating",
        "drinking", "holding", "carrying", "riding", "playing",
        "looking", "wearing", "flying", "swimming", "jumping",
    ]
    if any(kw in lower for kw in action_keywords):
        return "action"

    # Attribute patterns (color, size, shape, material)
    attr_keywords = [
        "red", "blue", "green", "yellow", "white", "black",
        "brown", "orange", "pink", "purple", "gray", "grey",
        "large", "small", "big", "tiny", "tall", "short",
        "round", "square", "wooden", "metal", "glass", "plastic",
        "old", "new", "open", "closed", "empty", "full",
    ]
    if any(kw in lower for kw in attr_keywords):
        return "attribute"

    # Default: object existence
    return "object_existence"


def extract_and_classify(vlm_result: dict) -> dict:
    """
    Full extraction pipeline for a single VLM result.

    Args:
        vlm_result: dict with 'baseline' and 'belief_externalization' keys

    Returns:
        dict with extracted and classified claims for both modes
    """
    # From baseline: extract claims from observations
    baseline_obs = vlm_result.get("baseline", {}).get("observations", "")
    baseline_raw = vlm_result.get("baseline", {}).get("raw_response", "")
    text_to_extract = baseline_obs if baseline_obs else baseline_raw
    baseline_claims = extract_claims_from_text(text_to_extract)

    # From belief externalization: already structured
    beliefs = vlm_result.get("belief_externalization", {}).get("beliefs", [])
    belief_claims = extract_claims_from_beliefs(beliefs)

    # Also extract from baseline text and classify
    baseline_classified = [
        {"claim": c, "type": classify_claim_type(c)} for c in baseline_claims
    ]

    return {
        "question_id": vlm_result.get("question_id"),
        "baseline_claims": baseline_classified,
        "belief_claims": belief_claims,
        "num_baseline_claims": len(baseline_classified),
        "num_belief_claims": len(belief_claims),
    }
