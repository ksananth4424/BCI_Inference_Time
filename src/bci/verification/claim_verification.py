"""
BCI — Claim Verification Module
Verifies extracted visual claims against GQA scene graphs.

Verification statuses:
    SUPPORTED: claim is consistent with scene graph
    CONTRADICTED: claim is inconsistent with scene graph
    UNCERTAIN: cannot determine from scene graph
"""
import re
from difflib import SequenceMatcher

from bci.config import ATTRIBUTE_SIMILARITY_THRESHOLD, SPATIAL_TOLERANCE
from bci.data.data_loader import scene_graph_to_facts

# ─── Status Constants ────────────────────────────────────────────────────────
SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
UNCERTAIN = "UNCERTAIN"


def verify_claim(claim: dict, scene_graph: dict) -> dict:
    """
    Verify a single claim against a scene graph.

    Args:
        claim: {"claim": str, "type": str}
        scene_graph: GQA scene graph dict

    Returns:
        claim dict augmented with "status" and "evidence"
    """
    claim_text = claim["claim"]
    claim_type = claim["type"]
    facts = scene_graph_to_facts(scene_graph)

    if claim_type == "object_existence":
        status, evidence = verify_object_existence(claim_text, facts)
    elif claim_type == "attribute":
        status, evidence = verify_attribute(claim_text, facts)
    elif claim_type == "spatial":
        status, evidence = verify_spatial(claim_text, facts, scene_graph)
    elif claim_type == "counting":
        status, evidence = verify_counting(claim_text, facts)
    elif claim_type == "action":
        status, evidence = verify_action(claim_text, facts)
    elif claim_type == "text_ocr":
        # Cannot verify OCR from scene graphs
        status, evidence = UNCERTAIN, "OCR claims cannot be verified via scene graphs"
    else:
        status, evidence = UNCERTAIN, f"Unknown claim type: {claim_type}"

    return {
        **claim,
        "status": status,
        "evidence": evidence,
    }


# ─── Object Existence ───────────────────────────────────────────────────────


def verify_object_existence(claim_text: str, facts: list[dict]) -> tuple[str, str]:
    """Check if a claimed object exists in the scene graph."""
    # Extract object name from claim
    obj_name = extract_object_name(claim_text)
    if not obj_name:
        return UNCERTAIN, "Could not extract object name from claim"

    # Check against all objects in scene graph
    object_facts = [f for f in facts if f["type"] == "object"]
    matches = fuzzy_match_objects(obj_name, object_facts)

    if matches:
        best = matches[0]
        return SUPPORTED, f"Object '{best['subject']}' found in scene graph"
    else:
        # Check if it's a negative claim
        if is_negative_claim(claim_text):
            return SUPPORTED, f"Negative claim; '{obj_name}' correctly absent"
        return CONTRADICTED, f"Object '{obj_name}' not found in scene graph"


def extract_object_name(claim: str) -> str:
    """Extract the main object noun from a claim."""
    lower = claim.lower()

    # Common patterns
    patterns = [
        r"there (?:is|are) (?:a |an |the |some |no )?([\w\s]+?)(?:\s+(?:in|on|at|near|next|behind|left|right|above|below)|\s*$|\s*\.)",
        r"(?:a|an|the|one|two|three|four|five) ([\w\s]+?)(?:\s+(?:is|are|in|on|at)|\s*$)",
        r"(?:see|shows?|see|visible|contains?) (?:a |an |the )?([\w\s]+?)(?:\s+(?:in|on|at)|\s*$|\s*\.)",
        r"^([\w\s]+?)(?:\s+(?:is|are|has|have))",
    ]

    for pat in patterns:
        m = re.search(pat, lower)
        if m:
            return m.group(1).strip()

    # Fallback: take main nouns (simple heuristic)
    words = lower.split()
    # Remove common non-object words
    stop = {"the", "a", "an", "is", "are", "there", "in", "on", "at", "i", "can", "see", "it", "this", "that"}
    nouns = [w for w in words if w not in stop and len(w) > 2]
    return " ".join(nouns[:2]) if nouns else ""


def is_negative_claim(claim: str) -> bool:
    """Check if the claim asserts absence."""
    lower = claim.lower()
    return any(kw in lower for kw in ["no ", "none", "not ", "isn't", "aren't", "doesn't", "don't", "without", "absent", "missing", "empty"])


def fuzzy_match_objects(
    target: str, object_facts: list[dict], threshold: float = 0.6
) -> list[dict]:
    """Find objects in scene graph that match the target name."""
    target_lower = target.lower().strip()
    matches = []

    for fact in object_facts:
        name = fact["subject"].lower().strip()
        # Exact match
        if target_lower == name or target_lower in name or name in target_lower:
            matches.append(fact)
            continue
        # Fuzzy match
        ratio = SequenceMatcher(None, target_lower, name).ratio()
        if ratio >= threshold:
            matches.append(fact)

    return matches


# ─── Attribute Verification ────────────────────────────────────────────────


def verify_attribute(claim_text: str, facts: list[dict]) -> tuple[str, str]:
    """Verify an attribute claim (e.g., 'The car is red')."""
    lower = claim_text.lower()

    # Extract subject and attribute
    subj, attr = extract_subject_attribute(lower)
    if not subj or not attr:
        return UNCERTAIN, "Could not parse subject/attribute from claim"

    # Find the object
    object_facts = [f for f in facts if f["type"] == "object"]
    obj_matches = fuzzy_match_objects(subj, object_facts)

    if not obj_matches:
        return UNCERTAIN, f"Object '{subj}' not found in scene graph"

    # Check attributes for matched objects
    attr_facts = [f for f in facts if f["type"] == "attribute"]

    for obj_match in obj_matches:
        obj_id = obj_match.get("obj_id")
        obj_attrs = [
            f for f in attr_facts if f.get("obj_id") == obj_id
        ]

        for af in obj_attrs:
            sg_attr = af["object"].lower()
            if attr_matches(attr, sg_attr):
                return SUPPORTED, f"'{subj}' has attribute '{sg_attr}' in scene graph"

        # Check if a contradicting attribute exists
        for af in obj_attrs:
            sg_attr = af["object"].lower()
            if attrs_are_contradictory(attr, sg_attr):
                return (
                    CONTRADICTED,
                    f"Claim says '{attr}' but scene graph says '{sg_attr}'",
                )

    return UNCERTAIN, f"Attribute '{attr}' not found in scene graph for '{subj}'"


def extract_subject_attribute(claim_lower: str) -> tuple[str, str]:
    """Extract subject and attribute from 'The X is Y' style claims."""
    patterns = [
        r"(?:the |a |an )?([\w\s]+?)\s+(?:is|are|looks?)\s+(\w+)",
        r"(\w+)\s+(?:is|are)\s+(\w+)",
        r"(\w+)\s+(\w+)\s+(?:colored|shaped|sized)",
    ]
    for pat in patterns:
        m = re.search(pat, claim_lower)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return "", ""


def attr_matches(claimed: str, actual: str) -> bool:
    """Check if claimed attribute matches actual."""
    if claimed == actual:
        return True
    ratio = SequenceMatcher(None, claimed, actual).ratio()
    return ratio >= ATTRIBUTE_SIMILARITY_THRESHOLD


def attrs_are_contradictory(claimed: str, actual: str) -> bool:
    """Check if two attributes are contradictory (e.g., red vs blue)."""
    color_groups = [
        {"red", "crimson", "scarlet", "maroon"},
        {"blue", "navy", "azure", "cyan"},
        {"green", "lime", "olive", "emerald"},
        {"yellow", "gold", "golden"},
        {"white", "ivory", "cream"},
        {"black", "dark", "ebony"},
        {"brown", "tan", "beige", "khaki"},
        {"orange", "amber"},
        {"pink", "magenta", "fuchsia", "rose"},
        {"purple", "violet", "lavender", "indigo"},
        {"gray", "grey", "silver"},
    ]
    size_groups = [
        {"large", "big", "huge", "giant", "tall"},
        {"small", "tiny", "little", "short", "miniature"},
    ]

    all_groups = color_groups + size_groups

    claimed_group = None
    actual_group = None
    for group in all_groups:
        if claimed in group:
            claimed_group = group
        if actual in group:
            actual_group = group

    if claimed_group and actual_group and claimed_group != actual_group:
        # Both are in specific groups but different ones
        # Check they're in the same category (both colors or both sizes)
        for category in [color_groups, size_groups]:
            if claimed_group in category and actual_group in category:
                return True

    return False


# ─── Spatial Verification ──────────────────────────────────────────────────


def verify_spatial(
    claim_text: str, facts: list[dict], scene_graph: dict
) -> tuple[str, str]:
    """Verify spatial relation claims using bounding box positions."""
    lower = claim_text.lower()

    subj, relation, obj = extract_spatial_relation(lower)
    if not subj or not relation or not obj:
        return UNCERTAIN, "Could not parse spatial relation from claim"

    objects = scene_graph.get("objects", {})
    object_facts = [f for f in facts if f["type"] == "object"]

    # Find subject and object
    subj_matches = fuzzy_match_objects(subj, object_facts)
    obj_matches = fuzzy_match_objects(obj, object_facts)

    if not subj_matches:
        return UNCERTAIN, f"Subject '{subj}' not found in scene graph"
    if not obj_matches:
        return UNCERTAIN, f"Object '{obj}' not found in scene graph"

    # Get bounding boxes
    subj_id = subj_matches[0].get("obj_id")
    obj_id = obj_matches[0].get("obj_id")

    subj_obj = objects.get(subj_id, {})
    obj_obj = objects.get(obj_id, {})

    subj_bbox = (subj_obj.get("x", 0), subj_obj.get("y", 0),
                 subj_obj.get("w", 0), subj_obj.get("h", 0))
    obj_bbox = (obj_obj.get("x", 0), obj_obj.get("y", 0),
                obj_obj.get("w", 0), obj_obj.get("h", 0))

    # Also check scene graph relations directly
    rel_facts = [f for f in facts if f["type"] == "relation"]
    for rf in rel_facts:
        if (
            rf.get("obj_id") == subj_id
            and rf.get("target_obj_id") == obj_id
        ):
            sg_rel = rf["predicate"].lower()
            if spatial_relation_matches(relation, sg_rel):
                return SUPPORTED, f"Relation '{relation}' confirmed by scene graph relation '{sg_rel}'"
            if spatial_relations_contradict(relation, sg_rel):
                return CONTRADICTED, f"Claim '{relation}' contradicts scene graph '{sg_rel}'"

    # Fallback: geometric check
    geo_result = check_spatial_geometry(relation, subj_bbox, obj_bbox)
    if geo_result is not None:
        status = SUPPORTED if geo_result else CONTRADICTED
        return status, f"Geometric check: '{subj}' {relation} '{obj}' is {geo_result}"

    return UNCERTAIN, "Could not verify spatial relation"


def extract_spatial_relation(claim_lower: str) -> tuple[str, str, str]:
    """Extract (subject, relation, object) from a spatial claim."""
    spatial_preps = [
        "to the left of", "to the right of", "left of", "right of",
        "above", "below", "on top of", "under", "beneath",
        "behind", "in front of", "next to", "beside", "near",
        "between", "on", "inside", "outside",
    ]
    for prep in spatial_preps:
        pattern = rf"(?:the |a |an )?([\w\s]+?)\s+(?:is |are )?{re.escape(prep)}\s+(?:the |a |an )?([\w\s]+?)(?:\s*$|\s*\.)"
        m = re.search(pattern, claim_lower)
        if m:
            return m.group(1).strip(), prep, m.group(2).strip()
    return "", "", ""


def spatial_relation_matches(claimed: str, actual: str) -> bool:
    """Check if two spatial relation descriptions match."""
    synonyms = {
        "left of": {"to the left of", "left of"},
        "right of": {"to the right of", "right of"},
        "above": {"above", "over", "on top of"},
        "below": {"below", "under", "beneath", "underneath"},
        "next to": {"next to", "beside", "near", "adjacent to"},
        "behind": {"behind", "in back of"},
        "in front of": {"in front of", "before"},
    }
    for _, group in synonyms.items():
        if claimed in group and actual in group:
            return True
    return claimed == actual


def spatial_relations_contradict(claimed: str, actual: str) -> bool:
    """Check if two spatial relations contradict each other."""
    opposites = [
        ({"left of", "to the left of"}, {"right of", "to the right of"}),
        ({"above", "over", "on top of"}, {"below", "under", "beneath"}),
        ({"in front of", "before"}, {"behind", "in back of"}),
    ]
    for group_a, group_b in opposites:
        if (claimed in group_a and actual in group_b) or (
            claimed in group_b and actual in group_a
        ):
            return True
    return False


def check_spatial_geometry(
    relation: str, subj_bbox: tuple, obj_bbox: tuple
) -> bool | None:
    """
    Check spatial relation using bounding box geometry.
    bbox format: (x, y, w, h) where (x,y) is top-left corner.
    Returns True/False or None if uncertain.
    """
    sx, sy, sw, sh = subj_bbox
    ox, oy, ow, oh = obj_bbox

    if sw == 0 or sh == 0 or ow == 0 or oh == 0:
        return None

    subj_cx = sx + sw / 2
    subj_cy = sy + sh / 2
    obj_cx = ox + ow / 2
    obj_cy = oy + oh / 2

    tolerance = SPATIAL_TOLERANCE * max(sw + ow, sh + oh)

    if relation in {"left of", "to the left of"}:
        diff = obj_cx - subj_cx
        if diff > tolerance:
            return True
        elif diff < -tolerance:
            return False
        return None

    if relation in {"right of", "to the right of"}:
        diff = subj_cx - obj_cx
        if diff > tolerance:
            return True
        elif diff < -tolerance:
            return False
        return None

    if relation in {"above", "over", "on top of"}:
        diff = obj_cy - subj_cy
        if diff > tolerance:
            return True
        elif diff < -tolerance:
            return False
        return None

    if relation in {"below", "under", "beneath", "underneath"}:
        diff = subj_cy - obj_cy
        if diff > tolerance:
            return True
        elif diff < -tolerance:
            return False
        return None

    return None


# ─── Counting Verification ─────────────────────────────────────────────────


def verify_counting(claim_text: str, facts: list[dict]) -> tuple[str, str]:
    """Verify counting claims against scene graph object counts."""
    lower = claim_text.lower()

    obj_name, claimed_count = extract_count_claim(lower)
    if not obj_name or claimed_count is None:
        return UNCERTAIN, "Could not parse counting claim"

    # Count matching objects in scene graph
    object_facts = [f for f in facts if f["type"] == "object"]
    matching = fuzzy_match_objects(obj_name, object_facts)
    actual_count = len(matching)

    if actual_count == claimed_count:
        return SUPPORTED, f"Count matches: {claimed_count} '{obj_name}' found"
    else:
        return (
            CONTRADICTED,
            f"Count mismatch: claimed {claimed_count}, found {actual_count} '{obj_name}'",
        )


def extract_count_claim(claim_lower: str) -> tuple[str, int | None]:
    """Extract (object_name, count) from counting claims."""
    word_to_num = {
        "zero": 0, "no": 0, "one": 1, "a": 1, "an": 1,
        "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }

    # "there are N objects"
    m = re.search(r"there (?:is|are) (\w+)\s+([\w\s]+?)(?:\s*$|\s*\.)", claim_lower)
    if m:
        count_word = m.group(1)
        obj = m.group(2).strip()
        count = word_to_num.get(count_word)
        if count is None:
            try:
                count = int(count_word)
            except ValueError:
                return "", None
        return obj, count

    # "N objects"
    m = re.search(r"(\d+)\s+([\w\s]+?)(?:\s*$|\s*\.)", claim_lower)
    if m:
        return m.group(2).strip(), int(m.group(1))

    return "", None


# ─── Action Verification ──────────────────────────────────────────────────


def verify_action(claim_text: str, facts: list[dict]) -> tuple[str, str]:
    """Verify action claims. Scene graphs have limited action info."""
    lower = claim_text.lower()

    # Check if the relation facts contain matching actions
    rel_facts = [f for f in facts if f["type"] == "relation"]

    for rf in rel_facts:
        pred = rf["predicate"].lower()
        if pred in lower or any(
            w in lower for w in pred.split()
        ):
            return SUPPORTED, f"Action '{pred}' found in scene graph relations"

    return UNCERTAIN, "Action claims have limited scene graph coverage"


# ─── Main Verification Pipeline ────────────────────────────────────────────


def verify_all_claims(
    claims: list[dict], scene_graph: dict
) -> list[dict]:
    """Verify all claims for a single example."""
    return [verify_claim(c, scene_graph) for c in claims]


def compute_verification_summary(verified_claims: list[dict]) -> dict:
    """Compute summary statistics for verified claims."""
    total = len(verified_claims)
    if total == 0:
        return {"total": 0, "supported": 0, "contradicted": 0, "uncertain": 0}

    supported = sum(1 for c in verified_claims if c["status"] == SUPPORTED)
    contradicted = sum(1 for c in verified_claims if c["status"] == CONTRADICTED)
    uncertain = sum(1 for c in verified_claims if c["status"] == UNCERTAIN)

    return {
        "total": total,
        "supported": supported,
        "contradicted": contradicted,
        "uncertain": uncertain,
        "supported_rate": supported / total,
        "contradicted_rate": contradicted / total,
        "uncertain_rate": uncertain / total,
    }
