from bci.verification.claim_verification import verify_claim, SUPPORTED, CONTRADICTED


def toy_scene_graph():
    return {
        "objects": {
            "1": {
                "name": "apple",
                "x": 10,
                "y": 10,
                "w": 20,
                "h": 20,
                "attributes": ["red"],
                "relations": [],
            },
            "2": {
                "name": "bowl",
                "x": 100,
                "y": 10,
                "w": 30,
                "h": 30,
                "attributes": ["white"],
                "relations": [],
            },
        }
    }


def test_verify_object_supported():
    claim = {"claim": "There is an apple", "type": "object_existence"}
    out = verify_claim(claim, toy_scene_graph())
    assert out["status"] == SUPPORTED


def test_verify_attribute_contradicted():
    claim = {"claim": "The apple is blue", "type": "attribute"}
    out = verify_claim(claim, toy_scene_graph())
    assert out["status"] in {CONTRADICTED, "UNCERTAIN"}
