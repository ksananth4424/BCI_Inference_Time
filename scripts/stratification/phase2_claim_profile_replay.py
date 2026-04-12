"""Phase 2 for Task 4: Claim-type profile replay.

Consumes phase1 sample metadata and reconstructs per-sample claim-type signals via:
  belief externalization -> claim extraction -> claim verification.

Primary output:
  results/priorityA_task4_error_stratification/phase2_claim_profile.csv

This is the alignment input for phase 3 claim-type stratification.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bci_src.config import GQA_DIR
from bci_src.data.benchmarks.registry import get_benchmark_adapter
from bci_src.models.vlm_inference import VLMInference
from bci_src.verification.claim_extraction import extract_and_classify
from bci_src.verification.claim_verification import verify_claim

CLAIM_TYPES = [
    "object_existence",
    "attribute",
    "spatial",
    "counting",
    "action",
    "text_ocr",
]

STATUSES = ["SUPPORTED", "CONTRADICTED", "UNCERTAIN"]


def normalize_claim_type(value: Any) -> str:
    t = str(value or "object_existence").strip().lower()
    aliases = {
        "object": "object_existence",
        "existence": "object_existence",
        "ocr": "text_ocr",
        "text": "text_ocr",
        "relation": "spatial",
    }
    t = aliases.get(t, t)
    if t not in CLAIM_TYPES:
        return "object_existence"
    return t


def normalize_status(value: Any) -> str:
    s = str(value or "UNCERTAIN").strip().upper()
    if s not in STATUSES:
        return "UNCERTAIN"
    return s


def load_image(image_id: str) -> Image.Image:
    candidates = [
        GQA_DIR / "images" / f"{image_id}.jpg",
        Path("data/gqa/images") / f"{image_id}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(f"Image not found for image_id={image_id}")


def dominant_type_from_counts(counts: dict[str, int]) -> str:
    best_type = "none"
    best_value = 0
    for t in CLAIM_TYPES:
        v = int(counts.get(t, 0))
        if v > best_value:
            best_type = t
            best_value = v
    return best_type if best_value > 0 else "none"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Task4 Phase2: claim profile replay")
    parser.add_argument("--results-root", type=Path, default=repo_root / "results")
    parser.add_argument("--output-subdir", type=str, default="priorityA_task4_error_stratification")
    parser.add_argument(
        "--phase1-metadata",
        type=Path,
        default=None,
        help="Path to phase1_sample_metadata.csv (defaults to task output folder)",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample cap for dry-runs")
    args = parser.parse_args()

    out_dir = args.results_root.resolve() / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    phase1_metadata = args.phase1_metadata.resolve() if args.phase1_metadata else (out_dir / "phase1_sample_metadata.csv")
    if not phase1_metadata.exists():
        raise FileNotFoundError(
            f"Phase1 metadata not found: {phase1_metadata}. Run phase1_question_type_stratification.py first."
        )

    meta_df = pd.read_csv(phase1_metadata)
    required = {"sample_id", "image_id", "question", "ground_truth", "question_type_structural", "question_type_semantic"}
    missing = required - set(meta_df.columns)
    if missing:
        raise ValueError(f"Phase1 metadata missing columns: {sorted(missing)}")

    if args.limit is not None:
        meta_df = meta_df.head(int(args.limit)).copy()

    # Scene graph lookup
    adapter = get_benchmark_adapter("gqa")
    sg_raw = adapter.load_scene_graphs()
    scene_graphs = sg_raw if isinstance(sg_raw, dict) else {}

    vlm = VLMInference(device=args.device)

    rows: list[dict[str, Any]] = []
    n_total = len(meta_df)
    checkpoint = max(20, n_total // 10) if n_total else 20

    for i, row in meta_df.iterrows():
        sample_id = str(row["sample_id"])
        image_id = str(row["image_id"])
        question = str(row["question"])

        if (i + 1) % checkpoint == 0:
            print(f"[{i+1:4d}/{n_total}] replayed")

        try:
            image = load_image(image_id)
            beliefs_out = vlm.belief_externalization(image, question)

            extraction = extract_and_classify(
                {
                    "question_id": sample_id,
                    "baseline": {},
                    "belief_externalization": beliefs_out,
                }
            )
            claims = extraction.get("belief_claims", [])

            sg = scene_graphs.get(image_id) or scene_graphs.get(sample_id, {})

            status_counts_total = defaultdict(int)
            contradicted_counts = defaultdict(int)
            supported_counts = defaultdict(int)
            uncertain_counts = defaultdict(int)

            for c in claims:
                ctype = normalize_claim_type(c.get("type"))
                verified = verify_claim(c, sg)
                status = normalize_status(verified.get("status"))

                status_counts_total[ctype] += 1
                if status == "CONTRADICTED":
                    contradicted_counts[ctype] += 1
                elif status == "SUPPORTED":
                    supported_counts[ctype] += 1
                else:
                    uncertain_counts[ctype] += 1

            out = {
                "sample_id": sample_id,
                "image_id": image_id,
                "question": question,
                "ground_truth": row["ground_truth"],
                "question_type_structural": row["question_type_structural"],
                "question_type_semantic": row["question_type_semantic"],
                "n_claims_total": int(len(claims)),
            }

            for t in CLAIM_TYPES:
                out[f"claims_total_{t}"] = int(status_counts_total.get(t, 0))
                out[f"claims_supported_{t}"] = int(supported_counts.get(t, 0))
                out[f"claims_contradicted_{t}"] = int(contradicted_counts.get(t, 0))
                out[f"claims_uncertain_{t}"] = int(uncertain_counts.get(t, 0))

            out["dominant_contradicted_type"] = dominant_type_from_counts(contradicted_counts)
            out["has_any_contradiction"] = int(sum(contradicted_counts.values()) > 0)
            rows.append(out)

        except Exception as e:
            err = {
                "sample_id": sample_id,
                "image_id": image_id,
                "question": question,
                "ground_truth": row["ground_truth"],
                "question_type_structural": row["question_type_structural"],
                "question_type_semantic": row["question_type_semantic"],
                "n_claims_total": 0,
                "dominant_contradicted_type": "error",
                "has_any_contradiction": 0,
                "error": str(e),
            }
            for t in CLAIM_TYPES:
                err[f"claims_total_{t}"] = 0
                err[f"claims_supported_{t}"] = 0
                err[f"claims_contradicted_{t}"] = 0
                err[f"claims_uncertain_{t}"] = 0
            rows.append(err)

    out_df = pd.DataFrame(rows)
    out_csv = out_dir / "phase2_claim_profile.csv"
    out_df.to_csv(out_csv, index=False)

    # Lightweight totals table
    totals_rows = []
    for t in CLAIM_TYPES:
        totals_rows.append(
            {
                "claim_type": t,
                "total_claims": int(out_df[f"claims_total_{t}"].sum()),
                "supported": int(out_df[f"claims_supported_{t}"].sum()),
                "contradicted": int(out_df[f"claims_contradicted_{t}"].sum()),
                "uncertain": int(out_df[f"claims_uncertain_{t}"].sum()),
            }
        )
    totals_df = pd.DataFrame(totals_rows)
    totals_csv = out_dir / "phase2_claim_type_totals.csv"
    totals_df.to_csv(totals_csv, index=False)

    with open(out_dir / "phase2_claim_profile.json", "w") as f:
        json.dump(out_df.to_dict(orient="records"), f, indent=2)

    print("=" * 80)
    print("TASK4 PHASE2 COMPLETE")
    print("=" * 80)
    print(f"Claim profile CSV      : {out_csv}")
    print(f"Claim type totals CSV  : {totals_csv}")
    print(f"Rows processed         : {len(out_df)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
