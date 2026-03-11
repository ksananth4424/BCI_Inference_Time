"""
BCI — GQA Dataset Loader (Hybrid)
Uses HuggingFace datasets for QA pairs + images (fast CDN),
and Stanford downloads for scene graphs (needed for verification).
"""
import json
import os
import random
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from config import (
    GQA_DIR,
    GQA_SCENE_GRAPHS_URL,
    HF_CACHE_DIR,
    NUM_SAMPLES,
    RANDOM_SEED,
)


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress bar."""
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dest.name} ...")
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size):
            f.write(chunk)
            bar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file."""
    print(f"  Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def download_scene_graphs() -> None:
    """Download GQA scene graphs from Stanford (needed for claim verification)."""
    GQA_DIR.mkdir(parents=True, exist_ok=True)
    sg_zip = GQA_DIR / "sceneGraphs.zip"
    if not (GQA_DIR / "val_sceneGraphs.json").exists():
        download_file(GQA_SCENE_GRAPHS_URL, sg_zip)
        extract_zip(sg_zip, GQA_DIR)
    print("Scene graphs ready.")


def load_scene_graphs(split: str = "val") -> dict:
    """Load GQA scene graphs."""
    sg_path = GQA_DIR / f"{split}_sceneGraphs.json"
    if not sg_path.exists():
        raise FileNotFoundError(
            f"Scene graphs not found at {sg_path}. Run download_gqa_metadata() first."
        )
    print(f"Loading scene graphs from {sg_path.name} ...")
    with open(sg_path) as f:
        return json.load(f)


def load_gqa_from_hf(
    split: str = "val",
    config: str = "val_balanced_instructions",
) -> list[dict]:
    """
    Load GQA questions + answers from HuggingFace.
    Returns list of dicts with QA info (no images yet).
    """
    from datasets import load_dataset

    print(f"Loading GQA from HuggingFace ({config}, {split}) ...")
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    ds = load_dataset(
        "lmms-lab/GQA",
        config,
        split=split,
        cache_dir=str(HF_CACHE_DIR),
    )
    print(f"  Total questions: {len(ds)}")

    all_questions = []
    for item in ds:
        q_type = item.get("types", {})
        if isinstance(q_type, str):
            try:
                q_type = json.loads(q_type)
            except (json.JSONDecodeError, TypeError):
                q_type = {}

        all_questions.append({
            "question_id": str(item["id"]),
            "question": item["question"],
            "answer": item["answer"],
            "image_id": str(item["imageId"]),
            "full_answer": item.get("fullAnswer", ""),
            "question_type_structural": q_type.get("structural", ""),
            "question_type_semantic": q_type.get("semantic", ""),
        })

    return all_questions


def load_gqa_images_from_hf(
    image_ids: set[str],
    image_dir: Path,
    split: str = "val",
    config: str = "val_balanced_images",
) -> int:
    """
    Download GQA images from HuggingFace for specific image IDs.
    Only downloads images we don't already have.
    Returns number of new images downloaded.
    """
    from datasets import load_dataset

    image_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in image_dir.glob("*.jpg")}
    needed = image_ids - existing
    if not needed:
        print(f"All {len(image_ids)} images already downloaded.")
        return 0

    print(f"Need {len(needed)} images ({len(existing)} already exist).")
    print(f"Loading GQA images from HuggingFace ({config}) ...")

    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    ds = load_dataset(
        "lmms-lab/GQA",
        config,
        split=split,
        cache_dir=str(HF_CACHE_DIR),
    )

    found = 0
    for item in tqdm(ds, desc="Saving images", total=len(ds)):
        img_id = str(item["id"])
        if img_id in needed:
            img = item["image"]
            img.save(image_dir / f"{img_id}.jpg")
            found += 1
            needed.discard(img_id)
            if not needed:
                break

    print(f"Downloaded {found} new images.")
    if needed:
        print(f"Warning: {len(needed)} images not found in HF dataset.")
    return found


def sample_questions(
    questions: list[dict],
    scene_graphs: dict,
    n: int = NUM_SAMPLES,
    seed: int = RANDOM_SEED,
    question_types: Optional[list[str]] = None,
) -> list[dict]:
    """
    Sample n questions that have matching scene graphs.
    Optionally filter by question type.
    """
    random.seed(seed)

    candidates = []
    for q in questions:
        img_id = q["image_id"]
        if img_id not in scene_graphs:
            continue

        if question_types:
            if (q["question_type_structural"] not in question_types
                    and q["question_type_semantic"] not in question_types):
                continue

        candidates.append({
            **q,
            "scene_graph": scene_graphs[img_id],
        })

    if len(candidates) < n:
        print(
            f"Warning: only {len(candidates)} candidates found, requested {n}. "
            "Using all candidates."
        )
        return candidates

    sampled = random.sample(candidates, n)
    print(f"Sampled {len(sampled)} questions (from {len(candidates)} candidates).")
    return sampled


def scene_graph_to_facts(sg: dict) -> list[dict]:
    """
    Convert a GQA scene graph into a list of structured facts.
    Used as ground truth for claim verification.

    Returns list of dicts:
        {"type": "object"|"attribute"|"relation",
         "subject": ..., "predicate": ..., "object": ...,
         "bbox": ...}
    """
    facts = []
    objects = sg.get("objects", {})

    for obj_id, obj in objects.items():
        obj_name = obj.get("name", "unknown")
        bbox = {
            "x": obj.get("x", 0),
            "y": obj.get("y", 0),
            "w": obj.get("w", 0),
            "h": obj.get("h", 0),
        }

        # Object existence
        facts.append(
            {
                "type": "object",
                "subject": obj_name,
                "predicate": "exists",
                "object": None,
                "bbox": bbox,
                "obj_id": obj_id,
            }
        )

        # Attributes
        for attr in obj.get("attributes", []):
            facts.append(
                {
                    "type": "attribute",
                    "subject": obj_name,
                    "predicate": "has_attribute",
                    "object": attr,
                    "bbox": bbox,
                    "obj_id": obj_id,
                }
            )

        # Relations
        for rel in obj.get("relations", []):
            target_id = rel.get("object", "")
            target_obj = objects.get(target_id, {})
            target_name = target_obj.get("name", "unknown")
            facts.append(
                {
                    "type": "relation",
                    "subject": obj_name,
                    "predicate": rel.get("name", ""),
                    "object": target_name,
                    "bbox": bbox,
                    "obj_id": obj_id,
                    "target_obj_id": target_id,
                }
            )

    return facts


def setup_gqa_data(n_samples: int = NUM_SAMPLES) -> tuple[list[dict], dict, Path]:
    """
    Full setup: load scene graphs, load questions from HF,
    sample, and download needed images.

    Returns:
        (sampled_questions, scene_graphs, image_dir)
    """
    sg = load_scene_graphs("val")
    questions = load_gqa_from_hf()
    samples = sample_questions(questions, sg, n=n_samples)

    image_dir = GQA_DIR / "images"
    needed_ids = {s["image_id"] for s in samples}
    load_gqa_images_from_hf(needed_ids, image_dir)

    return samples, sg, image_dir


if __name__ == "__main__":
    # Quick test
    sg = load_scene_graphs("val")
    questions = load_gqa_from_hf()
    samples = sample_questions(questions, sg, n=5)
    for s in samples:
        print(f"\nQ: {s['question']}")
        print(f"A: {s['answer']}")
        facts = scene_graph_to_facts(s["scene_graph"])
        print(f"Scene graph facts: {len(facts)}")
        for f in facts[:3]:
            print(f"  {f['type']}: {f['subject']} {f['predicate']} {f.get('object', '')}")
