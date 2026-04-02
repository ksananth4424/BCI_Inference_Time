"""Run matched competitor-style inference-time baselines on the same samples.

Methods:
- vanilla: direct baseline inference
- self_correct: ask model to re-check and revise its own answer
- cove_lite: draft -> generate verification checks -> revise
- resample_lite: stochastic multi-sample decoding + majority vote

This script is compute-focused and produces one CSV plus summary JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bci_src.analysis.error_classification import answers_match
from bci_src.config import GQA_DIR, RANDOM_SEED
from bci_src.data.benchmarks.registry import get_benchmark_adapter
from bci_src.models.vlm_inference import VLMInference, parse_baseline_response
import bci_src.models.vlm_inference as vlm_module


SELF_CORRECT_PROMPT = """You previously answered this visual question.

Question: {question}
Previous answer: {draft_answer}

Re-check the image carefully. If the previous answer is wrong, correct it.
Return only one concise final answer phrase.

Answer:"""


COVE_PLAN_PROMPT = """Given the image and question below, propose 3 short verification checks
that would validate whether a draft answer is visually grounded.

Question: {question}
Draft answer: {draft_answer}

Output exactly 3 bullet checks."""


COVE_REVISE_PROMPT = """Use these verification checks to revise the final answer.

Question: {question}
Draft answer: {draft_answer}
Verification checks:
{checks}

If the draft is unsupported, correct it. Return only a concise final answer phrase.

Answer:"""


def load_image(image_id: str) -> Image.Image:
    candidates = [
        GQA_DIR / "images" / f"{image_id}.jpg",
        Path("data/gqa/images") / f"{image_id}.jpg",
    ]
    for path in candidates:
        if path.exists():
            return Image.open(path).convert("RGB")
    raise FileNotFoundError(f"Image not found for image_id={image_id}")


def run_vanilla(vlm: VLMInference, image: Image.Image, question: str) -> str:
    return vlm.baseline_inference(image, question).get("answer", "")


def run_self_correct(vlm: VLMInference, image: Image.Image, question: str) -> str:
    draft = vlm.baseline_inference(image, question).get("answer", "")
    prompt = SELF_CORRECT_PROMPT.format(question=question, draft_answer=draft)
    revised = vlm.generate(image, prompt)
    parsed = parse_baseline_response(revised)
    return parsed.get("answer", "").strip() or revised.strip()


def run_cove_lite(vlm: VLMInference, image: Image.Image, question: str) -> str:
    draft = vlm.baseline_inference(image, question).get("answer", "")
    checks = vlm.generate(image, COVE_PLAN_PROMPT.format(question=question, draft_answer=draft))
    revised = vlm.generate(
        image,
        COVE_REVISE_PROMPT.format(question=question, draft_answer=draft, checks=checks),
    )
    parsed = parse_baseline_response(revised)
    return parsed.get("answer", "").strip() or revised.strip()


def run_resample_lite(
    vlm: VLMInference,
    image: Image.Image,
    question: str,
    k: int,
    temperature: float,
) -> str:
    old_temp = vlm_module.VLM_TEMPERATURE
    try:
        vlm_module.VLM_TEMPERATURE = float(temperature)
        answers: list[str] = []
        for _ in range(k):
            ans = vlm.baseline_inference(image, question).get("answer", "").strip()
            if ans:
                answers.append(ans)
        if not answers:
            return ""
        return Counter(answers).most_common(1)[0][0]
    finally:
        vlm_module.VLM_TEMPERATURE = old_temp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run competitor baseline methods")
    parser.add_argument("--benchmark", default="gqa")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--methods",
        default="vanilla,self_correct,cove_lite,resample_lite",
        help="Comma-separated methods",
    )
    parser.add_argument("--resample-k", type=int, default=3)
    parser.add_argument("--resample-temperature", type=float, default=0.7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/competitors"))
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    valid = {"vanilla", "self_correct", "cove_lite", "resample_lite"}
    unknown = [m for m in methods if m not in valid]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading adapter={args.benchmark}, n_samples={args.n_samples}")
    adapter = get_benchmark_adapter(args.benchmark)
    samples = adapter.load_samples(args.n_samples, seed=args.seed)
    adapter.ensure_assets(samples)
    print(f"Loaded {len(samples)} samples")

    print(f"Initializing VLM on {args.device}")
    vlm = VLMInference(device=args.device)

    rows: list[dict[str, Any]] = []
    total_steps = len(samples) * len(methods)
    step = 0

    for i, sample in enumerate(samples):
        if (i + 1) % max(50, len(samples) // 10) == 0:
            print(f"[{i+1:4d}/{len(samples)}] samples processed...")

        try:
            image = load_image(sample.image_id)
        except Exception as e:
            print(f"Warning: image load failed for {sample.sample_id}: {e}")
            continue

        for method in methods:
            step += 1
            try:
                if method == "vanilla":
                    pred = run_vanilla(vlm, image, sample.question)
                elif method == "self_correct":
                    pred = run_self_correct(vlm, image, sample.question)
                elif method == "cove_lite":
                    pred = run_cove_lite(vlm, image, sample.question)
                elif method == "resample_lite":
                    pred = run_resample_lite(
                        vlm,
                        image,
                        sample.question,
                        args.resample_k,
                        args.resample_temperature,
                    )
                else:
                    continue

                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "image_id": sample.image_id,
                        "question": sample.question,
                        "ground_truth": sample.answer,
                        "method": method,
                        "prediction": pred,
                        "correct": int(answers_match(pred, sample.answer)),
                    }
                )
            except Exception as e:
                print(f"Warning: {method} failed on sample {sample.sample_id}: {e}")

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / f"competitor_results_{args.benchmark}_{args.n_samples}.csv"
    df.to_csv(csv_path, index=False)

    summary: dict[str, Any] = {
        "benchmark": args.benchmark,
        "n_samples_requested": args.n_samples,
        "n_rows": len(df),
        "methods": methods,
        "metrics": {},
    }
    if not df.empty:
        for method, group in df.groupby("method"):
            summary["metrics"][method] = {
                "n": int(len(group)),
                "accuracy": float(group["correct"].mean()),
                "n_correct": int(group["correct"].sum()),
            }

    summary_path = args.output_dir / f"competitor_summary_{args.benchmark}_{args.n_samples}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("COMPETITOR RESULTS")
    print("=" * 72)
    for m in methods:
        metric = summary["metrics"].get(m)
        if metric:
            print(f"{m:15s}: {metric['accuracy']:.2%} ({metric['n_correct']}/{metric['n']})")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
