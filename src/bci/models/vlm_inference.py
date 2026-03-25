"""
BCI — VLM Inference Module
Runs VLM on GQA questions to generate reasoning and answers.
Supports two modes:
  1. Free reasoning (baseline): generate answer with reasoning
  2. Belief externalization: generate atomic visual claims first

Supports:
  - Qwen2.5-VL-7B-Instruct (default, recommended)
  - LLaVA-1.5-7b-hf (legacy)
"""
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

from bci.config import (
    RESULTS_DIR,
    VLM_DEVICE,
    VLM_MAX_NEW_TOKENS,
    VLM_MODEL_ID,
    VLM_TEMPERATURE,
)

# ─── Prompt Templates ───────────────────────────────────────────────────────

BASELINE_PROMPT = """Look at this image and answer the following question.

Question: {question}

Think step by step. First describe what you observe in the image that is relevant to the question, then give your answer.

Format:
Observations: <what you see>
Reasoning: <your logic>
Answer: <your final answer>"""

BELIEF_EXTERNALIZATION_PROMPT = """Look at this image very carefully.

Before answering the question, you MUST list between 5 and 10 specific visual facts you are relying on. Each fact must be:
- Atomic: one single claim only (e.g. "The apple is red", NOT "The apple is red and round")
- Grounded: directly visible in the image (NOT an inference)
- Falsifiable: could potentially be wrong (specific, not vague)

BAD examples (too vague): "There are some objects", "I can see things"
GOOD examples: "The cup is to the left of the plate", "There are 3 apples on the table", "The car is red", "The man is standing next to the bicycle"

Question: {question}

Visual beliefs:
1. <specific visual fact>
2. <specific visual fact>
3. <specific visual fact>
4. <specific visual fact>
5. <specific visual fact>

Reasoning: <your logic based ONLY on the beliefs above>
Answer: <your final answer, one phrase or word>"""

CONSTRAINED_REASONING_PROMPT = """Given ONLY these verified visual facts:
{beliefs}

Answer the following question using ONLY the facts above. Do not assume anything not listed.

Question: {question}

Reasoning: <your logic based ONLY on the verified facts>
Answer: <your final answer, one phrase or word>"""


class VLMInference:
    """
    Wrapper for VLM inference.
    Supports Qwen2.5-VL-7B-Instruct and LLaVA-1.5.
    """

    def __init__(
        self,
        model_id: str = VLM_MODEL_ID,
        device: str = VLM_DEVICE,
    ):
        print(f"Loading model {model_id} on {device} ...")
        self.device = device
        self.model_id = model_id
        self._load_model(model_id, device)
        print("Model loaded.")

    def _load_model(self, model_id: str, device: str):
        """Load the appropriate model class."""
        if "qwen" in model_id.lower() or "Qwen" in model_id:
            from transformers import Qwen2_5_VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info

            self.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                device_map=device,
            )
            self.model_type = "qwen"
            self._process_vision_info = process_vision_info

        elif "llava" in model_id.lower() or "LLaVA" in model_id:
            from transformers import LlavaForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=device,
                low_cpu_mem_usage=True,
            )
            self.model_type = "llava"

        elif "internvl" in model_id.lower() or "InternVL" in model_id:
            from transformers import AutoModel, AutoTokenizer

            self.model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )
            self.model_type = "internvl"
        else:
            raise ValueError(f"Unsupported model: {model_id}")

        self.model.eval()

    @torch.no_grad()
    def generate(self, image: Image.Image, prompt: str) -> str:
        """Run inference on a single image + prompt."""
        if self.model_type == "qwen":
            return self._generate_qwen(image, prompt)
        elif self.model_type == "llava":
            return self._generate_llava(image, prompt)
        elif self.model_type == "internvl":
            return self._generate_internvl(image, prompt)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _generate_qwen(self, image: Image.Image, prompt: str) -> str:
        """Qwen2.5-VL inference."""
        import tempfile, os
        from qwen_vl_utils import process_vision_info

        # Save image to temp file (Qwen needs a path or URL)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name, "JPEG")
            tmp_path = tmp.name

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": tmp_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            gen_kwargs = {
                "max_new_tokens": VLM_MAX_NEW_TOKENS,
                "do_sample": VLM_TEMPERATURE > 0,
            }
            if VLM_TEMPERATURE > 0:
                gen_kwargs["temperature"] = VLM_TEMPERATURE

            output_ids = self.model.generate(**inputs, **gen_kwargs)
            generated_ids = [
                out[len(inp):]
                for inp, out in zip(inputs.input_ids, output_ids)
            ]
            return self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        finally:
            os.unlink(tmp_path)

    def _generate_llava(self, image: Image.Image, prompt: str) -> str:
        """LLaVA-1.5 inference."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.device, torch.float16)

        gen_kwargs = {
            "max_new_tokens": VLM_MAX_NEW_TOKENS,
            "do_sample": VLM_TEMPERATURE > 0,
        }
        if VLM_TEMPERATURE > 0:
            gen_kwargs["temperature"] = VLM_TEMPERATURE

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
        return self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

    def _generate_internvl(self, image: Image.Image, prompt: str) -> str:
        """InternVL3 inference."""
        import numpy as np
        from transformers import AutoTokenizer

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device, torch.bfloat16)

        gen_config = {"max_new_tokens": VLM_MAX_NEW_TOKENS, "do_sample": False}
        return self.model.chat(
            self.processor.tokenizer,
            pixel_values,
            f"<image>\n{prompt}",
            gen_config,
        )

    def baseline_inference(self, image: Image.Image, question: str) -> dict:
        """Standard VLM inference with reasoning."""
        prompt = BASELINE_PROMPT.format(question=question)
        response = self.generate(image, prompt)
        return parse_baseline_response(response)

    def belief_externalization(self, image: Image.Image, question: str) -> dict:
        """Generate atomic visual beliefs before reasoning."""
        prompt = BELIEF_EXTERNALIZATION_PROMPT.format(question=question)
        response = self.generate(image, prompt)
        return parse_belief_response(response)

    def constrained_reasoning(
        self, image: Image.Image, question: str, beliefs: list[str]
    ) -> dict:
        """Reason using only the provided verified beliefs."""
        belief_str = "\n".join(f"- {b}" for b in beliefs)
        prompt = CONSTRAINED_REASONING_PROMPT.format(
            beliefs=belief_str, question=question
        )
        response = self.generate(image, prompt)
        return parse_baseline_response(response)


# ─── Response Parsers ────────────────────────────────────────────────────────


def parse_baseline_response(response: str) -> dict:
    """Parse a baseline VLM response into components."""
    result = {"raw_response": response, "observations": "", "reasoning": "", "answer": ""}

    lines = response.split("\n")
    current_section = None

    for line in lines:
        line_stripped = line.strip()
        lower = line_stripped.lower()
        if lower.startswith("observations:") or lower.startswith("observation:"):
            current_section = "observations"
            result["observations"] = line_stripped.split(":", 1)[1].strip()
        elif lower.startswith("reasoning:"):
            current_section = "reasoning"
            result["reasoning"] = line_stripped.split(":", 1)[1].strip()
        elif lower.startswith("answer:"):
            current_section = "answer"
            result["answer"] = line_stripped.split(":", 1)[1].strip()
        elif current_section and line_stripped:
            result[current_section] += " " + line_stripped

    # Fallback: if no structured answer found, use last non-empty line
    if not result["answer"]:
        for line in reversed(lines):
            if line.strip():
                result["answer"] = line.strip()
                break

    return result


def parse_belief_response(response: str) -> dict:
    """Parse a belief externalization response."""
    result = {
        "raw_response": response,
        "beliefs": [],
        "reasoning": "",
        "answer": "",
    }

    lines = response.split("\n")
    current_section = None
    in_beliefs = False

    for line in lines:
        line_stripped = line.strip()
        lower = line_stripped.lower()

        if lower.startswith("visual belief") or lower.startswith("beliefs"):
            in_beliefs = True
            current_section = "beliefs"
            continue
        elif lower.startswith("reasoning:"):
            in_beliefs = False
            current_section = "reasoning"
            result["reasoning"] = line_stripped.split(":", 1)[1].strip()
            continue
        elif lower.startswith("answer:"):
            in_beliefs = False
            current_section = "answer"
            result["answer"] = line_stripped.split(":", 1)[1].strip()
            continue

        if in_beliefs and line_stripped:
            # Strip numbering like "1. ", "- ", "* "
            claim = line_stripped.lstrip("0123456789.-*) ").strip()
            if claim:
                result["beliefs"].append(claim)
        elif current_section == "reasoning" and line_stripped:
            result["reasoning"] += " " + line_stripped
        elif current_section == "answer" and line_stripped:
            result["answer"] += " " + line_stripped

    # Fallback for answer
    if not result["answer"]:
        for line in reversed(lines):
            if line.strip():
                result["answer"] = line.strip()
                break

    return result


def run_experiment_1(
    samples: list[dict],
    image_dir: Path,
    output_path: Path | None = None,
) -> list[dict]:
    """
    Run Experiment 1: Baseline + Belief Externalization on GQA samples.

    Args:
        samples: list of sample dicts from data_loader.sample_questions()
        image_dir: directory containing GQA images
        output_path: where to save results JSON

    Returns:
        list of result dicts
    """
    from tqdm import tqdm

    vlm = VLMInference()

    results = []
    output_path = output_path or RESULTS_DIR / "experiment1_vlm_outputs.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for sample in tqdm(samples, desc="VLM Inference"):
        img_path = image_dir / f"{sample['image_id']}.jpg"
        if not img_path.exists():
            print(f"  Image not found: {img_path}, skipping.")
            continue

        image = Image.open(img_path).convert("RGB")

        # Run baseline
        baseline = vlm.baseline_inference(image, sample["question"])

        # Run belief externalization
        belief = vlm.belief_externalization(image, sample["question"])

        result = {
            "question_id": sample["question_id"],
            "question": sample["question"],
            "ground_truth": sample["answer"],
            "full_answer": sample["full_answer"],
            "image_id": sample["image_id"],
            "question_type_structural": sample["question_type_structural"],
            "question_type_semantic": sample["question_type_semantic"],
            "baseline": baseline,
            "belief_externalization": belief,
        }
        results.append(result)

        # Save incrementally
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"Saved {len(results)} results to {output_path}")
    return results
