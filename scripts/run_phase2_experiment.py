"""
Unified Phase 2 Experiment Runner.

Orchestrates causal intervention experiments with versioned configs.
Supports multiple intervention policies and stratified analysis.

Usage:
    python scripts/run_phase2_experiment.py \\
        --exp E1 \\
        --config configs/phase2/e1_gqa_replace_all.yaml \\
        --output results/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml

# Add project root to path so bci_src can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bci_src.data.benchmarks.registry import get_benchmark_adapter
from bci_src.verification.profiles import get_profile
from bci_src.models.vlm_inference import VLMInference
from bci_src.verification.claim_extraction import extract_and_classify
from bci_src.verification.claim_verification import verify_claim
import bci_src.verification.claim_verification as claim_verification_module
from bci_src.analysis.error_classification import classify_error, answers_match
from bci_src.runtime.run_manifest import RunManifest
from bci_src.config import RANDOM_SEED, GQA_DIR
from bci_src.data.data_loader import scene_graph_to_facts
import pandas as pd
import numpy as np
from PIL import Image


# =============================================================================
# INTERVENTION POLICIES
# =============================================================================

class InterventionPolicy:
    """Base class for intervention policies."""
    
    def apply(
        self,
        verified_claims: List[Dict],
        scene_graph: Optional[Dict] = None,
    ) -> List[Dict]:
        """Transform verified claims based on policy."""
        raise NotImplementedError


class ReplaceAllPolicy(InterventionPolicy):
    """Replace ALL contradicted beliefs with ground truth (oracle)."""
    
    def apply(self, verified_claims, scene_graph=None):
        # Keep all non-contradicted claims
        corrected = [c for c in verified_claims if c["status"] != "CONTRADICTED"]
        
        # Extract facts from scene graph and add as ground-truth replacements
        if scene_graph:
            facts = scene_graph_to_facts(scene_graph)
            # Convert facts to natural language claims
            for fact in facts:
                claim_text = self._fact_to_claim(fact)
                if claim_text:
                    corrected.append({
                        "claim": claim_text,
                        "type": fact["type"],
                        "status": "SUPPORTED",
                        "confidence": 1.0,
                        "source": "ground_truth",  # Mark as GT injection
                    })
        
        return corrected
    
    @staticmethod
    def _fact_to_claim(fact: dict) -> str:
        """Convert a scene graph fact to a natural language claim."""
        fact_type = fact.get("type", "")
        subject = fact.get("subject", "")
        predicate = fact.get("predicate", "")
        obj = fact.get("object")
        
        if fact_type == "object":
            if predicate == "exists":
                return f"There is a {subject}."
            return f"A {subject} exists."
        
        elif fact_type == "attribute":
            if obj:
                return f"The {subject} is {obj}."
            return f"The {subject} has attribute {predicate}."
        
        elif fact_type == "relation":
            if obj:
                return f"The {subject} is {predicate} the {obj}."
            return f"The {subject} {predicate}."
        
        return ""


class ReplaceConfidentPolicy(InterventionPolicy):
    """Replace only high-confidence contradictions (practical version)."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.threshold = confidence_threshold
    
    def apply(self, verified_claims, scene_graph=None):
        corrected = []
        for c in verified_claims:
            if c["status"] == "CONTRADICTED" and c.get("confidence", 0) >= self.threshold:
                # Skip this claim (treat as corrected)
                continue
            corrected.append(c)
        return corrected


class RemoveOnlyPolicy(InterventionPolicy):
    """Remove contradicted claims; don't add GT (conservative)."""
    
    def apply(self, verified_claims, scene_graph=None):
        return [c for c in verified_claims if c["status"] != "CONTRADICTED"]


def get_policy(name: str) -> InterventionPolicy:
    """Factory for intervention policies."""
    policies = {
        "replace_contradicted_with_ground_truth": ReplaceAllPolicy,
        "replace_confident_contradictions": ReplaceConfidentPolicy,
        "remove_contradicted_only": RemoveOnlyPolicy,
    }
    return policies[name]()


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class Phase2ExpRunner:
    """Main orchestrator for Phase 2 experiments."""
    
    def __init__(
        self,
        exp_id: str,
        config_path: str,
        output_dir: Optional[Path] = None,
        device_override: Optional[str] = None,
        n_samples_override: Optional[int] = None,
    ):
        """
        Initialize experiment.
        
        Args:
            exp_id: Experiment ID (E1, E2, E3, etc.)
            config_path: Path to YAML config
            output_dir: Results directory
        """
        self.exp_id = exp_id
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir or "results")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load config
        with open(self.config_path) as f:
            self.cfg = yaml.safe_load(f)

            # Optional CLI overrides for fast/controlled execution.
            if device_override:
                self.cfg.setdefault("model", {})["device"] = device_override
            if n_samples_override is not None:
                self.cfg.setdefault("dataset", {})["n_samples"] = int(n_samples_override)
        
        self.manifest = RunManifest(
            self.cfg["experiment"]["id"],
            str(self.config_path),
            self.output_dir,
        )
        
        self.results: List[Dict[str, Any]] = []

    @staticmethod
    def _normalize_scene_graphs(scene_graphs: Any) -> Dict[str, Dict[str, Any]]:
        """Normalize scene graphs to an image-id keyed dictionary."""
        if isinstance(scene_graphs, dict):
            return scene_graphs
        if isinstance(scene_graphs, list):
            normalized: Dict[str, Dict[str, Any]] = {}
            for item in scene_graphs:
                if not isinstance(item, dict):
                    continue
                image_id = (
                    item.get("image_id")
                    or item.get("imageId")
                    or item.get("id")
                )
                if image_id is None:
                    continue
                normalized[str(image_id)] = item
            return normalized
        return {}
    
    def run(self) -> None:
        """Execute experiment end-to-end."""
        print(f"\n{'='*70}")
        print(f"PHASE 2 EXPERIMENT: {self.exp_id}")
        print(f"{'='*70}\n")
        
        # Load components
        print(f"[1/6] Loading benchmark adapter...")
        adapter = get_benchmark_adapter(self.cfg["dataset"]["name"])
        
        print(f"[2/6] Loading verifier profile...")
        profile = get_profile(self.cfg["pipeline"]["stages"][3]["verifier_profile"])

        # Wire profile thresholds into verifier runtime for this experiment run.
        claim_verification_module.ATTRIBUTE_SIMILARITY_THRESHOLD = (
            profile.ATTRIBUTE_SIMILARITY_THRESHOLD
        )
        claim_verification_module.SPATIAL_TOLERANCE = profile.SPATIAL_TOLERANCE
        
        print(f"[3/6] Initializing VLM...")
        vlm = VLMInference(device=self.cfg["model"].get("device", "cuda:1"))
        
        print(f"[4/6] Loading and preparing samples...")
        samples = adapter.load_samples(
            self.cfg["dataset"]["n_samples"],
            seed=self.cfg["dataset"]["random_seed"],
        )
        adapter.ensure_assets(samples)
        print(f"      ✓ Loaded {len(samples)} samples")
        
        print(f"[5/6] Running experiment pipeline...")
        
        scene_graphs = self._normalize_scene_graphs(adapter.load_scene_graphs())
        
        for i, sample in enumerate(samples):
            if (i + 1) % max(50, len(samples) // 10) == 0:
                print(f"      [{i+1:4d}/{len(samples)}] samples processed...")
            
            try:
                result = self._process_sample(
                    sample, vlm, profile, scene_graphs, adapter
                )
                self.results.append(result)
            except Exception as e:
                print(f"      Warning: Error on sample {sample.sample_id}: {e}")
                continue
        
        print(f"[6/6] Analyzing and saving results...\n")
        self._analyze_and_save()
    
    def _process_sample(
        self,
        sample: Any,
        vlm: VLMInference,
        profile: Any,
        scene_graphs: Dict,
        adapter: Any,
    ) -> Dict[str, Any]:
        """Process single sample through full pipeline."""
        
        # Load image
        image = self._load_image(sample, adapter)
        
        # Stage 1: Baseline inference
        baseline_out = vlm.baseline_inference(image, sample.question)
        
        # Stage 2: Belief externalization
        beliefs_out = vlm.belief_externalization(image, sample.question)
        
        # Stage 3: Claim extraction + verification
        extraction = extract_and_classify(
            {
                "question_id": sample.sample_id,
                "baseline": baseline_out,
                "belief_externalization": beliefs_out,
            }
        )
        claims = extraction.get("belief_claims", [])

        # GQA scene graphs are keyed by image_id, not question_id.
        scene_graph = scene_graphs.get(sample.image_id) or scene_graphs.get(sample.sample_id, {})
        verified_claims = []
        for claim in claims:
            verified = verify_claim(claim, scene_graph)
            verified_claims.append(
                {
                    **verified,
                    "confidence": 0.9 if verified.get("status") != "UNCERTAIN" else 0.5,
                }
            )
        
        # Stage 4: Intervention
        policy_name = self.cfg["pipeline"]["stages"][4]["policy"]
        policy = get_policy(policy_name)
        corrected_claims = policy.apply(verified_claims, scene_graph)
        
        # Stage 5: Constrained re-reasoning
        corrected_beliefs = [c.get("claim", "") for c in corrected_claims if c.get("claim")]
        corrected_out = vlm.constrained_reasoning(
            image, sample.question, corrected_beliefs
        )
        
        # Stage 6: Error classification
        baseline_correct = answers_match(baseline_out["answer"], sample.answer)
        corrected_correct = answers_match(corrected_out["answer"], sample.answer)
        
        flip_to_correct = (
            not baseline_correct and corrected_correct
        )
        
        return {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "ground_truth": sample.answer,
            "baseline_answer": baseline_out.get("answer"),
            "corrected_answer": corrected_out.get("answer"),
            "baseline_correct": baseline_correct,
            "corrected_correct": corrected_correct,
            "flip_to_correct": int(flip_to_correct),
            "n_claims": len(claims),
            "n_claims_supported": sum(1 for c in verified_claims if c["status"] == "SUPPORTED"),
            "n_claims_contradicted": sum(1 for c in verified_claims if c["status"] == "CONTRADICTED"),
            "n_claims_uncertain": sum(1 for c in verified_claims if c["status"] == "UNCERTAIN"),
        }
    
    def _load_image(self, sample: Any, adapter: Any) -> Image.Image:
        """Load image for sample."""
        try:
            # Prefer config-driven absolute data roots; keep relative fallbacks.
            possible_dirs = [
                GQA_DIR / "images",
                Path("data/gqa/images"),
                Path("data/mmmu/images"),
                Path("data/mathvista/images"),
            ]
            
            for img_dir in possible_dirs:
                img_path = img_dir / f"{sample.image_id}.jpg"
                if img_path.exists():
                    return Image.open(img_path).convert("RGB")
            
            # Fallback
            raise FileNotFoundError(f"Image not found: {sample.image_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {sample.image_id}: {e}")
    
    def _analyze_and_save(self) -> None:
        """Compute metrics and save results."""
        
        results_df = pd.DataFrame(self.results)

        if results_df.empty:
            summary = {
                "flip_to_correct_rate": 0.0,
                "n_samples": 0,
                "n_flipped": 0,
                "n_baseline_correct": 0,
                "n_corrected_correct": 0,
                "warning": "No valid samples processed. Check image availability and data paths.",
            }

            results_csv = self.output_dir / f"results_{self.cfg['experiment']['id']}.csv"
            summary_json = self.output_dir / f"summary_{self.cfg['experiment']['id']}.json"
            results_df.to_csv(results_csv, index=False)
            with open(summary_json, "w") as f:
                json.dump(summary, f, indent=2)

            manifest_path = self.manifest.finalize(summary)
            print(f"{'='*70}")
            print("RESULTS SUMMARY")
            print(f"{'='*70}")
            print("No valid samples were processed.")
            print(f"CSV:      {results_csv}")
            print(f"Summary:  {summary_json}")
            print(f"Manifest: {manifest_path}")
            print(f"{'='*70}\n")
            return
        
        # Summary metrics
        flip_rate = results_df["flip_to_correct"].mean()
        n_flipped = results_df["flip_to_correct"].sum()
        
        summary = {
            "flip_to_correct_rate": float(flip_rate),
            "n_samples": len(results_df),
            "n_flipped": int(n_flipped),
            "n_baseline_correct": int(results_df["baseline_correct"].sum()),
            "n_corrected_correct": int(results_df["corrected_correct"].sum()),
        }
        
        # Save results CSV
        results_csv = self.output_dir / f"results_{self.cfg['experiment']['id']}.csv"
        results_df.to_csv(results_csv, index=False)
        
        # Save summary JSON
        summary_json = self.output_dir / f"summary_{self.cfg['experiment']['id']}.json"
        with open(summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Finalize manifest
        manifest_path = self.manifest.finalize(summary)
        
        # Print summary
        print(f"{'='*70}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"Samples processed:        {len(results_df):6d}")
        print(f"Baseline correct:         {results_df['baseline_correct'].sum():6d} ({results_df['baseline_correct'].mean():.1%})")
        print(f"After correction:         {results_df['corrected_correct'].sum():6d} ({results_df['corrected_correct'].mean():.1%})")
        print(f"Flip-to-correct rate:     {flip_rate:6.1%} ({int(n_flipped)} samples)")
        print(f"\nResults saved to:")
        print(f"  CSV:      {results_csv}")
        print(f"  Summary:  {summary_json}")
        print(f"  Manifest: {manifest_path}")
        print(f"{'='*70}\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run Phase 2 experiments with versioned configs"
    )
    parser.add_argument(
        "--exp",
        required=True,
        help="Experiment ID (E1, E2, E3, etc.)",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="results",
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override (e.g., cuda:0)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Optional sample-count override",
    )
    
    args = parser.parse_args()
    
    try:
        runner = Phase2ExpRunner(
            args.exp,
            args.config,
            args.output_dir,
            device_override=args.device,
            n_samples_override=args.n_samples,
        )
        runner.run()
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
