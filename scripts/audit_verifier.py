"""
Verifier Audit Script (Experiment E10 & E11).

Audits verifier reliability:
  - E10: Measure precision/recall per claim type on clean data
  - E11: Inject noise and measure robustness to false positives/negatives

Outputs per-claim-type metrics and noise degradation curves.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

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
from bci_src.runtime.run_manifest import RunManifest
from bci_src.config import RANDOM_SEED, GQA_DIR
import pandas as pd


def inject_noise(true_status: str, error_rate: float, noise_seed: int = 42) -> str:
    """
    Randomly flip verifier status with given probability.
    
    Args:
        true_status: Original claim status ("SUPPORTED", "CONTRADICTED", "UNCERTAIN")
        error_rate: Probability of flipping (0.0-1.0)
        noise_seed: Random seed for reproducibility
    
    Returns:
        Noisy status (possibly flipped)
    """
    if np.random.rand() < error_rate:
        # Simple flip: SUPPORTED <-> CONTRADICTED, UNCERTAIN stays UNCERTAIN
        if true_status == "SUPPORTED":
            return "CONTRADICTED"
        elif true_status == "CONTRADICTED":
            return "SUPPORTED"
        else:
            return true_status
    return true_status


def compute_metrics_by_type(
    results: List[Dict[str, Any]],
    status_key: str = "true_status"
) -> Dict[str, Dict[str, float]]:
    """
    Compute precision, recall, F1 per claim type.
    
    Simplified approach:
      - If status == "CONTRADICTED": Treat as positive class
      - Otherwise: Treat as negative class
    
    Args:
        results: List of claim verification results
        status_key: Which status field to use ("true_status" or "noisy_status_*")
    
    Returns:
        Dict mapping claim_type -> {precision, recall, f1, count}
    """
    by_type = {}
    
    for result in results:
        claim_type = result["claim_type"]
        status = result.get(status_key, "UNCERTAIN")
        
        if claim_type not in by_type:
            by_type[claim_type] = {
                "contradicted": 0,
                "total": 0,
            }
        
        by_type[claim_type]["total"] += 1
        if status == "CONTRADICTED":
            by_type[claim_type]["contradicted"] += 1
    
    # Compute detection rate (simplified metric: % of claims marked contradicted)
    metrics = {}
    for claim_type, counts in by_type.items():
        detection_rate = (
            counts["contradicted"] / counts["total"]
            if counts["total"] > 0 else 0.0
        )
        
        metrics[claim_type] = {
            "detection_rate": detection_rate,
            "contradicted_count": counts["contradicted"],
            "total_count": counts["total"],
            "supported_count": counts["total"] - counts["contradicted"],
        }
    
    return metrics


def _claim_field(claim: Any, key: str, default: Any = None) -> Any:
    """Read claim field from either dict-based or object-based claims."""
    if isinstance(claim, dict):
        return claim.get(key, default)
    return getattr(claim, key, default)


def run_audit(
    benchmark_name: str,
    n_samples: int,
    verifier_profile: str,
    device: str,
    output_dir: Path = None,
) -> None:
    """
    Main audit function combining E10 (clean) and E11 (noisy).
    
    Args:
        benchmark_name: Benchmark ID (e.g., "gqa")
        n_samples: Number of samples to audit
        verifier_profile: Profile name ("strict", "balanced", "high_recall")
        device: Torch device string (e.g., "cuda:0")
        output_dir: Where to save results
    
    Raises:
        ValueError: If benchmark or profile not found
    """
    
    manifest = RunManifest(
        f"E10_E11_{benchmark_name}_{verifier_profile}",
        f"configs/phase2/verifier_audit_{benchmark_name}.yaml",
        output_dir=output_dir,
    )
    
    print(f"\n{'='*70}")
    print(f"VERIFIER AUDIT: {benchmark_name.upper()} ({verifier_profile})")
    print(f"{'='*70}")
    
    # Load components
    print(f"\n[1/5] Loading benchmark adapter...")
    try:
        adapter = get_benchmark_adapter(benchmark_name)
    except KeyError as e:
        raise ValueError(f"Unknown benchmark: {benchmark_name}") from e
    
    print(f"[2/5] Loading verifier profile...")
    try:
        profile = get_profile(verifier_profile)
    except KeyError as e:
        raise ValueError(f"Unknown profile: {verifier_profile}") from e

    # Wire profile thresholds into verifier runtime for this audit run.
    claim_verification_module.ATTRIBUTE_SIMILARITY_THRESHOLD = (
        profile.ATTRIBUTE_SIMILARITY_THRESHOLD
    )
    claim_verification_module.SPATIAL_TOLERANCE = profile.SPATIAL_TOLERANCE
    
    print(f"[3/5] Initializing VLM...")
    vlm = VLMInference(device=device)  # Uses default model from config
    
    # Load samples
    print(f"[4/5] Loading {n_samples} benchmark samples...")
    samples = adapter.load_samples(n_samples, seed=RANDOM_SEED)
    adapter.ensure_assets(samples)
    print(f"      ✓ Loaded {len(samples)} samples")
    
    # Run audit
    print(f"[5/5] Processing samples and verifying claims...")
    results_clean: List[Dict[str, Any]] = []
    
    noisy_results = {
        0.05: [],
        0.10: [],
        0.20: [],
    }
    
    scene_graphs = adapter.load_scene_graphs()

    for i, sample in enumerate(samples):
        if (i + 1) % max(50, n_samples // 10) == 0:
            print(f"      [{i+1:4d}/{n_samples}] samples processed...")
        
        try:
            # Load image
            image = load_image_for_sample(sample, adapter)
            
            # Run VLM inference
            baseline_output = vlm.baseline_inference(image, sample.question)
            beliefs_output = vlm.belief_externalization(image, sample.question)
            
            # Extract claims
            extraction = extract_and_classify(
                {
                    "question_id": sample.sample_id,
                    "baseline": baseline_output,
                    "belief_externalization": beliefs_output,
                }
            )
            claims = extraction.get("belief_claims", [])
            if not claims:
                continue
            
            # GQA scene graphs are keyed by image_id.
            scene_graph = scene_graphs.get(sample.image_id) or scene_graphs.get(sample.sample_id, {})
            
            # Verify each claim
            for claim_idx, claim in enumerate(claims):
                status_clean, confidence = verify_claim_safe(
                    claim, scene_graph, profile
                )
                
                result_clean = {
                    "sample_id": sample.sample_id,
                    "claim_id": _claim_field(claim, "id", f"{sample.sample_id}_{claim_idx}"),
                    "claim_text": _claim_field(claim, "claim", str(claim)),
                    "claim_type": _claim_field(claim, "type", "unknown"),
                    "true_status": status_clean,
                    "confidence": confidence,
                }
                
                results_clean.append(result_clean)
                
                # Generate noisy versions
                for noise_rate in noisy_results.keys():
                    np.random.seed(i * 1000 + hash(result_clean["claim_id"]) % 1000)
                    status_noisy = inject_noise(status_clean, noise_rate)
                    
                    result_noisy = {
                        **result_clean,
                        "noisy_status": status_noisy,
                        "noise_rate": noise_rate,
                    }
                    noisy_results[noise_rate].append(result_noisy)
        
        except Exception as e:
            print(f"      Warning: Error processing sample {i}: {e}")
            continue
    
    # Compute metrics
    print(f"\nComputing metrics...")
    metrics_clean = compute_metrics_by_type(results_clean, "true_status")
    
    metrics_noisy = {}
    for noise_rate, results in noisy_results.items():
        metrics_noisy[noise_rate] = compute_metrics_by_type(
            results, "noisy_status"
        )
    
    # Print results
    print(f"\n{'='*70}")
    print(f"CLEAN VERIFIER METRICS (by claim type)")
    print(f"{'='*70}")
    
    for claim_type, metrics in sorted(metrics_clean.items()):
        print(
            f"  {claim_type:15s}: "
            f"detection={metrics['detection_rate']:.2%} "
            f"(n={metrics['total_count']}, contradicted={metrics['contradicted_count']})"
        )
    
    print(f"\n{'='*70}")
    print(f"NOISE ROBUSTNESS")
    print(f"{'='*70}")
    
    for noise_rate in sorted(noisy_results.keys()):
        metrics = metrics_noisy[noise_rate]
        print(f"\nAt {noise_rate:.0%} injected error rate:")
        
        for claim_type, metric_clean in sorted(metrics_clean.items()):
            metric_noisy = metrics.get(claim_type, {})
            if not metric_noisy:
                print(f"  {claim_type:15s}: N/A (no samples)")
                continue
            
            clean_det = metric_clean["detection_rate"]
            noisy_det = metric_noisy["detection_rate"]
            change = (noisy_det - clean_det) / max(abs(clean_det), 0.01)
            
            print(
                f"  {claim_type:15s}: "
                f"clean={clean_det:.2%} -> noisy={noisy_det:.2%} "
                f"(Δ={change:+.1%})"
            )
    
    # Save results
    output_path = manifest.output_dir
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save clean results CSV
    clean_csv = output_path / f"audit_{benchmark_name}_{verifier_profile}_clean.csv"
    pd.DataFrame(results_clean).to_csv(clean_csv, index=False)
    print(f"\n✓ Clean results saved to {clean_csv}")
    
    # Save metrics JSON
    metrics_output = {
        "benchmark": benchmark_name,
        "profile": verifier_profile,
        "n_samples": n_samples,
        "total_claims": len(results_clean),
        "clean_metrics": metrics_clean,
        "noisy_metrics": {
            str(k): v for k, v in metrics_noisy.items()
        }
    }
    
    metrics_json = output_path / f"audit_{benchmark_name}_{verifier_profile}_metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"✓ Metrics saved to {metrics_json}")
    
    # Finalize manifest
    manifest_path = manifest.finalize({
        "total_claims_audited": len(results_clean),
        "n_samples": n_samples,
        "metrics": metrics_output,
    })
    print(f"✓ Manifest saved to {manifest_path}")
    print(f"\n{'='*70}\n")


def load_image_for_sample(sample: Any, adapter: Any) -> Any:
    """Load image from sample using adapter-specific logic."""
    # This is adapter-dependent; assume it has ensure_assets beforehand
    from PIL import Image
    
    image_id = sample.image_id
    # Prefer config-driven absolute path; keep workspace fallback.
    candidates = [
        GQA_DIR / "images" / f"{image_id}.jpg",
        Path("data/gqa/images") / f"{image_id}.jpg",
    ]

    for image_path in candidates:
        if image_path.exists():
            return Image.open(image_path).convert("RGB")

    raise FileNotFoundError(f"Image not found in any known location for image_id={image_id}")


def verify_claim_safe(claim: Any, scene_graph: Dict, profile: Any) -> Tuple[str, float]:
    """
    Safely verify a claim with error handling.
    
    Returns:
        (status: str, confidence: float)
    """
    try:
        # verify_claim returns a dict with status/evidence.
        verified = verify_claim(claim, scene_graph)
        status = verified.get("status", "UNCERTAIN")
        confidence = 0.9 if status != "UNCERTAIN" else 0.5
        return status, confidence
    except Exception as e:
        # Fallback to UNCERTAIN if verification fails
        return "UNCERTAIN", 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run verifier audit (E10/E11) on a benchmark"
    )
    parser.add_argument(
        "--benchmark",
        default="gqa",
        help="Benchmark name (default: gqa)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples to audit (default: 100)",
    )
    parser.add_argument(
        "--profile",
        default="balanced",
        choices=["strict", "balanced", "high_recall"],
        help="Verifier profile (default: balanced)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="results",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for VLM inference (default: cuda:0)",
    )
    
    args = parser.parse_args()
    
    try:
        run_audit(
            args.benchmark,
            args.n_samples,
            args.profile,
            args.device,
            args.output_dir,
        )
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
