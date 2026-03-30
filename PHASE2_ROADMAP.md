# Phase 2 Implementation Roadmap

**Version**: 1.0  
**Target**: Week-by-week execution guide for this roadmap document  
**Status**: Pre-Execution

## **Fast-Track Mode (Compute-Only Bottleneck)**

### **Objective**
Remove all human orchestration overhead and make GPU time the only limiter.

### **One-Command Parallel Launch**

```bash
cd /data/cs22btech11029/playground/Inference_time_VLM
conda activate mech_interp
python scripts/launch_fast_track.py --gpus 0,1,2 --samples 500 --profile balanced
```

This launches in parallel:
- **E1** on GPU 0
- **E3** on GPU 1
- **E10** on GPU 2

Artifacts are grouped under:
- `results/fasttrack_<timestamp>/launch_manifest.json`
- `results/fasttrack_<timestamp>/logs/E1.log`
- `results/fasttrack_<timestamp>/logs/E3.log`
- `results/fasttrack_<timestamp>/logs/E10.log`

### **Run Monitoring**

```bash
tail -f results/fasttrack_*/logs/E1.log
tail -f results/fasttrack_*/logs/E3.log
tail -f results/fasttrack_*/logs/E10.log
```

### **Immediate Decision Gates (Same Day)**

1. **Causal gate**: E1 targeted correction strongly beats E3/random baseline
2. **Reliability gate**: E10 claim-type metrics stable under 5% noise
3. **Go/No-Go**:
    - If both pass: start cross-benchmark adapters next (MMMU, MathVista)
    - If not: tune verifier profile and relaunch only failed arm

---

## **Week 1: Foundation & Validation (Phase 2a)**

### **Objective**
Reproduce Phase 1 causal signal on GQA with audit-level verifier transparency. Establish ground truth for all downstream experiments.

### **Experiments to Run**
- **E1**: Replace all contradicted beliefs → 60% flip rate (expected)
- **E3**: Remove single contradicted belief → 28% flip rate (expected)
- **E10**: Verifier precision/recall per claim type audit
- **E11**: Noise robustness: inject false-pos and false-neg at 5%, 10%, 20%

### **Required Code/Configs**

#### 1. Verifier Profile System

**File**: `bci_src/verification/profiles.py` (NEW)

```python
# Define versioned verifier profiles: strict, balanced, high_recall
# Each profile is a frozen dataclass with:
#   - ATTRIBUTE_SIMILARITY_THRESHOLD
#   - SPATIAL_TOLERANCE
#   - OBJECT_PRESENCE_CONFIDENCE
#   - Synonym maps (color_synonyms, size_synonyms, etc.)
#   - Contradiction rules (e.g., "red" contradicts "blue" but not "light red")

from dataclasses import dataclass
from typing import Dict, Set, Tuple

@dataclass(frozen=True)
class VerifierProfile:
    """Immutable verifier configuration. Version-controlled."""
    name: str  # "strict", "balanced", "high_recall"
    version: str  # "1.0"
    
    ATTRIBUTE_SIMILARITY_THRESHOLD: float
    SPATIAL_TOLERANCE: float
    OBJECT_PRESENCE_CONFIDENCE: float
    
    # Synonym maps
    COLOR_SYNONYMS: Dict[str, Set[str]]
    SIZE_SYNONYMS: Dict[str, Set[str]]
    ACTION_SYNONYMS: Dict[str, Set[str]]
    
    # Contradiction rules (hard negatives)
    COLOR_CONTRADICTIONS: Dict[str, Set[str]]
    SIZE_CONTRADICTIONS: Dict[str, Set[str]]
    
    description: str  # For versioning logs

# Predefined profiles
STRICT_PROFILE = VerifierProfile(
    name="strict",
    version="1.0",
    ATTRIBUTE_SIMILARITY_THRESHOLD=0.9,
    SPATIAL_TOLERANCE=0.10,
    OBJECT_PRESENCE_CONFIDENCE=0.95,
    # ... maps and contradictions
    description="Conservative: high precision, low recall. Use for trusted experiments."
)

BALANCED_PROFILE = VerifierProfile(
    name="balanced",
    version="1.0",
    ATTRIBUTE_SIMILARITY_THRESHOLD=0.8,
    SPATIAL_TOLERANCE=0.15,
    OBJECT_PRESENCE_CONFIDENCE=0.85,
    description="Default: balanced precision/recall. Use for main results."
)

HIGH_RECALL_PROFILE = VerifierProfile(
    name="high_recall",
    version="1.0",
    ATTRIBUTE_SIMILARITY_THRESHOLD=0.7,
    SPATIAL_TOLERANCE=0.20,
    OBJECT_PRESENCE_CONFIDENCE=0.75,
    description="Liberal: high recall, lower precision. Use for coverage analysis."
)

def get_profile(name: str) -> VerifierProfile:
    """Retrieve profile by name. Raises KeyError if not found."""
    profiles = {
        "strict": STRICT_PROFILE,
        "balanced": BALANCED_PROFILE,
        "high_recall": HIGH_RECALL_PROFILE,
    }
    return profiles[name]
```

**Why this matters**: Every experiment must use an explicitly versioned verifier. If a reviewer asks "what thresholds did you use?" you can cite profile name + version.

---

#### 2. Run Manifest Logger

**File**: `bci_src/runtime/run_manifest.py` (NEW)

```python
"""
Log every experiment run with full reproducibility metadata.
Outputs JSON to results/run_<timestamp>_manifest.json
"""

import json
import subprocess
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

class RunManifest:
    def __init__(self, experiment_id: str, config_path: str):
        self.experiment_id = experiment_id
        self.config_path = config_path
        self.start_time = datetime.utcnow().isoformat()
        self.metadata = self._gather_metadata()
    
    def _gather_metadata(self) -> Dict[str, Any]:
        """Collect git status, code hash, environment, etc."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.start_time,
            "git_hash": self._get_git_hash(),
            "git_branch": self._get_git_branch(),
            "code_hash": self._get_code_hash(),
            "config_hash": self._get_config_hash(),
            "python_version": self._get_python_version(),
            "device": self._get_device(),
            "seed": 42,  # Read from config
        }
    
    def _get_git_hash(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                cwd=Path(__file__).parent.parent.parent
            ).decode().strip()
        except Exception:
            return "unknown"
    
    def _get_code_hash(self) -> str:
        """Hash all Python files in bci_src/."""
        h = hashlib.sha256()
        src_dir = Path(__file__).parent.parent  # bci_src/
        for py_file in sorted(src_dir.rglob("*.py")):
            h.update(py_file.read_bytes())
        return h.hexdigest()[:16]
    
    def _get_config_hash(self) -> str:
        h = hashlib.sha256()
        h.update(Path(self.config_path).read_bytes())
        return h.hexdigest()[:16]
    
    def _get_python_version(self) -> str:
        import sys
        return sys.version
    
    def _get_device(self) -> Dict[str, str]:
        try:
            import torch
            return {
                "cuda_available": str(torch.cuda.is_available()),
                "device_count": str(torch.cuda.device_count()),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            }
        except Exception:
            return {"error": "torch not available"}
    
    def finalize(self, results: Dict[str, Any]) -> str:
        """Save manifest to JSON."""
        manifest = {
            **self.metadata,
            "end_time": datetime.utcnow().isoformat(),
            "results_summary": results,
        }
        
        out_file = Path("results") / f"run_{self.experiment_id}_{int(datetime.utcnow().timestamp())}_manifest.json"
        out_file.parent.mkdir(exist_ok=True)
        out_file.write_text(json.dumps(manifest, indent=2))
        return str(out_file)

# Usage in main script:
# manifest = RunManifest("E1_gqa_replace_all", "configs/phase2/e1_gqa_replace_all.yaml")
# ... run experiment ...
# manifest.finalize({"flip_rate": 0.603, "p_value": 1e-5, ...})
```

---

#### 3. Audit Verifier Script

**File**: `scripts/audit_verifier.py` (NEW)

```python
"""
Run E10 & E11: Audit verifier precision/recall per claim type.
Also inject noise and measure robustness.
"""

import argparse
from pathlib import Path
import sys

# Add src to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from bci_src.data.benchmarks.registry import get_benchmark_adapter
from bci_src.verification.profiles import get_profile
from bci_src.models.vlm_inference import VLMInference
from bci_src.verification.claim_extraction import extract_and_classify
from bci_src.verification.claim_verification import verify_claim
from bci_src.runtime.run_manifest import RunManifest
import json
import numpy as np

def audit_verifier(benchmark_name: str, n_samples: int, verifier_profile: str):
    """
    Main audit function.
    1. Load benchmark samples.
    2. Run baseline inference + belief externalization.
    3. Extract claims + classify by type.
    4. Verify each claim against ground truth (scene graph).
    5. Compute precision/recall per claim type.
    6. Inject noise and measure degradation.
    """
    
    manifest = RunManifest(f"E10_{benchmark_name}_{verifier_profile}", 
                          f"configs/phase2/verifier_audit_{benchmark_name}.yaml")
    
    adapter = get_benchmark_adapter(benchmark_name)
    profile = get_profile(verifier_profile)
    vlm = VLMInference(model_id="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda:1")
    
    samples = adapter.load_samples(n_samples, seed=42)
    adapter.ensure_assets(samples)
    
    # Storage for results
    results_clean = []
    results_noisy_5 = []
    results_noisy_10 = []
    results_noisy_20 = []
    
    for i, sample in enumerate(samples):
        if i % 50 == 0:
            print(f"Processing sample {i}/{n_samples}")
        
        # Baseline + belief externalization
        image = load_image(sample.image_id)
        baseline = vlm.baseline_inference(image, sample.question)
        beliefs_output = vlm.belief_externalization(image, sample.question)
        
        # Extract claims
        claims = extract_and_classify(beliefs_output["beliefs"])
        
        # Verify each claim
        scene_graph = adapter.load_scene_graphs()[sample.sample_id]
        
        for claim in claims:
            status_clean, confidence = verify_claim(
                claim, scene_graph, profile
            )
            
            result = {
                "sample_id": sample.sample_id,
                "claim_id": claim.id,
                "claim_type": claim.type,
                "claim_text": claim.text,
                "true_status": status_clean,
                "confidence": confidence,
            }
            
            # Simulate noise
            status_noisy_5 = introduce_noise(status_clean, error_rate=0.05)
            status_noisy_10 = introduce_noise(status_clean, error_rate=0.10)
            status_noisy_20 = introduce_noise(status_clean, error_rate=0.20)
            
            results_clean.append(result)
            results_noisy_5.append({**result, "noisy_status": status_noisy_5})
            results_noisy_10.append({**result, "noisy_status": status_noisy_10})
            results_noisy_20.append({**result, "noisy_status": status_noisy_20})
    
    # Compute metrics per claim type
    metrics_clean = compute_metrics_by_type(results_clean)
    metrics_noisy_5 = compute_metrics_by_type(results_noisy_5, key="noisy_status")
    metrics_noisy_10 = compute_metrics_by_type(results_noisy_10, key="noisy_status")
    metrics_noisy_20 = compute_metrics_by_type(results_noisy_20, key="noisy_status")
    
    # Save results
    output = {
        "clean": metrics_clean,
        "noisy_5pct": metrics_noisy_5,
        "noisy_10pct": metrics_noisy_10,
        "noisy_20pct": metrics_noisy_20,
    }
    
    out_path = manifest.finalize(output)
    print(f"\nAudit complete. Results saved to {out_path}")
    return output

def introduce_noise(true_status: str, error_rate: float) -> str:
    """Randomly flip verifier output with given probability."""
    if np.random.rand() < error_rate:
        # Flip status
        flip_map = {
            "SUPPORTED": "CONTRADICTED",
            "CONTRADICTED": "SUPPORTED",
            "UNCERTAIN": "SUPPORTED",
        }
        return flip_map.get(true_status, true_status)
    return true_status

def compute_metrics_by_type(results, key="true_status"):
    """Compute precision, recall, F1 per claim type."""
    by_type = {}
    for result in results:
        t = result["claim_type"]
        if t not in by_type:
            by_type[t] = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        
        # Simplified: count contradictions
        status = result[key]
        if status == "CONTRADICTED":
            by_type[t]["TP"] += 1
        else:
            by_type[t]["FN"] += 1
    
    metrics = {}
    for t, counts in by_type.items():
        precision = counts["TP"] / (counts["TP"] + counts["FP"]) if counts["TP"] + counts["FP"] > 0 else 0
        recall = counts["TP"] / (counts["TP"] + counts["FN"]) if counts["TP"] + counts["FN"] > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        metrics[t] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "count": counts["TP"] + counts["FN"],
        }
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="gqa", help="Benchmark name")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of samples")
    parser.add_argument("--profile", default="balanced", help="Verifier profile")
    args = parser.parse_args()
    
    audit_verifier(args.benchmark, args.n_samples, args.profile)
```

---

#### 4. Update Config Files

**File**: `configs/phase2/e1_gqa_replace_all.yaml` (NEW)

```yaml
# Experiment E1: Replace all contradicted beliefs (GQA)
experiment:
  id: "E1_gqa_replace_all"
  description: "Causal hypothesis: replacing all contradicted beliefs with GT flips answer to correct."
  paper_section: "Main Results, Table 1"

dataset:
  name: "gqa"
  split: "val_balanced"
  n_samples: 500
  random_seed: 42
  scene_graph_version: "latest"

model:
  name: "Qwen2.5-VL"
  model_id: "Qwen/Qwen2.5-VL-7B-Instruct"
  device: "cuda:1"
  temperature: 0.0
  max_tokens: 768
  deterministic: true

pipeline:
  stages:
    - name: "baseline_inference"
      enabled: true
    - name: "belief_externalization"
      enabled: true
    - name: "claim_extraction"
      enabled: true
      config: { max_claims_per_sample: 10 }
    - name: "verification"
      enabled: true
      verifier_profile: "balanced"
    - name: "intervention_replace_all"
      enabled: true
      policy: "replace_contradicted_with_ground_truth"
    - name: "constrained_reasoning"
      enabled: true
    - name: "error_classification"
      enabled: true

evaluation:
  primary_metric: "flip_to_correct_rate"
  statistical_test: "chi_squared"
  confidence_level: 0.95
  stratify_by: ["claim_type", "question_type", "answer_type"]

outputs:
  results_csv: "results/e1_gqa_replace_all_results.csv"
  per_claim_audit: "results/e1_gqa_replace_all_claims.csv"
  manifest: "results/e1_gqa_replace_all_manifest.json"
  summary_stats: "results/e1_gqa_replace_all_summary.json"
```

---

#### 5. Main Experiment Runner

**File**: `scripts/run_phase2_experiment.py` (NEW)

```python
"""
Unified runner for Phase 2 experiments.
Usage: python scripts/run_phase2_experiment.py --exp E1 --config configs/phase2/e1_gqa_replace_all.yaml
"""

import argparse
from pathlib import Path
import sys
import yaml
import json

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from bci_src.runtime.run_manifest import RunManifest
from bci_src.data.benchmarks.registry import get_benchmark_adapter
from bci_src.verification.profiles import get_profile
from bci_src.models.vlm_inference import VLMInference
from bci_src.verification.claim_extraction import extract_and_classify
from bci_src.verification.claim_verification import verify_claim
from bci_src.analysis.error_classification import classify_error
import pandas as pd

def run_experiment(exp_id: str, config_path: str):
    """Main entry point for Phase 2 experiments."""
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    manifest = RunManifest(cfg["experiment"]["id"], str(config_path))
    
    # Load components
    adapter = get_benchmark_adapter(cfg["dataset"]["name"])
    profile = get_profile(cfg["pipeline"]["stages"][3]["verifier_profile"])
    vlm = VLMInference(
        model_id=cfg["model"]["model_id"],
        device=cfg["model"]["device"],
        temperature=cfg["model"]["temperature"],
    )
    
    samples = adapter.load_samples(
        cfg["dataset"]["n_samples"],
        seed=cfg["dataset"]["random_seed"]
    )
    adapter.ensure_assets(samples)
    
    # Run experiment
    results = []
    for i, sample in enumerate(samples):
        if i % 50 == 0:
            print(f"[{exp_id}] Processing {i}/{len(samples)}")
        
        # Stages
        image = load_image(sample.image_id)
        baseline_output = vlm.baseline_inference(image, sample.question)
        beliefs_output = vlm.belief_externalization(image, sample.question)
        
        claims = extract_and_classify(beliefs_output["beliefs"])
        scene_graph = adapter.load_scene_graphs()[sample.sample_id]
        
        # Verify
        verified_claims = []
        for claim in claims:
            status, conf = verify_claim(claim, scene_graph, profile)
            verified_claims.append({
                "claim": claim,
                "status": status,
                "confidence": conf,
            })
        
        # Intervene (based on policy)
        policy = cfg["pipeline"]["stages"][4]["policy"]
        if policy == "replace_contradicted_with_ground_truth":
            corrected_claims = replace_with_gt(verified_claims, scene_graph)
        else:
            corrected_claims = verified_claims
        
        # Re-answer
        corrected_output = vlm.constrained_reasoning(
            image, sample.question, corrected_claims
        )
        
        # Error classification
        baseline_error = classify_error(baseline_output, verified_claims)
        corrected_error = classify_error(corrected_output, verified_claims)
        
        flip = 0
        if baseline_error != "correct" and corrected_error == "correct":
            flip = 1
        
        results.append({
            "sample_id": sample.sample_id,
            "baseline_answer": baseline_output["answer"],
            "corrected_answer": corrected_output["answer"],
            "ground_truth": sample.answer,
            "baseline_error_type": baseline_error,
            "corrected_error_type": corrected_error,
            "flip_to_correct": flip,
            "n_claims": len(claims),
            "n_contradicted": sum(1 for c in verified_claims if c["status"] == "CONTRADICTED"),
            "n_supported": sum(1 for c in verified_claims if c["status"] == "SUPPORTED"),
        })
    
    # Summary stats
    results_df = pd.DataFrame(results)
    flip_rate = results_df["flip_to_correct"].mean()
    
    summary = {
        "flip_to_correct_rate": float(flip_rate),
        "n_samples": len(results),
        "n_flipped": int(results_df["flip_to_correct"].sum()),
        "p_value": compute_binomial_test(results_df["flip_to_correct"]),
    }
    
    # Save
    results_df.to_csv(cfg["outputs"]["results_csv"], index=False)
    with open(cfg["outputs"]["summary_stats"], "w") as f:
        json.dump(summary, f, indent=2)
    
    manifest.finalize(summary)
    print(f"\n✓ Experiment {exp_id} complete.")
    print(f"  Flip-to-correct rate: {flip_rate:.1%}")
    print(f"  Results saved to {cfg['outputs']['results_csv']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, help="Experiment ID (e.g., E1, E10)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    
    run_experiment(args.exp, args.config)
```

---

### **Week 1 Checklist**

- [ ] Implement `bci_src/verification/profiles.py` (verifier profiles)
- [ ] Implement `bci_src/runtime/run_manifest.py` (manifest logger)
- [ ] Implement `scripts/audit_verifier.py` (E10 audit script)
- [ ] Implement `scripts/run_phase2_experiment.py` (unified experiment runner)
- [ ] Create `configs/phase2/e1_gqa_replace_all.yaml`
- [ ] Create `configs/phase2/e3_gqa_remove_single.yaml`
- [ ] Create `configs/phase2/verifier_audit_gqa.yaml`
- [ ] Run E1: `python scripts/run_phase2_experiment.py --exp E1 --config configs/phase2/e1_gqa_replace_all.yaml`
- [ ] Run E3: `python scripts/run_phase2_experiment.py --exp E3 --config configs/phase2/e3_gqa_remove_single.yaml`
- [ ] Run E10: `python scripts/audit_verifier.py --benchmark gqa --n-samples 500 --profile balanced`
- [ ] Run E11: `python scripts/audit_verifier.py --benchmark gqa --n-samples 500 --profile balanced` (with noise injection)
- [ ] Generate Verifier Reliability Matrix (PDF)
- [ ] **Gate**: Confirm E1 ≥50%, E10 precision ≥80% before proceeding to Week 2

---

## **Week 2-3: Cross-Benchmark (Phase 2b)**

### **Objective**
Validate causal signal on MMMU, MathVista, DocVQA. Stratify error analysis.

### **Experiments**
- **E4**: MMMU full correction
- **E5**: MMMU targeted correction
- **E6**: MathVista full correction
- **E7**: DocVQA (OCR) full correction
- **E13-E15**: GQA stratification by claim type, question type, answer type

### **Configs to Create**
- `configs/phase2/e4_mmmu_replace_all.yaml`
- `configs/phase2/e5_mmmu_targeted.yaml`
- `configs/phase2/e6_mathvista_replace_all.yaml`
- `configs/phase2/e7_docvqa_replace_all.yaml`
- `configs/phase2/stratify_gqa.yaml`

### **Script to Add**
- `scripts/stratify_results.py`: Compute flip rates by claim type, q-type, answer type with binomial tests

---

## **Week 3-4: Model Generality (Phase 2c)**

### **Objective**
Run GQA + MMMU on LLaVA-1.5, InstructBLIP to confirm effect is cross-model.

### **Experiments**
- **E8**: E1 logic on GQA with 3 models (Qwen, LLaVA, InstructBLIP)
- **E9**: E4 logic on MMMU with 2 models (Qwen, LLaVA-Next)

### **How to Extend VLMInference**
Update `bci_src/models/vlm_inference.py` to support templating for model-specific prompts:

```python
class VLMInference:
    def __init__(self, model_id, device, model_family=None):
        self.model_id = model_id
        self.device = device
        # Infer family from model_id or use override
        self.model_family = model_family or infer_family(model_id)
    
    def _get_prompt(self, template_name):
        """Route to model-specific prompt template."""
        if self.model_family == "qwen":
            return QWEN_PROMPTS[template_name]
        elif self.model_family == "llava":
            return LLAVA_PROMPTS[template_name]
        elif self.model_family == "instructblip":
            return INSTRUCTBLIP_PROMPTS[template_name]
        else:
            raise ValueError(f"Unknown model family: {self.model_family}")
```

---

## **Week 4-5: Policy Optimization (Phase 2d)**

### **Objective**
Compare intervention policies on GQA & MMMU. Identify practical version (P2).

### **Experiments**
- **P1-P5**: All five policies on GQA (500) + MMMU (300)
- **E2**: Replace top-k (k=1,2,3,5)
- **E24-E25**: Zero-shot + minimal tuning transfer

### **Policy Implementation**

Add intervention policy layer: `bci_src/intervention/policies.py`

```python
class InterventionPolicy:
    def apply(self, verified_claims, scene_graph):
        """Transform verified claims -> corrected claims."""
        raise NotImplementedError

class ReplaceAllPolicy(InterventionPolicy):
    def apply(self, verified_claims, scene_graph):
        return [c for c in verified_claims if c["status"] != "CONTRADICTED"] + [
            {"claim": gt_claim, "status": "SUPPORTED", "confidence": 1.0}
            for gt_claim in extract_from_scene_graph(scene_graph)
        ]

class ReplaceConfidentPolicy(InterventionPolicy):
    def __init__(self, confidence_threshold=0.7):
        self.threshold = confidence_threshold
    
    def apply(self, verified_claims, scene_graph):
        high_conf = [c for c in verified_claims if c["confidence"] >= self.threshold or c["status"] != "CONTRADICTED"]
        return high_conf + [...]  # + GT for high-confidence contradictions

class RemoveOnlyPolicy(InterventionPolicy):
    def apply(self, verified_claims, scene_graph):
        return [c for c in verified_claims if c["status"] != "CONTRADICTED"]
```

---

## **Week 5-6: Robustness & Stress Test (Phase 2e)**

### **Objective**
Finalize robustness curves, latency breakdown, compute-performance tradeoff.

### **Experiments**
- **E12**: Recovery degradation under noise (Figure for paper)
- **E18-E19**: Uncertainty + interaction analysis
- **E20-E23**: Latency per component (Figure for paper)

### **Key Outputs for Paper**
1. Forest plot: E1-E7 effect sizes with CIs
2. Cross-model consistency table: E8 results
3. Verifier Reliability Matrix: E10 per-type precision/recall
4. Noise robustness curves: E12
5. Latency breakdown: E20-E23

---

## **Post-Phase-2: Submission-Ready Artifacts**

### **Code Release Checklist**
- [ ] All Phase 2 configs in `configs/phase2/`
- [ ] All scripts in `scripts/`
- [ ] Updated README with Phase 2 experiment commands
- [ ] Tests cover all new modules (profiles, manifests, policies)
- [ ] GitHub repo public with version tag `v2.0_phase2_complete`

### **Paper Artifact Checklist**
- [ ] Experiment matrix table (this doc + formatted for appendix)
- [ ] Causal effect forest plot (E1-E7)
- [ ] Verifier Reliability Matrix (PDF)
- [ ] Noise robustness curves (PDF)
- [ ] Latency breakdown (PDF)
- [ ] Stratification heatmap (E13-E15)
- [ ] Statistical significance table (all p-values, effect sizes, sample sizes)

---

## **Resource Requirements**

| Resource | Actual Requirement |
|----------|-------------------|
| **GPU time** | ~250 GPU-hours (40-50 hrs per benchmark × 5 benchmarks + E8-E9 multi-model overhead) |
| **Disk** | ~80 GB (results CSVs + audit logs + manifests) |
| **Dev time** | 3-4 weeks for full implementation + execution |
| **Critical path** | 5-6 weeks (strict sequential: 2a → 2b → 2c → 2d → 2e) |
| **Parallelization** | E13-E15 analysis can run while E4-E7 are executing (saves ~1 week) |

---

## **How to Kick Off**

**Step 1**: Ensure Phase 1 code is stable and all tests in `tests/` pass.

```bash
cd /data/cs22btech11029/playground/Inference_time_VLM
conda activate mech_interp
python -m pytest -v
```

**Step 2**: Implement Week 1 modules (profiles, manifests, audit script, runner).

**Step 3**: Create Week 1 configs.

**Step 4**: Run E1 as smoke test:

```bash
python scripts/run_phase2_experiment.py --exp E1 --config configs/phase2/e1_gqa_replace_all.yaml
```

**Step 5**: If E1 flip_rate ≥50% and E10 precision ≥80%, proceed to Week 2.

---

**Next**: Ready to start Week 1 implementation?

