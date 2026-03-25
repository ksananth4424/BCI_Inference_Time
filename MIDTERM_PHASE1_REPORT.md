# Belief-Constrained Inference for VLMs (BCI)
## Mid-Term Project Report (Phase 1)

### Authors
- K S Ananth (`CS22BTECH11029`)
- Bonthu Mani Hemanth Reddy (`CS22BTECH11013`)
- Kartikeya Mandapati (`CS22BTECH11032`)

### Date
14 March 2026

---

## 1. Problem Statement
Vision-Language Models (VLMs) often fail by reasoning correctly from **incorrect visual premises**. In standard inference, once the image is encoded, the model can continue reasoning without explicitly re-checking perceptual assumptions.

This project studies whether a substantial fraction of VLM errors are caused by **epistemic failures** (wrong perceptual beliefs) rather than logical reasoning failures, and whether correcting premises can recover answers.

---

## 2. Core Hypothesis

### Main Hypothesis (H1)
A significant portion of VLM answer errors are due to false visual premises. If these premises are corrected, many wrong answers should flip to correct without retraining model weights.

### Supporting Hypotheses
- **H2 (Constraint Hypothesis):** Constraining reasoning using verified beliefs improves answer reliability.
- **H3 (Weak Verifier Sufficiency):** Even imperfect deterministic verification can improve outcomes.
- **H4 (Belief Minimality):** In a subset of failures, removing a small number of false beliefs is sufficient to fix the answer.

---

## 3. Method: Belief-Constrained Inference (BCI)

BCI uses an inference-time loop with explicit premise handling:

1. **Belief Externalization**
   - The VLM must output 5-10 atomic, grounded, falsifiable visual beliefs before answering.

2. **Belief Verification**
   - Claims are checked against structured visual evidence (in Phase 1: GQA scene graphs).
   - Status per claim: `SUPPORTED`, `CONTRADICTED`, `UNCERTAIN`.

3. **Belief Revision**
   - Contradicted beliefs are removed or replaced (depending on experiment).

4. **Constrained Reasoning**
   - The model answers using only the revised/verified belief set.

---

## 4. Benchmarks and Datasets

## 4.1 Primary Benchmark
- **GQA** (balanced validation setup)

## 4.2 Data Sources Used in Phase 1
- **Questions + answers + images:** HuggingFace `lmms-lab/GQA`
  - `val_balanced_instructions` (QA metadata)
  - `val_balanced_images` (images)
- **Scene graph supervision:** Stanford GQA release
  - `val_sceneGraphs.json` (used as verifier evidence)

## 4.3 Sampling
- Total evaluated samples: **500**
- Sampling seed: `42`
- Candidate pool: GQA val-balanced entries with matching scene graph image IDs

## 4.4 Why GQA for Phase 1
- Contains structured scene-graph annotations (objects, attributes, relations)
- Enables automated claim verification and error decomposition
- Appropriate for hypothesis testing before scaling to harder benchmarks

---

## 5. Experimental Setup

## 5.1 Models Used
- **Initial run:** `llava-hf/llava-1.5-7b-hf`
- **Final Phase 1 run:** `Qwen/Qwen2.5-VL-7B-Instruct`

Reason for switch:
- LLaVA externalized too few beliefs (avg ~2.17), causing poor claim coverage and under-detection of premise errors.
- Qwen2.5-VL produced richer belief sets (avg ~4.96), improving verifiability.

## 5.2 Inference Config
From `config.py`:
- `VLM_MAX_NEW_TOKENS = 768`
- `VLM_TEMPERATURE = 0.0`
- `NUM_SAMPLES = 500`
- `SPATIAL_TOLERANCE = 0.15`
- `ATTRIBUTE_SIMILARITY_THRESHOLD = 0.8`

## 5.3 Compute Environment
- GPU: `4 x NVIDIA RTX 6000 Ada (49 GB each)`
- Active inference device in config: `cuda:1`

---

## 6. Evaluation Metrics

## 6.1 Core Error Decomposition (Experiment 1)
For each question:
- Determine answer correctness
- Verify extracted beliefs against scene graph
- Classify error type:
  - **Premise Error:** at least one contradicted belief in an incorrect case
  - **Reasoning Error:** incorrect answer with no contradicted beliefs detected

Primary metric:
- `Premise Error Rate (of incorrect)`

## 6.2 Claim Coverage (Experiment 3)
- Fraction of question-relevant scene-graph facts surfaced by belief externalization.
- Used to interpret whether premise-error rate is conservative due to missing beliefs.

## 6.3 Failure Recovery (Experiment 2)
- **Flip Rate:** percentage of recoverable premise-error cases that become correct after replacing beliefs with ground-truth relevant beliefs.

## 6.4 Belief Minimality (Experiment 4)
- Fraction of premise-error cases fixed by removing one contradicted belief at a time.

## 6.5 Placebo Control (Random Ablation)
- Remove random beliefs (same removal budget) and measure flip rate.
- Purpose: show targeted correction does better than generic belief deletion.

---

## 7. Phase 1 Experiments and Results

## 7.1 Experiment 1: Premise Error Quantification (Final Qwen Run)
From `results/experiment1_analysis.json`:

- Total samples: **500**
- Correct: **296** (59.2%)
- Incorrect: **204**

Among incorrect answers:
- **Premise errors:** **68 / 204 = 33.3%**
- **Reasoning errors:** **136 / 204 = 66.7%**

Coverage:
- Mean coverage: **0.49**
- Median coverage: **0.50**

Belief externalization quality (from outputs):
- Qwen avg beliefs/question: **4.964**
- LLaVA avg beliefs/question: **2.172**

## 7.2 Model Comparison (LLaVA vs Qwen)

| Metric | LLaVA-1.5-7B | Qwen2.5-VL-7B |
|---|---:|---:|
| Accuracy | 52.2% | 59.2% |
| Premise errors (of incorrect) | 18.8% | 33.3% |
| Reasoning errors (of incorrect) | 81.2% | 66.7% |
| Avg beliefs per question | 2.172 | 4.964 |

Interpretation:
- Qwen improves both answer accuracy and premise-error observability by producing richer, more specific belief sets.

## 7.3 Question-Type Breakdown (Qwen)
Premise-error fraction among incorrect per structural type:

- `logical`: **50.0%**
- `compare`: **40.0%**
- `query`: **32.9%**
- `verify`: **30.4%**
- `choose`: **14.3%**

This indicates premise-driven failures are especially strong in logical/relational questions.

## 7.4 Experiment 2: Premise Correction Test
From `results/experiment2_premise_correction.json`:

- Recoverable premise-error cases: **68**
- Flipped to correct after ground-truth premise replacement: **41**
- **Flip rate: 60.3%**

This is the strongest causal evidence in Phase 1.

## 7.5 Experiment 4: Belief Minimality
From `results/experiment4_belief_minimality.json`:

- Cases evaluated: **68**
- Cases fixed by removing one contradicted belief candidate: **19**
- **Single-belief minimality rate: 27.9%**

Interpretation:
- A non-trivial subset of failures is sensitive to a single bad premise.
- Many failures likely involve multiple interacting false premises.

## 7.6 Random Ablation (Placebo Control)
From `results/experiment_random_ablation.json`:

- Mean random flip rate: **25.0%**

Comparison:
- Targeted full premise correction (Exp 2): **60.3%**
- Random ablation: **25.0%**
- Gap: **+35.3 percentage points**

This supports that targeted premise correction is not equivalent to random belief deletion.

---

## 8. Artifacts Generated

## 8.1 Core Result Files
- `results/experiment1_analysis.json`
- `results/experiment1_vlm_outputs.json`
- `results/experiment2_premise_correction.json`
- `results/experiment4_belief_minimality.json`
- `results/experiment_random_ablation.json`

## 8.2 Figures
- `results/figures/error_breakdown.pdf`
- `results/figures/claim_coverage.pdf`
- `results/figures/premise_correction.pdf`
- `results/figures/belief_minimality.pdf`
- `results/figures/random_vs_targeted.pdf`

---

## 9. Reproducibility (Commands)

From project root:

```bash
python run_pipeline.py download      # download scene graphs
python run_pipeline.py setup         # sample questions + pull images from HF
python run_pipeline.py experiment1   # run VLM baseline + belief externalization
python run_pipeline.py analyze       # claim verification + error decomposition
python run_pipeline.py experiment2   # premise correction
python run_pipeline.py experiment4   # belief minimality
python run_pipeline.py random_ablation
```

---

## 10. Current Conclusions (Phase 1)

1. **Premise errors are substantial** in the final setup (33.3% of incorrect answers), and even higher in some categories (logical: 50%).
2. **Causal evidence is strong:** replacing false premises flips 60.3% of premise-error cases to correct.
3. **Targeted correction matters:** large advantage over random ablation (+35.3 pp).
4. **Belief externalization quality is crucial:** stronger instruction-following VLMs yield better premise-error observability.

Overall, Phase 1 supports the BCI thesis that many VLM failures are epistemic and that premise-level intervention is an effective correction axis.

---

## 11. Limitations in Phase 1

1. **Single benchmark focus:** only GQA was used in this phase.
2. **Verifier dependence:** scene graph quality/noise can affect verification labels.
3. **Coverage is incomplete (49%):** measured premise error rate is likely conservative.
4. **No full Woodpecker/VCD runtime baseline yet:** comparison in this phase is primarily mechanism-focused.

---

## 12. Next Steps (Phase 2 Plan)

1. Add at least one modern benchmark for headline generalization (e.g., MMMU / MathVista / MM-Vet).
2. Add explicit verifier-noise robustness curves (false negative/positive injection).
3. Strengthen claim extraction quality and coverage.
4. Run direct head-to-head with post-hoc correction baselines (Woodpecker-style ordering comparison).
5. Consolidate into paper-ready sections: hypothesis validation, mechanism analysis, robustness, benchmark generalization.

---

## Appendix A: Quick Result Snapshot

- `Exp1 premise-error rate (incorrect only): 33.3%`
- `Exp2 flip rate after GT premise correction: 60.3%`
- `Exp4 single-belief minimality rate: 27.9%`
- `Random ablation flip rate: 25.0%`
- `Qwen accuracy: 59.2%`
- `LLaVA accuracy: 52.2%`

These numbers represent the current state of the project at mid-term milestone completion.
