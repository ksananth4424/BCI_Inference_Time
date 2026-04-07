# Inference-Time VLM Hallucination Correction: Current Project State

Last updated: 2026-04-07

## 1. Project Objective

This project studies whether inference-time belief correction can causally reduce visual hallucinations and improve answer accuracy in vision-language models (VLMs), without retraining the base model.

Core hypothesis family:

- H1: A substantial fraction of wrong VLM answers is caused by erroneous intermediate visual beliefs.
- H2: Correcting these beliefs before final reasoning should improve final answer correctness.
- H3: This can be achieved with practical, compute-bounded intervention policies (not only oracle settings).

Primary model and benchmark used in current experiments:

- Model: Qwen/Qwen2.5-VL-7B-Instruct
- Benchmark: GQA (val_balanced style sampling)

Method class: post-hoc, inference-time intervention pipeline.


## 2. Pipeline and Methodology (Implemented)

The main execution path is implemented in scripts/run_phase2_experiment.py.

Per-sample pipeline:

1. Baseline inference
- Generate baseline answer from image + question.

2. Belief externalization
- Prompt model to output explicit visual claims used for reasoning.

3. Claim extraction and verification
- Convert beliefs into atomic claims.
- Verify each claim against GQA scene-graph evidence.
- Label claims as SUPPORTED, CONTRADICTED, or UNCERTAIN.

4. Intervention policy
- Modify claim set according to policy.
- Major policies tested:
	- E1: replace contradicted beliefs with ground-truth facts (strong intervention).
	- E3: remove contradicted claims only (conservative intervention).
	- E12/E12b: confidence-gated intervention with localized GT edits.

5. Constrained re-reasoning
- Re-answer using corrected claims.

6. Evaluation
- Compare baseline vs corrected correctness and paired flip behavior.
- Use paired statistics (McNemar exact) where applicable.


### 2.1 Full Pipeline Diagram of the Novel Contribution

The diagram below captures the complete contribution stack used in this project, from data and model inputs to intervention logic, correction generation, and statistical validation.

```mermaid
flowchart TD
	A[Input: image + question] --> B[Stage 1: Baseline inference]
	A --> C[Stage 2: Belief externalization]
	C --> D[Stage 3a: Claim extraction]
	D --> E[Stage 3b: Claim verification using scene graph]
	E --> F{Claim labels}

	F --> F1[SUPPORTED]
	F --> F2[CONTRADICTED]
	F --> F3[UNCERTAIN]

	F1 --> G[Stage 4: Intervention policy]
	F2 --> G
	F3 --> G

	G --> G1[E1: Replace contradicted claims with GT facts]
	G --> G2[E3: Remove contradicted claims only]
	G --> G3[E12/E12b: Confidence-gated localized replacement]

	G1 --> H[Stage 5: Constrained re-reasoning]
	G2 --> H
	G3 --> H

	H --> I[Corrected answer]
	B --> J[Baseline answer]

	J --> K[Stage 6: Paired evaluation]
	I --> K

	K --> K1[Accuracy delta]
	K --> K2[Flip-to-correct / flip-to-wrong]
	K --> K3[McNemar exact significance]
	K --> K4[Error-type and claim-type analysis]

	E --> L[Verifier audit branch (E10/E11)]
	L --> L1[Profile sweep: balanced/high_recall/strict]
	L --> L2[Noise robustness: 0.05/0.1/0.2]
	L1 --> K4
	L2 --> K4

	K --> M[Final artifacts: CSV + JSON + manifests + paper tables]
```


## 3. What Was Fixed and Stabilized in the Codebase

Critical wiring bugs were identified and fixed before final experiments:

- Claim extraction input-shape mismatches causing runtime failures.
- Scene graph keying mismatches (image_id/sample_id handling).
- Verification output handling mismatches.
- Profile threshold propagation into verifier runtime.
- E1 policy behavior upgraded from filtering-only to actual GT injection logic.

As a result, E1/E3/E12/E12b runs are now stable and reproducible under deterministic settings.


## 4. Completed Core Experiments and Results

### 4.0 Experiment ID Glossary (E1, E2, ...)

This glossary explains all experiment IDs that appear in logs, scripts, and artifacts, including both current Phase-2 IDs and earlier pilot IDs.

Current Phase-2 IDs:

- E1: Replace contradicted claims with ground-truth facts (strong intervention / causal upper-bound style setting).
- E2: Reserved ID in the Phase-2 naming scheme; no finalized E2 run artifacts are currently part of the stable result set.
- E3: Remove contradicted claims only, without GT replacement (conservative intervention).
- E10/E11: Verifier audit and robustness suite (profile sweeps and synthetic noise stress tests).
- E12: Confidence-gated practical intervention (initial threshold setting, more conservative gate).
- E12b: Tuned confidence-gated practical intervention (threshold adjusted; stronger practical performance than E12).

Legacy pilot IDs (pre-Phase-2 naming, still useful for historical context):

- Experiment 1: Baseline error analysis and error taxonomy construction.
	- Artifact examples: results/experiment1_analysis.json, results/experiment1_vlm_outputs.json
- Experiment 2: Premise correction pilot (early correction trials before standardized Phase-2 runner).
	- Artifact examples: results/experiment2_premise_correction.json, results/experiment2_log.txt
- Experiment 4: Belief minimality/random-ablation style pilots (removal sensitivity and intervention behavior checks).
	- Artifact examples: results/experiment4_belief_minimality.json, results/experiment_random_ablation.json

Why this matters:

- Current paper-facing claims should prioritize Phase-2 IDs (E1/E3/E10/E11/E12/E12b) because those runs use the stabilized, versioned pipeline.
- Legacy experiments are best used as developmental evidence and motivation, not as headline benchmark claims.

### 4.1 E1 (Strong GT Injection), n=500

Source artifact: results/phase2_intervention_analysis.json

- Baseline accuracy: 59.2% (296/500)
- Corrected accuracy: 77.2% (386/500)
- Delta: +18.0 percentage points
- Flip-to-correct: 117 (23.4%)
- Flip-to-wrong: 27 (5.4%)
- Net gain: +90 samples (+18.0 pp net)
- McNemar exact p-value: 1.487e-14

Interpretation:

- Strong causal evidence for H1/H2.
- Largest observed gain among tested methods.


### 4.2 E3 (Remove Contradicted Only), n=500

Source artifact: results/phase2_intervention_analysis.json

- Baseline accuracy: 59.2% (296/500)
- Corrected accuracy: 58.6% (293/500)
- Delta: -0.6 percentage points
- Flip-to-correct: 47 (9.4%)
- Flip-to-wrong: 50 (10.0%)
- Net gain: -3 samples
- McNemar exact p-value: 0.839

Interpretation:

- Conservative removal alone is not sufficient.
- Supports the need for replacement/synthesis, not only pruning.


### 4.3 Confidence-Gated Variant E12, n=500

Source artifact: results/e12_main/summary_E12_gqa_confidence_gated_e1.json

- Baseline accuracy: 59.2% (296/500)
- Corrected accuracy: 59.6% (298/500)
- Flip-to-correct: 25 (5.0%)

Interpretation:

- Initial gate threshold was too strict for strong gain.


### 4.4 Tuned Confidence-Gated Variant E12b, n=500

Source artifact: results/e12b_main/summary_E12b_gqa_confidence_gated_e1_tuned.json

- Baseline accuracy: 59.2% (296/500)
- Corrected accuracy: 62.4% (312/500)
- Delta vs baseline: +3.2 percentage points
- Flip-to-correct: 49 (9.8%)

Paired comparison E12b vs E12 on same 500 samples:

Source artifact: results/e12_vs_e12b_comparison.md

- E12 accuracy: 59.6%
- E12b accuracy: 62.4%
- Absolute gain: +2.8 pp
- Wins (E12 wrong -> E12b right): 25
- Losses (E12 right -> E12b wrong): 11
- McNemar exact p-value: 0.0288

Interpretation:

- Tuned gate materially improves the practical variant.


### 4.5 Tuned Confidence-Gated Variant E12b, n=2000

Source artifacts:

- results/e12b_n2000/summary_E12b_gqa_confidence_gated_e1_tuned.json
- results/e12b_n2000_analysis.md

Headline:

- Baseline accuracy: 62.9% (1258/2000)
- Corrected accuracy: 66.35% (1327/2000)
- Delta: +3.45 percentage points
- Flip-to-correct: 158 (7.9%)

Paired significance baseline vs E12b corrected:

- Wins: 158
- Losses: 89
- Net wins: +69
- McNemar exact p-value: 1.34e-05

Intervention diagnostics:

- Intervened rate: 59.85%
- Fallback rate: 0.10%

Interpretation:

- Practical gated method is robust and significant at larger sample size.


## 5. Competitor Matrix (Matched In-House Baselines)

Source artifacts:

- results/competitors_main/competitor_matrix_summary.json
- results/competitors_main/competitor_summary_gqa_500.json

Methods included:

- vanilla
- self_correct
- cove_lite
- resample_lite
- bci_e3
- bci_e1

n=500 leaderboard (accuracy):

1. bci_e1: 0.772 (386/500)
2. resample_lite: 0.606 (303/500)
3. vanilla: 0.592 (296/500)
4. bci_e3: 0.586 (293/500)
5. cove_lite: 0.572 (286/500)
6. self_correct: 0.492 (246/500)

Interpretation:

- BCI-E1 clearly dominates matched lightweight baselines.
- Strong evidence that claim-level intervention is more effective than generic self-correction on this setup.


## 6. Verifier Audit and Robustness (E10/E11)

Purpose:

- Characterize verifier contradiction detection across profiles and noise levels.

Artifacts:

- results/e10_e11_gqa_balanced/audit_gqa_balanced_metrics.json (n=500)
- results/e10_e11_gqa_high_recall_v2/audit_gqa_high_recall_metrics.json (n=500)
- results/e10_e11_gqa_balanced_n2000/audit_gqa_balanced_metrics.json (n=2000)
- results/e10_e11_gqa_high_recall_n2000/audit_gqa_high_recall_metrics.json (n=2000)
- results/e10_e11_gqa_strict_n2000/audit_gqa_strict_metrics.json (n=2000)

Key observations (n=2000):

- Object existence detection is strongest among major claim families and rises under injected noise.
	- Balanced clean: 25.29% (844/3337)
	- Balanced noise 0.2: 34.67% (1157/3337)

- Action contradiction detection is near zero on clean but rises sharply with noise.
	- Balanced clean: 0.00% (0/1502)
	- Balanced noise 0.2: 20.17% (303/1502)

- Attribute detection remains low but non-zero.
	- Balanced clean: 2.13% (62/2913)
	- Balanced noise 0.2: 3.81% (111/2913)

- OCR detection is consistently 0 in these audited runs (known weak spot).

- Strict and high_recall profiles are similar to balanced on clean object detection in this audited setup, but differ under noise especially for spatial/action.

Interpretation:

- Verifier is useful and stable for high-volume object existence contradictions.
- OCR and some fine-grained relation/action cases are still bottlenecks.


## 7. Literature Review Integration Status

Paper extraction and synthesis artifacts:

- results/paper_review_index.json
- results/paper_snippets.json
- results/paper_synthesis_action_plan.md

Status:

- 13 papers parsed and indexed with benchmark/method snippets.
- Actionable ideas translated into concrete roadmap items (confidence gating, decomposition, fallback/resampling, selective tool usage).
- Current E12/E12b work directly reflects this literature-informed roadmap.


## 8. Main Scientific Conclusions at Current Stage

1. Strong causal signal is established.
- E1 shows large, highly significant gains, supporting the premise-correction hypothesis.

2. Practical selective intervention is feasible.
- E12b achieves significant gains at n=2000 (+3.45 pp) with bounded intervention and minimal fallback.

3. Intervention design matters.
- Removal-only (E3) can hurt.
- Replacement and calibrated gating outperform conservative filtering.

4. Against matched lightweight baselines, current method family is strong.
- BCI-E1 is best in the internal matrix.


## 9. Limitations and Fairness Caveats (Important for Paper Writing)

1. Headline E1 is a strong intervention upper bound.
- It is valid as causal evidence, but should be framed as high-assistance/oracle-like relative to practical deployments.

2. External paper comparison is not yet full reproduction-equivalent.
- Current cross-paper positioning is strongest in matched in-house baseline space.
- Avoid claiming direct SOTA over differently trained systems without matched protocol reproduction.

3. Some claim families remain under-detected in verifier audit.
- OCR and specific fine-grained categories require further dedicated handling.


## 10. Experiment Inventory (Completed)

Completed and usable now:

- E1 (GT replacement), n=500
- E3 (remove-only), n=500
- E12 (gated initial), n=500
- E12b (gated tuned), n=500 and n=2000
- E10/E11 verifier audits, n=500 and n=2000 across profiles
- Competitor matrix, n=500 (vanilla, self_correct, cove_lite, resample_lite)
- Literature extraction and synthesis pipeline over 13 papers


## 11. Remaining Tests and Experiments (Prioritized)

Below is the exact list of experiments still recommended to make the paper maximally strong.

### Priority A (High impact, should run)

1. E1 at n=2000 (same protocol as E12b n=2000)
- Why: confirms headline method at larger scale and allows clean large-n comparison vs E12b.

2. Multi-seed stability for E1 and E12b (at least 3 seeds)
- Why: enables mean ± std reporting and reduces single-seed criticism.

3. Paired significance table across all major methods
- Compare baseline, E1, E3, E12, E12b, and matched competitors with McNemar exact tests.

4. Error-type stratified gains for E1 and E12b
- Break down gains by claim type and question type to strengthen mechanistic explanation.

### Priority B (Strong paper polish)

5. Cost-efficiency analysis
- Report runtime/tokens per sample and gain per unit compute.
- Critical for the stated project goal of inference-time efficiency.

6. Robustness sweep for E12b threshold
- Evaluate at least 3 threshold values around 0.18 on same sample split.
- Show monotonicity or trade-off curve.

7. Calibration and abstention analysis
- Use intervention_score as risk proxy and report reliability bins.

### Priority C (Cross-paper strengthening)

8. One external hallucination benchmark transfer run
- Example: POPE-style or MMHal-style setup if feasible in current infrastructure.
- Goal: strengthen external validity beyond GQA.

9. Closer reproduction of one external method
- Pick one practical comparator from literature and reproduce under your exact model/split.
- This tightens comparison against other papers.


## 12. Recommended Final Paper Positioning (Current Best)

Primary claim:

- Inference-time belief intervention yields large and statistically significant gains; strongest with full replacement policy (E1), and still significant with practical gated policy (E12b).

Secondary claim:

- Under matched in-house baselines, the method substantially outperforms self-correction and lightweight verification competitors.

Caveat framing:

- Distinguish clearly between causal upper-bound (E1) and practical bounded policy (E12b).


## 13. Artifact Map (Fast Navigation)

Core summaries:

- results/phase2_intervention_analysis.json
- results/e12_main/summary_E12_gqa_confidence_gated_e1.json
- results/e12b_main/summary_E12b_gqa_confidence_gated_e1_tuned.json
- results/e12b_n2000/summary_E12b_gqa_confidence_gated_e1_tuned.json
- results/e12_vs_e12b_comparison.md
- results/e12b_n2000_analysis.md

Competitors:

- results/competitors_main/competitor_summary_gqa_500.json
- results/competitors_main/competitor_matrix_summary.json

Verifier audits:

- results/e10_e11_gqa_balanced/audit_gqa_balanced_metrics.json
- results/e10_e11_gqa_high_recall_v2/audit_gqa_high_recall_metrics.json
- results/e10_e11_gqa_balanced_n2000/audit_gqa_balanced_metrics.json
- results/e10_e11_gqa_high_recall_n2000/audit_gqa_high_recall_metrics.json
- results/e10_e11_gqa_strict_n2000/audit_gqa_strict_metrics.json

Literature synthesis:

- results/paper_review_index.json
- results/paper_snippets.json
- results/paper_synthesis_action_plan.md


## 14. Short Handoff Summary for New Collaborators

If you are joining this project now, the key points are:

1. The pipeline is stable and reproducible.
2. E1 is the strongest result and already highly significant on n=500.
3. E12b is the practical gated variant and is significant on n=2000.
4. Matched competitor results strongly favor BCI-E1.
5. To finish a top-tier submission, run large-n E1, multi-seed stability, and final unified significance/cost tables.

