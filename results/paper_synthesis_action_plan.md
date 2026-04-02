# Paper-by-Paper Synthesis and Action Plan

Scope: 13 papers in papers/ folder, synthesized for immediate integration into this repository's Phase-2 BCI pipeline.

## Executive Summary

- Your strongest empirical axis (E1 GT injection with verifier-guided correction) is already aligned with the most robust direction in the literature: claim-level verification + targeted correction.
- The highest ROI next additions are inference-time only and can be implemented without retraining:
  - confidence-gated correction triggers,
  - selective resampling/backtracking for uncertain spans,
  - contrastive decoding fallback,
  - lightweight programmatic tool calls for counting/attributes.
- For NeurIPS positioning, frame your method as a practical, compute-efficient variant of claim verification that is stronger than lightweight baselines under matched protocol, with paired significance.

## Per-Paper Notes and What To Reuse

1. VOLCANO (2024.naacl-long.23.pdf)
- Core idea: self-feedback guided revision (draft -> critique -> revise -> decide).
- Reuse now: add an explicit critique step before final correction in E1/E3, but keep it short and schema-constrained.
- Minimal implementation: add a critique prompt template that outputs only contradicted claims + reason tags.

2. CoVe (2309.11495v2.pdf)
- Core idea: draft response, plan verification questions, answer independently, then rewrite.
- Reuse now: generate per-claim verification questions and answer each independently to reduce coupling bias.
- Minimal implementation: replace single-pass verifier call with per-claim independent checks in the existing verifier loop.

3. Woodpecker (2310.16045v2.pdf)
- Core idea: training-free post-hoc correction with a 5-stage pipeline.
- Reuse now: maintain your post-remedy architecture, but add explicit key-concept extraction and yes/no micro-checks for entities.
- Minimal implementation: before correction, extract entity/attribute/relation triples and validate each with short QA probes.

4. Pelican (2407.02352v2.pdf)
- Core idea: claim decomposition into predicate-level sub-claims + Program-of-Thought verification.
- Reuse now: decompose complex claims into atomic predicates and score each sub-claim confidence.
- Minimal implementation: augment claim representation to include sub_claims[] and aggregate confidence for correction decisions.

5. DeGF / Self-Correcting Decoding with Generative Feedback (2502.06130v2.pdf)
- Core idea: generated visual feedback + complementary/contrastive decoding.
- Reuse now: skip expensive text-to-image generation; keep only the contrastive decoding fallback for uncertain claims.
- Minimal implementation: when verifier confidence is low, decode with a contrastive penalty against language-prior-heavy tokens.

6. REVERSE (2504.13169v3.pdf)
- Core idea: hallucination-aware generation + retrospective resampling/backtracking.
- Reuse now: inference-time retrospective resampling only (no retraining).
- Minimal implementation: after initial answer, identify uncertain spans and resample only those spans under stricter verifier thresholds.

7. Sherlock (2505.22651v2.pdf)
- Core idea: trajectory-level self-correction and preference-driven improvement.
- Reuse now: trajectory-level edit, not full rewrite. Correct only erroneous spans/claims.
- Minimal implementation: enforce edit-locality in correction prompt (preserve unaffected clauses).

8. Aha Moment Revisited (2506.17417v3.pdf)
- Core finding: in VLMs, simple majority voting often beats verification-centric methods in visual math; self-verification gains are limited.
- Reuse now: add a cheap majority-vote branch as a safety baseline in uncertain cases.
- Minimal implementation: run k=3 short decodes only when verifier uncertainty > threshold and select majority at claim level.

9. CoRGI (2508.00378v1.pdf)
- Core idea: verify CoT steps with explicit visual grounding evidence.
- Reuse now: store evidence snippets/regions linked to corrected claims for interpretability.
- Minimal implementation: output structured evidence fields per claim (supporting object/attribute/relation evidence).

10. DST (2510.06107v1.pdf)
- Core idea: mechanistic tracing to locate commitment layers in hallucination.
- Reuse now: use as analysis framing, not immediate implementation.
- Minimal implementation: optional future probe study on your strongest/weakest examples.

11. VIB-Probe (2601.05547v1.pdf)
- Core idea: detect hallucination from internal attention-head signals via information bottleneck.
- Reuse now: emulate with lightweight uncertainty proxy first, postpone head-probe training.
- Minimal implementation: derive per-claim uncertainty from verifier disagreement + entropy-like token signals.

12. VCD (Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.pdf)
- Core idea: training-free visual contrastive decoding using original vs distorted visuals.
- Reuse now: add a togglable VCD-like decode mode for object-claim-heavy prompts.
- Minimal implementation: only apply for object existence/count claims to cap compute overhead.

13. ViperGPT (Suris_ViperGPT_Visual_Inference_via_Python_Execution_for_Reasoning_ICCV_2023_paper.pdf)
- Core idea: generate executable programs over visual tools.
- Reuse now: use selective tool calls for hard claim types (counting, spatial relation), not full agentic pipeline.
- Minimal implementation: add a tool-router for claim types and call deterministic utilities when uncertainty is high.

## Prioritized Integration Roadmap (Low Compute First)

P0 (Immediate, inference-time only)
- Add confidence-gated correction trigger.
- Add edit-locality prompt constraints (patch wrong spans only).
- Add majority-vote fallback for high-uncertainty samples.

P1 (Next)
- Add claim decomposition (predicate-level sub-claims).
- Add per-claim independent verification questions.
- Add evidence logging fields for interpretability.

P2 (Optional but high upside)
- Add contrastive decoding fallback for object claims.
- Add retrospective span resampling/backtracking.
- Add selective tool-router for counting/spatial claims.

P3 (Research extension)
- Probe-based detector (VIB-style) and mechanistic tracing (DST-style).

## Suggested New Experiments (Matched Protocol)

- E12: Confidence-Gated E1
  - Compare unconditional E1 vs gated E1 on same sample split.
  - Report corrected accuracy, flip-to-correct, flip-to-wrong, net gain, McNemar p-value.

- E13: E1 + Majority Fallback
  - Apply majority voting only on uncertain items.
  - Report compute overhead and net gain per extra decode.

- E14: E1 + Claim Decomposition
  - Verify atomic sub-claims independently before rewrite.
  - Report gains by claim type (existence, count, attribute, relation).

- E15: E1 + Contrastive Fallback
  - Trigger only for object-heavy uncertain cases.
  - Report gains and failure modes versus baseline E1.

## NeurIPS Positioning Notes

- Strong claim to make:
  - Compute-efficient, inference-time intervention with statistically significant gains over matched lightweight baselines.
- Claim to avoid:
  - Directly claiming SOTA over papers with different models/training data unless reproduced under matched protocol.
- Recommended framing:
  - "Matched in-house baselines + literature-positioned comparison with explicit protocol caveats."

## Final Recommendation

If only one next change is implemented, do this:
- Confidence-gated E1 with edit-locality + uncertainty-only majority fallback.

Reason:
- It is fully compatible with the current code, low engineering risk, and directly aligned with the strongest findings across CoVe, VOLCANO, REVERSE, and Aha-Revisited.