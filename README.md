# Inference-time VLM Reliability (BCI)

Belief-Constrained Inference (BCI) investigates whether Vision-Language Model failures are primarily caused by incorrect visual premises, and whether correcting those premises at inference-time improves answer reliability.

## Repository Layout

```
Inference_time_VLM/
├── src/
│   └── bci/
│       ├── config.py
│       ├── data/
│       │   └── data_loader.py
│       ├── models/
│       │   └── vlm_inference.py
│       ├── verification/
│       │   ├── claim_extraction.py
│       │   └── claim_verification.py
│       ├── analysis/
│       │   ├── error_classification.py
│       │   └── reporting.py
│       └── experiments/
│           └── phase1_experiments.py
├── scripts/
│   └── run_phase1.py
├── run_pipeline.py              # backward-compatible entrypoint
├── MIDTERM_PHASE1_REPORT.md
└── README.md
```

## Quick Start

Run from repository root:

```bash
python run_pipeline.py download
python run_pipeline.py setup
python run_pipeline.py experiment1
python run_pipeline.py analyze
python run_pipeline.py experiment2
python run_pipeline.py experiment4
python run_pipeline.py random_ablation
```

You can also call the script directly:

```bash
python scripts/run_phase1.py <step>
```

Config-driven execution:

```bash
python scripts/run_experiment.py --config configs/phase1/qwen_gqa_full.yaml
```

## Testing

Run deterministic unit tests:

```bash
python -m pytest -q
```

Current tests cover:
- answer normalization and error bucketing
- claim extraction and type classification
- basic claim verification behavior on toy scene graphs

## What Phase 1 Contains

- Dataset setup (GQA questions/images + scene graphs)
- Baseline VLM inference
- Belief externalization
- Claim verification against scene graphs
- Error decomposition (premise vs reasoning)
- Premise correction and minimality experiments
- Random ablation control

## Scaling Roadmap (NeurIPS-focused)

1. Add multi-benchmark evaluation (MMMU, MathVista, MM-Vet)
2. Add verifier robustness sweeps (noise injection and calibration)
3. Add stronger claim extraction and uncertainty modeling
4. Add post-hoc baseline parity runs (Woodpecker-style comparisons)
5. Add standardized experiment config + tracking (YAML + seeds + manifests)
6. Add tests for parser/verification modules and CI checks

## Notes

- `run_pipeline.py` remains available for compatibility.
- Core implementation now lives under `src/bci/` for cleaner scaling.
