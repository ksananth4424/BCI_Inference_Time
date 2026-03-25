"""Config-driven experiment runner.

Usage:
    python scripts/run_experiment.py --config configs/phase1/qwen_gqa_full.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bci.runtime_config import get_pipeline_steps, load_config
from scripts.run_phase1 import get_step_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BCI experiment from config")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML/JSON experiment config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    steps = get_pipeline_steps(cfg)

    step_map = get_step_map()
    unknown = [s for s in steps if s not in step_map]
    if unknown:
        raise ValueError(f"Unknown steps in config: {unknown}")

    exp_name = cfg.get("experiment", {}).get("name", "unnamed_experiment")
    print(f"=== Running experiment: {exp_name} ===")
    print(f"Steps: {steps}")

    for step in steps:
        print(f"\n>>> Step: {step}")
        step_map[step]()

    print("\n=== Experiment complete ===")


if __name__ == "__main__":
    main()
