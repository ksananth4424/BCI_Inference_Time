"""Runtime config loading for experiment orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML experiment config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text())

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for .yaml configs. Install with: pip install pyyaml"
            ) from exc
        with open(path) as f:
            return yaml.safe_load(f)

    raise ValueError(f"Unsupported config extension: {suffix}")


def get_pipeline_steps(cfg: dict[str, Any]) -> list[str]:
    """Extract ordered pipeline steps from config."""
    steps = cfg.get("pipeline", {}).get("steps", [])
    if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
        raise ValueError("Config pipeline.steps must be a list of step names")
    return steps
