"""Backward-compatible entrypoint for Phase 1 pipeline.

Usage remains unchanged:
    python run_pipeline.py <step>

Internally delegates to scripts/run_phase1.py.
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "scripts" / "run_phase1.py"
    runpy.run_path(str(script), run_name="__main__")
