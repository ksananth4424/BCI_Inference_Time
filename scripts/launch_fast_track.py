"""Fast-track launcher for Phase 2 experiments.

Goal: eliminate human orchestration bottlenecks so compute is the only limiter.

What this script does:
1) Creates run-stamped output/log directories
2) Generates device-specific temporary configs from base configs
3) Launches E1, E3, and E10 in parallel as background processes
4) Writes a manifest JSON with command, PID, log path, and config path

Usage examples:
  python scripts/launch_fast_track.py
  python scripts/launch_fast_track.py --gpus 0,1,2 --samples 500 --profile balanced
  python scripts/launch_fast_track.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_yaml(path: Path) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _dump_yaml(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _make_temp_config(
    base_config: Path,
    out_config: Path,
    exp_suffix: str,
    device: str,
    n_samples: int,
    out_dir: Path,
) -> Dict:
    cfg = _load_yaml(base_config)

    # Keep experiment IDs unique per fast-track run.
    cfg["experiment"]["id"] = f"{cfg['experiment']['id']}_{exp_suffix}"
    cfg["dataset"]["n_samples"] = int(n_samples)
    cfg["model"]["device"] = device

    outputs = cfg.get("outputs", {})
    outputs["results_csv"] = str(out_dir / f"results_{cfg['experiment']['id']}.csv")
    outputs["summary_json"] = str(out_dir / f"summary_{cfg['experiment']['id']}.json")
    outputs["manifest_json"] = str(out_dir / f"manifest_{cfg['experiment']['id']}.json")
    cfg["outputs"] = outputs

    _dump_yaml(out_config, cfg)
    return cfg


def _launch(cmd: List[str], log_file: Path, dry_run: bool) -> Dict:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd_str = " ".join(cmd)

    if dry_run:
        return {
            "cmd": cmd_str,
            "pid": None,
            "log": str(log_file),
            "status": "dry-run",
        }

    fh = open(log_file, "w")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=fh,
        stderr=subprocess.STDOUT,
    )

    return {
        "cmd": cmd_str,
        "pid": proc.pid,
        "log": str(log_file),
        "status": "running",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch fast-track Phase 2 run set")
    parser.add_argument("--gpus", default="0,1,2", help="GPU indices for E1,E3,E10 (default: 0,1,2)")
    parser.add_argument("--samples", type=int, default=500, help="Sample count for E1/E3/E10 (default: 500)")
    parser.add_argument("--profile", default="balanced", choices=["strict", "balanced", "high_recall"], help="Verifier profile for E10")
    parser.add_argument("--dry-run", action="store_true", help="Print launch plan without starting jobs")
    args = parser.parse_args()

    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if len(gpu_list) < 3:
        raise ValueError("Need three GPU indices for parallel E1,E3,E10 (e.g., --gpus 0,1,2)")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_name = f"fasttrack_{timestamp}"
    run_root = PROJECT_ROOT / "results" / run_name
    logs_dir = run_root / "logs"
    cfg_dir = run_root / "configs"

    e1_base = PROJECT_ROOT / "configs" / "phase2" / "e1_gqa_replace_all.yaml"
    e3_base = PROJECT_ROOT / "configs" / "phase2" / "e3_gqa_remove_single.yaml"

    e1_cfg_path = cfg_dir / "e1.yaml"
    e3_cfg_path = cfg_dir / "e3.yaml"

    e1_cfg = _make_temp_config(
        base_config=e1_base,
        out_config=e1_cfg_path,
        exp_suffix=run_name,
        device=f"cuda:{gpu_list[0]}",
        n_samples=args.samples,
        out_dir=run_root,
    )
    e3_cfg = _make_temp_config(
        base_config=e3_base,
        out_config=e3_cfg_path,
        exp_suffix=run_name,
        device=f"cuda:{gpu_list[1]}",
        n_samples=args.samples,
        out_dir=run_root,
    )

    cmds = {
        "E1": [
            sys.executable,
            "scripts/run_phase2_experiment.py",
            "--exp",
            "E1",
            "--config",
            str(e1_cfg_path),
            "--output-dir",
            str(run_root),
        ],
        "E3": [
            sys.executable,
            "scripts/run_phase2_experiment.py",
            "--exp",
            "E3",
            "--config",
            str(e3_cfg_path),
            "--output-dir",
            str(run_root),
        ],
        "E10": [
            sys.executable,
            "scripts/audit_verifier.py",
            "--benchmark",
            "gqa",
            "--n-samples",
            str(args.samples),
            "--profile",
            args.profile,
            "--device",
            f"cuda:{gpu_list[2]}",
            "--output-dir",
            str(run_root),
        ],
    }

    launches = {}
    for name, cmd in cmds.items():
        launches[name] = _launch(cmd, logs_dir / f"{name}.log", args.dry_run)

    manifest = {
        "run_name": run_name,
        "created_utc": datetime.utcnow().isoformat(),
        "gpus": {
            "E1": gpu_list[0],
            "E3": gpu_list[1],
            "E10": gpu_list[2],
        },
        "samples": args.samples,
        "profile": args.profile,
        "configs": {
            "E1": str(e1_cfg_path),
            "E3": str(e3_cfg_path),
        },
        "launches": launches,
    }

    manifest_path = run_root / "launch_manifest.json"
    _write_json(manifest_path, manifest)

    print("=" * 72)
    print("FAST-TRACK LAUNCH PLAN")
    print("=" * 72)
    for name in ["E1", "E3", "E10"]:
        info = launches[name]
        print(f"{name}: {info['status']}")
        print(f"  PID: {info['pid']}")
        print(f"  LOG: {info['log']}")
    print(f"\nManifest: {manifest_path}")
    print(f"Run root: {run_root}")
    print("=" * 72)


if __name__ == "__main__":
    main()
