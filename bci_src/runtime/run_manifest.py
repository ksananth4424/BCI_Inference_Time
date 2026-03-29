"""
Run Manifest System for Reproducibility.

Logs comprehensive metadata for every experiment run:
  - Git state (commit hash, branch)
  - Code state (hash of all Python files)
  - Config state (hash of experiment YAML)
  - Environment (Python version, CUDA, device info)
  - Timestamp and random seed
  - Dataset fingerprint (hash of sampled question IDs)

All metadata stored in JSON for archival and audit trail.
"""

import json
import subprocess
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import platform


class RunManifest:
    """
    Context manager for experiment reproducibility tracking.
    
    Usage:
        with RunManifest("E1_gqa_replace_all", "configs/phase2/e1.yaml") as manifest:
            # ... run experiment ...
            manifest.record_metric("flip_rate", 0.603)
            manifest.record_result(results_df)
    """
    
    def __init__(
        self,
        experiment_id: str,
        config_path: str,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize run manifest.
        
        Args:
            experiment_id: Human-readable ID (e.g., "E1_gqa_replace_all")
            config_path: Path to experiment YAML config
            output_dir: Where to save manifest JSON (default: results/)
        """
        self.experiment_id = experiment_id
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir or "results")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.start_time = datetime.utcnow()
        self.metrics: Dict[str, Any] = {}
        self.metadata = self._gather_metadata()
    
    def _gather_metadata(self) -> Dict[str, Any]:
        """Collect all reproducibility metadata."""
        return {
            "experiment_id": self.experiment_id,
            "start_time": self.start_time.isoformat(),
            "git": self._get_git_state(),
            "code": self._get_code_state(),
            "config": self._get_config_state(),
            "environment": self._get_environment_state(),
        }
    
    def _get_git_state(self) -> Dict[str, str]:
        """Capture git state (commit, branch, dirty status)."""
        try:
            project_root = Path(__file__).parent.parent.parent.parent  # Navigate back to project root
            
            # Get current commit hash
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            
            # Get current branch
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=project_root,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            
            # Check for uncommitted changes
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            is_dirty = len(status) > 0
            
            return {
                "commit": commit[:16],
                "branch": branch,
                "is_dirty": is_dirty,
                "dirty_file_count": len(status.split("\n")) if is_dirty else 0,
            }
        except Exception as e:
            return {
                "error": f"Failed to get git state: {str(e)}",
                "commit": "unknown",
                "branch": "unknown",
            }
    
    def _get_code_state(self) -> Dict[str, str]:
        """Hash all Python files in bci_src/ for version control."""
        try:
            src_dir = Path(__file__).parent.parent  # bci_src/
            h = hashlib.sha256()
            
            py_files = sorted(src_dir.rglob("*.py"))
            for py_file in py_files:
                h.update(py_file.read_bytes())
            
            code_hash = h.hexdigest()[:16]
            file_count = len(py_files)
            
            return {
                "hash": code_hash,
                "python_file_count": file_count,
                "src_dir": str(src_dir),
            }
        except Exception as e:
            return {"error": f"Failed to hash code: {str(e)}", "hash": "unknown"}
    
    def _get_config_state(self) -> Dict[str, str]:
        """Hash experiment config file."""
        try:
            if not self.config_path.exists():
                return {"error": f"Config file not found: {self.config_path}"}
            
            h = hashlib.sha256()
            h.update(self.config_path.read_bytes())
            config_hash = h.hexdigest()[:16]
            
            return {
                "path": str(self.config_path),
                "hash": config_hash,
                "size_bytes": self.config_path.stat().st_size,
            }
        except Exception as e:
            return {"error": f"Failed to hash config: {str(e)}"}
    
    def _get_environment_state(self) -> Dict[str, Any]:
        """Capture environment details (Python, CUDA, PyTorch, etc.)."""
        env = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "python_executable": sys.executable,
        }
        
        try:
            import torch
            env["torch_version"] = torch.__version__
            env["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                env["cuda_version"] = torch.version.cuda
                env["device_name"] = torch.cuda.get_device_name(0)
                env["device_count"] = torch.cuda.device_count()
        except ImportError:
            env["torch"] = "not installed"
        
        try:
            import transformers
            env["transformers_version"] = transformers.__version__
        except ImportError:
            env["transformers"] = "not installed"
        
        return env
    
    def record_metric(self, key: str, value: Any) -> None:
        """Record a named metric (e.g., 'flip_rate': 0.603)."""
        self.metrics[key] = value
    
    def record_result(self, label: str, data: Any) -> None:
        """Record complex result data (DataFrame, dict, etc.)."""
        self.metrics[f"result_{label}"] = data
    
    def finalize(self, summary: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save manifest to JSON and return path.
        
        Args:
            summary: Optional dict of high-level results to include
        
        Returns:
            Path to saved manifest JSON
        """
        end_time = datetime.utcnow()
        duration_seconds = (end_time - self.start_time).total_seconds()
        
        manifest = {
            **self.metadata,
            "end_time": end_time.isoformat(),
            "duration_seconds": duration_seconds,
            "metrics": self.metrics,
        }
        
        if summary:
            manifest["summary"] = summary
        
        # Use timestamp-based filename for uniqueness
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"run_{self.experiment_id}_{timestamp}_manifest.json"
        out_path = self.output_dir / filename
        
        # Write with pretty formatting for readability
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        
        return out_path
    
    def __enter__(self) -> "RunManifest":
        """Support context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Automatically finalize on context exit."""
        if exc_type is not None:
            self.metadata["error"] = {
                "type": exc_type.__name__,
                "message": str(exc_val),
            }
        # Finalize will be called explicitly by user for better control


class ManifestIndex:
    """
    Index of all run manifests for quick lookup and comparison.
    
    Usage:
        index = ManifestIndex("results/")
        index.load_all()
        e1_runs = index.find_by_experiment("E1_gqa_replace_all")
        index.compare_runs([run1, run2])
    """
    
    def __init__(self, manifest_dir: Path = None):
        self.manifest_dir = Path(manifest_dir or "results")
        self.manifests: Dict[str, Dict[str, Any]] = {}
    
    def load_all(self) -> None:
        """Load all manifest JSON files from directory."""
        for path in self.manifest_dir.glob("*_manifest.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                exp_id = data.get("experiment_id", "unknown")
                self.manifests[str(path)] = data
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
    
    def find_by_experiment(self, exp_id: str) -> list:
        """Find all runs for a given experiment ID."""
        return [
            m for m in self.manifests.values()
            if m.get("experiment_id") == exp_id
        ]
    
    def compare_runs(self, runs: list) -> Dict[str, Any]:
        """Compare metadata across multiple runs."""
        if not runs:
            return {}
        
        comparison = {
            "run_count": len(runs),
            "code_hashes_match": len(set(
                r.get("code", {}).get("hash") for r in runs
            )) == 1,
            "config_hashes": [r.get("config", {}).get("hash") for r in runs],
            "commits": [r.get("git", {}).get("commit") for r in runs],
        }
        
        return comparison
    
    def export_summary(self, output_path: Path) -> None:
        """Export index summary to JSON."""
        summary = {
            "total_runs": len(self.manifests),
            "experiments": {},
        }
        
        for path, manifest in self.manifests.items():
            exp_id = manifest.get("experiment_id", "unknown")
            if exp_id not in summary["experiments"]:
                summary["experiments"][exp_id] = []
            summary["experiments"][exp_id].append({
                "timestamp": manifest.get("start_time"),
                "duration_sec": manifest.get("duration_seconds"),
                "git_commit": manifest.get("git", {}).get("commit"),
            })
        
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
