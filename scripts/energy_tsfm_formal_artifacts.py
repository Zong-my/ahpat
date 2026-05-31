#!/usr/bin/env python3
"""Artifact helpers for formal energy-TSFM training runs.

This module centralizes the reproducibility contract used by formal runners:
stable run paths, JSON/JSONL logs, curve files, atomic checkpoints and
one-command inference scripts. It intentionally avoids destructive cleanup.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_FORMAL_ROOT = PROJECT / "results" / "energy_tsfm_formal"
PROJECT_PYTHON = Path(sys.executable)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(dumps_json(value) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, default=json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class FormalRunPaths:
    run_dir: Path

    @property
    def manifest(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def config(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def window_subsets(self) -> Path:
        return self.run_dir / "window_subsets.json"

    @property
    def train_log_jsonl(self) -> Path:
        return self.run_dir / "train_log.jsonl"

    @property
    def train_log_csv(self) -> Path:
        return self.run_dir / "train_log.csv"

    @property
    def stdout_log(self) -> Path:
        return self.run_dir / "train_stdout.log"

    @property
    def stderr_log(self) -> Path:
        return self.run_dir / "train_stderr.log"

    @property
    def curves_dir(self) -> Path:
        return self.run_dir / "curves"

    @property
    def loss_curve_csv(self) -> Path:
        return self.curves_dir / "loss_curve.csv"

    @property
    def loss_curve_png(self) -> Path:
        return self.curves_dir / "loss_curve_latest.png"

    @property
    def metric_curve_png(self) -> Path:
        return self.curves_dir / "metric_curve_latest.png"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def best_checkpoint(self) -> Path:
        return self.checkpoints_dir / "best.pt"

    @property
    def last_checkpoint(self) -> Path:
        return self.checkpoints_dir / "last.pt"

    @property
    def previous_checkpoint(self) -> Path:
        return self.checkpoints_dir / "previous.pt"

    @property
    def predictions_dir(self) -> Path:
        return self.run_dir / "predictions"

    @property
    def validation_predictions(self) -> Path:
        return self.predictions_dir / "validation_predictions.parquet"

    @property
    def test_predictions(self) -> Path:
        return self.predictions_dir / "test_predictions.parquet"

    @property
    def metrics_dir(self) -> Path:
        return self.run_dir / "metrics"

    @property
    def validation_metrics_csv(self) -> Path:
        return self.metrics_dir / "validation_metrics.csv"

    @property
    def validation_metrics_json(self) -> Path:
        return self.metrics_dir / "validation_metrics.json"

    @property
    def test_metrics_csv(self) -> Path:
        return self.metrics_dir / "test_metrics.csv"

    @property
    def test_metrics_json(self) -> Path:
        return self.metrics_dir / "test_metrics.json"

    @property
    def inference_dir(self) -> Path:
        return self.run_dir / "inference"

    @property
    def inference_script(self) -> Path:
        return self.inference_dir / "run_inference.sh"

    @property
    def inference_config(self) -> Path:
        return self.inference_dir / "inference_config.json"

    @property
    def failure_status(self) -> Path:
        return self.run_dir / "failure_status.json"


def make_run_paths(
    *,
    output_root: Path,
    model_id: str,
    domain_id: str,
    horizon: str,
    config_id: str,
    seed: int,
) -> FormalRunPaths:
    safe_horizon = horizon.replace("/", "_")
    return FormalRunPaths(
        run_dir=output_root / model_id / domain_id / safe_horizon / config_id / f"seed{seed}"
    )


def ensure_formal_run_dirs(paths: FormalRunPaths, *, resume: bool) -> None:
    if paths.run_dir.exists() and not resume:
        raise FileExistsError(
            f"formal run directory already exists: {paths.run_dir}. "
            "Use --resume to continue the run or choose a new --config-id."
        )
    for directory in [
        paths.run_dir,
        paths.curves_dir,
        paths.checkpoints_dir,
        paths.predictions_dir,
        paths.metrics_dir,
        paths.inference_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    paths.stdout_log.touch(exist_ok=True)
    paths.stderr_log.touch(exist_ok=True)


def log_stdout(paths: FormalRunPaths, message: str) -> None:
    print(message, flush=True)
    with paths.stdout_log.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def log_stderr(paths: FormalRunPaths, message: str) -> None:
    with paths.stderr_log.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def update_curve_artifacts(
    paths: FormalRunPaths,
    rows: list[dict[str, Any]],
    *,
    primary_metric_col: str = "validation_primary_wape",
) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    paths.curves_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.train_log_csv, index=False)
    df.to_csv(paths.loss_curve_csv, index=False)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - only used on minimal envs
        log_stderr(paths, f"matplotlib unavailable; skipped curve PNGs: {exc}")
        return

    if "epoch" not in df.columns:
        return

    plt.figure(figsize=(7, 4))
    if "train_loss" in df.columns:
        plt.plot(df["epoch"], df["train_loss"], marker="o", label="train_loss")
    if "validation_loss" in df.columns:
        plt.plot(df["epoch"], df["validation_loss"], marker="s", label="validation_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.loss_curve_png, dpi=160)
    plt.close()

    if primary_metric_col in df.columns:
        plt.figure(figsize=(7, 4))
        plt.plot(df["epoch"], df[primary_metric_col], marker="o", label=primary_metric_col)
        plt.xlabel("epoch")
        plt.ylabel(primary_metric_col)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(paths.metric_curve_png, dpi=160)
        plt.close()


def package_versions() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def cuda_metadata(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_max_memory_allocated_bytes": None,
        }
    return {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def write_inference_entrypoint(
    paths: FormalRunPaths,
    *,
    runner_script: Path,
    python_path: Path = PROJECT_PYTHON,
) -> None:
    paths.inference_dir.mkdir(parents=True, exist_ok=True)
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f'"{python_path}" "{runner_script}" --inference-only --run-dir "{paths.run_dir}"',
            "",
        ]
    )
    paths.inference_script.write_text(script, encoding="utf-8")
    paths.inference_script.chmod(0o755)
    write_json_atomic(
        paths.inference_config,
        {
            "runner_script": str(runner_script),
            "python": str(python_path),
            "run_dir": str(paths.run_dir),
            "checkpoint": str(paths.best_checkpoint),
            "command": str(paths.inference_script),
        },
    )
