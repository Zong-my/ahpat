#!/usr/bin/env python3
"""Run P3-4an minimal TimesFM 2.5 H1 validation-only screen.

This is the TimesFM-side companion to P3-4am. It runs three validation-only
arms on the same representative windows:

- TimesFM 2.5 target-only zero-shot;
- TimesFM 2.5 ordinary calendar XReg;
- TimesFM 2.5 calendar XReg plus origin-time energy static covariates.

The Ours arm is a public-API fallback, not hidden-state adapter insertion and
not final H1 evidence. The test split is never accessed.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import timesfm
import torch

from energy_conditioned_adapter_ours import records_to_conditioning_batch
from energy_tsfm_p2_core import P1cWindowDataset, build_prediction_stub, load_window_index, serialize_series, validate_prediction_against_windows
from evaluate_energy_tsfm_predictions import evaluate_prediction_frame, write_metrics
from run_timesfm2p5_xreg_shared_dev import (
    DEFAULT_MODEL_NAME,
    DEFAULT_SEED,
    build_dynamic_calendar_covariates,
    load_timesfm2p5,
    numeric_target_history,
    quantile_columns,
)


PROJECT = Path(__file__).resolve().parents[1]
PLAN_ID = "p3_4an_timesfm2p5_h1_minimal_validation_v0_codex_20260516"
DEFAULT_OUT_ROOT = PROJECT / "results" / "energy_tsfm_tuning" / PLAN_ID
DEFAULT_SUBSET_MANIFEST = PROJECT / "data" / "energy_tsfm_tuning" / "p3_4k_stride_tuning_policy_v0_codex_20260515" / "stride4" / "subset_manifest.json"
DEFAULT_RUN_ID = "p3_4an_timesfm2p5_h1_minimal_val_aluminum_arena_4h_gpu_evidence_v1"
ZERO_CONFIG_ID = "timesfm2p5_zero_shot_h1_minimal_val_v0_codex_20260516"
XREG_CONFIG_ID = "timesfm2p5_xreg_ordinary_h1_minimal_val_v0_codex_20260516"
OURS_CONFIG_ID = "timesfm2p5_xreg_energy_conditioned_fallback_h1_minimal_val_v0_codex_20260516"
MODEL_FAMILY = "tsfm"
MODEL_ID = "timesfm2p5"
OURS_MODEL_FAMILY = "ours"
OURS_MODEL_ID = "energy_conditioned_adapter"
SELECTION_ID = "p3_4an_timesfm2p5_h1_minimal_shared_windows"
FALLBACK_TYPE = "xreg_static_energy_conditioning_fallback"
ARMS = [
    "timesfm2p5_zero_shot_h1_rep",
    "timesfm2p5_xreg_ordinary_h1_rep",
    "timesfm2p5_xreg_energy_adapter_ours_h1_rep",
]


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_indexes(n: int, limit: int, seed_key: str) -> list[int]:
    if n <= 0:
        return []
    if limit <= 0 or limit >= n:
        return list(range(n))
    anchor_order = [0, n - 1, n // 2]
    selected = set(anchor_order[: min(limit, len(anchor_order))])
    remaining = [idx for idx in range(n) if idx not in selected]
    rng = random.Random(seed_key)
    selected.update(rng.sample(remaining, k=min(max(0, limit - len(selected)), len(remaining))))
    return sorted(selected)


def manifest_positions(subset_manifest: dict[str, Any], domain: str, horizon: str, split: str, *, limit: int, selection_id: str) -> dict[str, Any]:
    split_payload = subset_manifest["subsets"][domain][horizon][split]
    positions = [int(value) for value in split_payload["positions"]]
    window_ids = [str(value) for value in split_payload["window_ids"]]
    indexes = sample_indexes(len(positions), limit, f"{selection_id}:{domain}:{horizon}:{split}")
    return {
        "positions": [positions[idx] for idx in indexes],
        "window_ids": [window_ids[idx] for idx in indexes],
        "manifest_indexes": indexes,
        "source_count": int(len(positions)),
        "selected_count": int(len(indexes)),
    }


def batches_from_positions(domain: str, horizon: str, split: str, positions: list[int]) -> list[Any]:
    dataset = P1cWindowDataset(domain, horizon, split=split)
    return [dataset.get(pos) for pos in positions]


def require_cuda(accelerator: str) -> dict[str, Any]:
    if str(accelerator).lower() != "cuda":
        raise ValueError("TimesFM 2.5 H1 minimal validation must use --accelerator cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; TimesFM 2.5 H1 minimal validation is GPU-only")
    index = torch.cuda.current_device()
    return {
        "cuda_required": True,
        "cuda_available": True,
        "device": "cuda",
        "cuda_device_index": int(index),
        "cuda_device_name": torch.cuda.get_device_name(index),
    }


def cuda_runtime_metadata() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    index = torch.cuda.current_device()
    return {
        "cuda_available": True,
        "cuda_device_index": int(index),
        "cuda_device_name": torch.cuda.get_device_name(index),
        "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(index)),
        "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
        "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(index)),
        "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
    }


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def domain_family_record(domain: str) -> dict[str, float]:
    return {
        "domain_family_load": 1.0 if "load" in domain else 0.0,
        "domain_family_pv": 1.0 if "pv" in domain else 0.0,
        "domain_family_industrial": 1.0 if "aluminum" in domain else 0.0,
        "domain_family_microgrid": 1.0 if "microgrid" in domain else 0.0,
        "domain_family_aidc": 1.0 if "aidc" in domain else 0.0,
    }


def condition_record(batch: Any) -> dict[str, float]:
    context_target = pd.to_numeric(batch.context["target"], errors="coerce").astype(float)
    context_target = context_target.ffill().bfill().fillna(0.0)
    values = context_target.to_numpy(dtype=np.float32)
    mean = float(np.mean(values)) if len(values) else 0.0
    std = float(np.std(values)) if len(values) else 0.0
    cv = float(std / (abs(mean) + 1e-6))
    recent_ramp = float(values[-1] - values[0]) if len(values) >= 2 else 0.0
    peak_concentration = float(np.max(np.abs(values)) / (np.sum(np.abs(values)) + 1e-6)) if len(values) else 0.0
    timestamps = pd.to_datetime(batch.target["timestamp"], errors="raise")
    hour = timestamps.dt.hour.astype(float) + timestamps.dt.minute.astype(float) / 60.0
    calendar_phase = float(np.mean(np.sin(2 * np.pi * hour / 24.0)))
    horizon = str(batch.metadata["horizon"])
    return {
        **domain_family_record(str(batch.metadata["domain_id"])),
        "horizon_4h": 1.0 if horizon == "4h" else 0.0,
        "horizon_24h": 1.0 if horizon == "24h" else 0.0,
        "native_resolution_minutes": float(batch.metadata["native_step_minutes"]),
        "context_length_steps": float(batch.metadata["context_steps"]),
        "horizon_steps": float(batch.metadata["horizon_steps"]),
        "context_mean": mean,
        "context_std": std,
        "context_cv": cv,
        "context_zero_fraction": float(np.mean(np.isclose(values, 0.0))) if len(values) else 0.0,
        "context_recent_ramp": recent_ramp,
        "context_peak_concentration": peak_concentration,
        "label_reliability": 1.0,
        "known_future_calendar_phase": calendar_phase,
    }


def energy_static_covariates(batches: list[Any]) -> tuple[dict[str, list[float]], tuple[str, ...]]:
    conditioning = records_to_conditioning_batch([condition_record(batch) for batch in batches])
    values = conditioning.values.detach().cpu().numpy()
    static: dict[str, list[float]] = {}
    for col_idx, name in enumerate(conditioning.feature_names):
        static[f"energy_{name}"] = [float(v) for v in values[:, col_idx]]
    return static, conditioning.feature_names


def build_row(
    batch: Any,
    *,
    y_pred: np.ndarray,
    quantiles: np.ndarray,
    config_id: str,
    model_family: str,
    model_id: str,
    route_label: str,
    adapter_fallback_type: str,
    notes: str,
) -> dict[str, Any]:
    row = build_prediction_stub(
        batch,
        model_family=model_family,
        model_id=model_id,
        config_id=config_id,
        seed=DEFAULT_SEED,
        y_pred=pd.Series(np.asarray(y_pred, dtype=float)),
        notes=notes,
    )
    row["split"] = "validation"
    row["route_label"] = route_label
    row["adapter_fallback_type"] = adapter_fallback_type
    row.update(quantile_columns(np.asarray(quantiles, dtype=float)))
    return row


def predict_zero_shot(model: Any, batches: list[Any], *, batch_size: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    horizon_steps = int(batches[0].metadata["horizon_steps"])
    for start in range(0, len(batches), int(batch_size)):
        chunk = batches[start : start + int(batch_size)]
        point_forecast, quantile_forecast = model.forecast(
            horizon=horizon_steps,
            inputs=[numeric_target_history(batch.context) for batch in chunk],
        )
        for batch, y_pred, q_pred in zip(chunk, np.asarray(point_forecast), np.asarray(quantile_forecast), strict=True):
            rows.append(
                build_row(
                    batch,
                    y_pred=np.asarray(y_pred, dtype=float)[:horizon_steps],
                    quantiles=np.asarray(q_pred, dtype=float)[:horizon_steps],
                    config_id=ZERO_CONFIG_ID,
                    model_family=MODEL_FAMILY,
                    model_id=MODEL_ID,
                    route_label="timesfm2p5_zero_shot_h1_rep",
                    adapter_fallback_type="",
                    notes="P3-4an TimesFM 2.5 target-only zero-shot validation; no future measured exogenous covariates.",
                )
            )
    return pd.DataFrame(rows)


def predict_xreg(
    model: Any,
    batches: list[Any],
    *,
    batch_size: int,
    xreg_mode: str,
    ridge: float,
    max_rows_per_col: int,
    use_energy_static: bool,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    rows: list[dict[str, Any]] = []
    horizon_steps = int(batches[0].metadata["horizon_steps"])
    feature_names: tuple[str, ...] = ()
    for start in range(0, len(batches), int(batch_size)):
        chunk = batches[start : start + int(batch_size)]
        static_covariates = None
        if use_energy_static:
            static_covariates, feature_names = energy_static_covariates(chunk)
        point_forecast, quantile_forecast = model.forecast_with_covariates(
            inputs=[numeric_target_history(batch.context).tolist() for batch in chunk],
            dynamic_numerical_covariates=build_dynamic_calendar_covariates(chunk),
            static_numerical_covariates=static_covariates,
            xreg_mode=xreg_mode,
            normalize_xreg_target_per_input=True,
            ridge=ridge,
            max_rows_per_col=max_rows_per_col,
            force_on_cpu=False,
        )
        for batch, y_pred, q_pred in zip(chunk, np.asarray(point_forecast), np.asarray(quantile_forecast), strict=True):
            if use_energy_static:
                rows.append(
                    build_row(
                        batch,
                        y_pred=np.asarray(y_pred, dtype=float)[:horizon_steps],
                        quantiles=np.asarray(q_pred, dtype=float)[:horizon_steps],
                        config_id=OURS_CONFIG_ID,
                        model_family=OURS_MODEL_FAMILY,
                        model_id=OURS_MODEL_ID,
                        route_label="timesfm2p5_xreg_energy_adapter_ours_h1_rep",
                        adapter_fallback_type=FALLBACK_TYPE,
                        notes=(
                            "P3-4an TimesFM 2.5 calendar XReg plus origin-time energy static covariates; "
                            "public-API fallback, not hidden-state adapter insertion and not final H1 evidence."
                        ),
                    )
                )
            else:
                rows.append(
                    build_row(
                        batch,
                        y_pred=np.asarray(y_pred, dtype=float)[:horizon_steps],
                        quantiles=np.asarray(q_pred, dtype=float)[:horizon_steps],
                        config_id=XREG_CONFIG_ID,
                        model_family=MODEL_FAMILY,
                        model_id=MODEL_ID,
                        route_label="timesfm2p5_xreg_ordinary_h1_rep",
                        adapter_fallback_type="",
                        notes=(
                            "P3-4an TimesFM 2.5 ordinary calendar XReg validation; "
                            f"xreg_mode={xreg_mode}; ridge={ridge:g}; no future measured exogenous covariates."
                        ),
                    )
                )
    return pd.DataFrame(rows), feature_names


def run_cell(args: argparse.Namespace, *, model: Any, domain: str, horizon: str, cell_index: int, subset_manifest: dict[str, Any], run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.time()
    seed = int(DEFAULT_SEED + cell_index)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    val_sel = manifest_positions(subset_manifest, domain, horizon, "validation", limit=args.max_validation_windows, selection_id=args.selection_id)
    val_batches = batches_from_positions(domain, horizon, "validation", val_sel["positions"])
    cell_dir = run_dir / "cells" / f"{domain}_{horizon}"
    pred_dir = cell_dir / "predictions"
    metric_dir = cell_dir / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    zero_val = predict_zero_shot(model, val_batches, batch_size=args.batch_size)
    xreg_val, _ = predict_xreg(
        model,
        val_batches,
        batch_size=args.batch_size,
        xreg_mode=args.xreg_mode,
        ridge=float(args.ridge),
        max_rows_per_col=int(args.max_rows_per_col),
        use_energy_static=False,
    )
    ours_val, feature_names = predict_xreg(
        model,
        val_batches,
        batch_size=args.batch_size,
        xreg_mode=args.xreg_mode,
        ridge=float(args.ridge),
        max_rows_per_col=int(args.max_rows_per_col),
        use_energy_static=True,
    )

    predictions = pd.concat([zero_val, xreg_val, ours_val], ignore_index=True)
    predictions["split"] = "validation"
    window_index = load_window_index(domain, horizon, split="validation")
    validate_prediction_against_windows(predictions.drop(columns=["split"], errors="ignore"), window_index)
    pred_path = pred_dir / "validation_predictions_all_arms.parquet"
    predictions.to_parquet(pred_path, index=False)
    metrics = evaluate_prediction_frame(predictions)
    metric_paths = write_metrics(metrics, metric_dir, stem="validation_metrics")

    cell_manifest = {
        "status": "ok",
        "domain": domain,
        "horizon": horizon,
        "seed": seed,
        "selection_id": args.selection_id,
        "validation_selection": val_sel,
        "validation_window_id_fingerprint": json.dumps(val_sel["window_ids"], sort_keys=True),
        "arms": ARMS,
        "zero_config_id": ZERO_CONFIG_ID,
        "xreg_config_id": XREG_CONFIG_ID,
        "ours_config_id": OURS_CONFIG_ID,
        "ordinary_adaptation_baseline": "timesfm2p5_xreg_or_plain_adaptation",
        "ours_fallback_type": FALLBACK_TYPE,
        "ours_hidden_state_insertion": False,
        "conditioning_feature_names": list(feature_names),
        "prediction_rows": int(len(predictions)),
        "metric_rows": int(len(metrics)),
        "predictions": str(pred_path),
        "metrics": metric_paths,
        "test_predictions_generated": False,
        "important_boundary": "P3-4an minimal validation-only cell; no test access and not final H1 evidence.",
        "elapsed_sec": round(time.time() - started, 3),
    }
    (cell_dir / "cell_manifest.json").write_text(dumps(cell_manifest) + "\n", encoding="utf-8")
    return predictions, cell_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", action="append", default=None)
    parser.add_argument("--horizon", action="append", default=None)
    parser.add_argument("--subset-manifest", type=Path, default=DEFAULT_SUBSET_MANIFEST)
    parser.add_argument("--max-validation-windows", type=int, default=2)
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument("--max-horizon", type=int, default=256)
    parser.add_argument("--per-core-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--selection-id", default=SELECTION_ID)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--accelerator", choices=["cuda"], default="cuda")
    parser.add_argument("--xreg-mode", choices=["xreg + timesfm", "timesfm + xreg"], default="xreg + timesfm")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--max-rows-per-col", type=int, default=0)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domains = args.domain or ["aluminum_load", "arena_pv"]
    horizons = args.horizon or ["4h"]
    started = time.time()
    torch.set_float32_matmul_precision("high")
    cuda = require_cuda(args.accelerator)
    torch.cuda.reset_peak_memory_stats()
    cuda_metadata_start = cuda_runtime_metadata()
    subset_manifest = read_json(args.subset_manifest)
    if subset_manifest.get("status") != "ok" or subset_manifest.get("stride") != 4:
        raise ValueError("P3-4an expects the approved P3-4k stride4 subset manifest")
    run_dir = args.out_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_load_started = time.time()
    model = load_timesfm2p5(args.model_name, allow_download=bool(args.allow_download), torch_compile=bool(args.torch_compile))
    forecast_config = timesfm.ForecastConfig(
        max_context=args.max_context,
        max_horizon=args.max_horizon,
        normalize_inputs=True,
        per_core_batch_size=args.per_core_batch_size,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
        return_backcast=True,
    )
    model.compile(forecast_config)
    model_load_elapsed = round(time.time() - model_load_started, 3)

    all_predictions: list[pd.DataFrame] = []
    cell_manifests: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for cell_index, (domain, horizon) in enumerate((d, h) for d in domains for h in horizons):
        try:
            predictions, cell_manifest = run_cell(
                args,
                model=model,
                domain=domain,
                horizon=horizon,
                cell_index=cell_index,
                subset_manifest=subset_manifest,
                run_dir=run_dir,
            )
            all_predictions.append(predictions)
            cell_manifests.append(cell_manifest)
            print(dumps({"status": "cell_ok", "domain": domain, "horizon": horizon}))
        except Exception as exc:
            failure = {
                "status": "cell_failed",
                "domain": domain,
                "horizon": horizon,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            print(dumps(failure))
            raise
        finally:
            release_cuda()

    predictions_all = pd.concat(all_predictions, ignore_index=True)
    pred_dir = run_dir / "predictions"
    metric_dir = run_dir / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    combined_path = pred_dir / "validation_predictions_all_cells_all_arms.parquet"
    predictions_all.to_parquet(combined_path, index=False)
    metrics = evaluate_prediction_frame(predictions_all)
    metric_paths = write_metrics(metrics, metric_dir, stem="validation_metrics_all_cells")
    cuda_metadata_end = cuda_runtime_metadata()
    manifest = {
        "status": "ok" if not failures else "failed",
        "plan_id": PLAN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "checkpoint_is_full_power": False,
        "route_role": "timesfm2p5_200m_newer_xreg_adaptation_variant_not_timesfm_family_largest_checkpoint",
        "run_dir": str(run_dir),
        "subset_manifest": str(args.subset_manifest),
        "subset_id": subset_manifest.get("subset_id"),
        "selection_id": args.selection_id,
        "domains": domains,
        "horizons": horizons,
        "eval_splits": ["validation"],
        "arms": ARMS,
        "ordinary_adaptation_baseline": "timesfm2p5_xreg_or_plain_adaptation",
        "ours_fallback_type": FALLBACK_TYPE,
        "ours_hidden_state_insertion": False,
        "ours_is_h1_sufficient_final_evidence": False,
        "max_validation_windows": int(args.max_validation_windows),
        "forecast_config": dataclasses.asdict(model.forecast_config),
        "xreg_mode": args.xreg_mode,
        "ridge": float(args.ridge),
        "max_rows_per_col": int(args.max_rows_per_col),
        "force_xreg_cpu": False,
        "model_load_elapsed_sec": model_load_elapsed,
        "prediction_rows": int(len(predictions_all)),
        "metric_rows": int(len(metrics)),
        "predictions_all": str(combined_path),
        "metrics": metric_paths,
        "cell_manifests": cell_manifests,
        "failures": failures,
        **cuda,
        "cuda_metadata_start": cuda_metadata_start,
        "cuda_metadata_end": cuda_metadata_end,
        "cuda_max_memory_allocated_bytes": int(cuda_metadata_end.get("cuda_max_memory_allocated_bytes", 0) or 0),
        "cuda_max_memory_reserved_bytes": int(cuda_metadata_end.get("cuda_max_memory_reserved_bytes", 0) or 0),
        "model_loading_launched": True,
        "real_project_window_data_read": True,
        "training_launched": False,
        "fine_tuning_launched": False,
        "xreg_adaptation_launched": True,
        "inference_launched": True,
        "forecast_metrics_computed": True,
        "prediction_artifact_saved": True,
        "test_predictions_generated": False,
        "test_artifacts_created": False,
        "important_boundary": "Minimal validation-only H1 path check; no test access, no final H1 claim, Ours arm is public-API XReg static-energy fallback.",
        "elapsed_sec": round(time.time() - started, 3),
    }
    (run_dir / "manifest.json").write_text(dumps(manifest) + "\n", encoding="utf-8")
    print(dumps(manifest))


if __name__ == "__main__":
    main()
