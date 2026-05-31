#!/usr/bin/env python3
"""Run TimesFM 2.5 parity feature-family ablations.

P3-4bo proved that the full LightGBM-equivalent feature bundle can be passed to
TimesFM 2.5 XReg, but it was worse than the previous energy-static baseline on
validation. P3-4bp keeps the same validation-only boundary and tests a compact
set of interpretable feature-family/ridge variants.

No test split is read. No supervised TimesFM fine-tuning is performed.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import random
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import timesfm
import torch

from energy_tsfm_context_feature_bundle import build_lightgbm_equivalent_bundle
from energy_tsfm_p2_core import (
    P1cWindowDataset,
    build_prediction_stub,
    load_window_index,
    serialize_series,
    validate_prediction_against_windows,
)
from evaluate_energy_tsfm_predictions import evaluate_prediction_frame, write_metrics
from run_p3_4an_timesfm2p5_h1_minimal_validation import (
    energy_static_covariates,
    manifest_positions,
    read_json,
)
from run_timesfm2p5_xreg_shared_dev import (
    DEFAULT_MODEL_NAME,
    DEFAULT_SEED,
    build_dynamic_calendar_covariates,
    load_timesfm2p5,
    numeric_target_history,
)


PROJECT = Path(__file__).resolve().parents[1]
PLAN_ID = "p3_4bp_timesfm2p5_parity_ablation_validation_v0_codex_20260517"
DEFAULT_OUT_ROOT = PROJECT / "results" / "energy_tsfm_tuning" / PLAN_ID
DEFAULT_SUBSET_MANIFEST = (
    PROJECT
    / "data"
    / "energy_tsfm_tuning"
    / "p3_4k_stride_tuning_policy_v0_codex_20260515"
    / "stride4"
    / "subset_manifest.json"
)
DEFAULT_RUN_ID = "p3_4bp_timesfm2p5_parity_ablation_smoke_aluminum_4h_val2_codex_20260517"
MODEL_ID = "timesfm2p5"
MODEL_FAMILY = "tsfm"
BASELINE_CONFIG_ID = "timesfm2p5_p3_4bp_baseline_energy_static_ridge1e_3"
SELECTION_ID = "p3_4bp_timesfm2p5_parity_ablation_shared_windows"
PARITY_BUNDLE_ID = "p3_4bn_lightgbm_equivalent_context_feature_bundle_v0_codex_20260516"


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    config_id: str
    scalar_families: tuple[str, ...]
    dynamic_families: tuple[str, ...]
    ridge: float
    xreg_mode: str = "xreg + timesfm"


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        variant_id="baseline_energy_static_ridge1e_3",
        config_id=BASELINE_CONFIG_ID,
        scalar_families=(),
        dynamic_families=(),
        ridge=1e-3,
    ),
    VariantSpec(
        variant_id="lead_only_ridge1e_3",
        config_id="timesfm2p5_p3_4bp_lead_only_ridge1e_3",
        scalar_families=(),
        dynamic_families=("lead_position",),
        ridge=1e-3,
    ),
    VariantSpec(
        variant_id="target_global_plus_lead_ridge1e_3",
        config_id="timesfm2p5_p3_4bp_target_global_plus_lead_ridge1e_3",
        scalar_families=("target_global",),
        dynamic_families=("lead_position",),
        ridge=1e-3,
    ),
    VariantSpec(
        variant_id="target_recent_plus_lead_ridge1e_3",
        config_id="timesfm2p5_p3_4bp_target_recent_plus_lead_ridge1e_3",
        scalar_families=("target_global", "target_lags", "target_rolling"),
        dynamic_families=("lead_position",),
        ridge=1e-3,
    ),
    VariantSpec(
        variant_id="target_recent_meta_origin_plus_lead_ridge1e_3",
        config_id="timesfm2p5_p3_4bp_target_recent_meta_origin_plus_lead_ridge1e_3",
        scalar_families=("metadata_origin", "target_global", "target_lags", "target_rolling"),
        dynamic_families=("lead_position",),
        ridge=1e-3,
    ),
    VariantSpec(
        variant_id="context_covariates_plus_lead_ridge1e_3",
        config_id="timesfm2p5_p3_4bp_context_covariates_plus_lead_ridge1e_3",
        scalar_families=("context_covariate_summaries",),
        dynamic_families=("lead_position",),
        ridge=1e-3,
    ),
    VariantSpec(
        variant_id="target_recent_plus_lead_ridge1e_2",
        config_id="timesfm2p5_p3_4bp_target_recent_plus_lead_ridge1e_2",
        scalar_families=("target_global", "target_lags", "target_rolling"),
        dynamic_families=("lead_position",),
        ridge=1e-2,
    ),
    VariantSpec(
        variant_id="full_parity_ridge1e_2",
        config_id="timesfm2p5_p3_4bp_full_parity_ridge1e_2",
        scalar_families=(
            "metadata_origin",
            "target_global",
            "target_lags",
            "target_rolling",
            "context_covariate_summaries",
        ),
        dynamic_families=("lead_position", "target_calendar"),
        ridge=1e-2,
    ),
    VariantSpec(
        variant_id="full_parity_ridge1e_1",
        config_id="timesfm2p5_p3_4bp_full_parity_ridge1e_1",
        scalar_families=(
            "metadata_origin",
            "target_global",
            "target_lags",
            "target_rolling",
            "context_covariate_summaries",
        ),
        dynamic_families=("lead_position", "target_calendar"),
        ridge=1e-1,
    ),
)


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def allowed_subset_manifest(subset_manifest: dict[str, Any]) -> bool:
    if subset_manifest.get("status") != "ok":
        return False
    if subset_manifest.get("stride") == 4:
        return True
    return (
        subset_manifest.get("subset_id") == "p3_target_pure_v0_codex_20260514"
        and subset_manifest.get("policy") == "target_pure_validation_test"
    )


def require_cuda(accelerator: str) -> dict[str, Any]:
    if str(accelerator).lower() != "cuda":
        raise ValueError("P3-4bp TimesFM parity ablation must use --accelerator cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; P3-4bp TimesFM parity ablation is GPU-only")
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


def torch_model_cuda_metadata(model: Any) -> dict[str, Any]:
    module = getattr(model, "model", model)
    if not hasattr(module, "parameters"):
        raise RuntimeError("TimesFM 2.5 torch route does not expose torch parameters for CUDA verification")
    total_params = 0
    cuda_params = 0
    devices: set[str] = set()
    dtypes: set[str] = set()
    for param in module.parameters():
        n_params = int(param.numel())
        total_params += n_params
        if param.is_cuda:
            cuda_params += n_params
        devices.add(str(param.device))
        dtypes.add(str(param.dtype))
    if total_params <= 0:
        raise RuntimeError("TimesFM 2.5 torch route exposes zero parameters")
    if cuda_params <= 0:
        raise RuntimeError(f"TimesFM 2.5 torch parameters are not on CUDA; devices={sorted(devices)}")
    return {
        "torch_model_class": type(module).__name__,
        "torch_parameter_count": int(total_params),
        "torch_cuda_parameter_count": int(cuda_params),
        "torch_parameter_devices": sorted(devices),
        "torch_parameter_dtypes": sorted(dtypes),
    }


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def batches_from_positions(domain: str, horizon: str, split: str, positions: list[int]) -> list[Any]:
    dataset = P1cWindowDataset(domain, horizon, split=split)
    return [dataset.get(pos) for pos in positions]


def quantile_columns(quantile_forecast: np.ndarray) -> dict[str, str]:
    if quantile_forecast.ndim != 2 or quantile_forecast.shape[1] < 10:
        return {}
    return {
        "q10": serialize_series(pd.Series(quantile_forecast[:, 1])),
        "q50": serialize_series(pd.Series(quantile_forecast[:, 5])),
        "q90": serialize_series(pd.Series(quantile_forecast[:, 9])),
    }


def scalar_family(name: str) -> str:
    if name in {"context_steps", "horizon_steps", "native_step_minutes"} or name.startswith("origin_"):
        return "metadata_origin"
    if name.startswith("target_lag_"):
        return "target_lags"
    if name.startswith("target_roll"):
        return "target_rolling"
    if name.startswith("target_"):
        return "target_global"
    if name.startswith("ctx_"):
        return "context_covariate_summaries"
    return "other"


def dynamic_family(name: str) -> str:
    if name in {"lead_step", "lead_fraction"}:
        return "lead_position"
    if name.startswith("target_"):
        return "target_calendar"
    return "other"


def selected_static_covariates(
    batches: list[Any],
    spec: VariantSpec,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    base_static, energy_feature_names = energy_static_covariates(batches)
    bundles = [build_lightgbm_equivalent_bundle(batch) for batch in batches]
    selected_names: list[str] = []
    for bundle in bundles[:1]:
        selected_names = [
            name for name in bundle.scalar_feature_names if scalar_family(name) in set(spec.scalar_families)
        ]
    covariates = dict(base_static)
    missing_indicator_count = 0
    for name in selected_names:
        values: list[float] = []
        missing: list[float] = []
        for bundle in bundles:
            value = float(bundle.scalar_features.get(name, np.nan))
            is_missing = not np.isfinite(value)
            values.append(0.0 if is_missing else value)
            missing.append(1.0 if is_missing else 0.0)
        covariates[f"parity_scalar_{name}"] = values
        if any(v > 0 for v in missing):
            covariates[f"parity_scalar_{name}_is_missing"] = missing
            missing_indicator_count += 1
    return covariates, {
        "energy_static_feature_count": int(len(energy_feature_names)),
        "parity_static_feature_count": int(len(covariates) - len(base_static)),
        "selected_scalar_families": list(spec.scalar_families),
        "selected_scalar_feature_count_before_missing_indicators": int(len(selected_names)),
        "missing_indicator_count": int(missing_indicator_count),
        "feature_bundle_hashes": [bundle.feature_bundle_hash for bundle in bundles],
    }


def selected_dynamic_covariates(
    batches: list[Any],
    spec: VariantSpec,
) -> tuple[dict[str, list[list[float]]], dict[str, Any]]:
    covariates = build_dynamic_calendar_covariates(batches)
    bundles = [build_lightgbm_equivalent_bundle(batch) for batch in batches]
    selected_names: list[str] = []
    for bundle in bundles[:1]:
        selected_names = [
            name for name in bundle.future_known_feature_names if dynamic_family(name) in set(spec.dynamic_families)
        ]
    for name in selected_names:
        key = f"parity_future_{name}"
        values_by_batch: list[list[float]] = []
        for batch, bundle in zip(batches, bundles, strict=True):
            future_values = pd.to_numeric(bundle.future_known_features[name], errors="raise").astype(float).to_list()
            values_by_batch.append([0.0] * len(batch.context) + future_values)
        covariates[key] = values_by_batch
    return covariates, {
        "base_dynamic_feature_count": int(len(covariates) - len(selected_names)),
        "parity_dynamic_feature_count": int(len(selected_names)),
        "selected_dynamic_families": list(spec.dynamic_families),
    }


def predict_variant(
    model: Any,
    batches: list[Any],
    *,
    spec: VariantSpec,
    batch_size: int,
    seed: int,
    split: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not batches:
        return pd.DataFrame(), {}
    horizon_steps = int(batches[0].metadata["horizon_steps"])
    feature_meta: dict[str, Any] = {
        "variant_id": spec.variant_id,
        "config_id": spec.config_id,
        "ridge": float(spec.ridge),
        "xreg_mode": spec.xreg_mode,
        "static_feature_counts": [],
        "dynamic_feature_counts": [],
        "feature_bundle_hash_count": 0,
    }
    all_hashes: set[str] = set()

    for start in range(0, len(batches), int(batch_size)):
        chunk = batches[start : start + int(batch_size)]
        static_covariates, static_meta = selected_static_covariates(chunk, spec)
        dynamic_covariates, dynamic_meta = selected_dynamic_covariates(chunk, spec)
        all_hashes.update(static_meta.get("feature_bundle_hashes", []))
        feature_meta["static_feature_counts"].append(len(static_covariates))
        feature_meta["dynamic_feature_counts"].append(len(dynamic_covariates))
        point_forecast, quantile_forecast = model.forecast_with_covariates(
            inputs=[numeric_target_history(batch.context).tolist() for batch in chunk],
            dynamic_numerical_covariates=dynamic_covariates,
            static_numerical_covariates=static_covariates,
            xreg_mode=spec.xreg_mode,
            normalize_xreg_target_per_input=True,
            ridge=float(spec.ridge),
            max_rows_per_col=0,
            force_on_cpu=False,
        )
        for batch, y_pred, q_pred in zip(chunk, np.asarray(point_forecast), np.asarray(quantile_forecast), strict=True):
            row = build_prediction_stub(
                batch,
                model_family=MODEL_FAMILY,
                model_id=MODEL_ID,
                config_id=spec.config_id,
                seed=seed,
                y_pred=pd.Series(np.asarray(y_pred, dtype=float)[:horizon_steps]),
                notes=(
                    f"P3-4bp TimesFM 2.5 parity ablation; variant={spec.variant_id}; "
                    f"scalar_families={','.join(spec.scalar_families) or 'none'}; "
                    f"dynamic_families={','.join(spec.dynamic_families) or 'none'}; "
                    f"ridge={spec.ridge:g}; no future measured covariates; split={split}."
                ),
            )
            row["split"] = split
            row["route_label"] = spec.variant_id
            row["parity_bundle_id"] = PARITY_BUNDLE_ID if spec.scalar_families or spec.dynamic_families else ""
            row.update(quantile_columns(np.asarray(q_pred, dtype=float)[:horizon_steps]))
            rows.append(row)
        feature_meta["last_static_meta"] = {k: v for k, v in static_meta.items() if k != "feature_bundle_hashes"}
        feature_meta["last_dynamic_meta"] = dynamic_meta

    feature_meta["static_feature_count_min"] = int(min(feature_meta["static_feature_counts"]))
    feature_meta["static_feature_count_max"] = int(max(feature_meta["static_feature_counts"]))
    feature_meta["dynamic_feature_count_min"] = int(min(feature_meta["dynamic_feature_counts"]))
    feature_meta["dynamic_feature_count_max"] = int(max(feature_meta["dynamic_feature_counts"]))
    feature_meta["feature_bundle_hash_count"] = int(len(all_hashes))
    feature_meta.pop("static_feature_counts", None)
    feature_meta.pop("dynamic_feature_counts", None)
    return pd.DataFrame(rows), feature_meta


def run_cell(
    args: argparse.Namespace,
    *,
    model: Any,
    domain: str,
    horizon: str,
    cell_index: int,
    subset_manifest: dict[str, Any],
    run_dir: Path,
    variants: tuple[VariantSpec, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.time()
    seed = int(DEFAULT_SEED + cell_index)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    eval_splits = tuple(args.eval_split or ("validation",))

    split_limits = {
        "validation": int(args.max_validation_windows),
        "test": int(args.max_test_windows),
    }
    eval_selections = {
        split: manifest_positions(
            subset_manifest,
            domain,
            horizon,
            split,
            limit=split_limits[split],
            selection_id=f"{args.selection_id}:{split}",
        )
        for split in eval_splits
    }
    eval_batches = {
        split: batches_from_positions(domain, horizon, split, selection["positions"])
        for split, selection in eval_selections.items()
    }
    cell_dir = run_dir / "cells" / f"{domain}_{horizon}"
    pred_dir = cell_dir / "predictions"
    metric_dir = cell_dir / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    variant_meta: list[dict[str, Any]] = []
    split_outputs: dict[str, Any] = {}
    split_prediction_frames: list[pd.DataFrame] = []
    for split, batches in eval_batches.items():
        prediction_frames: list[pd.DataFrame] = []
        for spec in variants:
            pred, meta = predict_variant(
                model,
                batches,
                spec=spec,
                batch_size=args.batch_size,
                seed=seed,
                split=split,
            )
            prediction_frames.append(pred)
            if split == "validation":
                variant_meta.append(meta)
        split_predictions = pd.concat(prediction_frames, ignore_index=True)
        window_index = load_window_index(domain, horizon, split=split)
        validate_prediction_against_windows(split_predictions.drop(columns=["split"], errors="ignore"), window_index)
        split_pred_path = pred_dir / f"{split}_predictions_all_variants.parquet"
        split_predictions.to_parquet(split_pred_path, index=False)
        split_metrics = evaluate_prediction_frame(split_predictions)
        split_metric_paths = write_metrics(split_metrics, metric_dir, stem=f"{split}_metrics")
        split_outputs[split] = {
            "selection": eval_selections[split],
            "prediction_rows": int(len(split_predictions)),
            "metric_rows": int(len(split_metrics)),
            "predictions": str(split_pred_path),
            "metrics": split_metric_paths,
        }
        split_prediction_frames.append(split_predictions)

    predictions = pd.concat(split_prediction_frames, ignore_index=True)
    pred_path = pred_dir / "requested_split_predictions_all_variants.parquet"
    predictions.to_parquet(pred_path, index=False)
    metrics = evaluate_prediction_frame(predictions)
    metric_paths = write_metrics(metrics, metric_dir, stem="requested_split_metrics")

    cell_manifest = {
        "status": "ok",
        "domain": domain,
        "horizon": horizon,
        "seed": seed,
        "selection_id": args.selection_id,
        "eval_splits": list(eval_splits),
        "eval_outputs": split_outputs,
        "validation_selection": split_outputs.get("validation", {}).get("selection"),
        "test_selection": split_outputs.get("test", {}).get("selection"),
        "variant_ids": [spec.variant_id for spec in variants],
        "variant_meta": variant_meta,
        "parity_bundle_id": PARITY_BUNDLE_ID,
        "prediction_rows": int(len(predictions)),
        "metric_rows": int(len(metrics)),
        "predictions": str(pred_path),
        "metrics": metric_paths,
        "test_predictions_generated": "test" in eval_splits,
        "important_boundary": "TimesFM 2.5 parity ablation runner; test split is only evaluated when explicitly requested by P5 executor.",
        "elapsed_sec": round(time.time() - started, 3),
    }
    (cell_dir / "cell_manifest.json").write_text(dumps(cell_manifest) + "\n", encoding="utf-8")
    return predictions, cell_manifest


def variant_subset(names: list[str] | None) -> tuple[VariantSpec, ...]:
    if not names:
        return VARIANTS
    lookup = {spec.variant_id: spec for spec in VARIANTS}
    missing = sorted(set(names) - set(lookup))
    if missing:
        raise ValueError(f"unknown variant ids: {missing}")
    return tuple(lookup[name] for name in names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", action="append", default=None)
    parser.add_argument("--horizon", action="append", default=None)
    parser.add_argument("--variant", action="append", default=None)
    parser.add_argument("--subset-manifest", type=Path, default=DEFAULT_SUBSET_MANIFEST)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-validation-windows", type=int, default=2)
    parser.add_argument("--max-test-windows", type=int, default=0)
    parser.add_argument("--eval-split", action="append", choices=["validation", "test"])
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument("--max-horizon", type=int, default=256)
    parser.add_argument("--per-core-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--selection-id", default=SELECTION_ID)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--accelerator", choices=["cuda"], default="cuda")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domains = args.domain or ["aluminum_load"]
    horizons = args.horizon or ["4h"]
    variants = variant_subset(args.variant)
    eval_splits = tuple(args.eval_split or ("validation",))
    started = time.time()
    torch.set_float32_matmul_precision("high")
    cuda = require_cuda(args.accelerator)
    torch.cuda.reset_peak_memory_stats()
    cuda_metadata_start = cuda_runtime_metadata()
    subset_manifest = read_json(args.subset_manifest)
    if not allowed_subset_manifest(subset_manifest):
        raise ValueError("P3-4bp expects an approved P3-4k stride4 or P3 target-pure subset manifest")
    run_dir = args.out_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_load_started = time.time()
    model = load_timesfm2p5(args.model_name, allow_download=bool(args.allow_download), torch_compile=bool(args.torch_compile))
    model_cuda_metadata = torch_model_cuda_metadata(model)
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
                variants=variants,
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
    combined_path = pred_dir / "validation_predictions_all_cells_all_variants.parquet"
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
        "route_role": "timesfm2p5_200m_parity_feature_family_ablation_validation",
        "run_dir": str(run_dir),
        "subset_manifest": str(args.subset_manifest),
        "subset_id": subset_manifest.get("subset_id"),
        "selection_id": args.selection_id,
        "domains": domains,
        "horizons": horizons,
        "eval_splits": list(eval_splits),
        "variant_ids": [spec.variant_id for spec in variants],
        "variant_specs": [dataclasses.asdict(spec) for spec in variants],
        "parity_bundle_id": PARITY_BUNDLE_ID,
        "max_train_windows": int(args.max_train_windows),
        "supervised_training_used": False,
        "max_validation_windows": int(args.max_validation_windows),
        "max_test_windows": int(args.max_test_windows),
        "forecast_config": dataclasses.asdict(model.forecast_config),
        "model_load_elapsed_sec": model_load_elapsed,
        "model_cuda_metadata": model_cuda_metadata,
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
        "test_predictions_generated": "test" in eval_splits,
        "test_artifacts_created": "test" in eval_splits,
        "important_boundary": "TimesFM 2.5 parity ablation runner; test split is only evaluated when explicitly requested by P5 executor. 200M route is not TimesFM-family 500M full-power route.",
        "elapsed_sec": round(time.time() - started, 3),
    }
    (run_dir / "manifest.json").write_text(dumps(manifest) + "\n", encoding="utf-8")
    print(dumps({k: manifest[k] for k in ["status", "plan_id", "run_dir", "prediction_rows", "metric_rows", "elapsed_sec", "device", "test_predictions_generated"]}))


if __name__ == "__main__":
    main()
