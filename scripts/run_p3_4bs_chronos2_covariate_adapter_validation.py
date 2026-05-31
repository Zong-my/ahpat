#!/usr/bin/env python3
"""Run P3-4bs Chronos-2 covariate-aware hidden adapter validation.

This branch extends the P3-4bf hidden adapter by allowing P3-4bn
LightGBM-equivalent context-only feature bundles to enter the adapter
condition vector. It is a validation-only hardening branch; the installed
Chronos package is not modified.

Boundary: validation-only screen, no test access. Full-train formal evidence
must be generated later before main-result claims.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from einops import rearrange
from torch import nn

from energy_conditioned_adapter_ours import EnergyConditionedAdapter, records_to_conditioning_batch
from energy_tsfm_context_feature_bundle import (
    ROLLING_WINDOWS,
    TARGET_LAGS,
    calendar_features,
    context_covariate_features,
    lightgbm_scalar_features,
    numeric_context_columns,
    safe_feature_name,
    target_history_features,
)
from energy_tsfm_p2_core import (
    P1cWindowDataset,
    build_prediction_stub,
    load_canonical,
    load_window_index,
    serialize_series,
    validate_prediction_against_windows,
)
from evaluate_energy_tsfm_predictions import evaluate_prediction_frame, write_metrics
from run_chronos2_lora_preflight import DEFAULT_MODEL_NAME, DEFAULT_SEED, clean_numeric


PROJECT = Path(__file__).resolve().parents[1]
PLAN_ID = "p3_4bs_chronos2_covariate_adapter_validation_v0_codex_20260517"
DEFAULT_OUT_ROOT = PROJECT / "results" / "energy_tsfm_tuning" / PLAN_ID
DEFAULT_SUBSET_MANIFEST = (
    PROJECT
    / "data"
    / "energy_tsfm_tuning"
    / "p3_4k_stride_tuning_policy_v0_codex_20260515"
    / "stride4"
    / "subset_manifest.json"
)
DEFAULT_RUN_ID = "p3_4bs_chronos2_covariate_adapter_val_5domain_2h_val128_target_recent_codex_20260517"
MODEL_ID = "chronos2"
MODEL_FAMILY = "tsfm"
BASE_CONFIG_ID = "chronos2_frozen_target_only_h1_p3_4bs_v0_codex_20260517"
ADAPTER_CONFIG_ID = "chronos2_covariate_aware_hidden_adapter_h1_p3_4bs_v0_codex_20260517"
ROUTE_BASE = "chronos2_frozen_target_only_h1_p3_4bs"
ROUTE_ADAPTER = "chronos2_covariate_aware_hidden_adapter_h1_p3_4bs"
SELECTION_ID = "p3_4bs_chronos2_covariate_adapter_shared_windows"
EPS = 1e-12
EXTRA_CALENDAR_CONDITIONING_FEATURES: tuple[str, ...] = (
    "future_hour_sin_mean",
    "future_hour_cos_mean",
    "future_doy_sin_mean",
    "future_doy_cos_mean",
    "future_weekend_fraction",
    "future_daylight_fraction_proxy",
)
FUTURE_STEP_CONDITION_FEATURES: tuple[str, ...] = (
    "future_hour_sin",
    "future_hour_cos",
    "future_dow_sin",
    "future_dow_cos",
    "future_doy_sin",
    "future_doy_cos",
    "future_is_weekend",
    "future_daylight_proxy",
    "future_lead_fraction",
)
CONDITION_FEATURE_MODES: tuple[str, ...] = (
    "energy_base",
    "target_recent",
    "target_recent_context_covariates",
    "full_scalar",
)


class ConditionNormalizer:
    """Train-split z-score normalizer for origin-time conditioning tensors."""

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor, feature_names: tuple[str, ...]) -> None:
        self.mean = mean.detach().float().cpu()
        self.scale = scale.detach().float().cpu()
        self.feature_names = tuple(feature_names)

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        scale = self.scale.to(device=values.device, dtype=values.dtype)
        return (values - mean) / scale

    def summary(self) -> dict[str, Any]:
        return {
            "kind": "train_zscore",
            "feature_names": list(self.feature_names),
            "mean": [float(v) for v in self.mean.view(-1).numpy().tolist()],
            "scale": [float(v) for v in self.scale.view(-1).numpy().tolist()],
        }


@dataclass
class ChronosWindowCache:
    """Cell-local cached arrays for repeated Chronos train/eval passes."""

    batches: list[Any]
    context_np: np.ndarray
    future_target_np: np.ndarray
    condition_records: list[dict[str, float]]
    metadata: pd.DataFrame | None = None
    condition_values_np: np.ndarray | None = None
    future_step_condition_np: np.ndarray | None = None
    condition_feature_names: tuple[str, ...] = ()
    future_step_condition_feature_names: tuple[str, ...] = ()

    def __len__(self) -> int:
        return int(self.context_np.shape[0])


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def allowed_subset_manifest(subset_manifest: dict[str, Any]) -> bool:
    if subset_manifest.get("status") != "ok":
        return False
    if subset_manifest.get("stride") == 4:
        return True
    return (
        subset_manifest.get("subset_id") == "p3_target_pure_v0_codex_20260514"
        and subset_manifest.get("policy") == "target_pure_validation_test"
    )


def require_cuda(device_map: str) -> dict[str, Any]:
    if str(device_map).lower() != "cuda":
        raise ValueError("P3-4bs Chronos-2 covariate-aware hidden adapter branch must use --device-map cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; Chronos-2 covariate-aware hidden adapter branch is GPU-only")
    index = torch.cuda.current_device()
    return {
        "cuda_required": True,
        "cuda_available": True,
        "device": "cuda",
        "cuda_device_index": int(index),
        "cuda_device_name": torch.cuda.get_device_name(index),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sample_indexes(n: int, limit: int, seed_key: str) -> list[int]:
    if n <= 0:
        return []
    if limit <= 0 or limit >= n:
        return list(range(n))
    anchors = [0, n - 1, n // 2]
    selected = set(anchors[: min(limit, len(anchors))])
    remaining = [idx for idx in range(n) if idx not in selected]
    rng = random.Random(seed_key)
    selected.update(rng.sample(remaining, k=min(max(0, limit - len(selected)), len(remaining))))
    return sorted(selected)


def manifest_positions(
    subset_manifest: dict[str, Any],
    *,
    domain: str,
    horizon: str,
    split: str,
    limit: int,
    selection_id: str,
) -> dict[str, Any]:
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


def batches_from_positions(
    domain: str,
    horizon: str,
    split: str,
    positions: list[int],
    *,
    workers: int = 1,
) -> list[Any]:
    dataset = P1cWindowDataset(domain, horizon, split=split)
    if workers <= 1 or len(positions) <= 1:
        return [dataset.get(pos) for pos in positions]
    with ThreadPoolExecutor(max_workers=min(int(workers), len(positions))) as pool:
        return list(pool.map(dataset.get, positions))


def domain_family_record(domain: str) -> dict[str, float]:
    return {
        "domain_family_load": 1.0 if "load" in domain else 0.0,
        "domain_family_pv": 1.0 if "pv" in domain else 0.0,
        "domain_family_industrial": 1.0 if "aluminum" in domain else 0.0,
        "domain_family_microgrid": 1.0 if "microgrid" in domain else 0.0,
        "domain_family_aidc": 1.0 if "aidc" in domain else 0.0,
    }


def parity_scalar_family(name: str) -> str:
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


def fast_context_covariate_features(context: pd.DataFrame, numeric_cols: tuple[str, ...]) -> dict[str, float]:
    feats: dict[str, float] = {}
    for col in numeric_cols:
        if col == "target":
            continue
        values = context[col].to_numpy(dtype=float, copy=False)
        if np.isfinite(values).sum() == 0:
            continue
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in str(col))
        last = float(values[-1]) if np.isfinite(values[-1]) else np.nan
        feats[f"ctx_{safe_name}_last"] = last
        feats[f"ctx_{safe_name}_mean"] = float(np.nanmean(values))
        feats[f"ctx_{safe_name}_std"] = float(np.nanstd(values))
    return feats


def fast_lightgbm_scalar_features(batch: Any, numeric_cols: tuple[str, ...]) -> dict[str, float]:
    row = batch.metadata
    return {
        "context_steps": float(row["context_steps"]),
        "horizon_steps": float(row["horizon_steps"]),
        "native_step_minutes": float(row["native_step_minutes"]),
        **calendar_features(batch.origin_time, "origin"),
        **target_history_features(batch.context),
        **fast_context_covariate_features(batch.context, numeric_cols),
    }


def selected_parity_condition_features(
    batch: Any,
    mode: str,
    *,
    numeric_context_cols: tuple[str, ...] | None = None,
) -> dict[str, float]:
    if mode == "energy_base":
        return {}
    if mode == "target_recent":
        families = {"target_global", "target_lags", "target_rolling"}
        scalar_features = target_history_features(batch.context)
    elif mode == "target_recent_context_covariates":
        families = {"target_global", "target_lags", "target_rolling", "context_covariate_summaries"}
        scalar_features = {
            **target_history_features(batch.context),
            **(
                fast_context_covariate_features(batch.context, numeric_context_cols)
                if numeric_context_cols is not None
                else context_covariate_features(batch.context)
            ),
        }
    elif mode == "full_scalar":
        families = {
            "metadata_origin",
            "target_global",
            "target_lags",
            "target_rolling",
            "context_covariate_summaries",
        }
        scalar_features = (
            fast_lightgbm_scalar_features(batch, numeric_context_cols)
            if numeric_context_cols is not None
            else lightgbm_scalar_features(batch)
        )
    else:
        raise ValueError(f"unsupported condition feature mode: {mode}")

    out: dict[str, float] = {}
    for name, value in scalar_features.items():
        if parity_scalar_family(name) not in families:
            continue
        safe_value = float(value)
        is_missing = not np.isfinite(safe_value)
        out[f"parity_{name}"] = 0.0 if is_missing else safe_value
        out[f"parity_{name}_is_missing"] = 1.0 if is_missing else 0.0
    return out


def finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def condition_record(
    batch: Any,
    *,
    condition_feature_mode: str,
    numeric_context_cols: tuple[str, ...] | None = None,
) -> dict[str, float]:
    context_target = pd.to_numeric(batch.context["target"], errors="coerce").astype(float)
    context_target = context_target.ffill().bfill().fillna(0.0)
    values = context_target.to_numpy(dtype=np.float32)
    mean = float(np.mean(values)) if len(values) else 0.0
    std = float(np.std(values)) if len(values) else 0.0
    timestamps = pd.to_datetime(batch.target["timestamp"], errors="raise")
    hour = timestamps.dt.hour.astype(float) + timestamps.dt.minute.astype(float) / 60.0
    doy = timestamps.dt.dayofyear.astype(float)
    dow = timestamps.dt.dayofweek.astype(float)
    horizon = str(batch.metadata["horizon"])
    base = {
        **domain_family_record(str(batch.metadata["domain_id"])),
        "horizon_4h": 1.0 if horizon == "4h" else 0.0,
        "horizon_24h": 1.0 if horizon == "24h" else 0.0,
        "native_resolution_minutes": float(batch.metadata["native_step_minutes"]),
        "context_length_steps": float(batch.metadata["context_steps"]),
        "horizon_steps": float(batch.metadata["horizon_steps"]),
        "context_mean": mean,
        "context_std": std,
        "context_cv": float(std / (abs(mean) + 1e-6)),
        "context_zero_fraction": float(np.mean(np.isclose(values, 0.0))) if len(values) else 0.0,
        "context_recent_ramp": float(values[-1] - values[0]) if len(values) >= 2 else 0.0,
        "context_peak_concentration": float(np.max(np.abs(values)) / (np.sum(np.abs(values)) + 1e-6)) if len(values) else 0.0,
        "label_reliability": 1.0,
        "known_future_calendar_phase": float(np.mean(np.sin(2 * np.pi * hour / 24.0))),
        "future_hour_sin_mean": float(np.mean(np.sin(2 * np.pi * hour / 24.0))),
        "future_hour_cos_mean": float(np.mean(np.cos(2 * np.pi * hour / 24.0))),
        "future_doy_sin_mean": float(np.mean(np.sin(2 * np.pi * doy / 366.0))),
        "future_doy_cos_mean": float(np.mean(np.cos(2 * np.pi * doy / 366.0))),
        "future_weekend_fraction": float(np.mean(dow >= 5.0)),
        "future_daylight_fraction_proxy": float(np.mean((hour >= 6.0) & (hour <= 18.0))),
    }
    base.update(
        selected_parity_condition_features(
            batch,
            condition_feature_mode,
            numeric_context_cols=numeric_context_cols,
        )
    )
    return base


def p3_4bf_condition_tensor(
    records: list[dict[str, float]],
    *,
    device: torch.device | None = None,
    extended_calendar_condition: bool,
    feature_names: tuple[str, ...] | None = None,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    base = records_to_conditioning_batch(records, device=device)
    if feature_names is None:
        reserved = set(base.feature_names) | set(EXTRA_CALENDAR_CONDITIONING_FEATURES)
        parity_names = tuple(sorted({name for record in records for name in record if name.startswith("parity_") and name not in reserved}))
        feature_names = tuple(base.feature_names) + parity_names
        if extended_calendar_condition:
            feature_names = feature_names + EXTRA_CALENDAR_CONDITIONING_FEATURES
    values = torch.tensor(
        [[finite_float(record.get(name, 0.0)) for name in feature_names] for record in records],
        dtype=torch.float32,
        device=device,
    )
    return values, tuple(feature_names)


def fit_condition_normalizer(
    batches: list[Any],
    *,
    mode: str,
    extended_calendar_condition: bool,
    condition_feature_mode: str,
) -> ConditionNormalizer | None:
    if mode == "none":
        return None
    if mode != "train_zscore":
        raise ValueError(f"unsupported condition standardization mode: {mode}")
    values, feature_names = p3_4bf_condition_tensor(
        [condition_record(batch, condition_feature_mode=condition_feature_mode) for batch in batches],
        extended_calendar_condition=extended_calendar_condition,
    )
    values = values.float()
    mean = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, unbiased=False, keepdim=True)
    scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
    return ConditionNormalizer(mean=mean, scale=scale, feature_names=feature_names)


def tensor_batch_from_batches(
    batches: list[Any],
    device: torch.device,
    *,
    condition_normalizer: ConditionNormalizer | None = None,
    extended_calendar_condition: bool,
    condition_feature_mode: str,
) -> dict[str, Any]:
    contexts = [clean_numeric(batch.context["target"]) for batch in batches]
    targets = [clean_numeric(batch.target["target"]) for batch in batches]
    future_timestamps = np.stack(
        [pd.to_datetime(batch.target["timestamp"], errors="raise").to_numpy() for batch in batches]
    )
    condition_values, condition_feature_names = p3_4bf_condition_tensor(
        [condition_record(batch, condition_feature_mode=condition_feature_mode) for batch in batches],
        device=device,
        extended_calendar_condition=extended_calendar_condition,
        feature_names=condition_normalizer.feature_names if condition_normalizer is not None else None,
    )
    if condition_normalizer is not None:
        condition_values = condition_normalizer.apply(condition_values)
    return {
        "context": torch.tensor(np.stack(contexts), dtype=torch.float32, device=device),
        "future_target": torch.tensor(np.stack(targets), dtype=torch.float32, device=device),
        "condition_values": condition_values,
        "condition_feature_names": condition_feature_names,
        "future_step_conditions": torch.tensor(
            _future_step_condition_array(future_timestamps),
            dtype=torch.float32,
            device=device,
        ),
        "future_step_condition_feature_names": FUTURE_STEP_CONDITION_FEATURES,
    }


def build_chronos_window_cache(
    batches: list[Any],
    *,
    condition_feature_mode: str,
    keep_batches: bool,
    workers: int = 1,
) -> ChronosWindowCache:
    numeric_cols: tuple[str, ...] | None = None
    if batches and condition_feature_mode in {"target_recent_context_covariates", "full_scalar"}:
        numeric_cols = tuple(numeric_context_columns(batches[0].context))

    def encode_one(batch: Any) -> tuple[np.ndarray, np.ndarray, dict[str, float], np.ndarray]:
        future_timestamps = pd.to_datetime(batch.target["timestamp"], errors="raise").to_numpy()[None, :]
        return (
            clean_numeric(batch.context["target"]),
            clean_numeric(batch.target["target"]),
            condition_record(
                batch,
                condition_feature_mode=condition_feature_mode,
                numeric_context_cols=numeric_cols,
            ),
            _future_step_condition_array(future_timestamps)[0],
        )

    if workers <= 1 or len(batches) <= 1:
        encoded = [encode_one(batch) for batch in batches]
    else:
        with ThreadPoolExecutor(max_workers=min(int(workers), len(batches))) as pool:
            encoded = list(pool.map(encode_one, batches))
    contexts = [item[0] for item in encoded]
    targets = [item[1] for item in encoded]
    records = [item[2] for item in encoded]
    future_step_conditions = [item[3] for item in encoded]
    return ChronosWindowCache(
        batches=list(batches) if keep_batches else [],
        context_np=np.stack(contexts).astype(np.float32, copy=False),
        future_target_np=np.stack(targets).astype(np.float32, copy=False),
        condition_records=records,
        future_step_condition_np=np.stack(future_step_conditions).astype(np.float32, copy=False),
        future_step_condition_feature_names=FUTURE_STEP_CONDITION_FEATURES,
    )


def _clean_2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if np.isfinite(values).all():
        return np.array(values, dtype=np.float32, copy=True)
    cleaned = np.empty(values.shape, dtype=np.float32)
    for idx, row in enumerate(values):
        cleaned[idx] = clean_numeric(pd.Series(row))
    return cleaned


def _calendar_feature_arrays(timestamps: Any, prefix: str) -> dict[str, np.ndarray]:
    ts = pd.to_datetime(pd.Series(timestamps), errors="raise")
    hour = ts.dt.hour.astype(float) + ts.dt.minute.astype(float) / 60.0
    dayofweek = ts.dt.dayofweek.astype(float)
    month = ts.dt.month.astype(float)
    dayofyear = ts.dt.dayofyear.astype(float)
    return {
        f"{prefix}_hour_sin": np.sin(2 * np.pi * hour / 24.0).to_numpy(dtype=np.float64),
        f"{prefix}_hour_cos": np.cos(2 * np.pi * hour / 24.0).to_numpy(dtype=np.float64),
        f"{prefix}_dow_sin": np.sin(2 * np.pi * dayofweek / 7.0).to_numpy(dtype=np.float64),
        f"{prefix}_dow_cos": np.cos(2 * np.pi * dayofweek / 7.0).to_numpy(dtype=np.float64),
        f"{prefix}_month_sin": np.sin(2 * np.pi * month / 12.0).to_numpy(dtype=np.float64),
        f"{prefix}_month_cos": np.cos(2 * np.pi * month / 12.0).to_numpy(dtype=np.float64),
        f"{prefix}_doy_sin": np.sin(2 * np.pi * dayofyear / 366.0).to_numpy(dtype=np.float64),
        f"{prefix}_doy_cos": np.cos(2 * np.pi * dayofyear / 366.0).to_numpy(dtype=np.float64),
        f"{prefix}_is_weekend": (dayofweek >= 5.0).to_numpy(dtype=np.float64),
    }


def _future_calendar_mean_arrays(timestamp_windows: np.ndarray) -> dict[str, np.ndarray]:
    n_rows, horizon_steps = timestamp_windows.shape
    flat = pd.to_datetime(pd.Series(timestamp_windows.reshape(-1)), errors="raise")
    hour = (flat.dt.hour.astype(float) + flat.dt.minute.astype(float) / 60.0).to_numpy().reshape(n_rows, horizon_steps)
    doy = flat.dt.dayofyear.astype(float).to_numpy().reshape(n_rows, horizon_steps)
    dow = flat.dt.dayofweek.astype(float).to_numpy().reshape(n_rows, horizon_steps)
    return {
        "known_future_calendar_phase": np.mean(np.sin(2 * np.pi * hour / 24.0), axis=1),
        "future_hour_sin_mean": np.mean(np.sin(2 * np.pi * hour / 24.0), axis=1),
        "future_hour_cos_mean": np.mean(np.cos(2 * np.pi * hour / 24.0), axis=1),
        "future_doy_sin_mean": np.mean(np.sin(2 * np.pi * doy / 366.0), axis=1),
        "future_doy_cos_mean": np.mean(np.cos(2 * np.pi * doy / 366.0), axis=1),
        "future_weekend_fraction": np.mean(dow >= 5.0, axis=1),
        "future_daylight_fraction_proxy": np.mean((hour >= 6.0) & (hour <= 18.0), axis=1),
    }


def _future_step_condition_array(timestamp_windows: np.ndarray) -> np.ndarray:
    n_rows, horizon_steps = timestamp_windows.shape
    flat = pd.to_datetime(pd.Series(timestamp_windows.reshape(-1)), errors="raise")
    hour = (flat.dt.hour.astype(float) + flat.dt.minute.astype(float) / 60.0).to_numpy().reshape(n_rows, horizon_steps)
    dow = flat.dt.dayofweek.astype(float).to_numpy().reshape(n_rows, horizon_steps)
    doy = flat.dt.dayofyear.astype(float).to_numpy().reshape(n_rows, horizon_steps)
    lead_fraction = np.linspace(0.0, 1.0, horizon_steps, dtype=np.float64)[None, :]
    features = [
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
        np.sin(2 * np.pi * doy / 366.0),
        np.cos(2 * np.pi * doy / 366.0),
        (dow >= 5.0).astype(np.float64),
        ((hour >= 6.0) & (hour <= 18.0)).astype(np.float64),
        np.broadcast_to(lead_fraction, (n_rows, horizon_steps)),
    ]
    return np.stack(features, axis=2).astype(np.float32, copy=False)


def _target_history_feature_arrays(context_values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(context_values, dtype=np.float64)
    n_rows, width = values.shape
    feats: dict[str, np.ndarray] = {
        "target_last": values[:, -1],
        "target_first": values[:, 0],
        "target_mean": np.nanmean(values, axis=1),
        "target_std": np.nanstd(values, axis=1),
        "target_min": np.nanmin(values, axis=1),
        "target_max": np.nanmax(values, axis=1),
        "target_median": np.nanmedian(values, axis=1),
        "target_q25": np.nanquantile(values, 0.25, axis=1),
        "target_q75": np.nanquantile(values, 0.75, axis=1),
        "target_sum": np.nansum(values, axis=1),
        "target_positive_share": np.nanmean(values > 0, axis=1),
        "target_zero_share": np.nanmean(values == 0, axis=1),
        "target_slope_last_first": (values[:, -1] - values[:, 0]) / max(1, width - 1),
        "target_last_diff": values[:, -1] - values[:, -2] if width >= 2 else np.zeros(n_rows, dtype=np.float64),
    }
    for lag in TARGET_LAGS:
        feats[f"target_lag_{lag}"] = values[:, -lag] if width >= lag else np.full(n_rows, np.nan, dtype=np.float64)
    for window in ROLLING_WINDOWS:
        recent = values[:, -min(window, width) :]
        feats[f"target_roll{window}_mean"] = np.nanmean(recent, axis=1)
        feats[f"target_roll{window}_std"] = np.nanstd(recent, axis=1)
        feats[f"target_roll{window}_min"] = np.nanmin(recent, axis=1)
        feats[f"target_roll{window}_max"] = np.nanmax(recent, axis=1)
    return feats


def _context_covariate_feature_arrays(
    seg: pd.DataFrame,
    covariate_cols: tuple[str, ...],
    starts: np.ndarray,
    ends: np.ndarray,
) -> dict[str, np.ndarray]:
    if not covariate_cols:
        return {}
    arr = seg.loc[:, list(covariate_cols)].to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(arr)
    filled = np.where(finite, arr, 0.0)
    counts = np.vstack([np.zeros((1, len(covariate_cols)), dtype=np.float64), np.cumsum(finite.astype(np.float64), axis=0)])
    sums = np.vstack([np.zeros((1, len(covariate_cols)), dtype=np.float64), np.cumsum(filled, axis=0)])
    sumsq = np.vstack([np.zeros((1, len(covariate_cols)), dtype=np.float64), np.cumsum(filled * filled, axis=0)])
    right = ends + 1
    count = counts[right] - counts[starts]
    total = sums[right] - sums[starts]
    total_sq = sumsq[right] - sumsq[starts]
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    variance = np.divide(total_sq, count, out=np.full_like(total_sq, np.nan), where=count > 0) - mean * mean
    std = np.sqrt(np.maximum(variance, 0.0))
    last = arr[ends]
    out: dict[str, np.ndarray] = {}
    for col_index, col in enumerate(covariate_cols):
        safe = safe_feature_name(col)
        out[f"ctx_{safe}_last"] = last[:, col_index]
        out[f"ctx_{safe}_mean"] = mean[:, col_index]
        out[f"ctx_{safe}_std"] = std[:, col_index]
    return out


def _add_parity(record: dict[str, float], name: str, value: float) -> None:
    safe_value = float(value)
    is_missing = not np.isfinite(safe_value)
    record[f"parity_{name}"] = 0.0 if is_missing else safe_value
    record[f"parity_{name}_is_missing"] = 1.0 if is_missing else 0.0


def build_chronos_window_cache_from_arrays(
    domain: str,
    horizon: str,
    split: str,
    positions: list[int],
    *,
    condition_feature_mode: str,
    keep_metadata: bool,
) -> ChronosWindowCache:
    windows = load_window_index(domain, horizon, split=split)
    selected = windows.iloc[positions].reset_index(drop=True).copy()
    selected["_cache_order"] = np.arange(len(selected), dtype=np.int64)
    canonical = load_canonical(domain)
    numeric_cols = tuple(numeric_context_columns(canonical))
    covariate_cols = tuple(col for col in numeric_cols if col != "target")
    context_steps = int(selected["context_steps"].iloc[0])
    horizon_steps = int(selected["horizon_steps"].iloc[0])
    context_np = np.empty((len(selected), context_steps), dtype=np.float32)
    future_np = np.empty((len(selected), horizon_steps), dtype=np.float32)
    future_step_condition_np = np.empty((len(selected), horizon_steps, len(FUTURE_STEP_CONDITION_FEATURES)), dtype=np.float32)
    records: list[dict[str, float] | None] = [None] * len(selected)
    family = domain_family_record(domain)

    for segment_id, sub in selected.groupby("segment_id", sort=False):
        seg = canonical[canonical["segment_id"].astype(str) == str(segment_id)].sort_values("segment_row_index").reset_index(drop=True)
        target_arr = pd.to_numeric(seg["target"], errors="coerce").to_numpy(dtype=np.float64)
        timestamp_arr = pd.to_datetime(seg["timestamp"], errors="raise").to_numpy()
        ctx_starts = sub["context_start_row_index"].to_numpy(dtype=np.int64)
        ctx_ends = sub["context_end_row_index"].to_numpy(dtype=np.int64)
        fut_starts = sub["forecast_start_row_index"].to_numpy(dtype=np.int64)
        order = sub["_cache_order"].to_numpy(dtype=np.int64)

        context_view = np.lib.stride_tricks.sliding_window_view(target_arr, context_steps)
        future_view = np.lib.stride_tricks.sliding_window_view(target_arr, horizon_steps)
        timestamp_future_view = np.lib.stride_tricks.sliding_window_view(timestamp_arr, horizon_steps)
        raw_context_values = np.asarray(context_view[ctx_starts], dtype=np.float64)
        context_values = _clean_2d(raw_context_values)
        future_values = _clean_2d(future_view[fut_starts])
        context_np[order] = context_values
        future_np[order] = future_values
        future_step_condition_np[order] = _future_step_condition_array(timestamp_future_view[fut_starts])

        target_feats = _target_history_feature_arrays(raw_context_values)
        cov_feats: dict[str, np.ndarray] = {}
        if condition_feature_mode in {"target_recent_context_covariates", "full_scalar"}:
            cov_feats = _context_covariate_feature_arrays(seg, covariate_cols, ctx_starts, ctx_ends)
        origin_feats: dict[str, np.ndarray] = {}
        if condition_feature_mode == "full_scalar":
            origin_feats = _calendar_feature_arrays(sub["context_end_timestamp"], "origin")
        future_feats = _future_calendar_mean_arrays(timestamp_future_view[fut_starts])
        scalar_arrays_by_mode: dict[str, np.ndarray] = {}
        allowed_families: set[str] = set()
        if condition_feature_mode == "target_recent":
            scalar_arrays_by_mode.update(target_feats)
            allowed_families = {"target_global", "target_lags", "target_rolling"}
        elif condition_feature_mode == "target_recent_context_covariates":
            scalar_arrays_by_mode.update(target_feats)
            scalar_arrays_by_mode.update(cov_feats)
            allowed_families = {"target_global", "target_lags", "target_rolling", "context_covariate_summaries"}
        elif condition_feature_mode == "full_scalar":
            scalar_arrays_by_mode.update(
                {
                    "context_steps": sub["context_steps"].to_numpy(dtype=np.float64),
                    "horizon_steps": sub["horizon_steps"].to_numpy(dtype=np.float64),
                    "native_step_minutes": sub["native_step_minutes"].to_numpy(dtype=np.float64),
                }
            )
            scalar_arrays_by_mode.update(origin_feats)
            scalar_arrays_by_mode.update(target_feats)
            scalar_arrays_by_mode.update(cov_feats)
            allowed_families = {
                "metadata_origin",
                "target_global",
                "target_lags",
                "target_rolling",
                "context_covariate_summaries",
            }

        for local_index, global_index in enumerate(order):
            row = sub.iloc[local_index]
            values = context_values[local_index]
            mean = float(np.mean(values)) if len(values) else 0.0
            std = float(np.std(values)) if len(values) else 0.0
            record: dict[str, float] = {
                **family,
                "horizon_4h": 1.0 if str(row["horizon"]) == "4h" else 0.0,
                "horizon_24h": 1.0 if str(row["horizon"]) == "24h" else 0.0,
                "native_resolution_minutes": float(row["native_step_minutes"]),
                "context_length_steps": float(row["context_steps"]),
                "horizon_steps": float(row["horizon_steps"]),
                "context_mean": mean,
                "context_std": std,
                "context_cv": float(std / (abs(mean) + 1e-6)),
                "context_zero_fraction": float(np.mean(np.isclose(values, 0.0))) if len(values) else 0.0,
                "context_recent_ramp": float(values[-1] - values[0]) if len(values) >= 2 else 0.0,
                "context_peak_concentration": float(np.max(np.abs(values)) / (np.sum(np.abs(values)) + 1e-6)) if len(values) else 0.0,
                "label_reliability": 1.0,
            }
            for name, arr in future_feats.items():
                record[name] = float(arr[local_index])
            if condition_feature_mode != "energy_base":
                for name, arr in scalar_arrays_by_mode.items():
                    if parity_scalar_family(name) not in allowed_families:
                        continue
                    _add_parity(record, name, float(arr[local_index]))
            records[int(global_index)] = record

    if any(record is None for record in records):
        raise RuntimeError(f"{domain}/{horizon}/{split}: failed to build all fast array condition records")
    return ChronosWindowCache(
        batches=[],
        context_np=context_np,
        future_target_np=future_np,
        condition_records=[record for record in records if record is not None],
        metadata=selected.drop(columns=["_cache_order"]) if keep_metadata else None,
        future_step_condition_np=future_step_condition_np,
        future_step_condition_feature_names=FUTURE_STEP_CONDITION_FEATURES,
    )


def fit_condition_normalizer_from_records(
    records: list[dict[str, float]],
    *,
    mode: str,
    extended_calendar_condition: bool,
) -> ConditionNormalizer | None:
    if mode == "none":
        return None
    if mode != "train_zscore":
        raise ValueError(f"unsupported condition standardization mode: {mode}")
    values, feature_names = p3_4bf_condition_tensor(
        records,
        extended_calendar_condition=extended_calendar_condition,
    )
    values = values.float()
    mean = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, unbiased=False, keepdim=True)
    scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
    return ConditionNormalizer(mean=mean, scale=scale, feature_names=feature_names)


def finalize_condition_cache(
    cache: ChronosWindowCache,
    *,
    condition_normalizer: ConditionNormalizer | None,
    extended_calendar_condition: bool,
) -> ChronosWindowCache:
    values, feature_names = p3_4bf_condition_tensor(
        cache.condition_records,
        extended_calendar_condition=extended_calendar_condition,
        feature_names=condition_normalizer.feature_names if condition_normalizer is not None else None,
    )
    values = values.float()
    if condition_normalizer is not None:
        values = condition_normalizer.apply(values)
    cache.condition_values_np = np.array(values.cpu().numpy(), dtype=np.float32, copy=True)
    cache.condition_feature_names = tuple(feature_names)
    return cache


def tensor_batch_from_cache(
    cache: ChronosWindowCache,
    indexes: list[int],
    device: torch.device,
) -> dict[str, Any]:
    if cache.condition_values_np is None:
        raise RuntimeError("condition cache has not been finalized")
    idx = np.asarray(indexes, dtype=np.int64)
    out = {
        "context": torch.as_tensor(cache.context_np[idx], dtype=torch.float32).to(device, non_blocking=True),
        "future_target": torch.as_tensor(cache.future_target_np[idx], dtype=torch.float32).to(device, non_blocking=True),
        "condition_values": torch.as_tensor(cache.condition_values_np[idx], dtype=torch.float32).to(device, non_blocking=True),
        "condition_feature_names": cache.condition_feature_names,
    }
    if cache.future_step_condition_np is not None:
        out["future_step_conditions"] = torch.as_tensor(cache.future_step_condition_np[idx], dtype=torch.float32).to(
            device,
            non_blocking=True,
        )
        out["future_step_condition_feature_names"] = cache.future_step_condition_feature_names
    return out


def iter_chunks(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def iter_index_chunks(n_items: int, batch_size: int) -> list[list[int]]:
    return [list(range(start, min(start + batch_size, n_items))) for start in range(0, n_items, batch_size)]


def parameter_counts(module: nn.Module) -> dict[str, int]:
    total = int(sum(param.numel() for param in module.parameters()))
    trainable = int(sum(param.numel() for param in module.parameters() if param.requires_grad))
    return {"total": total, "trainable": trainable}


def grad_norm(module: nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum().item())
    return float(total**0.5)


def grad_norm_parameters(parameters: list[nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum().item())
    return float(total**0.5)


def zero_init_identity_adapter(adapter: EnergyConditionedAdapter) -> None:
    """Make the initial adapter contribution exactly zero."""
    with torch.no_grad():
        adapter.up.weight.zero_()
        adapter.up.bias.zero_()
        adapter.shift.weight.zero_()
        adapter.shift.bias.zero_()
        adapter.scale.weight.zero_()
        adapter.scale.bias.zero_()


class Chronos2HiddenEnergyAdapter(nn.Module):
    """Frozen Chronos-2 backbone with adapter on forecast hidden states."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        cond_dim: int,
        adapter_bottleneck: int,
        adapter_dropout: float,
        zero_init: bool,
        position_bias: bool,
        future_patch_dim: int = 0,
        max_position_patches: int = 512,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad_(False)
        self.base_model.eval()
        d_model = int(getattr(self.base_model, "model_dim"))
        dtype = next(self.base_model.parameters()).dtype
        device = next(self.base_model.parameters()).device
        self.adapter = EnergyConditionedAdapter(
            d_model=d_model,
            cond_dim=int(cond_dim),
            bottleneck=int(adapter_bottleneck),
            dropout=float(adapter_dropout),
        ).to(device=device, dtype=dtype)
        if zero_init:
            zero_init_identity_adapter(self.adapter)
        self.position_bias_enabled = bool(position_bias)
        self.position_bias = (
            nn.Parameter(torch.zeros(1, int(max_position_patches), d_model, device=device, dtype=dtype))
            if self.position_bias_enabled
            else None
        )
        self.future_patch_dim = int(future_patch_dim)
        self.future_patch_projection = (
            nn.Sequential(
                nn.LayerNorm(self.future_patch_dim),
                nn.Linear(self.future_patch_dim, d_model),
            ).to(device=device, dtype=dtype)
            if self.future_patch_dim > 0
            else None
        )
        if zero_init and self.future_patch_projection is not None:
            linear = self.future_patch_projection[-1]
            assert isinstance(linear, nn.Linear)
            with torch.no_grad():
                linear.weight.zero_()
                linear.bias.zero_()
        self.hidden_insert_location = "Chronos2Model.encode.last_num_output_patches.before_output_patch_embedding"

    def trainable_adapter_parameters(self) -> list[nn.Parameter]:
        params = list(self.adapter.parameters())
        if self.position_bias is not None:
            params.append(self.position_bias)
        if self.future_patch_projection is not None:
            params.extend(self.future_patch_projection.parameters())
        return params

    @property
    def output_patch_size(self) -> int:
        return int(self.base_model.chronos_config.output_patch_size)

    def num_output_patches(self, horizon_steps: int) -> int:
        return int(math.ceil(int(horizon_steps) / self.output_patch_size))

    def future_patch_condition(
        self,
        future_step_conditions: torch.Tensor | None,
        *,
        horizon_steps: int,
        num_output_patches: int,
    ) -> torch.Tensor | None:
        if self.future_patch_projection is None:
            return None
        if future_step_conditions is None:
            raise ValueError("future patch projection is enabled but future_step_conditions were not provided")
        values = future_step_conditions[:, :horizon_steps].to(
            device=next(self.future_patch_projection.parameters()).device,
            dtype=next(self.future_patch_projection.parameters()).dtype,
        )
        if values.ndim != 3 or values.shape[2] != self.future_patch_dim:
            raise ValueError(
                f"future_step_conditions shape {tuple(values.shape)} incompatible with dim {self.future_patch_dim}"
            )
        patch_values: list[torch.Tensor] = []
        for patch_index in range(num_output_patches):
            start = patch_index * self.output_patch_size
            stop = min((patch_index + 1) * self.output_patch_size, int(horizon_steps))
            patch_values.append(values[:, start:stop, :].mean(dim=1))
        return torch.stack(patch_values, dim=1)

    def encode_forecast_embeds(
        self,
        context: torch.Tensor,
        *,
        horizon_steps: int,
        future_target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor, int]:
        num_output_patches = self.num_output_patches(horizon_steps)
        with torch.no_grad():
            encoder_outputs, loc_scale, patched_future_covariates_mask, _num_context_patches = self.base_model.encode(
                context=context,
                context_mask=None,
                group_ids=None,
                future_covariates=None,
                future_covariates_mask=None,
                num_output_patches=num_output_patches,
                future_target=future_target,
                future_target_mask=None,
                output_attentions=False,
            )
            hidden_states: torch.Tensor = encoder_outputs[0]
            forecast_embeds = hidden_states[:, -num_output_patches:].detach()
        return forecast_embeds, loc_scale, patched_future_covariates_mask, num_output_patches

    def forward(
        self,
        context: torch.Tensor,
        condition_values: torch.Tensor,
        *,
        horizon_steps: int,
        future_target: torch.Tensor | None = None,
        future_step_conditions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        self.base_model.eval()
        forecast_embeds, loc_scale, patched_future_covariates_mask, num_output_patches = self.encode_forecast_embeds(
            context,
            horizon_steps=horizon_steps,
            future_target=future_target,
        )
        condition_values = condition_values.to(device=forecast_embeds.device, dtype=forecast_embeds.dtype)
        adapted = self.adapter(forecast_embeds, condition_values)
        future_patch_values = self.future_patch_condition(
            future_step_conditions,
            horizon_steps=horizon_steps,
            num_output_patches=num_output_patches,
        )
        if future_patch_values is not None:
            adapted = adapted + self.future_patch_projection(future_patch_values)
        if self.position_bias is not None:
            if num_output_patches > self.position_bias.shape[1]:
                raise ValueError(
                    f"num_output_patches={num_output_patches} exceeds position bias width {self.position_bias.shape[1]}"
                )
            adapted = adapted + self.position_bias[:, :num_output_patches].to(
                device=adapted.device,
                dtype=adapted.dtype,
            )
        quantile_preds: torch.Tensor = self.base_model.output_patch_embedding(adapted)
        quantile_preds = rearrange(
            quantile_preds,
            "b n (q p) -> b q (n p)",
            n=num_output_patches,
            q=self.base_model.num_quantiles,
            p=self.output_patch_size,
        )
        loss = (
            self.base_model._compute_loss(
                quantile_preds=quantile_preds,
                future_target=future_target,
                future_target_mask=None,
                patched_future_covariates_mask=patched_future_covariates_mask,
                loc_scale=loc_scale,
                num_output_patches=num_output_patches,
            )
            if future_target is not None
            else None
        )
        batch_size = context.shape[0]
        quantile_preds = rearrange(
            quantile_preds,
            "b q h -> b (q h)",
            b=batch_size,
            q=self.base_model.num_quantiles,
            h=num_output_patches * self.output_patch_size,
        )
        quantile_preds = self.base_model.instance_norm.inverse(quantile_preds, loc_scale)
        quantile_preds = rearrange(
            quantile_preds,
            "b (q h) -> b q h",
            q=self.base_model.num_quantiles,
            h=num_output_patches * self.output_patch_size,
        )
        return {"loss": loss, "quantile_preds": quantile_preds}


def quantile_indices(base_model: nn.Module) -> dict[str, int]:
    quantiles = base_model.quantiles.detach().float().cpu().numpy()
    return {
        "q10": int(np.argmin(np.abs(quantiles - 0.10))),
        "q50": int(np.argmin(np.abs(quantiles - 0.50))),
        "q90": int(np.argmin(np.abs(quantiles - 0.90))),
    }


@torch.no_grad()
def frozen_quantiles(base_model: nn.Module, context: torch.Tensor, horizon_steps: int) -> torch.Tensor:
    base_model.eval()
    num_output_patches = int(math.ceil(int(horizon_steps) / int(base_model.chronos_config.output_patch_size)))
    out = base_model(
        context=context,
        context_mask=None,
        group_ids=None,
        future_covariates=None,
        future_covariates_mask=None,
        num_output_patches=num_output_patches,
        future_target=None,
        future_target_mask=None,
        output_attentions=False,
    )
    return out.quantile_preds[:, :, :horizon_steps]


@torch.no_grad()
def adapter_quantiles(
    wrapper: Chronos2HiddenEnergyAdapter,
    context: torch.Tensor,
    condition_values: torch.Tensor,
    horizon_steps: int,
    future_step_conditions: torch.Tensor | None = None,
) -> torch.Tensor:
    wrapper.base_model.eval()
    wrapper.adapter.eval()
    if wrapper.future_patch_projection is not None:
        wrapper.future_patch_projection.eval()
    out = wrapper(
        context,
        condition_values,
        horizon_steps=horizon_steps,
        future_target=None,
        future_step_conditions=future_step_conditions,
    )
    quantiles = out["quantile_preds"]
    assert quantiles is not None
    return quantiles[:, :, :horizon_steps]


def adapter_training_loss(
    wrapper: Chronos2HiddenEnergyAdapter,
    out: dict[str, torch.Tensor | None],
    future_target: torch.Tensor,
    *,
    horizon_steps: int,
    loss_kind: str,
) -> torch.Tensor:
    if loss_kind == "chronos_internal":
        loss = out["loss"]
        if loss is None:
            raise RuntimeError("adapter training loss was not computed")
        return loss
    quantiles = out["quantile_preds"]
    if quantiles is None:
        raise RuntimeError("adapter q50 loss requested but quantile predictions were not returned")
    q50_index = quantile_indices(wrapper.base_model)["q50"]
    pred = quantiles[:, q50_index, :horizon_steps]
    target = future_target[:, :horizon_steps].to(device=pred.device, dtype=pred.dtype)
    if loss_kind == "mae_q50":
        return torch.nn.functional.l1_loss(pred, target)
    if loss_kind == "huber_q50":
        return torch.nn.functional.smooth_l1_loss(pred, target, beta=1.0)
    raise ValueError(f"unsupported adapter loss: {loss_kind}")


def prediction_rows_from_quantiles(
    *,
    batches: list[Any],
    quantile_tensor: torch.Tensor,
    base_model: nn.Module,
    config_id: str,
    route_label: str,
    seed: int,
    split: str,
    notes: str,
) -> list[dict[str, Any]]:
    indexes = quantile_indices(base_model)
    quantile_np = quantile_tensor.detach().float().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for batch, one in zip(batches, quantile_np, strict=True):
        horizon_steps = int(batch.metadata["horizon_steps"])
        row = build_prediction_stub(
            batch,
            model_family=MODEL_FAMILY,
            model_id=MODEL_ID,
            config_id=config_id,
            seed=seed,
            y_pred=pd.Series(one[indexes["q50"], :horizon_steps]),
            notes=notes,
        )
        row["q10"] = serialize_series(pd.Series(one[indexes["q10"], :horizon_steps]))
        row["q50"] = serialize_series(pd.Series(one[indexes["q50"], :horizon_steps]))
        row["q90"] = serialize_series(pd.Series(one[indexes["q90"], :horizon_steps]))
        row["split"] = split
        row["route_label"] = route_label
        row["h1_branch"] = PLAN_ID
        rows.append(row)
    return rows


def prediction_rows_from_quantiles_cache(
    *,
    cache: ChronosWindowCache,
    chunk_indexes: list[int],
    quantile_tensor: torch.Tensor,
    base_model: nn.Module,
    config_id: str,
    route_label: str,
    seed: int,
    split: str,
    notes: str,
) -> list[dict[str, Any]]:
    if cache.batches:
        return prediction_rows_from_quantiles(
            batches=[cache.batches[idx] for idx in chunk_indexes],
            quantile_tensor=quantile_tensor,
            base_model=base_model,
            config_id=config_id,
            route_label=route_label,
            seed=seed,
            split=split,
            notes=notes,
        )
    if cache.metadata is None:
        raise ValueError(f"{split} cache must retain batches or metadata for prediction row construction")
    indexes = quantile_indices(base_model)
    quantile_np = quantile_tensor.detach().float().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for local_pos, one in zip(chunk_indexes, quantile_np, strict=True):
        meta = cache.metadata.iloc[local_pos]
        horizon_steps = int(meta["horizon_steps"])
        y_pred = pd.Series(one[indexes["q50"], :horizon_steps])
        row = {
            "domain_id": str(meta["domain_id"]),
            "series_id": str(meta["series_id"]),
            "segment_id": str(meta["segment_id"]),
            "horizon": str(meta["horizon"]),
            "window_id": str(meta["window_id"]),
            "model_family": MODEL_FAMILY,
            "model_id": MODEL_ID,
            "config_id": config_id,
            "seed": int(seed),
            "origin_time": pd.Timestamp(meta["context_end_timestamp"]).isoformat(),
            "target_start_time": pd.Timestamp(meta["forecast_start_timestamp"]).isoformat(),
            "target_end_time": pd.Timestamp(meta["forecast_end_timestamp"]).isoformat(),
            "y_true": serialize_series(pd.Series(cache.future_target_np[local_pos, :horizon_steps])),
            "y_pred": serialize_series(y_pred),
            "notes": notes,
            "q10": serialize_series(pd.Series(one[indexes["q10"], :horizon_steps])),
            "q50": serialize_series(pd.Series(one[indexes["q50"], :horizon_steps])),
            "q90": serialize_series(pd.Series(one[indexes["q90"], :horizon_steps])),
            "split": split,
            "route_label": route_label,
            "h1_branch": PLAN_ID,
        }
        rows.append(row)
    return rows


def predict_frozen(
    base_model: nn.Module,
    batches: list[Any],
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for chunk in iter_chunks(batches, batch_size):
        tensor_batch = tensor_batch_from_batches(
            chunk,
            device,
            extended_calendar_condition=False,
            condition_feature_mode="energy_base",
        )
        horizon_steps = int(chunk[0].metadata["horizon_steps"])
        quantiles = frozen_quantiles(base_model, tensor_batch["context"], horizon_steps)
        rows.extend(
            prediction_rows_from_quantiles(
                batches=chunk,
                quantile_tensor=quantiles,
                base_model=base_model,
                config_id=BASE_CONFIG_ID,
                route_label=ROUTE_BASE,
                seed=seed,
                split=split,
                notes=f"P3-4bs Chronos-2 frozen target-only {split} prediction.",
            )
        )
    return pd.DataFrame(rows)


def predict_adapter(
    wrapper: Chronos2HiddenEnergyAdapter,
    batches: list[Any],
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    split: str,
    condition_normalizer: ConditionNormalizer | None,
    extended_calendar_condition: bool,
    condition_feature_mode: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for chunk in iter_chunks(batches, batch_size):
        tensor_batch = tensor_batch_from_batches(
            chunk,
            device,
            condition_normalizer=condition_normalizer,
            extended_calendar_condition=extended_calendar_condition,
            condition_feature_mode=condition_feature_mode,
        )
        horizon_steps = int(chunk[0].metadata["horizon_steps"])
        quantiles = adapter_quantiles(
            wrapper,
            tensor_batch["context"],
            tensor_batch["condition_values"],
            horizon_steps,
            tensor_batch.get("future_step_conditions"),
        )
        rows.extend(
            prediction_rows_from_quantiles(
                batches=chunk,
                quantile_tensor=quantiles,
                base_model=wrapper.base_model,
                config_id=ADAPTER_CONFIG_ID,
                route_label=ROUTE_ADAPTER,
                seed=seed,
                split=split,
                notes=(
                    f"P3-4bs Chronos-2 covariate-aware hidden adapter {split} prediction; "
                    "adapter inserted after encode and before output_patch_embedding; frozen backbone."
                ),
            )
        )
    return pd.DataFrame(rows)


def predict_frozen_cached(
    base_model: nn.Module,
    cache: ChronosWindowCache,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    horizon_steps = int(cache.future_target_np.shape[1])
    for chunk_indexes in iter_index_chunks(len(cache), batch_size):
        tensor_batch = tensor_batch_from_cache(cache, chunk_indexes, device)
        quantiles = frozen_quantiles(base_model, tensor_batch["context"], horizon_steps)
        rows.extend(
            prediction_rows_from_quantiles_cache(
                cache=cache,
                chunk_indexes=chunk_indexes,
                quantile_tensor=quantiles,
                base_model=base_model,
                config_id=BASE_CONFIG_ID,
                route_label=ROUTE_BASE,
                seed=seed,
                split=split,
                notes=f"P3-4bs Chronos-2 frozen target-only {split} prediction.",
            )
        )
    return pd.DataFrame(rows)


def predict_adapter_cached(
    wrapper: Chronos2HiddenEnergyAdapter,
    cache: ChronosWindowCache,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    horizon_steps = int(cache.future_target_np.shape[1])
    for chunk_indexes in iter_index_chunks(len(cache), batch_size):
        tensor_batch = tensor_batch_from_cache(cache, chunk_indexes, device)
        quantiles = adapter_quantiles(
            wrapper,
            tensor_batch["context"],
            tensor_batch["condition_values"],
            horizon_steps,
            tensor_batch.get("future_step_conditions"),
        )
        rows.extend(
            prediction_rows_from_quantiles_cache(
                cache=cache,
                chunk_indexes=chunk_indexes,
                quantile_tensor=quantiles,
                base_model=wrapper.base_model,
                config_id=ADAPTER_CONFIG_ID,
                route_label=ROUTE_ADAPTER,
                seed=seed,
                split=split,
                notes=(
                    f"P3-4bs Chronos-2 covariate-aware hidden adapter {split} prediction; "
                    "adapter inserted after encode and before output_patch_embedding; frozen backbone."
                ),
            )
        )
    return pd.DataFrame(rows)


def train_adapter(
    wrapper: Chronos2HiddenEnergyAdapter,
    train_batches: list[Any],
    *,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
    condition_normalizer: ConditionNormalizer | None,
    extended_calendar_condition: bool,
    condition_feature_mode: str,
) -> list[dict[str, Any]]:
    trainable_params = wrapper.trainable_adapter_parameters()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    history: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for epoch in range(int(args.epochs)):
        wrapper.base_model.eval()
        wrapper.adapter.train()
        if wrapper.future_patch_projection is not None:
            wrapper.future_patch_projection.train()
        order = list(range(len(train_batches)))
        rng.shuffle(order)
        losses: list[float] = []
        grad_norms: list[float] = []
        chunks = [order[start : start + int(args.batch_size)] for start in range(0, len(order), int(args.batch_size))]
        for step, chunk_indexes in enumerate(chunks):
            if args.max_train_batches is not None and step >= int(args.max_train_batches):
                break
            chunk = [train_batches[idx] for idx in chunk_indexes]
            tensor_batch = tensor_batch_from_batches(
                chunk,
                device,
                condition_normalizer=condition_normalizer,
                extended_calendar_condition=extended_calendar_condition,
                condition_feature_mode=condition_feature_mode,
            )
            horizon_steps = int(chunk[0].metadata["horizon_steps"])
            optimizer.zero_grad(set_to_none=True)
            out = wrapper(
                tensor_batch["context"],
                tensor_batch["condition_values"],
                horizon_steps=horizon_steps,
                future_target=tensor_batch["future_target"],
                future_step_conditions=tensor_batch.get("future_step_conditions"),
            )
            loss = adapter_training_loss(
                wrapper,
                out,
                tensor_batch["future_target"],
                horizon_steps=horizon_steps,
                loss_kind=str(args.adapter_loss),
            )
            loss.backward()
            grad = grad_norm_parameters(trainable_params)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.gradient_clip))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            grad_norms.append(float(grad))
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)) if losses else None,
                "batches": int(len(losses)),
                "adapter_grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else None,
            }
        )
    return history


def train_adapter_cached(
    wrapper: Chronos2HiddenEnergyAdapter,
    train_cache: ChronosWindowCache,
    *,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
) -> list[dict[str, Any]]:
    trainable_params = wrapper.trainable_adapter_parameters()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    history: list[dict[str, Any]] = []
    rng = random.Random(seed)
    horizon_steps = int(train_cache.future_target_np.shape[1])
    for epoch in range(int(args.epochs)):
        wrapper.base_model.eval()
        wrapper.adapter.train()
        if wrapper.future_patch_projection is not None:
            wrapper.future_patch_projection.train()
        order = list(range(len(train_cache)))
        rng.shuffle(order)
        losses: list[float] = []
        grad_norms: list[float] = []
        chunks = [order[start : start + int(args.batch_size)] for start in range(0, len(order), int(args.batch_size))]
        for step, chunk_indexes in enumerate(chunks):
            if args.max_train_batches is not None and step >= int(args.max_train_batches):
                break
            tensor_batch = tensor_batch_from_cache(train_cache, chunk_indexes, device)
            optimizer.zero_grad(set_to_none=True)
            out = wrapper(
                tensor_batch["context"],
                tensor_batch["condition_values"],
                horizon_steps=horizon_steps,
                future_target=tensor_batch["future_target"],
                future_step_conditions=tensor_batch.get("future_step_conditions"),
            )
            loss = adapter_training_loss(
                wrapper,
                out,
                tensor_batch["future_target"],
                horizon_steps=horizon_steps,
                loss_kind=str(args.adapter_loss),
            )
            loss.backward()
            grad = grad_norm_parameters(trainable_params)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.gradient_clip))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            grad_norms.append(float(grad))
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)) if losses else None,
                "batches": int(len(losses)),
                "adapter_grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else None,
            }
        )
    return history


def summarize_full_day(metrics: pd.DataFrame) -> dict[str, float]:
    mapping = {BASE_CONFIG_ID: ROUTE_BASE, ADAPTER_CONFIG_ID: ROUTE_ADAPTER}
    out: dict[str, float] = {}
    scoped = metrics[metrics["metric_scope"].astype(str) == "full_day"]
    for row in scoped.to_dict(orient="records"):
        if pd.notna(row.get("wape")):
            out[mapping.get(str(row["config_id"]), str(row["config_id"]))] = float(row["wape"])
    return out


def load_pipeline(args: argparse.Namespace) -> Chronos2Pipeline:
    return Chronos2Pipeline.from_pretrained(
        args.model_name,
        local_files_only=not args.allow_download,
        device_map=args.device_map,
    )


def run_cell(
    args: argparse.Namespace,
    *,
    domain: str,
    horizon: str,
    cell_index: int,
    subset_manifest: dict[str, Any],
    run_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.time()
    seed = int(args.seed + cell_index)
    set_seed(seed)
    device = torch.device("cuda")
    eval_splits = tuple(args.eval_split or ("validation",))

    train_sel = manifest_positions(
        subset_manifest,
        domain=domain,
        horizon=horizon,
        split="train",
        limit=args.max_train_windows,
        selection_id=f"{args.selection_id}:train",
    )
    split_limits = {
        "train": int(args.max_train_windows),
        "validation": int(args.max_validation_windows),
        "test": int(args.max_test_windows),
    }
    eval_selections = {
        split: manifest_positions(
            subset_manifest,
            domain=domain,
            horizon=horizon,
            split=split,
            limit=split_limits[split],
            selection_id=f"{args.selection_id}:{split}",
        )
        for split in eval_splits
    }
    phase_timing: dict[str, float] = {}
    phase_started = time.time()
    if args.array_preprocess:
        train_cache = build_chronos_window_cache_from_arrays(
            domain,
            horizon,
            "train",
            train_sel["positions"],
            condition_feature_mode=args.condition_feature_mode,
            keep_metadata=False,
        )
        eval_caches = {
            split: build_chronos_window_cache_from_arrays(
                domain,
                horizon,
                split,
                eval_selections[split]["positions"],
                condition_feature_mode=args.condition_feature_mode,
                keep_metadata=True,
            )
            for split in eval_splits
        }
        phase_timing["array_window_tensor_condition_cache_sec"] = round(time.time() - phase_started, 3)
    else:
        train_batches = batches_from_positions(
            domain,
            horizon,
            "train",
            train_sel["positions"],
            workers=int(args.preprocess_workers),
        )
        eval_batches = {
            split: batches_from_positions(
                domain,
                horizon,
                split,
                eval_selections[split]["positions"],
                workers=int(args.preprocess_workers),
            )
            for split in eval_splits
        }
        phase_timing["window_batch_materialization_sec"] = round(time.time() - phase_started, 3)
        if not train_batches or any(not batches for batches in eval_batches.values()):
            raise ValueError(f"{domain}/{horizon}: empty train or eval selection")
        phase_started = time.time()
        train_cache = build_chronos_window_cache(
            train_batches,
            condition_feature_mode=args.condition_feature_mode,
            keep_batches=False,
            workers=int(args.preprocess_workers),
        )
        eval_caches = {
            split: build_chronos_window_cache(
                batches,
                condition_feature_mode=args.condition_feature_mode,
                keep_batches=True,
                workers=int(args.preprocess_workers),
            )
            for split, batches in eval_batches.items()
        }
        del train_batches
        release_cuda()
        phase_timing["window_tensor_condition_cache_sec"] = round(time.time() - phase_started, 3)
    if len(train_cache) == 0 or any(len(cache) == 0 for cache in eval_caches.values()):
        raise ValueError(f"{domain}/{horizon}: empty train or eval selection")

    cell_dir = run_dir / "cells" / f"{domain}_{horizon}"
    pred_dir = cell_dir / "predictions"
    metric_dir = cell_dir / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    pipeline = load_pipeline(args)
    base_model = pipeline.model
    base_model.eval()
    base_dtype = str(next(base_model.parameters()).dtype)
    base_device = str(next(base_model.parameters()).device)
    base_parameter_summary_loaded = parameter_counts(base_model)

    phase_started = time.time()
    condition_normalizer = fit_condition_normalizer_from_records(
        train_cache.condition_records,
        mode=args.condition_standardization,
        extended_calendar_condition=bool(args.extended_calendar_condition),
    )
    finalize_condition_cache(
        train_cache,
        condition_normalizer=condition_normalizer,
        extended_calendar_condition=bool(args.extended_calendar_condition),
    )
    for eval_cache in eval_caches.values():
        finalize_condition_cache(
            eval_cache,
            condition_normalizer=condition_normalizer,
            extended_calendar_condition=bool(args.extended_calendar_condition),
        )
    phase_timing["condition_normalizer_and_matrix_sec"] = round(time.time() - phase_started, 3)
    condition_features = list(train_cache.condition_feature_names)
    cond_dim = int(train_cache.condition_values_np.shape[1]) if train_cache.condition_values_np is not None else 0
    future_patch_dim = (
        int(train_cache.future_step_condition_np.shape[2])
        if bool(args.adapter_future_patch_condition) and train_cache.future_step_condition_np is not None
        else 0
    )

    wrapper = Chronos2HiddenEnergyAdapter(
        base_model,
        cond_dim=cond_dim,
        adapter_bottleneck=args.adapter_bottleneck,
        adapter_dropout=args.adapter_dropout,
        zero_init=bool(args.zero_init_adapter),
        position_bias=bool(args.adapter_position_bias),
        future_patch_dim=future_patch_dim,
    ).to(device)
    base_parameter_summary_frozen = parameter_counts(wrapper.base_model)
    adapter_parameter_summary = parameter_counts(wrapper.adapter)
    position_bias_parameter_count = int(wrapper.position_bias.numel()) if wrapper.position_bias is not None else 0
    future_patch_parameter_summary = (
        parameter_counts(wrapper.future_patch_projection)
        if wrapper.future_patch_projection is not None
        else {"total": 0, "trainable": 0}
    )
    trainable_adapter_parameter_count = int(sum(param.numel() for param in wrapper.trainable_adapter_parameters()))
    combined_parameter_summary = parameter_counts(wrapper)

    adapter_history = train_adapter_cached(
        wrapper,
        train_cache,
        device=device,
        args=args,
        seed=seed,
    )
    split_outputs: dict[str, Any] = {}
    split_prediction_frames: list[pd.DataFrame] = []
    for split, eval_cache in eval_caches.items():
        base_pred = predict_frozen_cached(
            wrapper.base_model,
            eval_cache,
            device=device,
            batch_size=args.eval_batch_size,
            seed=seed,
            split=split,
        )
        adapter_pred = predict_adapter_cached(
            wrapper,
            eval_cache,
            device=device,
            batch_size=args.eval_batch_size,
            seed=seed,
            split=split,
        )
        split_predictions = pd.concat([base_pred, adapter_pred], ignore_index=True)
        window_index = load_window_index(domain, horizon, split=split)
        validate_prediction_against_windows(split_predictions.drop(columns=["split"], errors="ignore"), window_index)
        split_pred_path = pred_dir / f"{split}_predictions_all_arms.parquet"
        split_predictions.to_parquet(split_pred_path, index=False)
        split_metrics = evaluate_prediction_frame(split_predictions)
        split_metric_paths = write_metrics(split_metrics, metric_dir, stem=f"{split}_metrics")
        split_outputs[split] = {
            "selection": eval_selections[split],
            "prediction_rows": int(len(split_predictions)),
            "metric_rows": int(len(split_metrics)),
            "predictions": str(split_pred_path),
            "metrics": split_metric_paths,
            "full_day_wape_by_route": summarize_full_day(split_metrics),
        }
        split_prediction_frames.append(split_predictions)

    predictions = pd.concat(split_prediction_frames, ignore_index=True)
    pred_path = pred_dir / "requested_split_predictions_all_arms.parquet"
    predictions.to_parquet(pred_path, index=False)
    metrics = evaluate_prediction_frame(predictions)
    metric_paths = write_metrics(metrics, metric_dir, stem="requested_split_metrics")
    wape = summarize_full_day(metrics)

    cell_manifest = {
        "status": "ok",
        "plan_id": PLAN_ID,
        "domain": domain,
        "horizon": horizon,
        "seed": seed,
        "selection_id": args.selection_id,
        "arms": [ROUTE_BASE, ROUTE_ADAPTER],
        "base_config_id": BASE_CONFIG_ID,
        "adapter_config_id": ADAPTER_CONFIG_ID,
        "true_chronos_hidden_state_insertion": True,
        "hidden_insert_location": wrapper.hidden_insert_location,
        "installed_package_modified": False,
        "base_backbone_frozen": int(base_parameter_summary_frozen["trainable"]) == 0,
        "checkpoint_is_full_power": True,
        "base_dtype": base_dtype,
        "base_device": base_device,
        "condition_feature_names": condition_features,
        "condition_feature_mode": args.condition_feature_mode,
        "condition_dim": cond_dim,
        "extended_calendar_condition": bool(args.extended_calendar_condition),
        "extra_calendar_conditioning_features": list(EXTRA_CALENDAR_CONDITIONING_FEATURES)
        if args.extended_calendar_condition
        else [],
        "adapter_zero_initialized": bool(args.zero_init_adapter),
        "adapter_bottleneck": int(args.adapter_bottleneck),
        "adapter_dropout": float(args.adapter_dropout),
        "adapter_position_bias": bool(args.adapter_position_bias),
        "adapter_future_patch_condition": bool(args.adapter_future_patch_condition),
        "future_step_condition_feature_names": list(train_cache.future_step_condition_feature_names)
        if bool(args.adapter_future_patch_condition)
        else [],
        "future_patch_dim": future_patch_dim,
        "adapter_loss": str(args.adapter_loss),
        "condition_standardization": args.condition_standardization,
        "condition_normalizer_summary": condition_normalizer.summary() if condition_normalizer is not None else {"kind": "none"},
        "fast_window_cache_enabled": True,
        "fast_window_cache_version": "p3_4bx_chronos_cache_v1",
        "array_preprocess_enabled": bool(args.array_preprocess),
        "preprocess_workers": int(args.preprocess_workers),
        "phase_timing_sec": phase_timing,
        "train_selection": train_sel,
        "eval_splits": list(eval_splits),
        "eval_outputs": split_outputs,
        "validation_selection": split_outputs.get("validation", {}).get("selection"),
        "test_selection": split_outputs.get("test", {}).get("selection"),
        "adapter_train_history": adapter_history,
        "base_parameter_summary_loaded": base_parameter_summary_loaded,
        "base_parameter_summary_frozen": base_parameter_summary_frozen,
        "adapter_parameter_summary": adapter_parameter_summary,
        "position_bias_parameter_count": position_bias_parameter_count,
        "future_patch_parameter_summary": future_patch_parameter_summary,
        "trainable_adapter_parameter_count": trainable_adapter_parameter_count,
        "combined_parameter_summary": combined_parameter_summary,
        "full_day_wape_by_route": wape,
        "prediction_rows": int(len(predictions)),
        "metric_rows": int(len(metrics)),
        "predictions": str(pred_path),
        "metrics": metric_paths,
        "test_predictions_generated": "test" in eval_splits,
        "test_artifacts_created": "test" in eval_splits,
        "important_boundary": "Chronos-2 covariate-aware hidden adapter runner; test split is only evaluated when explicitly requested by P5 executor.",
        "elapsed_sec": round(time.time() - started, 3),
    }
    (cell_dir / "cell_manifest.json").write_text(dumps(cell_manifest) + "\n", encoding="utf-8")

    del wrapper, pipeline
    release_cuda()
    return predictions, cell_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", action="append", default=None)
    parser.add_argument("--horizon", action="append", default=None)
    parser.add_argument("--subset-manifest", type=Path, default=DEFAULT_SUBSET_MANIFEST)
    parser.add_argument("--max-train-windows", type=int, default=128)
    parser.add_argument("--max-validation-windows", type=int, default=128)
    parser.add_argument("--max-test-windows", type=int, default=0)
    parser.add_argument("--eval-split", action="append", choices=["train", "validation", "test"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--array-preprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preprocess-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--adapter-bottleneck", type=int, default=16)
    parser.add_argument("--adapter-dropout", type=float, default=0.05)
    parser.add_argument("--adapter-position-bias", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adapter-future-patch-condition", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adapter-loss", choices=["chronos_internal", "mae_q50", "huber_q50"], default="chronos_internal")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--zero-init-adapter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--condition-standardization", choices=["train_zscore", "none"], default="train_zscore")
    parser.add_argument("--condition-feature-mode", choices=list(CONDITION_FEATURE_MODES), default="target_recent")
    parser.add_argument("--extended-calendar-condition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--selection-id", default=SELECTION_ID)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domains = args.domain or ["aluminum_load", "arena_pv"]
    horizons = args.horizon or ["4h"]
    eval_splits = tuple(args.eval_split or ("validation",))
    started = time.time()
    cuda = require_cuda(args.device_map)
    torch.cuda.reset_peak_memory_stats()
    subset_manifest = read_json(args.subset_manifest)
    if not allowed_subset_manifest(subset_manifest):
        raise ValueError("P3-4bs expects an approved P3-4k stride4 or P3 target-pure subset manifest")

    run_dir = args.out_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    all_predictions: list[pd.DataFrame] = []
    cell_manifests: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for cell_index, (domain, horizon) in enumerate((d, h) for d in domains for h in horizons):
        try:
            predictions, cell_manifest = run_cell(
                args,
                domain=domain,
                horizon=horizon,
                cell_index=cell_index,
                subset_manifest=subset_manifest,
                run_dir=run_dir,
            )
            all_predictions.append(predictions)
            cell_manifests.append(cell_manifest)
            print(dumps({"status": "cell_ok", "domain": domain, "horizon": horizon, "wape": cell_manifest["full_day_wape_by_route"]}))
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

    predictions_all = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    pred_dir = run_dir / "predictions"
    metric_dir = run_dir / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    combined_path = pred_dir / "validation_predictions_all_cells_all_arms.parquet"
    predictions_all.to_parquet(combined_path, index=False)
    metrics = evaluate_prediction_frame(predictions_all)
    metric_paths = write_metrics(metrics, metric_dir, stem="validation_metrics_all_cells")
    manifest = {
        "status": "ok" if not failures else "failed",
        "plan_id": PLAN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_family": MODEL_FAMILY,
        "model_name": args.model_name,
        "checkpoint_is_full_power": True,
        "run_dir": str(run_dir),
        "subset_manifest": str(args.subset_manifest),
        "subset_id": subset_manifest.get("subset_id"),
        "selection_id": args.selection_id,
        "domains": domains,
        "horizons": horizons,
        "eval_splits": list(eval_splits),
        "arms": [ROUTE_BASE, ROUTE_ADAPTER],
        "true_chronos_hidden_state_insertion": True,
        "hidden_insert_location": "Chronos2Model.encode.last_num_output_patches.before_output_patch_embedding",
        "installed_package_modified": False,
        "ordinary_adaptation_baseline_included": False,
        "requires_later_ordinary_adaptation_comparison_if_positive": True,
        "epochs": int(args.epochs),
        "max_train_windows": int(args.max_train_windows),
        "max_validation_windows": int(args.max_validation_windows),
        "max_test_windows": int(args.max_test_windows),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "array_preprocess": bool(args.array_preprocess),
        "preprocess_workers": int(args.preprocess_workers),
        "adapter_bottleneck": int(args.adapter_bottleneck),
        "adapter_dropout": float(args.adapter_dropout),
        "adapter_loss": str(args.adapter_loss),
        "adapter_position_bias": bool(args.adapter_position_bias),
        "adapter_future_patch_condition": bool(args.adapter_future_patch_condition),
        "future_step_condition_feature_names": list(FUTURE_STEP_CONDITION_FEATURES)
        if bool(args.adapter_future_patch_condition)
        else [],
        "adapter_zero_initialized": bool(args.zero_init_adapter),
        "condition_standardization": args.condition_standardization,
        "condition_feature_mode": args.condition_feature_mode,
        "extended_calendar_condition": bool(args.extended_calendar_condition),
        "extra_calendar_conditioning_features": list(EXTRA_CALENDAR_CONDITIONING_FEATURES)
        if args.extended_calendar_condition
        else [],
        "prediction_rows": int(len(predictions_all)),
        "metric_rows": int(len(metrics)),
        "predictions_all": str(combined_path),
        "metrics": metric_paths,
        "cell_manifests": cell_manifests,
        "failures": failures,
        **cuda,
        "max_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "model_loading_launched": True,
        "real_project_window_data_read": True,
        "training_launched": True,
        "fine_tuning_launched": True,
        "inference_launched": True,
        "forecast_metrics_computed": True,
        "prediction_artifact_saved": True,
        "test_predictions_generated": "test" in eval_splits,
        "test_artifacts_created": "test" in eval_splits,
        "important_boundary": "Chronos-2 covariate-aware hidden adapter runner; test split is only evaluated when explicitly requested by P5 executor.",
        "elapsed_sec": round(time.time() - started, 3),
    }
    (run_dir / "manifest.json").write_text(dumps(manifest) + "\n", encoding="utf-8")
    print(dumps(manifest))


if __name__ == "__main__":
    main()
