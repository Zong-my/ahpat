#!/usr/bin/env python3
"""Context-only feature bundles shared by energy TSFM runners.

The first contract in this module mirrors the current LightGBM feature
exposure exactly: origin-time/window metadata, target history statistics,
target lags, target rolling summaries, context numeric covariate summaries,
and deterministic future lead/calendar features. These helpers intentionally
read only the frozen context rows plus deterministic timestamps known at the
forecast origin.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from energy_tsfm_p2_core import WindowBatch


IDENTIFIER_COLUMNS = {
    "domain_id",
    "series_id",
    "segment_id",
    "split",
    "timestamp",
    "window_id",
}
NON_FEATURE_NUMERIC_COLUMNS = {"segment_row_index"}
TARGET_LAGS = (1, 2, 4, 8, 16, 24, 48, 96, 144)
ROLLING_WINDOWS = (4, 8, 16, 32)


@dataclass(frozen=True)
class ContextFeatureBundle:
    """A leakage-safe feature bundle for one frozen forecast window."""

    window_id: str
    scalar_features: dict[str, float]
    future_known_features: pd.DataFrame
    scalar_feature_names: tuple[str, ...]
    future_known_feature_names: tuple[str, ...]
    feature_bundle_hash: str
    leakage_contract: dict[str, Any]


def safe_feature_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name))


def calendar_features(ts: pd.Timestamp, prefix: str) -> dict[str, float]:
    ts = pd.Timestamp(ts)
    hour = ts.hour + ts.minute / 60.0
    dayofweek = ts.dayofweek
    month = ts.month
    dayofyear = ts.dayofyear
    return {
        f"{prefix}_hour_sin": float(np.sin(2 * np.pi * hour / 24.0)),
        f"{prefix}_hour_cos": float(np.cos(2 * np.pi * hour / 24.0)),
        f"{prefix}_dow_sin": float(np.sin(2 * np.pi * dayofweek / 7.0)),
        f"{prefix}_dow_cos": float(np.cos(2 * np.pi * dayofweek / 7.0)),
        f"{prefix}_month_sin": float(np.sin(2 * np.pi * month / 12.0)),
        f"{prefix}_month_cos": float(np.cos(2 * np.pi * month / 12.0)),
        f"{prefix}_doy_sin": float(np.sin(2 * np.pi * dayofyear / 366.0)),
        f"{prefix}_doy_cos": float(np.cos(2 * np.pi * dayofyear / 366.0)),
        f"{prefix}_is_weekend": float(dayofweek >= 5),
    }


def numeric_context_columns(context: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in context.columns:
        if col in IDENTIFIER_COLUMNS or col in NON_FEATURE_NUMERIC_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(context[col]):
            cols.append(str(col))
    return cols


def target_history_features(context: pd.DataFrame) -> dict[str, float]:
    target = pd.to_numeric(context["target"], errors="coerce").astype(float)
    values = target.to_numpy(dtype=float)
    feats: dict[str, float] = {
        "target_last": float(values[-1]),
        "target_first": float(values[0]),
        "target_mean": float(np.nanmean(values)),
        "target_std": float(np.nanstd(values)),
        "target_min": float(np.nanmin(values)),
        "target_max": float(np.nanmax(values)),
        "target_median": float(np.nanmedian(values)),
        "target_q25": float(np.nanquantile(values, 0.25)),
        "target_q75": float(np.nanquantile(values, 0.75)),
        "target_sum": float(np.nansum(values)),
        "target_positive_share": float(np.nanmean(values > 0)),
        "target_zero_share": float(np.nanmean(values == 0)),
        "target_slope_last_first": float((values[-1] - values[0]) / max(1, len(values) - 1)),
    }
    feats["target_last_diff"] = float(values[-1] - values[-2]) if len(values) >= 2 else 0.0

    for lag in TARGET_LAGS:
        feats[f"target_lag_{lag}"] = float(values[-lag]) if len(values) >= lag else np.nan

    for window in ROLLING_WINDOWS:
        recent = values[-min(window, len(values)) :]
        feats[f"target_roll{window}_mean"] = float(np.nanmean(recent))
        feats[f"target_roll{window}_std"] = float(np.nanstd(recent))
        feats[f"target_roll{window}_min"] = float(np.nanmin(recent))
        feats[f"target_roll{window}_max"] = float(np.nanmax(recent))
    return feats


def context_covariate_features(context: pd.DataFrame) -> dict[str, float]:
    feats: dict[str, float] = {}
    for col in numeric_context_columns(context):
        if col == "target":
            continue
        values = pd.to_numeric(context[col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).sum() == 0:
            continue
        safe_name = safe_feature_name(col)
        feats[f"ctx_{safe_name}_last"] = float(values[-1]) if np.isfinite(values[-1]) else np.nan
        feats[f"ctx_{safe_name}_mean"] = float(np.nanmean(values))
        feats[f"ctx_{safe_name}_std"] = float(np.nanstd(values))
    return feats


def lightgbm_scalar_features(batch: WindowBatch) -> dict[str, float]:
    row = batch.metadata
    return {
        "context_steps": float(row["context_steps"]),
        "horizon_steps": float(row["horizon_steps"]),
        "native_step_minutes": float(row["native_step_minutes"]),
        **calendar_features(batch.origin_time, "origin"),
        **target_history_features(batch.context),
        **context_covariate_features(batch.context),
    }


def lightgbm_future_known_features(batch: WindowBatch) -> pd.DataFrame:
    target = batch.target.reset_index(drop=True)
    rows: list[dict[str, float]] = []
    for lead_step, target_row in target.iterrows():
        target_time = pd.Timestamp(target_row["timestamp"])
        rows.append(
            {
                "lead_step": float(lead_step + 1),
                "lead_fraction": float((lead_step + 1) / len(target)),
                **calendar_features(target_time, "target"),
            }
        )
    return pd.DataFrame(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value):
            return None
        if np.isposinf(value):
            return "Infinity"
        if np.isneginf(value):
            return "-Infinity"
        return float(value)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def stable_feature_hash(
    *,
    scalar_features: dict[str, float],
    future_known_features: pd.DataFrame,
    extra: dict[str, Any] | None = None,
) -> str:
    payload = {
        "scalar_features": {key: _jsonable(scalar_features[key]) for key in sorted(scalar_features)},
        "future_known_features": [
            {key: _jsonable(row[key]) for key in sorted(future_known_features.columns)}
            for row in future_known_features.to_dict(orient="records")
        ],
        "extra": {key: _jsonable(value) for key, value in sorted((extra or {}).items())},
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_lightgbm_equivalent_bundle(batch: WindowBatch) -> ContextFeatureBundle:
    scalar = lightgbm_scalar_features(batch)
    future = lightgbm_future_known_features(batch)
    leakage_contract = {
        "context_rows_only_for_measured_values": True,
        "future_known_features_are_deterministic_calendar_or_lead_position": True,
        "future_measured_covariates_used": False,
        "target_horizon_target_values_used_as_features": False,
    }
    digest = stable_feature_hash(
        scalar_features=scalar,
        future_known_features=future,
        extra={
            "window_id": batch.window_id,
            "domain_id": str(batch.metadata["domain_id"]),
            "horizon": str(batch.metadata["horizon"]),
            **leakage_contract,
        },
    )
    return ContextFeatureBundle(
        window_id=batch.window_id,
        scalar_features=scalar,
        future_known_features=future,
        scalar_feature_names=tuple(scalar.keys()),
        future_known_feature_names=tuple(future.columns),
        feature_bundle_hash=digest,
        leakage_contract=leakage_contract,
    )


def lightgbm_supervised_feature_rows(batch: WindowBatch) -> pd.DataFrame:
    bundle = build_lightgbm_equivalent_bundle(batch)
    rows = []
    for _, future_row in bundle.future_known_features.iterrows():
        row = dict(bundle.scalar_features)
        row.update({key: float(future_row[key]) for key in bundle.future_known_features.columns})
        rows.append(row)
    return pd.DataFrame(rows)


def domain_family_record(domain: str) -> dict[str, float]:
    return {
        "domain_family_load": 1.0 if "load" in domain else 0.0,
        "domain_family_pv": 1.0 if "pv" in domain else 0.0,
        "domain_family_industrial": 1.0 if "aluminum" in domain else 0.0,
        "domain_family_microgrid": 1.0 if "microgrid" in domain else 0.0,
        "domain_family_aidc": 1.0 if "aidc" in domain else 0.0,
    }


def bundles_to_static_covariates(
    bundles: Iterable[ContextFeatureBundle],
    *,
    prefix: str = "parity_",
    include_missing_indicators: bool = True,
    nan_fill_value: float = 0.0,
) -> tuple[dict[str, list[float]], tuple[str, ...]]:
    bundle_list = list(bundles)
    if not bundle_list:
        return {}, ()
    names = tuple(bundle_list[0].scalar_feature_names)
    covariates: dict[str, list[float]] = {}
    for name in names:
        values: list[float] = []
        missing: list[float] = []
        for bundle in bundle_list:
            value = float(bundle.scalar_features.get(name, np.nan))
            is_missing = not np.isfinite(value)
            values.append(float(nan_fill_value) if is_missing else value)
            missing.append(1.0 if is_missing else 0.0)
        covariates[f"{prefix}{name}"] = values
        if include_missing_indicators and any(v > 0 for v in missing):
            covariates[f"{prefix}{name}_is_missing"] = missing
    return covariates, tuple(covariates.keys())


def bundles_to_future_dynamic_covariates(
    batches: Iterable[WindowBatch],
    *,
    prefix: str = "parity_",
    context_fill_value: float = 0.0,
) -> tuple[dict[str, list[list[float]]], tuple[str, ...]]:
    batch_list = list(batches)
    if not batch_list:
        return {}, ()
    bundles = [build_lightgbm_equivalent_bundle(batch) for batch in batch_list]
    names = tuple(bundles[0].future_known_feature_names)
    covariates: dict[str, list[list[float]]] = {f"{prefix}{name}": [] for name in names}
    for batch, bundle in zip(batch_list, bundles, strict=True):
        context_len = len(batch.context)
        for name in names:
            future_values = pd.to_numeric(bundle.future_known_features[name], errors="raise").astype(float).to_list()
            covariates[f"{prefix}{name}"].append([float(context_fill_value)] * context_len + future_values)
    return covariates, tuple(covariates.keys())
