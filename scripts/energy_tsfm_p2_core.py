#!/usr/bin/env python3
"""Shared P2 infrastructure for the new energy-TSFM paper.

This module deliberately contains no model-specific training logic. It freezes
the common task contract that all P2/P3 runners must use:

- read P1c frozen window indexes;
- load context/target slices from segment-safe canonical P1b data;
- enforce leakage and segment-boundary assertions;
- expose a common prediction schema for downstream evaluators.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
P1B_CANONICAL_ROOT = PROJECT_ROOT / "data" / "energy_tsfm_canonical_p1b"
P1C_WINDOW_ROOT = PROJECT_ROOT / "data" / "energy_tsfm_windows_p1c"
MODEL_REGISTRY_PATH = PROJECT_ROOT / "config" / "energy_tsfm_model_registry.json"

MAIN_HORIZONS = ("4h", "24h")

REQUIRED_WINDOW_COLUMNS = {
    "window_id",
    "domain_id",
    "series_id",
    "segment_id",
    "horizon",
    "context_steps",
    "horizon_steps",
    "native_step_minutes",
    "context_start_row_index",
    "context_end_row_index",
    "forecast_start_row_index",
    "forecast_end_row_index",
    "context_start_timestamp",
    "context_end_timestamp",
    "forecast_start_timestamp",
    "forecast_end_timestamp",
    "split",
}

REQUIRED_CANONICAL_COLUMNS = {
    "domain_id",
    "series_id",
    "segment_id",
    "segment_row_index",
    "timestamp",
    "target",
    "split",
    "native_step_minutes",
}

PREDICTION_SCHEMA_COLUMNS = [
    "domain_id",
    "series_id",
    "segment_id",
    "horizon",
    "window_id",
    "model_family",
    "model_id",
    "config_id",
    "seed",
    "origin_time",
    "target_start_time",
    "target_end_time",
    "y_true",
    "y_pred",
]

OPTIONAL_PREDICTION_COLUMNS = ["q10", "q50", "q90", "prediction_path", "notes"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_model_registry(path: Path = MODEL_REGISTRY_PATH) -> dict[str, Any]:
    registry = read_json(path)
    model_ids = [m["model_id"] for m in registry.get("models", [])]
    duplicates = sorted({m for m in model_ids if model_ids.count(m) > 1})
    if duplicates:
        raise ValueError(f"duplicate model_id entries in registry: {duplicates}")
    return registry


def list_domains(window_root: Path = P1C_WINDOW_ROOT) -> list[str]:
    if not window_root.exists():
        raise FileNotFoundError(f"missing P1c window root: {window_root}")
    return sorted(p.name for p in window_root.iterdir() if p.is_dir())


def canonical_path(domain_id: str) -> Path:
    return P1B_CANONICAL_ROOT / domain_id / "canonical_segmented.parquet"


def window_index_path(domain_id: str, horizon: str) -> Path:
    if horizon not in MAIN_HORIZONS:
        raise ValueError(f"invalid horizon {horizon!r}; expected one of {MAIN_HORIZONS}")
    return P1C_WINDOW_ROOT / domain_id / f"window_index_{horizon}.parquet"


def _require_columns(df: pd.DataFrame, required: set[str], source: Path | str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def load_window_index(domain_id: str, horizon: str, split: str | None = None) -> pd.DataFrame:
    path = window_index_path(domain_id, horizon)
    if not path.exists():
        raise FileNotFoundError(f"missing P1c window index: {path}")
    df = pd.read_parquet(path)
    _require_columns(df, REQUIRED_WINDOW_COLUMNS, path)
    if split is not None:
        df = df[df["split"].astype(str) == split].copy()
    return df.reset_index(drop=True)


def load_canonical(domain_id: str) -> pd.DataFrame:
    path = canonical_path(domain_id)
    if not path.exists():
        raise FileNotFoundError(f"missing P1b canonical data: {path}")
    df = pd.read_parquet(path)
    _require_columns(df, REQUIRED_CANONICAL_COLUMNS, path)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    return df


@dataclass(frozen=True)
class WindowBatch:
    metadata: pd.Series
    context: pd.DataFrame
    target: pd.DataFrame

    @property
    def window_id(self) -> str:
        return str(self.metadata["window_id"])

    @property
    def origin_time(self) -> pd.Timestamp:
        return pd.Timestamp(self.metadata["context_end_timestamp"])

    @property
    def target_start_time(self) -> pd.Timestamp:
        return pd.Timestamp(self.metadata["forecast_start_timestamp"])

    @property
    def target_end_time(self) -> pd.Timestamp:
        return pd.Timestamp(self.metadata["forecast_end_timestamp"])


class P1cWindowDataset:
    """Reader for the frozen P1c rolling-window index."""

    def __init__(self, domain_id: str, horizon: str, split: str | None = None) -> None:
        self.domain_id = domain_id
        self.horizon = horizon
        self.split = split
        self.windows = load_window_index(domain_id, horizon, split=split)
        canonical = load_canonical(domain_id)
        self._segments = {
            str(segment_id): seg.sort_values("segment_row_index").reset_index(drop=True)
            for segment_id, seg in canonical.groupby("segment_id", sort=False)
        }

    def __len__(self) -> int:
        return int(len(self.windows))

    def get(self, index: int) -> WindowBatch:
        row = self.windows.iloc[index]
        segment_id = str(row["segment_id"])
        if segment_id not in self._segments:
            raise KeyError(f"segment {segment_id!r} not found for domain {self.domain_id}")
        seg = self._segments[segment_id]
        context_start = int(row["context_start_row_index"])
        context_end = int(row["context_end_row_index"])
        target_start = int(row["forecast_start_row_index"])
        target_end = int(row["forecast_end_row_index"])
        context = seg.iloc[context_start : context_end + 1].copy()
        target = seg.iloc[target_start : target_end + 1].copy()
        batch = WindowBatch(metadata=row, context=context, target=target)
        assert_window_contract(batch)
        return batch

    def iter_batches(self, limit: int | None = None) -> Iterable[WindowBatch]:
        n = len(self) if limit is None else min(int(limit), len(self))
        for idx in range(n):
            yield self.get(idx)


def assert_window_contract(batch: WindowBatch) -> None:
    row = batch.metadata
    context = batch.context
    target = batch.target

    if context.empty or target.empty:
        raise AssertionError(f"{row['window_id']} has empty context or target")

    expected_context_steps = int(row["context_steps"])
    expected_horizon_steps = int(row["horizon_steps"])
    if len(context) != expected_context_steps:
        raise AssertionError(
            f"{row['window_id']} context length {len(context)} != {expected_context_steps}"
        )
    if len(target) != expected_horizon_steps:
        raise AssertionError(
            f"{row['window_id']} target length {len(target)} != {expected_horizon_steps}"
        )

    segment_id = str(row["segment_id"])
    if set(context["segment_id"].astype(str)) != {segment_id}:
        raise AssertionError(f"{row['window_id']} context crosses segment_id")
    if set(target["segment_id"].astype(str)) != {segment_id}:
        raise AssertionError(f"{row['window_id']} target crosses segment_id")

    context_max = pd.Timestamp(context["timestamp"].max())
    target_min = pd.Timestamp(target["timestamp"].min())
    target_max = pd.Timestamp(target["timestamp"].max())
    origin_time = pd.Timestamp(row["context_end_timestamp"])
    target_start = pd.Timestamp(row["forecast_start_timestamp"])
    target_end = pd.Timestamp(row["forecast_end_timestamp"])

    if context_max != origin_time:
        raise AssertionError(f"{row['window_id']} context endpoint mismatch")
    if target_min != target_start:
        raise AssertionError(f"{row['window_id']} target start mismatch")
    if target_max != target_end:
        raise AssertionError(f"{row['window_id']} target end mismatch")
    if context_max >= target_min:
        raise AssertionError(f"{row['window_id']} leaks future target into context")
    if target["target"].isna().any():
        raise AssertionError(f"{row['window_id']} target contains NaN")

    endpoint_split = str(target.iloc[-1]["split"])
    if endpoint_split != str(row["split"]):
        raise AssertionError(
            f"{row['window_id']} split mismatch: endpoint={endpoint_split}, index={row['split']}"
        )


def serialize_series(values: pd.Series) -> str:
    numeric = pd.to_numeric(values, errors="coerce")
    return json.dumps([None if pd.isna(v) else float(v) for v in numeric], separators=(",", ":"))


def build_prediction_stub(
    batch: WindowBatch,
    *,
    model_family: str,
    model_id: str,
    config_id: str,
    seed: int | None = None,
    y_pred: pd.Series | None = None,
    notes: str = "schema stub; not a model result",
) -> dict[str, Any]:
    if y_pred is None:
        y_pred_values = [None for _ in range(len(batch.target))]
        y_pred_serialized = json.dumps(y_pred_values, separators=(",", ":"))
    else:
        if len(y_pred) != len(batch.target):
            raise ValueError("y_pred length must match target length")
        y_pred_serialized = serialize_series(y_pred)

    return {
        "domain_id": str(batch.metadata["domain_id"]),
        "series_id": str(batch.metadata["series_id"]),
        "segment_id": str(batch.metadata["segment_id"]),
        "horizon": str(batch.metadata["horizon"]),
        "window_id": str(batch.metadata["window_id"]),
        "model_family": model_family,
        "model_id": model_id,
        "config_id": config_id,
        "seed": seed,
        "origin_time": str(batch.origin_time),
        "target_start_time": str(batch.target_start_time),
        "target_end_time": str(batch.target_end_time),
        "y_true": serialize_series(batch.target["target"]),
        "y_pred": y_pred_serialized,
        "notes": notes,
    }


def validate_prediction_schema(predictions: pd.DataFrame) -> None:
    missing = [c for c in PREDICTION_SCHEMA_COLUMNS if c not in predictions.columns]
    if missing:
        raise ValueError(f"prediction table missing required columns: {missing}")
    for col in ["origin_time", "target_start_time", "target_end_time"]:
        pd.to_datetime(predictions[col], errors="raise")
    if predictions["window_id"].isna().any():
        raise ValueError("prediction table contains missing window_id")
    if predictions["model_id"].isna().any():
        raise ValueError("prediction table contains missing model_id")


def validate_prediction_against_windows(predictions: pd.DataFrame, window_index: pd.DataFrame) -> None:
    validate_prediction_schema(predictions)
    _require_columns(window_index, REQUIRED_WINDOW_COLUMNS, "window_index")
    lookup = window_index.set_index("window_id", drop=False)
    missing_windows = sorted(set(predictions["window_id"]) - set(lookup.index))
    if missing_windows:
        raise ValueError(f"prediction table references unknown windows: {missing_windows[:5]}")

    for pred in predictions.itertuples(index=False):
        win = lookup.loc[getattr(pred, "window_id")]
        if str(getattr(pred, "domain_id")) != str(win["domain_id"]):
            raise ValueError(f"{pred.window_id}: domain_id mismatch")
        if str(getattr(pred, "series_id")) != str(win["series_id"]):
            raise ValueError(f"{pred.window_id}: series_id mismatch")
        if str(getattr(pred, "segment_id")) != str(win["segment_id"]):
            raise ValueError(f"{pred.window_id}: segment_id mismatch")
        if str(getattr(pred, "horizon")) != str(win["horizon"]):
            raise ValueError(f"{pred.window_id}: horizon mismatch")
        if pd.Timestamp(getattr(pred, "origin_time")) != pd.Timestamp(win["context_end_timestamp"]):
            raise ValueError(f"{pred.window_id}: origin_time mismatch")
        if pd.Timestamp(getattr(pred, "target_start_time")) != pd.Timestamp(win["forecast_start_timestamp"]):
            raise ValueError(f"{pred.window_id}: target_start_time mismatch")
        if pd.Timestamp(getattr(pred, "target_end_time")) != pd.Timestamp(win["forecast_end_timestamp"]):
            raise ValueError(f"{pred.window_id}: target_end_time mismatch")
        if pd.Timestamp(getattr(pred, "origin_time")) >= pd.Timestamp(getattr(pred, "target_start_time")):
            raise ValueError(f"{pred.window_id}: origin_time leaks into target interval")
