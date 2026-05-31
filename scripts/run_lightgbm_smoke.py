#!/usr/bin/env python3
"""Run small fixed-protocol LightGBM smoke tests on P1c frozen windows."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from energy_tsfm_p2_core import (
    MAIN_HORIZONS,
    P1cWindowDataset,
    build_prediction_stub,
    list_domains,
    load_window_index,
    validate_prediction_against_windows,
)
from evaluate_energy_tsfm_predictions import evaluate_prediction_frame, write_metrics


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = PROJECT / "results" / "energy_tsfm_p2_smoke" / "lightgbm"
DEFAULT_SEED = 20260514
MODEL_ID = "lightgbm"
MODEL_FAMILY = "tree"
CONFIG_ID = "lightgbm_smoke_standard_v1"

IDENTIFIER_COLUMNS = {
    "domain_id",
    "series_id",
    "segment_id",
    "split",
    "timestamp",
    "window_id",
}
NON_FEATURE_NUMERIC_COLUMNS = {"segment_row_index"}


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def sample_positions(n: int, limit: int, seed_key: str) -> list[int]:
    if n <= 0:
        return []
    if limit <= 0 or limit >= n:
        return list(range(n))
    anchor = {0, n // 2, n - 1}
    remaining = [i for i in range(n) if i not in anchor]
    rng = random.Random(seed_key)
    k = max(0, limit - len(anchor))
    selected = set(anchor)
    selected.update(rng.sample(remaining, k=min(k, len(remaining))))
    return sorted(selected)


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
            cols.append(col)
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
    if len(values) >= 2:
        feats["target_last_diff"] = float(values[-1] - values[-2])
    else:
        feats["target_last_diff"] = 0.0

    for lag in (1, 2, 4, 8, 16, 24, 48, 96, 144):
        feats[f"target_lag_{lag}"] = float(values[-lag]) if len(values) >= lag else np.nan

    for window in (4, 8, 16, 32):
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
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in col)
        feats[f"ctx_{safe_name}_last"] = float(values[-1]) if np.isfinite(values[-1]) else np.nan
        feats[f"ctx_{safe_name}_mean"] = float(np.nanmean(values))
        feats[f"ctx_{safe_name}_std"] = float(np.nanstd(values))
    return feats


def base_window_features(batch: Any) -> dict[str, float]:
    row = batch.metadata
    feats = {
        "context_steps": float(row["context_steps"]),
        "horizon_steps": float(row["horizon_steps"]),
        "native_step_minutes": float(row["native_step_minutes"]),
        **calendar_features(batch.origin_time, "origin"),
        **target_history_features(batch.context),
        **context_covariate_features(batch.context),
    }
    return feats


def build_supervised_rows(
    dataset: P1cWindowDataset,
    positions: list[int],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, Any]]:
    feature_rows: list[dict[str, float]] = []
    targets: list[float] = []
    row_meta: list[dict[str, Any]] = []
    batch_by_window: dict[str, Any] = {}

    for pos in positions:
        batch = dataset.get(pos)
        batch_by_window[batch.window_id] = batch
        base = base_window_features(batch)
        target = batch.target.reset_index(drop=True)
        for lead_step, target_row in target.iterrows():
            target_time = pd.Timestamp(target_row["timestamp"])
            feature_rows.append(
                {
                    **base,
                    "lead_step": float(lead_step + 1),
                    "lead_fraction": float((lead_step + 1) / len(target)),
                    **calendar_features(target_time, "target"),
                }
            )
            targets.append(float(target_row["target"]))
            row_meta.append({"window_id": batch.window_id, "lead_step": int(lead_step + 1)})

    return (
        pd.DataFrame(feature_rows),
        pd.Series(targets, name="target"),
        pd.DataFrame(row_meta),
        batch_by_window,
    )


def prediction_rows_from_model(
    model: LGBMRegressor,
    dataset: P1cWindowDataset,
    positions: list[int],
    *,
    seed: int,
    notes: str,
) -> tuple[pd.DataFrame, int]:
    x_pred, _, meta, batch_by_window = build_supervised_rows(dataset, positions)
    y_hat = np.asarray(model.predict(x_pred), dtype=float)
    meta = meta.copy()
    meta["y_pred"] = y_hat
    rows: list[dict[str, Any]] = []
    for window_id, group in meta.groupby("window_id", sort=False):
        group = group.sort_values("lead_step")
        batch = batch_by_window[str(window_id)]
        rows.append(
            build_prediction_stub(
                batch,
                model_family=MODEL_FAMILY,
                model_id=MODEL_ID,
                config_id=CONFIG_ID,
                seed=seed,
                y_pred=pd.Series(group["y_pred"].to_numpy(dtype=float)),
                notes=notes,
            )
        )
    return pd.DataFrame(rows), int(len(x_pred))


def fit_lightgbm(x_train: pd.DataFrame, y_train: pd.Series, seed: int) -> LGBMRegressor:
    model = LGBMRegressor(
        n_estimators=80,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="regression",
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    return model


def run_one(
    domain: str,
    horizon: str,
    *,
    max_train_windows: int,
    max_eval_windows: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.time()
    train_dataset = P1cWindowDataset(domain, horizon, split="train")
    train_positions = sample_positions(
        len(train_dataset), max_train_windows, f"{seed}:{domain}:{horizon}:train"
    )
    x_train, y_train, _, _ = build_supervised_rows(train_dataset, train_positions)
    if x_train.empty:
        raise ValueError(f"{domain}/{horizon}: no training rows for smoke run")

    model = fit_lightgbm(x_train, y_train, seed)
    prediction_frames: list[pd.DataFrame] = []
    split_meta: dict[str, Any] = {
        "train_windows_sampled": len(train_positions),
        "train_rows": int(len(x_train)),
    }
    for split in ("validation", "test"):
        dataset = P1cWindowDataset(domain, horizon, split=split)
        positions = sample_positions(len(dataset), max_eval_windows, f"{seed}:{domain}:{horizon}:{split}")
        preds, feature_rows = prediction_rows_from_model(
            model,
            dataset,
            positions,
            seed=seed,
            notes=(
                "LightGBM smoke run using P1c frozen windows and context-only tabular features; "
                "not a formal paper result."
            ),
        )
        if not preds.empty:
            index = load_window_index(domain, horizon, split=split)
            validate_prediction_against_windows(preds, index)
            prediction_frames.append(preds)
        split_meta[f"{split}_windows_sampled"] = len(positions)
        split_meta[f"{split}_feature_rows"] = feature_rows

    if not prediction_frames:
        raise ValueError(f"{domain}/{horizon}: no prediction rows")
    elapsed = time.time() - started
    manifest = {
        "domain": domain,
        "horizon": horizon,
        "model_id": MODEL_ID,
        "model_family": MODEL_FAMILY,
        "config_id": CONFIG_ID,
        "seed": seed,
        "max_train_windows": max_train_windows,
        "max_eval_windows": max_eval_windows,
        "elapsed_seconds": elapsed,
        "n_features": int(x_train.shape[1]),
        **split_meta,
    }
    return pd.concat(prediction_frames, ignore_index=True), manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="*", default=None, help="Domain IDs. Default: all P1c domains.")
    parser.add_argument("--horizons", nargs="*", default=list(MAIN_HORIZONS), choices=list(MAIN_HORIZONS))
    parser.add_argument("--max-train-windows", type=int, default=64)
    parser.add_argument("--max-eval-windows", type=int, default=24)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domains = args.domains if args.domains else list_domains()
    run_id = (
        f"{MODEL_ID}_smoke_seed{args.seed}_"
        f"tr{args.max_train_windows}_ev{args.max_eval_windows}"
    )
    out_dir = args.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    all_predictions: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for domain in domains:
        for horizon in args.horizons:
            preds, manifest = run_one(
                domain,
                horizon,
                max_train_windows=args.max_train_windows,
                max_eval_windows=args.max_eval_windows,
                seed=args.seed,
            )
            domain_dir = out_dir / domain / horizon
            domain_dir.mkdir(parents=True, exist_ok=True)
            pred_path = domain_dir / "predictions.parquet"
            manifest_path = domain_dir / "manifest.json"
            preds.to_parquet(pred_path, index=False)
            manifest["prediction_path"] = str(pred_path)
            manifest_path.write_text(dumps(manifest) + "\n", encoding="utf-8")
            manifests.append(manifest)
            all_predictions.append(preds)

    combined = pd.concat(all_predictions, ignore_index=True)
    combined_path = out_dir / "predictions_all.parquet"
    combined.to_parquet(combined_path, index=False)
    metrics = evaluate_prediction_frame(combined)
    metric_paths = write_metrics(metrics, out_dir, stem="metrics")

    run_manifest = {
        "run_id": run_id,
        "model_id": MODEL_ID,
        "config_id": CONFIG_ID,
        "domains": domains,
        "horizons": args.horizons,
        "seed": args.seed,
        "prediction_path": str(combined_path),
        "metric_paths": metric_paths,
        "runs": manifests,
    }
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(dumps(run_manifest) + "\n", encoding="utf-8")
    print(dumps({"status": "ok", "out_dir": str(out_dir), "prediction_path": str(combined_path), **metric_paths}))


if __name__ == "__main__":
    main()
