#!/usr/bin/env python3
"""Run covariate-aware NeuralForecast DL smoke tests on P1c windows.

This is the companion to `run_neuralforecast_dl_smoke.py`. The older runner is
target-only by design. This runner validates covariate paths for models that
support them under the P1c task contract. In NeuralForecast 3.1.7, `NBEATSx`
and `TFT` support the covariate path used here; `iTransformer` is retained as a
documented target-only backend fallback because this implementation rejects
historical and future exogenous variables at runtime.

Information boundary:

- historical exogenous variables come only from the context rows at evaluation;
- future inputs are limited to known-ahead calendar features;
- future measured weather, irradiance, PV, load, demand or power values are not
  used during validation/test prediction.

The outputs are smoke artifacts only and are not formal paper results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx, TFT, iTransformer

from energy_tsfm_p2_core import (
    MAIN_HORIZONS,
    P1cWindowDataset,
    build_prediction_stub,
    list_domains,
    load_canonical,
    load_window_index,
    validate_prediction_against_windows,
)
from evaluate_energy_tsfm_predictions import evaluate_prediction_frame, write_metrics


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = PROJECT / "results" / "energy_tsfm_p2_smoke" / "neuralforecast_dl_covariate"
DEFAULT_SEED = 20260514
MODEL_FAMILY = "dl"
SUPPORTED_MODELS = ("nbeatsx", "itransformer", "tft")

CALENDAR_FEATURES = [
    "cal_hour_sin",
    "cal_hour_cos",
    "cal_dow_sin",
    "cal_dow_cos",
    "cal_month_sin",
    "cal_month_cos",
    "cal_is_weekend",
]

EXCLUDED_HIST_EXOG_COLUMNS = {
    "segment_row_index",
    "target",
    "target_raw",
    "is_valid_target",
    "split",
    "native_step_minutes",
    "is_imputed_target",
}


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
    selected = set(anchor)
    selected.update(rng.sample(remaining, k=min(max(0, limit - len(anchor)), len(remaining))))
    return sorted(selected)


def calendar_features(timestamp: Any) -> dict[str, float]:
    ts = pd.Timestamp(timestamp)
    hour = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    dow = ts.dayofweek
    month = ts.month
    return {
        "cal_hour_sin": math.sin(2.0 * math.pi * hour / 24.0),
        "cal_hour_cos": math.cos(2.0 * math.pi * hour / 24.0),
        "cal_dow_sin": math.sin(2.0 * math.pi * dow / 7.0),
        "cal_dow_cos": math.cos(2.0 * math.pi * dow / 7.0),
        "cal_month_sin": math.sin(2.0 * math.pi * (month - 1) / 12.0),
        "cal_month_cos": math.cos(2.0 * math.pi * (month - 1) / 12.0),
        "cal_is_weekend": float(dow >= 5),
    }


def select_hist_exog_columns(
    domain: str,
    *,
    max_hist_exog: int,
    min_coverage: float,
) -> list[str]:
    canonical = load_canonical(domain)
    candidates: list[str] = []
    for col in canonical.columns:
        if col in EXCLUDED_HIST_EXOG_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(canonical[col]):
            continue
        series = pd.to_numeric(canonical[col], errors="coerce")
        coverage = float(series.notna().mean())
        nunique = int(series.nunique(dropna=True))
        if coverage >= min_coverage and nunique > 1:
            candidates.append(col)
    return candidates[:max_hist_exog]


def context_fill_values(context: pd.DataFrame, hist_exog_cols: list[str]) -> dict[str, float]:
    fill: dict[str, float] = {}
    for col in hist_exog_cols:
        values = pd.to_numeric(context[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(values.median()) if values.notna().any() else 0.0
        fill[col] = median if np.isfinite(median) else 0.0
    return fill


def hist_exog_values(row: pd.Series, hist_exog_cols: list[str], fill: dict[str, float]) -> dict[str, float]:
    values: dict[str, float] = {}
    for col in hist_exog_cols:
        value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
        if pd.isna(value) or not np.isfinite(float(value)):
            value = fill[col]
        values[col] = float(value)
    return values


def rows_for_windows(
    dataset: P1cWindowDataset,
    positions: list[int],
    *,
    hist_exog_cols: list[str],
    include_future_target: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    futr_rows: list[dict[str, Any]] = []
    uid_to_batch: dict[str, Any] = {}

    for ordinal, pos in enumerate(positions):
        batch = dataset.get(pos)
        uid = f"{dataset.domain_id}__{dataset.horizon}__{dataset.split}__{ordinal:06d}"
        uid_to_batch[uid] = batch

        context = batch.context.reset_index(drop=True)
        target = batch.target.reset_index(drop=True)
        fill = context_fill_values(context, hist_exog_cols)

        if include_future_target:
            frame = pd.concat([context, target], ignore_index=True)
        else:
            frame = context

        for ds, row in frame.iterrows():
            rec = {
                "unique_id": uid,
                "ds": int(ds),
                "y": float(pd.to_numeric(pd.Series([row["target"]]), errors="coerce").iloc[0]),
            }
            rec.update(calendar_features(row["timestamp"]))
            rec.update(hist_exog_values(row, hist_exog_cols, fill))
            rows.append(rec)

        if not include_future_target:
            context_len = len(context)
            for lead_step, row in target.iterrows():
                rec = {
                    "unique_id": uid,
                    "ds": int(context_len + lead_step),
                }
                rec.update(calendar_features(row["timestamp"]))
                futr_rows.append(rec)

    df = pd.DataFrame(rows)
    futr_df = None if include_future_target else pd.DataFrame(futr_rows)
    return df, futr_df, uid_to_batch


def make_model(
    model_id: str,
    *,
    h: int,
    input_size: int,
    max_steps: int,
    seed: int,
    hist_exog_cols: list[str],
    accelerator: str,
    devices: int | str,
) -> Any:
    model_hist_exog_cols = list(hist_exog_cols)
    model_futr_exog_cols: list[str] | None = CALENDAR_FEATURES
    if model_id == "itransformer":
        # NeuralForecast 3.1.7 exposes exogenous arguments in the signature, but
        # this iTransformer implementation raises at runtime when historical or
        # future exogs are supplied. Keep it as a backend-limited target-only
        # fallback in this smoke and record the limitation in the manifest.
        model_hist_exog_cols = []
        model_futr_exog_cols = None
    common = {
        "h": h,
        "input_size": input_size,
        "hist_exog_list": model_hist_exog_cols or None,
        "futr_exog_list": model_futr_exog_cols,
        "max_steps": max_steps,
        "batch_size": 4,
        "valid_batch_size": 4,
        "scaler_type": "identity",
        "random_seed": seed,
        "alias": model_id,
        "enable_progress_bar": False,
        "logger": False,
        "enable_checkpointing": False,
        "accelerator": accelerator,
        "devices": devices,
    }
    if model_id == "nbeatsx":
        return NBEATSx(
            **common,
            n_blocks=[1, 1, 1],
            mlp_units=[[32, 32], [32, 32], [32, 32]],
            windows_batch_size=16,
            inference_windows_batch_size=16,
        )
    if model_id == "itransformer":
        return iTransformer(
            **common,
            n_series=1,
            hidden_size=32,
            n_heads=2,
            e_layers=1,
            d_layers=1,
            d_ff=64,
            dropout=0.0,
            windows_batch_size=8,
            inference_windows_batch_size=8,
        )
    if model_id == "tft":
        return TFT(
            **common,
            hidden_size=16,
            n_head=1,
            n_rnn_layers=1,
            dropout=0.0,
            windows_batch_size=16,
            inference_windows_batch_size=16,
        )
    raise ValueError(f"unsupported model_id: {model_id}")


def predictions_from_forecast(
    forecast: pd.DataFrame,
    uid_to_batch: dict[str, Any],
    *,
    model_id: str,
    config_id: str,
    seed: int,
) -> pd.DataFrame:
    if model_id not in forecast.columns:
        raise ValueError(f"forecast table missing model output column {model_id!r}: {list(forecast.columns)}")
    rows: list[dict[str, Any]] = []
    for uid, group in forecast.groupby("unique_id", sort=False):
        if uid not in uid_to_batch:
            raise KeyError(f"forecast returned unknown unique_id: {uid}")
        batch = uid_to_batch[uid]
        group = group.sort_values("ds")
        y_pred = pd.to_numeric(group[model_id], errors="coerce")
        rows.append(
            build_prediction_stub(
                batch,
                model_family=MODEL_FAMILY,
                model_id=model_id,
                config_id=config_id,
                seed=seed,
                y_pred=y_pred.reset_index(drop=True),
                notes=(
                    "NeuralForecast covariate-aware smoke using context historical exogs "
                    "and known-future calendar features only; not a formal paper result."
                ),
            )
        )
    return pd.DataFrame(rows)


def run_one(
    model_id: str,
    domain: str,
    horizon: str,
    *,
    max_train_windows: int,
    max_eval_windows: int,
    max_steps: int,
    max_hist_exog: int,
    min_covariate_coverage: float,
    accelerator: str,
    devices: int | str,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.time()
    hist_exog_cols = select_hist_exog_columns(
        domain,
        max_hist_exog=max_hist_exog,
        min_coverage=min_covariate_coverage,
    )
    train_dataset = P1cWindowDataset(domain, horizon, split="train")
    h = int(train_dataset.windows.iloc[0]["horizon_steps"])
    input_size = int(train_dataset.windows.iloc[0]["context_steps"])
    train_positions = sample_positions(
        len(train_dataset), max_train_windows, f"{seed}:{model_id}:{domain}:{horizon}:train:cov"
    )
    train_df, _, _ = rows_for_windows(
        train_dataset,
        train_positions,
        hist_exog_cols=hist_exog_cols,
        include_future_target=True,
    )
    if model_id == "itransformer":
        train_df = train_df[["unique_id", "ds", "y"]].copy()
    if train_df.empty:
        raise ValueError(f"{model_id}/{domain}/{horizon}: empty training frame")

    config_id = f"{model_id}_neuralforecast_covariate_smoke_v1"
    model_hist_exog_cols = list(hist_exog_cols)
    model_futr_exog_cols = list(CALENDAR_FEATURES)
    covariate_mode = "historical_exog_plus_future_calendar"
    if model_id == "itransformer":
        model_hist_exog_cols = []
        model_futr_exog_cols = []
        covariate_mode = "target_only_backend_fallback"
    model = make_model(
        model_id,
        h=h,
        input_size=input_size,
        max_steps=max_steps,
        seed=seed,
        hist_exog_cols=hist_exog_cols,
        accelerator=accelerator,
        devices=devices,
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df=train_df, verbose=False)

    prediction_frames: list[pd.DataFrame] = []
    split_meta: dict[str, Any] = {
        "train_windows_sampled": len(train_positions),
        "train_rows": int(len(train_df)),
    }
    for split in ("validation", "test"):
        dataset = P1cWindowDataset(domain, horizon, split=split)
        positions = sample_positions(
            len(dataset), max_eval_windows, f"{seed}:{model_id}:{domain}:{horizon}:{split}:cov"
        )
        pred_df, futr_df, uid_to_batch = rows_for_windows(
            dataset,
            positions,
            hist_exog_cols=hist_exog_cols,
            include_future_target=False,
        )
        if model_id == "itransformer":
            pred_df = pred_df[["unique_id", "ds", "y"]].copy()
            futr_df = None
        forecast = nf.predict(
            df=pred_df,
            futr_df=None if model_id == "itransformer" else futr_df,
            verbose=False,
        )
        preds = predictions_from_forecast(
            forecast,
            uid_to_batch,
            model_id=model_id,
            config_id=config_id,
            seed=seed,
        )
        index = load_window_index(domain, horizon, split=split)
        validate_prediction_against_windows(preds, index)
        prediction_frames.append(preds)
        split_meta[f"{split}_windows_sampled"] = len(positions)
        split_meta[f"{split}_rows"] = int(len(pred_df))
        split_meta[f"{split}_future_calendar_rows"] = 0 if futr_df is None else int(len(futr_df))

    elapsed = time.time() - started
    manifest = {
        "model_id": model_id,
        "model_family": MODEL_FAMILY,
        "config_id": config_id,
        "implementation": "NeuralForecast official class",
        "domain": domain,
        "horizon": horizon,
        "seed": seed,
        "max_steps": max_steps,
        "max_train_windows": max_train_windows,
        "max_eval_windows": max_eval_windows,
        "context_steps": input_size,
        "horizon_steps": h,
        "hist_exog_cols": model_hist_exog_cols,
        "futr_exog_cols": model_futr_exog_cols,
        "hist_exog_count": len(model_hist_exog_cols),
        "futr_exog_count": len(model_futr_exog_cols),
        "covariate_mode": covariate_mode,
        "future_measured_exog_used_at_eval": False,
        "accelerator": accelerator,
        "devices": devices,
        "elapsed_seconds": elapsed,
        **split_meta,
        "interpretation": "covariate-aware smoke only; not a formal paper result",
    }
    return pd.concat(prediction_frames, ignore_index=True), manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", choices=SUPPORTED_MODELS, default=list(SUPPORTED_MODELS))
    parser.add_argument("--domains", nargs="*", default=None, help="Domain IDs. Default: all P1c domains.")
    parser.add_argument("--horizons", nargs="*", choices=list(MAIN_HORIZONS), default=list(MAIN_HORIZONS))
    parser.add_argument("--max-train-windows", type=int, default=8)
    parser.add_argument("--max-eval-windows", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-hist-exog", type=int, default=8)
    parser.add_argument("--min-covariate-coverage", type=float, default=0.95)
    parser.add_argument("--accelerator", default="auto", choices=["cuda", "auto", "gpu"])
    parser.add_argument("--devices", default="1")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/GPU is required for DL model smoke tests in this project")
    if args.accelerator == "cpu":
        raise ValueError("CPU execution is forbidden for DL/TSFM tests in this project")
    if args.accelerator in {"auto", "cuda"}:
        args.accelerator = "gpu"
    domains = args.domains if args.domains else list_domains()
    devices_tag = str(args.devices).replace(",", "-").replace("/", "-")
    run_id = (
        f"nf_dl_covariate_smoke_seed{args.seed}_tr{args.max_train_windows}_"
        f"ev{args.max_eval_windows}_steps{args.max_steps}_hist{args.max_hist_exog}_"
        f"acc{args.accelerator}_dev{devices_tag}"
    )
    out_dir = args.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    all_predictions: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for model_id in args.models:
        for domain in domains:
            for horizon in args.horizons:
                preds, manifest = run_one(
                    model_id,
                    domain,
                    horizon,
                    max_train_windows=args.max_train_windows,
                    max_eval_windows=args.max_eval_windows,
                    max_steps=args.max_steps,
                    max_hist_exog=args.max_hist_exog,
                    min_covariate_coverage=args.min_covariate_coverage,
                    accelerator=args.accelerator,
                    devices=args.devices,
                    seed=args.seed,
                )
                model_dir = out_dir / model_id / domain / horizon
                model_dir.mkdir(parents=True, exist_ok=True)
                pred_path = model_dir / "predictions.parquet"
                manifest_path = model_dir / "manifest.json"
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
        "models": args.models,
        "domains": domains,
        "horizons": args.horizons,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "max_train_windows": args.max_train_windows,
        "max_eval_windows": args.max_eval_windows,
        "max_hist_exog": args.max_hist_exog,
        "min_covariate_coverage": args.min_covariate_coverage,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "futr_exog_cols": CALENDAR_FEATURES,
        "future_measured_exog_used_at_eval": False,
        "prediction_path": str(combined_path),
        "metric_paths": metric_paths,
        "runs": manifests,
        "interpretation": "covariate-aware smoke only; not formal paper results",
    }
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(dumps(run_manifest) + "\n", encoding="utf-8")
    print(dumps({"status": "ok", "out_dir": str(out_dir), "prediction_path": str(combined_path), **metric_paths}))


if __name__ == "__main__":
    main()
