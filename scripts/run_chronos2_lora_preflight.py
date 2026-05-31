#!/usr/bin/env python3
"""Run a tiny Chronos-2 LoRA fine-tuning preflight on frozen P1c windows.

This is an adaptation-route preflight, not a manuscript result. It checks that
the full-power Chronos-2 checkpoint can be fine-tuned with LoRA under the
project's P1c task contract and compared against the same checkpoint's
covariate-aware zero-shot predictions.

Information boundary:

- train windows are used for fine-tuning only;
- validation windows are used only for validation/eval during fine-tuning and
  for validation predictions;
- test windows are predicted once after the LoRA configuration is fixed;
- prediction inputs use context target, past-only numeric context covariates and
  known-future calendar covariates only;
- validation/test prediction inputs do not use future measured weather,
  irradiance, load, PV, demand or power covariates.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

from energy_tsfm_p2_core import (
    P1cWindowDataset,
    build_prediction_stub,
    list_domains,
    load_window_index,
    serialize_series,
    validate_prediction_against_windows,
)
from evaluate_energy_tsfm_predictions import evaluate_prediction_frame, write_metrics


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = PROJECT / "results" / "energy_tsfm_p2_smoke" / "chronos2_lora"
DEFAULT_SEED = 20260514
MODEL_ID = "chronos2"
MODEL_FAMILY = "tsfm"
BASE_CONFIG_ID = "chronos2_covariate_zero_shot_preflight_v1"
LORA_CONFIG_ID = "chronos2_lora_finetune_preflight_v1"
DEFAULT_MODEL_NAME = "amazon/chronos-2"

IDENTIFIER_COLUMNS = {
    "domain_id",
    "series_id",
    "segment_id",
    "split",
    "timestamp",
    "window_id",
}
NON_COVARIATE_NUMERIC_COLUMNS = {
    "target",
    "segment_row_index",
    "native_step_minutes",
}


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def sample_positions(n: int, limit: int, seed_key: str) -> list[int]:
    if n <= 0:
        return []
    if limit <= 0 or limit >= n:
        return list(range(n))
    anchor_order = [0, n - 1, n // 2]
    anchor = set(anchor_order[: max(0, min(limit, len(anchor_order)))])
    remaining = [i for i in range(n) if i not in anchor]
    rng = random.Random(seed_key)
    k = max(0, limit - len(anchor))
    selected = set(anchor)
    selected.update(rng.sample(remaining, k=min(k, len(remaining))))
    return sorted(selected)


def calendar_features(timestamps: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    ts = pd.to_datetime(pd.Series(timestamps), errors="raise")
    hour = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek
    month = ts.dt.month
    doy = ts.dt.dayofyear
    return pd.DataFrame(
        {
            "cal_hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "cal_hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "cal_dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "cal_dow_cos": np.cos(2 * np.pi * dow / 7.0),
            "cal_month_sin": np.sin(2 * np.pi * month / 12.0),
            "cal_month_cos": np.cos(2 * np.pi * month / 12.0),
            "cal_doy_sin": np.sin(2 * np.pi * doy / 366.0),
            "cal_doy_cos": np.cos(2 * np.pi * doy / 366.0),
            "cal_is_weekend": (dow >= 5).astype(float),
        }
    )


def safe_covariate_name(name: str) -> str:
    return "cov_" + "".join(ch if ch.isalnum() else "_" for ch in name)


def numeric_past_covariate_columns(context: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in context.columns:
        if col in IDENTIFIER_COLUMNS or col in NON_COVARIATE_NUMERIC_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(context[col]):
            cols.append(col)
    return cols


def clean_numeric(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    numeric = numeric.ffill().bfill().fillna(0.0)
    return np.array(numeric.to_numpy(dtype=np.float32), dtype=np.float32, copy=True)


def build_chronos_frames(batch: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    context = batch.context.reset_index(drop=True)
    target = batch.target.reset_index(drop=True)

    history = pd.DataFrame(
        {
            "item_id": [batch.window_id] * len(context),
            "timestamp": pd.to_datetime(context["timestamp"], errors="raise"),
            "target": clean_numeric(context["target"]),
        }
    )
    history = pd.concat([history.reset_index(drop=True), calendar_features(history["timestamp"])], axis=1)

    for col in numeric_past_covariate_columns(context):
        values = clean_numeric(context[col])
        history[safe_covariate_name(col)] = values

    future = pd.DataFrame(
        {
            "item_id": [batch.window_id] * len(target),
            "timestamp": pd.to_datetime(target["timestamp"], errors="raise"),
        }
    )
    future = pd.concat([future.reset_index(drop=True), calendar_features(future["timestamp"])], axis=1)
    return history, future


def build_chronos_fit_task(batch: Any) -> dict[str, Any]:
    context = batch.context.reset_index(drop=True)
    target = batch.target.reset_index(drop=True)
    full_timestamps = pd.concat(
        [
            pd.to_datetime(context["timestamp"], errors="raise").reset_index(drop=True),
            pd.to_datetime(target["timestamp"], errors="raise").reset_index(drop=True),
        ],
        ignore_index=True,
    )
    full_target = np.concatenate(
        [clean_numeric(context["target"]), clean_numeric(target["target"])],
        axis=0,
    ).astype(np.float32)
    past_covariates: dict[str, np.ndarray] = {}
    future_covariates: dict[str, np.ndarray] = {}

    cal = calendar_features(full_timestamps)
    context_len = len(context)
    horizon_len = len(target)
    for col in cal.columns:
        values = np.array(pd.to_numeric(cal[col], errors="raise").to_numpy(dtype=np.float32), copy=True)
        past_covariates[col] = values
        future_covariates[col] = np.array(values[context_len : context_len + horizon_len], copy=True)

    for col in numeric_past_covariate_columns(context):
        context_values = clean_numeric(context[col])
        padded_values = np.concatenate(
            [context_values, np.full(horizon_len, np.nan, dtype=np.float32)],
            axis=0,
        ).astype(np.float32)
        past_covariates[safe_covariate_name(col)] = padded_values

    return {
        "target": full_target,
        "past_covariates": past_covariates,
        "future_covariates": future_covariates,
    }


def selected_batches(domain: str, horizon: str, split: str, limit: int, seed_label: str) -> tuple[list[Any], list[int]]:
    dataset = P1cWindowDataset(domain, horizon, split=split)
    positions = sample_positions(len(dataset), limit, f"{domain}:{horizon}:{split}:{seed_label}:{DEFAULT_SEED}")
    return [dataset.get(pos) for pos in positions], positions


def quantile_columns(forecast: pd.DataFrame) -> dict[str, str]:
    cols: dict[str, str] = {}
    if "0.1" in forecast.columns:
        cols["q10"] = serialize_series(forecast["0.1"])
    if "0.5" in forecast.columns:
        cols["q50"] = serialize_series(forecast["0.5"])
    if "0.9" in forecast.columns:
        cols["q90"] = serialize_series(forecast["0.9"])
    return cols


def predict_one(
    pipeline: Chronos2Pipeline,
    batch: Any,
    *,
    config_id: str,
    prediction_length: int,
    batch_size: int,
    context_length: int | None,
    notes: str,
) -> dict[str, Any]:
    history, future = build_chronos_frames(batch)
    forecast = pipeline.predict_df(
        history,
        future_df=future,
        target="target",
        prediction_length=prediction_length,
        batch_size=batch_size,
        context_length=context_length,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    forecast = forecast.sort_values("timestamp").reset_index(drop=True)
    y_pred = pd.Series(pd.to_numeric(forecast["predictions"], errors="raise").to_numpy(dtype=float))
    row = build_prediction_stub(
        batch,
        model_family=MODEL_FAMILY,
        model_id=MODEL_ID,
        config_id=config_id,
        seed=DEFAULT_SEED,
        y_pred=y_pred,
        notes=notes,
    )
    row.update(quantile_columns(forecast))
    return row


def predict_batches(
    pipeline: Chronos2Pipeline,
    batches: list[Any],
    *,
    split: str,
    config_id: str,
    batch_size: int,
    context_length: int | None,
    notes: str,
) -> pd.DataFrame:
    rows = []
    for batch in batches:
        rows.append(
            predict_one(
                pipeline,
                batch,
                config_id=config_id,
                prediction_length=int(batch.metadata["horizon_steps"]),
                batch_size=batch_size,
                context_length=context_length,
                notes=notes,
            )
        )
    predictions = pd.DataFrame(rows)
    if not predictions.empty:
        domain = str(batches[0].metadata["domain_id"])
        horizon = str(batches[0].metadata["horizon"])
        window_index = load_window_index(domain, horizon, split=split)
        validate_prediction_against_windows(predictions, window_index)
    return predictions


def trainable_parameter_summary(pipeline: Chronos2Pipeline) -> dict[str, int] | None:
    model = pipeline.model
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
        return {"trainable": int(trainable), "total": int(total)}
    try:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"trainable": int(trainable), "total": int(total)}
    except Exception:
        return None


def require_cuda_device_map(device_map: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/GPU is required for Chronos-2 LoRA tests in this project")
    if str(device_map).lower() != "cuda":
        raise ValueError("Chronos-2 DL/TSFM tests must run with --device-map cuda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=list_domains(), default="aluminum_load")
    parser.add_argument("--horizon", choices=["4h", "24h"], default="4h")
    parser.add_argument("--max-train-windows", type=int, default=16)
    parser.add_argument("--max-validation-windows", type=int, default=4)
    parser.add_argument("--max-test-windows", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face download if weights are not cached.")
    parser.add_argument("--device-map", default="cuda", help="Device map passed to from_pretrained; must be cuda.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default="chronos2_lora_preflight_aluminum_load_4h_seed20260514_steps5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.out_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    pred_dir = run_dir / "predictions"
    metric_dir = run_dir / "metrics"
    train_dir = run_dir / "finetune"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    random.seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)
    torch.manual_seed(DEFAULT_SEED)
    torch.set_float32_matmul_precision("high")
    require_cuda_device_map(args.device_map)

    train_batches, train_positions = selected_batches(
        args.domain, args.horizon, "train", args.max_train_windows, "lora_train"
    )
    validation_batches, validation_positions = selected_batches(
        args.domain, args.horizon, "validation", args.max_validation_windows, "lora_validation"
    )
    test_batches, test_positions = selected_batches(
        args.domain, args.horizon, "test", args.max_test_windows, "lora_test"
    )

    prediction_length = int(train_batches[0].metadata["horizon_steps"])
    train_inputs = [build_chronos_fit_task(batch) for batch in train_batches]
    validation_inputs = [build_chronos_fit_task(batch) for batch in validation_batches]

    base_pipeline = Chronos2Pipeline.from_pretrained(
        args.model_name,
        local_files_only=not args.allow_download,
        device_map=args.device_map,
    )

    fit_kwargs = {
        "prediction_length": prediction_length,
        "validation_inputs": validation_inputs,
        "finetune_mode": "lora",
        "learning_rate": float(args.learning_rate),
        "num_steps": int(args.num_steps),
        "batch_size": int(args.batch_size),
        "output_dir": train_dir,
        "context_length": args.context_length,
        "logging_steps": 1,
        "eval_steps": max(1, min(5, int(args.num_steps))),
        "save_steps": max(1, min(5, int(args.num_steps))),
        "remove_printer_callback": False,
    }
    lora_pipeline = base_pipeline.fit(inputs=train_inputs, **fit_kwargs)

    all_predictions = []
    summaries = []
    for split, batches, positions in [
        ("validation", validation_batches, validation_positions),
        ("test", test_batches, test_positions),
    ]:
        for config_id, pipeline, notes in [
            (
                BASE_CONFIG_ID,
                base_pipeline,
                "chronos2_lora_preflight_baseline; full-power covariate-aware zero-shot; no future measured exogenous covariates",
            ),
            (
                LORA_CONFIG_ID,
                lora_pipeline,
                "chronos2_lora_preflight; LoRA fine-tuned on P1c train windows; context target plus past-only numeric covariates; known-future calendar only",
            ),
        ]:
            started_pred = time.time()
            predictions = predict_batches(
                pipeline,
                batches,
                split=split,
                config_id=config_id,
                batch_size=args.batch_size,
                context_length=args.context_length,
                notes=notes,
            )
            path = pred_dir / f"{args.domain}_{args.horizon}_{split}_{config_id}_predictions.parquet"
            predictions.to_parquet(path, index=False)
            all_predictions.append(predictions)
            summaries.append(
                {
                    "split": split,
                    "config_id": config_id,
                    "selected_positions": positions,
                    "prediction_rows": int(len(predictions)),
                    "prediction_path": str(path),
                    "elapsed_sec": round(time.time() - started_pred, 3),
                }
            )

    predictions_all = pd.concat(all_predictions, ignore_index=True)
    combined_path = pred_dir / "predictions_all.parquet"
    predictions_all.to_parquet(combined_path, index=False)
    metrics = evaluate_prediction_frame(predictions_all)
    metric_paths = write_metrics(metrics, metric_dir, stem="metrics")

    manifest = {
        "status": "ok",
        "model_id": MODEL_ID,
        "model_family": MODEL_FAMILY,
        "model_name": args.model_name,
        "checkpoint_is_full_power": True,
        "domain": args.domain,
        "horizon": args.horizon,
        "prediction_length": prediction_length,
        "train_windows": int(len(train_batches)),
        "validation_windows": int(len(validation_batches)),
        "test_windows": int(len(test_batches)),
        "train_positions": train_positions,
        "validation_positions": validation_positions,
        "test_positions": test_positions,
        "finetune_mode": "lora",
        "learning_rate": float(args.learning_rate),
        "num_steps": int(args.num_steps),
        "batch_size": int(args.batch_size),
        "context_length": args.context_length,
        "device_map": args.device_map,
        "base_config_id": BASE_CONFIG_ID,
        "lora_config_id": LORA_CONFIG_ID,
        "base_parameter_summary": trainable_parameter_summary(base_pipeline),
        "lora_parameter_summary": trainable_parameter_summary(lora_pipeline),
        "finetuned_checkpoint": str(train_dir / "finetuned-ckpt"),
        "prediction_rows": int(len(predictions_all)),
        "metric_rows": int(len(metrics)),
        "predictions_all": str(combined_path),
        "metrics": metric_paths,
        "summaries": summaries,
        "elapsed_sec": round(time.time() - started, 3),
        "important_boundary": "Preflight only; not a manuscript performance result.",
        "covariate_policy": "context target plus past-only numeric context covariates; known-future calendar only; no future measured exogenous covariates at validation/test prediction time",
    }
    (run_dir / "manifest.json").write_text(dumps(manifest) + "\n", encoding="utf-8")
    print(dumps(manifest))


if __name__ == "__main__":
    main()
