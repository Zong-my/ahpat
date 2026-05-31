#!/usr/bin/env python3
"""Formal fixed-base wrapper for NeuralForecast N-BEATSx and TFT.

This runner upgrades the NeuralForecast covariate smoke path into a formal
fixed-base scaffold for the two covariate-aware supervised DL baselines. It
uses the frozen P1c task contract, external validation/test predictions, local
metrics, Lightning logs and NeuralForecast model snapshots.

Important backend caveats:

1. NeuralForecast exposes Lightning checkpoints, but its public
   `NeuralForecast.fit` wrapper does not expose optimizer-state resume via
   `ckpt_path`. This runner therefore supports best-checkpoint inference and
   weight-level resume through saved NeuralForecast models, while recording the
   optimizer-state resume gap in the manifest.
2. In the installed NeuralForecast version, explicit `val_df` currently requires
   validation and training frames to have the same group structure. P1c
   validation windows are independent frozen windows, so the runner builds a
   re-keyed validation frame that maps frozen validation windows onto the train
   group IDs for Lightning early stopping, while separately evaluating the
   original frozen P1c validation/test windows with the common evaluator.
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import HuberLoss, MAE
from neuralforecast.models import NBEATSx, TFT
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from energy_tsfm_formal_artifacts import (
    DEFAULT_FORMAL_ROOT,
    FormalRunPaths,
    cuda_metadata,
    dumps_json,
    ensure_formal_run_dirs,
    log_stderr,
    log_stdout,
    make_run_paths,
    package_versions,
    read_json,
    update_curve_artifacts,
    write_inference_entrypoint,
    write_json_atomic,
)
from energy_tsfm_p2_core import (
    MAIN_HORIZONS,
    P1cWindowDataset,
    build_prediction_stub,
    load_window_index,
    validate_prediction_against_windows,
)
from evaluate_energy_tsfm_predictions import evaluate_prediction_frame, write_metrics
from run_neuralforecast_dl_covariate_smoke import (
    CALENDAR_FEATURES,
    EXCLUDED_HIST_EXOG_COLUMNS,
    calendar_features,
    context_fill_values,
    hist_exog_values,
    sample_positions,
    select_hist_exog_columns,
)
from run_itransformer_covariate_fixed_base import (
    audit_reproduced_metrics,
    load_shared_window_subset_record,
    validate_window_subset_record,
    window_subset_record,
)


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 20260514
MODEL_FAMILY = "dl"
SUPPORTED_MODELS = ("nbeatsx", "tft")


def resolve_scaler_type(model_id: str, scaler_type: str) -> str:
    if scaler_type != "auto":
        return scaler_type
    if model_id == "nbeatsx":
        return "identity"
    if model_id == "tft":
        return "robust"
    raise ValueError(f"unsupported model {model_id!r}; expected {SUPPORTED_MODELS}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")


def safe_numeric(value: Any, fallback: float = 0.0) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed) or not np.isfinite(float(parsed)):
        return fallback
    return float(parsed)


def _rows_for_window_items(
    dataset: P1cWindowDataset,
    work_items: list[tuple[int, int, str]],
    *,
    hist_exog_cols: list[str],
    include_future_target: bool,
    keep_batches: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    futr_rows: list[dict[str, Any]] = []
    uid_to_batch: dict[str, Any] = {}

    for _ordinal, pos, uid in work_items:
        batch = dataset.get(pos)
        if keep_batches:
            uid_to_batch[uid] = batch
        context = batch.context.reset_index(drop=True)
        target = batch.target.reset_index(drop=True)
        fill = context_fill_values(context, hist_exog_cols)

        for ds, row in context.iterrows():
            rec = {
                "unique_id": uid,
                "ds": int(ds),
                "y": safe_numeric(row["target"]),
            }
            rec.update(calendar_features(row["timestamp"]))
            rec.update(hist_exog_values(row, hist_exog_cols, fill))
            rows.append(rec)

        context_len = len(context)
        if include_future_target:
            for lead_step, row in target.iterrows():
                rec = {
                    "unique_id": uid,
                    "ds": int(context_len + lead_step),
                    "y": safe_numeric(row["target"]),
                }
                rec.update(calendar_features(row["timestamp"]))
                rec.update({col: fill[col] for col in hist_exog_cols})
                rows.append(rec)
        else:
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


def _rows_for_windows_chunk_worker(
    domain_id: str,
    horizon: str,
    split: str | None,
    work_items: list[tuple[int, int, str]],
    hist_exog_cols: list[str],
    include_future_target: bool,
    keep_batches: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    dataset = P1cWindowDataset(domain_id, horizon, split=split)
    return _rows_for_window_items(
        dataset,
        work_items,
        hist_exog_cols=hist_exog_cols,
        include_future_target=include_future_target,
        keep_batches=keep_batches,
    )


def chunk_work_items(items: list[tuple[int, int, str]], *, workers: int, chunk_size: int) -> list[list[tuple[int, int, str]]]:
    if not items:
        return []
    workers = max(1, int(workers))
    auto_chunk_size = max(1, math.ceil(len(items) / workers))
    if chunk_size <= 0:
        chunk_size = auto_chunk_size
    elif workers > 1:
        # Treat user-provided chunk_size as an upper bound. A very large value
        # such as 65536 should not collapse full-P3 preprocessing to one worker
        # when --preprocess-workers requests parallel construction.
        chunk_size = max(1, min(int(chunk_size), auto_chunk_size))
    return [items[start : start + chunk_size] for start in range(0, len(items), chunk_size)]


def rows_for_windows_safe(
    dataset: P1cWindowDataset,
    positions: list[int],
    *,
    hist_exog_cols: list[str],
    include_future_target: bool,
    uid_list: list[str] | None = None,
    preprocess_workers: int = 1,
    preprocess_chunk_size: int = 0,
    keep_batches: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    """Build NeuralForecast rows without exposing future measured exogs.

    For train/validation frames that include target rows, measured historical
    exog columns in target rows are replaced with context-only fill values.
    Known-ahead calendar features remain available for all rows.
    """

    if uid_list is not None and include_future_target and not keep_batches:
        return rows_for_rekeyed_fit_windows_fast(
            dataset,
            positions,
            hist_exog_cols=hist_exog_cols,
            uid_list=uid_list,
        )

    iterable_positions = positions
    if uid_list is not None:
        if not positions:
            raise ValueError("positions cannot be empty when uid_list is provided")
        iterable_positions = [positions[idx % len(positions)] for idx in range(len(uid_list))]

    work_items: list[tuple[int, int, str]] = []
    for ordinal, pos in enumerate(iterable_positions):
        uid = uid_list[ordinal] if uid_list is not None else f"{dataset.domain_id}__{dataset.horizon}__{dataset.split}__{ordinal:06d}"
        work_items.append((int(ordinal), int(pos), str(uid)))

    workers = max(1, int(preprocess_workers))
    if workers <= 1 or len(work_items) < 2:
        return _rows_for_window_items(
            dataset,
            work_items,
            hist_exog_cols=hist_exog_cols,
            include_future_target=include_future_target,
            keep_batches=keep_batches,
        )

    chunks = chunk_work_items(work_items, workers=workers, chunk_size=int(preprocess_chunk_size))
    max_workers = min(workers, len(chunks))
    dfs: list[pd.DataFrame] = []
    futr_dfs: list[pd.DataFrame] = []
    uid_to_batch: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _rows_for_windows_chunk_worker,
                dataset.domain_id,
                dataset.horizon,
                dataset.split,
                chunk,
                hist_exog_cols,
                include_future_target,
                keep_batches,
            )
            for chunk in chunks
        ]
        for future in futures:
            df, futr_df, chunk_batches = future.result()
            if not df.empty:
                dfs.append(df)
            if futr_df is not None and not futr_df.empty:
                futr_dfs.append(futr_df)
            if keep_batches:
                uid_to_batch.update(chunk_batches)
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    futr_df = None if include_future_target else (pd.concat(futr_dfs, ignore_index=True) if futr_dfs else pd.DataFrame())
    return df, futr_df, uid_to_batch


def rows_for_rekeyed_fit_windows_fast(
    dataset: P1cWindowDataset,
    positions: list[int],
    *,
    hist_exog_cols: list[str],
    uid_list: list[str],
) -> tuple[pd.DataFrame, None, dict[str, Any]]:
    """Build the re-keyed NeuralForecast validation frame without repeated IO.

    NeuralForecast's installed ``val_df`` path requires validation groups to
    match train groups. The previous implementation honored that by cycling
    validation positions across ``uid_list`` and calling ``dataset.get`` once
    per train UID. That is semantically correct but very slow on full P3
    validation. This helper preserves the exact same UID-to-validation-position
    mapping, while materializing each unique validation position once and then
    copying its context/target row template under the required train UID.
    """

    if not positions:
        raise ValueError("positions cannot be empty when uid_list is provided")
    if not uid_list:
        return pd.DataFrame(), None, {}

    templates: list[list[dict[str, Any]]] = []
    for pos in positions:
        df, _futr_df, _batches = _rows_for_window_items(
            dataset,
            [(0, int(pos), "__template_uid__")],
            hist_exog_cols=hist_exog_cols,
            include_future_target=True,
            keep_batches=False,
        )
        if df.empty:
            templates.append([])
            continue
        templates.append(df.drop(columns=["unique_id"]).to_dict(orient="records"))

    rows: list[dict[str, Any]] = []
    for ordinal, uid in enumerate(uid_list):
        template = templates[ordinal % len(templates)]
        for rec in template:
            out = {"unique_id": str(uid)}
            out.update(rec)
            rows.append(out)
    return pd.DataFrame(rows), None, {}


def get_or_create_window_subsets(
    paths: FormalRunPaths,
    *,
    domain: str,
    horizon: str,
    model_id: str,
    seed: int,
    max_train_windows: int,
    max_val_windows: int,
    max_test_windows: int,
    window_subset_manifest: Path | None,
    resume: bool,
) -> tuple[P1cWindowDataset, P1cWindowDataset, P1cWindowDataset, dict[str, Any]]:
    train_base = P1cWindowDataset(domain, horizon, split="train")
    val_base = P1cWindowDataset(domain, horizon, split="validation")
    test_base = P1cWindowDataset(domain, horizon, split="test")
    if resume and paths.window_subsets.exists():
        saved = read_json(paths.window_subsets)
        validate_window_subset_record(train_base, saved["train"], split_name="train")
        validate_window_subset_record(val_base, saved["validation"], split_name="validation")
        validate_window_subset_record(test_base, saved["test"], split_name="test")
        return train_base, val_base, test_base, saved

    if window_subset_manifest is not None:
        record = load_shared_window_subset_record(
            window_subset_manifest,
            domain=domain,
            horizon=horizon,
            train_base=train_base,
            val_base=val_base,
            test_base=test_base,
        )
        record["model_id"] = model_id
        write_json_atomic(paths.window_subsets, record)
        return train_base, val_base, test_base, record

    train_positions = sample_positions(
        len(train_base),
        max_train_windows,
        f"{seed}:{model_id}:{domain}:{horizon}:formal:train",
    )
    val_positions = sample_positions(
        len(val_base),
        max_val_windows,
        f"{seed}:{model_id}:{domain}:{horizon}:formal:validation",
    )
    test_positions = sample_positions(
        len(test_base),
        max_test_windows,
        f"{seed}:{model_id}:{domain}:{horizon}:formal:test",
    )
    if not train_positions or not val_positions or not test_positions:
        raise ValueError(f"{model_id}/{domain}/{horizon}: empty train/validation/test subset")

    record = {
        "domain_id": domain,
        "horizon": horizon,
        "model_id": model_id,
        "seed": int(seed),
        "sampling_rule": "sample_positions; 0 means full split; includes first/middle/last anchors",
        "train": window_subset_record(train_base, train_positions),
        "validation": window_subset_record(val_base, val_positions),
        "test": window_subset_record(test_base, test_positions),
    }
    write_json_atomic(paths.window_subsets, record)
    return train_base, val_base, test_base, record


def make_nf_model(
    model_id: str,
    *,
    h: int,
    input_size: int,
    hist_exog_cols: list[str],
    args: argparse.Namespace,
    checkpoint_callback: ModelCheckpoint,
    logger: CSVLogger,
) -> Any:
    callbacks: list[Any] = [checkpoint_callback]
    if args.log_lr:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    common = {
        "h": h,
        "input_size": input_size,
        "hist_exog_list": hist_exog_cols or None,
        "futr_exog_list": CALENDAR_FEATURES,
        "loss": HuberLoss(delta=args.huber_delta),
        "valid_loss": MAE(),
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "num_lr_decays": -1,
        "early_stop_patience_steps": args.early_stop_patience_steps,
        "val_monitor": "ptl/val_loss",
        "val_check_steps": args.val_check_steps,
        "batch_size": args.batch_size,
        "valid_batch_size": args.valid_batch_size,
        "windows_batch_size": args.windows_batch_size,
        "inference_windows_batch_size": args.inference_windows_batch_size,
        "scaler_type": args.scaler_type,
        "random_seed": args.seed,
        "drop_last_loader": False,
        "alias": model_id,
        "optimizer": torch.optim.AdamW,
        "optimizer_kwargs": {"weight_decay": args.weight_decay},
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
        "lr_scheduler_kwargs": {"T_max": max(1, args.max_steps), "eta_min": args.min_lr},
        "enable_progress_bar": args.enable_progress_bar,
        "enable_checkpointing": True,
        "logger": logger,
        "callbacks": callbacks,
        "accelerator": args.accelerator,
        "devices": args.devices,
    }
    if model_id == "nbeatsx":
        return NBEATSx(
            **common,
            n_blocks=[args.nbeatsx_blocks] * 3,
            mlp_units=[[args.nbeatsx_width, args.nbeatsx_width]] * 3,
            dropout_prob_theta=args.dropout,
        )
    if model_id == "tft":
        return TFT(
            **common,
            hidden_size=args.tft_hidden_size,
            n_head=args.tft_heads,
            n_rnn_layers=args.tft_rnn_layers,
            dropout=args.dropout,
            attn_dropout=args.dropout,
        )
    raise ValueError(f"unsupported model_id: {model_id}")


def nf_model_parameter_count(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def predictions_from_forecast(
    forecast: pd.DataFrame,
    uid_to_batch: dict[str, Any],
    *,
    model_id: str,
    config_id: str,
    seed: int,
    hist_exog_cols: list[str],
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
                    "NeuralForecast fixed-base covariate runner;"
                    f"hist_exog_count={len(hist_exog_cols)};"
                    "future_calendar_only=true;"
                    "future_measured_exog=false"
                ),
            )
        )
    return pd.DataFrame(rows)


def save_nf_snapshot(nf: NeuralForecast, path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"NeuralForecast snapshot directory already exists and is non-empty: {path}")
    nf.save(str(path), save_dataset=False, overwrite=False)


def prepare_checkpoint_dirs(paths: FormalRunPaths, *, resume: bool) -> dict[str, Path]:
    ckpt_root = paths.checkpoints_dir
    dirs = {
        "lightning": ckpt_root / "lightning",
        "best_nf": ckpt_root / "best_nf",
        "last_nf": ckpt_root / "last_nf",
        "previous_nf": ckpt_root / "previous_nf",
        "best_lightning": ckpt_root / "best_lightning.ckpt",
        "last_lightning": ckpt_root / "last_lightning.ckpt",
        "previous_lightning": ckpt_root / "previous_lightning.ckpt",
    }
    dirs["lightning"].mkdir(parents=True, exist_ok=True)
    if resume and dirs["last_nf"].exists() and not dirs["previous_nf"].exists():
        shutil.copytree(dirs["last_nf"], dirs["previous_nf"])
    if resume and dirs["last_lightning"].exists() and not dirs["previous_lightning"].exists():
        shutil.copy2(dirs["last_lightning"], dirs["previous_lightning"])
    return dirs


def collect_lightning_train_rows(logger_dir: Path) -> list[dict[str, Any]]:
    metric_files = sorted(logger_dir.rglob("metrics.csv"))
    if not metric_files:
        return []
    df = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    if df.empty:
        return []
    if "step" not in df.columns:
        df["step"] = np.arange(len(df))
    rows: list[dict[str, Any]] = []
    by_step = df.groupby("step", dropna=False, sort=True)
    for ordinal, (step, group) in enumerate(by_step):
        row: dict[str, Any] = {
            "epoch": int(group["epoch"].dropna().iloc[-1]) if "epoch" in group and group["epoch"].notna().any() else ordinal,
            "global_step": int(step) if pd.notna(step) else ordinal,
        }
        for col in group.columns:
            if col in {"step", "epoch"}:
                continue
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            if not values.empty:
                row[col.replace("ptl/", "").replace("/", "_")] = float(values.iloc[-1])
        if "train_loss" in row or "val_loss" in row:
            rows.append(row)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "epoch": row.get("epoch", row.get("global_step", 0)),
                "global_step": row.get("global_step", row.get("epoch", 0)),
                "train_loss": row.get("train_loss"),
                "validation_loss": row.get("val_loss", row.get("valid_loss")),
                "validation_primary_wape": np.nan,
                **row,
            }
        )
    return normalized


def load_best_lightning_state_if_available(nf: NeuralForecast, best_path: Path) -> bool:
    if not best_path.exists():
        return False
    state = torch.load(best_path, map_location="cpu", weights_only=False)
    nf.models[0].load_state_dict(state["state_dict"])
    return True


def resolve_best_lightning_checkpoint(checkpoint_callback: ModelCheckpoint, lightning_dir: Path) -> Path | None:
    """Return the best Lightning checkpoint path under NeuralForecast wrappers."""
    if checkpoint_callback.best_model_path:
        candidate = Path(checkpoint_callback.best_model_path)
        if candidate.exists():
            return candidate
    candidate = lightning_dir / "best.ckpt"
    if candidate.exists():
        return candidate
    candidates = sorted(lightning_dir.glob("best*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def write_split_outputs(
    *,
    paths: FormalRunPaths,
    nf: NeuralForecast,
    val_dataset: P1cWindowDataset,
    test_dataset: P1cWindowDataset,
    subsets: dict[str, Any],
    hist_exog_cols: list[str],
    model_id: str,
    config_id: str,
    seed: int,
    horizon: str,
    domain: str,
    eval_splits: tuple[str, ...],
    preprocess_workers: int,
    preprocess_chunk_size: int,
) -> dict[str, Any]:
    full_index = load_window_index(domain, horizon)
    outputs: dict[str, Any] = {}
    for split, dataset, pred_path, metric_stem in [
        ("validation", val_dataset, paths.validation_predictions, "validation_metrics"),
        ("test", test_dataset, paths.test_predictions, "test_metrics"),
    ]:
        if split not in eval_splits:
            continue
        pred_df, futr_df, uid_to_batch = rows_for_windows_safe(
            dataset,
            [int(pos) for pos in subsets[split]["positions"]],
            hist_exog_cols=hist_exog_cols,
            include_future_target=False,
            preprocess_workers=preprocess_workers,
            preprocess_chunk_size=preprocess_chunk_size,
            keep_batches=True,
        )
        forecast = nf.predict(df=pred_df, futr_df=futr_df, verbose=False)
        preds = predictions_from_forecast(
            forecast,
            uid_to_batch,
            model_id=model_id,
            config_id=config_id,
            seed=seed,
            hist_exog_cols=hist_exog_cols,
        )
        validate_prediction_against_windows(preds, full_index)
        preds.to_parquet(pred_path, index=False)
        metrics = evaluate_prediction_frame(preds)
        metric_paths = write_metrics(metrics, paths.metrics_dir, stem=metric_stem)
        outputs[f"{split}_prediction_rows"] = int(len(preds))
        outputs[f"{split}_metric_rows"] = int(len(metrics))
        outputs[f"{split}_predictions"] = str(pred_path)
        outputs[f"{split}_metrics"] = metric_paths
    return outputs


def build_run_config(args: argparse.Namespace, *, hist_exog_cols: list[str], shape: dict[str, Any]) -> dict[str, Any]:
    return {
        "runner": str(Path(__file__).resolve()),
        "model_family": MODEL_FAMILY,
        "model_id": args.model,
        "implementation": "NeuralForecast official class",
        "domain_id": args.domain,
        "horizon": args.horizon,
        "config_id": args.config_id,
        "seed": int(args.seed),
        "eval_splits": list(args.eval_split or ("validation", "test")),
        "data_contract": {
            "window_source": "P1c frozen window index",
            "horizons_allowed": list(MAIN_HORIZONS),
            "no_future_measured_exog": True,
            "future_calendar_only": True,
            "split_rule": "forecast endpoint split from P1c",
            "segment_boundary_rule": "no window may cross segment_id",
        },
        "shape_config": shape,
        "hist_exog_cols": hist_exog_cols,
        "futr_exog_cols": CALENDAR_FEATURES,
        "training_config": {
            "loss": "HuberLoss",
            "valid_loss": "MAE",
            "external_primary_metric": "validation/test WAPE from common evaluator",
            "internal_checkpoint_monitor": "ptl/val_loss",
            "internal_checkpoint_validation_source": "frozen P1c validation windows re-keyed to match train unique_id groups for installed NeuralForecast val_df constraints; original P1c validation is evaluated externally after fit",
            "backend_resume_caveat": "NeuralForecast public fit wrapper does not expose optimizer-state ckpt_path resume.",
            "max_steps": int(args.max_steps),
            "val_check_steps": int(args.val_check_steps),
            "early_stop_patience_steps": int(args.early_stop_patience_steps),
            "batch_size": int(args.batch_size),
            "valid_batch_size": int(args.valid_batch_size),
            "windows_batch_size": int(args.windows_batch_size),
            "scaler_type": str(args.scaler_type),
            "requested_scaler_type": str(getattr(args, "requested_scaler_type", args.scaler_type)),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "lr_scheduler": "CosineAnnealingLR",
        },
        "model_config": {
            "nbeatsx_blocks": int(args.nbeatsx_blocks),
            "nbeatsx_width": int(args.nbeatsx_width),
            "tft_hidden_size": int(args.tft_hidden_size),
            "tft_heads": int(args.tft_heads),
            "tft_rnn_layers": int(args.tft_rnn_layers),
            "dropout": float(args.dropout),
            "max_hist_exog": int(args.max_hist_exog),
            "min_hist_exog_coverage": float(args.min_hist_exog_coverage),
        },
        "args": vars(args),
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if args.model not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported model {args.model!r}; expected {SUPPORTED_MODELS}")
    args.requested_scaler_type = args.scaler_type
    args.scaler_type = resolve_scaler_type(args.model, args.scaler_type)
    set_seed(args.seed)
    paths = make_run_paths(
        output_root=args.output_root,
        model_id=args.model,
        domain_id=args.domain,
        horizon=args.horizon,
        config_id=args.config_id,
        seed=args.seed,
    )
    ensure_formal_run_dirs(paths, resume=args.resume)
    start = time.time()
    log_stdout(paths, f"[{args.model}] NeuralForecast fixed-base run: {args.domain}/{args.horizon}")
    log_stdout(paths, f"run_dir={paths.run_dir}")
    log_stdout(paths, f"scaler_type={args.scaler_type} (requested={args.requested_scaler_type})")
    try:
        eval_splits = tuple(args.eval_split or ("validation", "test"))
        if "validation" not in eval_splits:
            raise ValueError("validation must be included in --eval-split")
        train_base, val_base, test_base, subsets = get_or_create_window_subsets(
            paths,
            domain=args.domain,
            horizon=args.horizon,
            model_id=args.model,
            seed=args.seed,
            max_train_windows=args.max_train_windows,
            max_val_windows=args.max_val_windows,
            max_test_windows=args.max_test_windows,
            window_subset_manifest=args.window_subset_manifest,
            resume=args.resume,
        )
        hist_exog_cols = select_hist_exog_columns(
            args.domain,
            max_hist_exog=args.max_hist_exog,
            min_coverage=args.min_hist_exog_coverage,
        )
        train_positions = [int(pos) for pos in subsets["train"]["positions"]]
        val_positions = [int(pos) for pos in subsets["validation"]["positions"]]
        train_df, _, _ = rows_for_windows_safe(
            train_base,
            train_positions,
            hist_exog_cols=hist_exog_cols,
            include_future_target=True,
            preprocess_workers=args.preprocess_workers,
            preprocess_chunk_size=args.preprocess_chunk_size,
            keep_batches=False,
        )
        train_uids = [str(uid) for uid in train_df["unique_id"].drop_duplicates().tolist()]
        val_fit_df, _, _ = rows_for_windows_safe(
            val_base,
            val_positions,
            hist_exog_cols=hist_exog_cols,
            include_future_target=True,
            uid_list=train_uids,
            preprocess_workers=args.preprocess_workers,
            preprocess_chunk_size=args.preprocess_chunk_size,
            keep_batches=False,
        )
        first_batch = train_base.get(train_positions[0])
        h = int(first_batch.metadata["horizon_steps"])
        input_size = int(first_batch.metadata["context_steps"])
        shape = {
            "lookback_steps": input_size,
            "horizon_steps": h,
            "train_rows": int(len(train_df)),
            "internal_validation_fit_rows": int(len(val_fit_df)),
            "internal_validation_unique_ids": int(len(train_uids)),
            "internal_validation_reuses_frozen_validation_windows": bool(len(train_uids) > len(val_positions)),
            "internal_validation_fast_rekey_enabled": True,
            "hist_exog_count": int(len(hist_exog_cols)),
            "future_calendar_count": int(len(CALENDAR_FEATURES)),
            "preprocess_workers": int(args.preprocess_workers),
            "preprocess_chunk_size": int(args.preprocess_chunk_size),
        }
        config = build_run_config(args, hist_exog_cols=hist_exog_cols, shape=shape)
        write_json_atomic(paths.config, config)
        write_inference_entrypoint(paths, runner_script=Path(__file__).resolve())

        ckpt_dirs = prepare_checkpoint_dirs(paths, resume=args.resume)
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(ckpt_dirs["lightning"]),
            filename="best",
            monitor="ptl/val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        )
        logger = CSVLogger(save_dir=str(paths.run_dir / "lightning_logs"), name="nf")
        model = make_nf_model(
            args.model,
            h=h,
            input_size=input_size,
            hist_exog_cols=hist_exog_cols,
            args=args,
            checkpoint_callback=checkpoint_callback,
            logger=logger,
        )
        parameter_count = nf_model_parameter_count(model)
        if args.resume and ckpt_dirs["last_nf"].exists():
            nf = NeuralForecast.load(str(ckpt_dirs["last_nf"]))
            nf.models[0].trainer_kwargs.update(model.trainer_kwargs)
            fit_use_init_models = True
            resume_mode = "weight_level_resume_without_optimizer_state"
        else:
            nf = NeuralForecast(models=[model], freq=1)
            fit_use_init_models = False
            resume_mode = "fresh"

        cuda_start = cuda_metadata(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        manifest = {
            "status": "running",
            "model_family": MODEL_FAMILY,
            "model_id": args.model,
            "domain_id": args.domain,
            "horizon": args.horizon,
            "config_id": args.config_id,
            "seed": int(args.seed),
            "device": cuda_start.get("device"),
            "accelerator": args.accelerator,
            "run_dir": str(paths.run_dir),
            "p1c_window_source": str(PROJECT / "data" / "energy_tsfm_windows_p1c" / args.domain / f"window_index_{args.horizon}.parquet"),
            "window_subset_manifest": str(args.window_subset_manifest) if args.window_subset_manifest else None,
            "train_windows": int(subsets["train"]["count"]),
            "validation_windows": int(subsets["validation"]["count"]),
            "test_windows": int(subsets["test"]["count"]),
            "hist_exog_cols": hist_exog_cols,
            "futr_exog_cols": CALENDAR_FEATURES,
            "parameter_count": int(parameter_count),
            "normalizer_policy": f"NeuralForecast scaler_type={args.scaler_type}; temporal rows built from P1c windows only",
            "loss_function": "HuberLoss",
            "validation_selection_metric": "NeuralForecast ptl/val_loss on re-keyed frozen P1c validation windows; external WAPE audited after fit",
            "internal_validation_source": "P1c validation windows re-keyed onto train unique_id groups because installed NeuralForecast val_df requires same group structure",
            "backend_resume_caveat": config["training_config"]["backend_resume_caveat"],
            "resume_mode": resume_mode,
            "started_at_unix": start,
            "cuda_metadata_start": cuda_start,
            "package_versions": package_versions(),
            "args": vars(args),
        }
        write_json_atomic(paths.manifest, manifest)

        nf.fit(
            df=train_df,
            val_df=val_fit_df,
            use_init_models=fit_use_init_models,
            verbose=args.verbose_fit,
        )

        best_lightning_path = resolve_best_lightning_checkpoint(checkpoint_callback, ckpt_dirs["lightning"])
        if best_lightning_path is not None and best_lightning_path.exists():
            shutil.copy2(best_lightning_path, ckpt_dirs["best_lightning"])
            loaded_best = load_best_lightning_state_if_available(nf, best_lightning_path)
        else:
            loaded_best = False
        if (ckpt_dirs["lightning"] / "last.ckpt").exists():
            shutil.copy2(ckpt_dirs["lightning"] / "last.ckpt", ckpt_dirs["last_lightning"])

        save_nf_snapshot(nf, ckpt_dirs["best_nf"])
        if not ckpt_dirs["last_nf"].exists():
            save_nf_snapshot(nf, ckpt_dirs["last_nf"])

        train_rows = collect_lightning_train_rows(paths.run_dir / "lightning_logs")
        update_curve_artifacts(paths, train_rows, primary_metric_col="validation_primary_wape")
        output_info = write_split_outputs(
            paths=paths,
            nf=nf,
            val_dataset=val_base,
            test_dataset=test_base,
            subsets=subsets,
            hist_exog_cols=hist_exog_cols,
            model_id=args.model,
            config_id=args.config_id,
            seed=args.seed,
            horizon=args.horizon,
            domain=args.domain,
            eval_splits=eval_splits,
            preprocess_workers=args.preprocess_workers,
            preprocess_chunk_size=args.preprocess_chunk_size,
        )
        cuda_end = cuda_metadata(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        manifest.update(
            {
                "status": "ok",
                "completed_at_unix": time.time(),
                "runtime_sec": float(time.time() - start),
                "device": cuda_end.get("device"),
                "accelerator": args.accelerator,
                "best_lightning_checkpoint": str(ckpt_dirs["best_lightning"]) if ckpt_dirs["best_lightning"].exists() else None,
                "last_lightning_checkpoint": str(ckpt_dirs["last_lightning"]) if ckpt_dirs["last_lightning"].exists() else None,
                "best_nf_checkpoint": str(ckpt_dirs["best_nf"]),
                "last_nf_checkpoint": str(ckpt_dirs["last_nf"]),
                "previous_nf_checkpoint": str(ckpt_dirs["previous_nf"]) if ckpt_dirs["previous_nf"].exists() else None,
                "best_lightning_state_loaded_for_predictions": bool(loaded_best),
                "train_log_csv": str(paths.train_log_csv),
                "curves": {
                    "loss_curve_csv": str(paths.loss_curve_csv),
                    "loss_curve_png": str(paths.loss_curve_png),
                    "metric_curve_png": str(paths.metric_curve_png),
                },
                "outputs": output_info,
                "one_command_inference": str(paths.inference_script),
                "cuda_metadata_end": cuda_end,
                "evaluator_validation_status": "validate_prediction_against_windows passed before metrics",
                "interpretation": "fixed-base NeuralForecast scaffold; dry-run limits, if used, are not paper results",
            }
        )
        write_json_atomic(paths.manifest, manifest)
        log_stdout(paths, dumps_json({"status": "ok", "run_dir": str(paths.run_dir), **output_info}))
        return manifest
    except Exception as exc:
        tb = traceback.format_exc()
        log_stderr(paths, tb)
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": tb,
            "failed_at_unix": time.time(),
        }
        write_json_atomic(paths.failure_status, failure)
        if paths.manifest.exists():
            manifest = read_json(paths.manifest)
            manifest.update(failure)
            write_json_atomic(paths.manifest, manifest)
        raise


def run_inference_only(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_dir is None:
        raise ValueError("--inference-only requires --run-dir")
    paths = FormalRunPaths(run_dir=args.run_dir)
    config = read_json(paths.config)
    subsets = read_json(paths.window_subsets)
    model_id = str(config["model_id"])
    domain = str(config["domain_id"])
    horizon = str(config["horizon"])
    seed = int(config["seed"])
    hist_exog_cols = [str(col) for col in config["hist_exog_cols"]]
    eval_splits = tuple(args.eval_split or config.get("eval_splits") or ("validation", "test"))
    nf_checkpoint = paths.checkpoints_dir / "best_nf"
    nf = NeuralForecast.load(str(nf_checkpoint))

    val_base = P1cWindowDataset(domain, horizon, split="validation")
    test_base = P1cWindowDataset(domain, horizon, split="test")
    full_index = load_window_index(domain, horizon)
    results: dict[str, Any] = {}
    metric_frames: dict[str, pd.DataFrame] = {}
    for split, dataset in [("validation", val_base), ("test", test_base)]:
        if split not in eval_splits:
            continue
        pred_df, futr_df, uid_to_batch = rows_for_windows_safe(
            dataset,
            [int(pos) for pos in subsets[split]["positions"]],
            hist_exog_cols=hist_exog_cols,
            include_future_target=False,
        )
        forecast = nf.predict(df=pred_df, futr_df=futr_df, verbose=False)
        preds = predictions_from_forecast(
            forecast,
            uid_to_batch,
            model_id=model_id,
            config_id=str(config["config_id"]),
            seed=seed,
            hist_exog_cols=hist_exog_cols,
        )
        validate_prediction_against_windows(preds, full_index)
        pred_path = paths.inference_dir / f"{split}_predictions_reproduced.parquet"
        preds.to_parquet(pred_path, index=False)
        metrics = evaluate_prediction_frame(preds)
        metric_frames[split] = metrics
        metric_paths = write_metrics(metrics, paths.inference_dir, stem=f"{split}_metrics_reproduced")
        results[f"{split}_prediction_rows"] = int(len(preds))
        results[f"{split}_metric_rows"] = int(len(metrics))
        results[f"{split}_predictions"] = str(pred_path)
        results[f"{split}_metrics"] = metric_paths

    reproducibility_audit = None
    if "test" in eval_splits:
        reproducibility_audit = audit_reproduced_metrics(
            original_validation_metrics_path=paths.validation_metrics_csv,
            reproduced_validation_metrics=metric_frames["validation"],
            original_test_metrics_path=paths.test_metrics_csv,
            reproduced_test_metrics=metric_frames["test"],
        )
    manifest = {
        "status": "ok",
        "run_dir": str(paths.run_dir),
        "checkpoint": str(nf_checkpoint),
        "model_id": model_id,
        "domain_id": domain,
        "horizon": horizon,
        "reproducibility_audit": reproducibility_audit,
        **results,
    }
    write_json_atomic(paths.inference_dir / "inference_manifest.json", manifest)
    print(dumps_json(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="nbeatsx")
    parser.add_argument("--domain", default="aluminum_load")
    parser.add_argument("--horizon", choices=list(MAIN_HORIZONS), default="4h")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--config-id", default="fixed_base_v0")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--max-test-windows", type=int, default=0)
    parser.add_argument("--window-subset-manifest", type=Path, default=None)
    parser.add_argument("--eval-split", action="append", choices=["validation", "test"])
    parser.add_argument("--max-hist-exog", type=int, default=16)
    parser.add_argument("--min-hist-exog-coverage", type=float, default=0.95)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--val-check-steps", type=int, default=100)
    parser.add_argument("--early-stop-patience-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--valid-batch-size", type=int, default=64)
    parser.add_argument("--windows-batch-size", type=int, default=1024)
    parser.add_argument("--inference-windows-batch-size", type=int, default=1024)
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=1,
        help="CPU workers for P1c window-to-NeuralForecast frame construction; training/inference still requires CUDA.",
    )
    parser.add_argument(
        "--preprocess-chunk-size",
        type=int,
        default=0,
        help="Windows per preprocessing task; 0 auto-splits work across preprocess workers.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--scaler-type",
        default="auto",
        help="Use 'auto' for model-specific defaults: nbeatsx=identity, tft=robust.",
    )
    parser.add_argument("--nbeatsx-blocks", type=int, default=3)
    parser.add_argument("--nbeatsx-width", type=int, default=256)
    parser.add_argument("--tft-hidden-size", type=int, default=128)
    parser.add_argument("--tft-heads", type=int, default=4)
    parser.add_argument("--tft-rnn-layers", type=int, default=1)
    parser.add_argument("--accelerator", choices=["cuda", "auto", "gpu"], default="auto")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--log-lr", action="store_true")
    parser.add_argument("--enable-progress-bar", action="store_true")
    parser.add_argument("--verbose-fit", action="store_true")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=False)
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args()


def enforce_gpu_execution(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/GPU is required for DL model training, validation, testing and inference")
    if args.accelerator == "cpu":
        raise ValueError("CPU execution is forbidden for DL/TSFM tests in this project")
    if args.accelerator in {"auto", "cuda"}:
        args.accelerator = "gpu"
    if args.preprocess_workers < 1:
        raise ValueError("--preprocess-workers must be >= 1")
    if args.preprocess_chunk_size < 0:
        raise ValueError("--preprocess-chunk-size must be >= 0")


def main() -> None:
    args = parse_args()
    enforce_gpu_execution(args)
    if args.inference_only:
        run_inference_only(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
