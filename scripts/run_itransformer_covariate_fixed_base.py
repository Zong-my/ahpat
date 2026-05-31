#!/usr/bin/env python3
"""Formal fixed-base runner for the local covariate-aware iTransformer.

This runner is the first formal DL baseline path built on top of the P1c frozen
window contract. It is still a development runner until full-domain jobs are
launched, but it implements the formal artifacts required for paper-grade runs:
validation monitoring, best/last/previous checkpoints, resume, curve files,
common-schema predictions, evaluator metrics and one-command best-checkpoint
inference.
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from energy_tsfm_formal_artifacts import (
    DEFAULT_FORMAL_ROOT,
    FormalRunPaths,
    append_jsonl,
    config_hash,
    cuda_metadata,
    dumps_json,
    ensure_formal_run_dirs,
    load_jsonl,
    log_stderr,
    log_stdout,
    make_run_paths,
    package_versions,
    read_json,
    save_checkpoint_atomic,
    update_curve_artifacts,
    write_inference_entrypoint,
    write_json_atomic,
)
from energy_tsfm_p2_core import (
    MAIN_HORIZONS,
    P1cWindowDataset,
    load_window_index,
    validate_prediction_against_windows,
)
from evaluate_energy_tsfm_predictions import evaluate_prediction_frame, write_metrics
from run_itransformer_covariate_smoke import (
    CALENDAR_FEATURES,
    MODEL_FAMILY,
    MODEL_ID,
    ITransformerWindowDataset,
    LocalCovariateITransformer,
    collate_tensor_windows,
    count_parameters,
    predict_split,
    resolve_device,
    sample_positions,
    select_hist_exog_columns,
)


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 20260514
EPS = 1e-12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")


def make_loss_fn(name: str, huber_delta: float) -> nn.Module:
    if name == "huber":
        return nn.HuberLoss(delta=huber_delta, reduction="mean")
    if name == "l1":
        return nn.L1Loss(reduction="mean")
    raise ValueError(f"unsupported loss: {name}")


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def window_subset_record(base: P1cWindowDataset, positions: list[int]) -> dict[str, Any]:
    window_ids = [str(base.windows.iloc[int(pos)]["window_id"]) for pos in positions]
    return {
        "count": int(len(positions)),
        "positions": [int(pos) for pos in positions],
        "window_ids": window_ids,
    }


def validate_window_subset_record(base: P1cWindowDataset, record: dict[str, Any], *, split_name: str) -> list[int]:
    positions = [int(pos) for pos in record["positions"]]
    expected_ids = [str(window_id) for window_id in record["window_ids"]]
    actual_ids = [str(base.windows.iloc[pos]["window_id"]) for pos in positions]
    if actual_ids != expected_ids:
        raise ValueError(f"{split_name} window subset does not match current P1c index")
    return positions


def load_shared_window_subset_record(
    manifest_path: Path,
    *,
    domain: str,
    horizon: str,
    train_base: P1cWindowDataset,
    val_base: P1cWindowDataset,
    test_base: P1cWindowDataset,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    try:
        record = manifest["subsets"][domain][horizon]
    except KeyError as exc:
        raise KeyError(f"shared subset manifest {manifest_path} missing {domain}/{horizon}") from exc
    validate_window_subset_record(train_base, record["train"], split_name="train")
    validate_window_subset_record(val_base, record["validation"], split_name="validation")
    validate_window_subset_record(test_base, record["test"], split_name="test")
    copied = dict(record)
    copied["source_window_subset_manifest"] = str(manifest_path)
    copied["source_subset_id"] = str(manifest.get("subset_id", record.get("subset_id", "unknown")))
    copied["sampling_rule"] = str(record.get("sampling_rule", manifest.get("sampling_rule", "shared subset manifest")))
    return copied


def get_or_create_window_subsets(
    paths: FormalRunPaths,
    *,
    domain: str,
    horizon: str,
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
        train_positions = validate_window_subset_record(train_base, saved["train"], split_name="train")
        val_positions = validate_window_subset_record(val_base, saved["validation"], split_name="validation")
        test_positions = validate_window_subset_record(test_base, saved["test"], split_name="test")
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
        write_json_atomic(paths.window_subsets, record)
        return train_base, val_base, test_base, record

    train_positions = sample_positions(
        len(train_base),
        max_train_windows,
        f"{seed}:{domain}:{horizon}:formal:train",
    )
    val_positions = sample_positions(
        len(val_base),
        max_val_windows,
        f"{seed}:{domain}:{horizon}:formal:validation",
    )
    test_positions = sample_positions(
        len(test_base),
        max_test_windows,
        f"{seed}:{domain}:{horizon}:formal:test",
    )
    if not train_positions or not val_positions or not test_positions:
        raise ValueError(f"{domain}/{horizon}: empty train/validation/test window subset")

    record = {
        "domain_id": domain,
        "horizon": horizon,
        "seed": int(seed),
        "sampling_rule": "sample_positions; 0 means full split; includes first/middle/last anchors",
        "train": window_subset_record(train_base, train_positions),
        "validation": window_subset_record(val_base, val_positions),
        "test": window_subset_record(test_base, test_positions),
    }
    write_json_atomic(paths.window_subsets, record)
    return train_base, val_base, test_base, record


def make_data_loaders(
    *,
    train_base: P1cWindowDataset,
    val_base: P1cWindowDataset,
    test_base: P1cWindowDataset,
    subsets: dict[str, Any],
    hist_exog_cols: list[str],
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 0,
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]], DataLoader[dict[str, Any]]]:
    train_set = ITransformerWindowDataset(
        train_base,
        [int(pos) for pos in subsets["train"]["positions"]],
        hist_exog_cols=hist_exog_cols,
    )
    val_set = ITransformerWindowDataset(
        val_base,
        [int(pos) for pos in subsets["validation"]["positions"]],
        hist_exog_cols=hist_exog_cols,
    )
    test_set = ITransformerWindowDataset(
        test_base,
        [int(pos) for pos in subsets["test"]["positions"]],
        hist_exog_cols=hist_exog_cols,
    )
    loader_kwargs: dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor > 0:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return (
        DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_tensor_windows,
            **loader_kwargs,
        ),
        DataLoader(
            val_set,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collate_tensor_windows,
            **loader_kwargs,
        ),
        DataLoader(
            test_set,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collate_tensor_windows,
            **loader_kwargs,
        ),
    )


def make_model(
    *,
    lookback_steps: int,
    horizon_steps: int,
    n_variates: int,
    args: argparse.Namespace,
    device: torch.device,
) -> LocalCovariateITransformer:
    return LocalCovariateITransformer(
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        n_variates=n_variates,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_future_calendar=not args.disable_future_calendar,
    ).to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    gradient_clip: float,
    max_train_batches: int | None,
) -> tuple[float, int]:
    model.train()
    weighted_loss_sum = 0.0
    seen = 0
    for step, batch in enumerate(loader):
        if max_train_batches is not None and step >= max_train_batches:
            break
        context_vars = batch["context_vars"].to(device, non_blocking=True)
        future_calendar = batch["future_calendar"].to(device, non_blocking=True)
        y_scaled = batch["y_scaled"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred_scaled = model(context_vars, future_calendar)
        loss = loss_fn(pred_scaled, y_scaled)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss: {loss.detach().cpu().item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()

        batch_size = int(context_vars.shape[0])
        weighted_loss_sum += float(loss.detach().cpu()) * batch_size
        seen += batch_size
    if seen <= 0:
        raise ValueError("training epoch consumed zero batches")
    return weighted_loss_sum / seen, seen


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    loss_fn: nn.Module,
    metric_scope: str,
) -> dict[str, Any]:
    model.eval()
    weighted_loss_sum = 0.0
    seen = 0
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []
    scoped_windows = 0

    for batch in loader:
        context_vars = batch["context_vars"].to(device, non_blocking=True)
        future_calendar = batch["future_calendar"].to(device, non_blocking=True)
        y_scaled = batch["y_scaled"].to(device, non_blocking=True)
        pred_scaled = model(context_vars, future_calendar)
        loss = loss_fn(pred_scaled, y_scaled)

        batch_size = int(context_vars.shape[0])
        weighted_loss_sum += float(loss.detach().cpu()) * batch_size
        seen += batch_size

        centers = batch["target_center"].view(-1, 1)
        scales = batch["target_scale"].view(-1, 1)
        pred_raw = (pred_scaled.detach().cpu() * scales + centers).numpy()
        true_raw = (batch["y_scaled"].detach().cpu() * scales + centers).numpy()

        for idx, batch_obj in enumerate(batch["batch_objects"]):
            include = True
            if metric_scope != "full_day":
                include = bool(batch_obj.metadata.get(f"metric_{metric_scope}", False))
            if include:
                y_true_chunks.append(true_raw[idx])
                y_pred_chunks.append(pred_raw[idx])
                scoped_windows += 1

    if seen <= 0:
        raise ValueError("validation loader consumed zero batches")

    if not y_true_chunks:
        primary_wape = math.inf
        n_points = 0
    else:
        y_true_all = np.concatenate(y_true_chunks)
        y_pred_all = np.concatenate(y_pred_chunks)
        finite = np.isfinite(y_true_all) & np.isfinite(y_pred_all)
        y_true_all = y_true_all[finite]
        y_pred_all = y_pred_all[finite]
        denom = float(np.sum(np.abs(y_true_all)))
        primary_wape = math.inf if denom <= EPS else float(np.sum(np.abs(y_pred_all - y_true_all)) / denom)
        n_points = int(len(y_true_all))

    return {
        "validation_loss": weighted_loss_sum / seen,
        "validation_primary_wape": primary_wape,
        "validation_metric_scope": metric_scope,
        "validation_scoped_windows": int(scoped_windows),
        "validation_scoped_points": int(n_points),
    }


def make_checkpoint_payload(
    *,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    best_metric: float,
    best_epoch: int | None,
    bad_epochs: int,
    config: dict[str, Any],
    hist_exog_cols: list[str],
    subsets: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_validation_metric": float(best_metric),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "bad_epochs": int(bad_epochs),
        "config": config,
        "hist_exog_cols": hist_exog_cols,
        "window_subsets": subsets,
        "rng_state_torch": torch.get_rng_state(),
        "rng_state_numpy": np.random.get_state(),
        "rng_state_random": random.getstate(),
    }
    if torch.cuda.is_available():
        payload["rng_state_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def save_last_and_previous(paths: FormalRunPaths, payload: dict[str, Any]) -> None:
    if paths.last_checkpoint.exists():
        shutil.copy2(paths.last_checkpoint, paths.previous_checkpoint)
    save_checkpoint_atomic(paths.last_checkpoint, payload)


def load_checkpoint(path: Path, *, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def restore_training_state(
    checkpoint: dict[str, Any],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    restore_rng: bool = False,
) -> dict[str, Any]:
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if restore_rng:
        if "rng_state_torch" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state_torch"])
        if "rng_state_numpy" in checkpoint:
            np.random.set_state(checkpoint["rng_state_numpy"])
        if "rng_state_random" in checkpoint:
            random.setstate(checkpoint["rng_state_random"])
        if torch.cuda.is_available() and "rng_state_cuda" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["rng_state_cuda"])
    return {
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", 0)),
        "best_validation_metric": float(checkpoint.get("best_validation_metric", math.inf)),
        "best_epoch": checkpoint.get("best_epoch"),
        "bad_epochs": int(checkpoint.get("bad_epochs", 0)),
    }


def write_split_outputs(
    *,
    paths: FormalRunPaths,
    model: nn.Module,
    val_loader: DataLoader[dict[str, Any]],
    test_loader: DataLoader[dict[str, Any]],
    domain: str,
    horizon: str,
    device: torch.device,
    config_id: str,
    seed: int,
    hist_exog_cols: list[str],
    eval_splits: tuple[str, ...],
) -> dict[str, Any]:
    full_index = load_window_index(domain, horizon)

    validation_predictions = predict_split(
        model,
        val_loader,
        device=device,
        config_id=config_id,
        seed=seed,
        hist_exog_cols=hist_exog_cols,
    )
    validate_prediction_against_windows(validation_predictions, full_index)
    validation_predictions.to_parquet(paths.validation_predictions, index=False)
    validation_metrics = evaluate_prediction_frame(validation_predictions)
    validation_metric_paths = write_metrics(validation_metrics, paths.metrics_dir, stem="validation_metrics")

    outputs = {
        "validation_prediction_rows": int(len(validation_predictions)),
        "validation_metric_rows": int(len(validation_metrics)),
        "validation_predictions": str(paths.validation_predictions),
        "validation_metrics": validation_metric_paths,
    }
    if "test" in eval_splits:
        test_predictions = predict_split(
            model,
            test_loader,
            device=device,
            config_id=config_id,
            seed=seed,
            hist_exog_cols=hist_exog_cols,
        )
        validate_prediction_against_windows(test_predictions, full_index)
        test_predictions.to_parquet(paths.test_predictions, index=False)
        test_metrics = evaluate_prediction_frame(test_predictions)
        test_metric_paths = write_metrics(test_metrics, paths.metrics_dir, stem="test_metrics")
        outputs.update(
            {
                "test_prediction_rows": int(len(test_predictions)),
                "test_metric_rows": int(len(test_metrics)),
                "test_predictions": str(paths.test_predictions),
                "test_metrics": test_metric_paths,
            }
        )
    return outputs


def build_run_config(args: argparse.Namespace, *, hist_exog_cols: list[str] | None = None) -> dict[str, Any]:
    config = {
        "runner": str(Path(__file__).resolve()),
        "model_family": MODEL_FAMILY,
        "model_id": MODEL_ID,
        "implementation": "local_project_implementation_inverted_variate_tokens",
        "official_reference": "https://github.com/thuml/iTransformer",
        "domain_id": args.domain,
        "horizon": args.horizon,
        "config_id": args.config_id,
        "seed": int(args.seed),
        "eval_splits": list(args.eval_split or ("validation", "test")),
        "data_contract": {
            "window_source": "P1c frozen window index",
            "horizons_allowed": list(MAIN_HORIZONS),
            "no_future_measured_exog": True,
            "future_calendar_only": not args.disable_future_calendar,
            "split_rule": "forecast endpoint split from P1c",
            "segment_boundary_rule": "no window may cross segment_id",
        },
        "window_limits": {
            "max_train_windows": int(args.max_train_windows),
            "max_val_windows": int(args.max_val_windows),
            "max_test_windows": int(args.max_test_windows),
        },
        "model_config": {
            "d_model": int(args.d_model),
            "n_heads": int(args.n_heads),
            "e_layers": int(args.e_layers),
            "d_ff": int(args.d_ff),
            "dropout": float(args.dropout),
            "max_hist_exog": int(args.max_hist_exog),
            "min_hist_exog_coverage": float(args.min_hist_exog_coverage),
            "calendar_features": CALENDAR_FEATURES,
        },
        "training_config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "num_workers": int(args.num_workers),
            "pin_memory": bool(args.pin_memory),
            "persistent_workers": bool(args.persistent_workers),
            "prefetch_factor": int(args.prefetch_factor),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "loss": args.loss,
            "huber_delta": float(args.huber_delta),
            "gradient_clip": float(args.gradient_clip),
            "lr_factor": float(args.lr_factor),
            "lr_patience": int(args.lr_patience),
            "min_lr": float(args.min_lr),
            "early_stopping_patience": int(args.patience),
            "early_stopping_min_delta": float(args.min_delta),
            "max_train_batches": args.max_train_batches,
            "validation_selection_metric": "validation WAPE",
            "validation_metric_scope": args.validation_metric_scope,
        },
        "artifact_contract": {
            "best_checkpoint": "checkpoints/best.pt",
            "last_checkpoint": "checkpoints/last.pt",
            "previous_checkpoint": "checkpoints/previous.pt",
            "curves": ["curves/loss_curve.csv", "curves/loss_curve_latest.png", "curves/metric_curve_latest.png"],
            "one_command_inference": "inference/run_inference.sh",
        },
        "args": vars(args),
    }
    if hist_exog_cols is not None:
        config["hist_exog_cols"] = hist_exog_cols
    config["config_hash"] = config_hash(config)
    return config


def audit_metric_reproduction(original_path: Path, reproduced: pd.DataFrame) -> dict[str, Any]:
    if not original_path.exists():
        return {
            "status": "missing_original",
            "original_path": str(original_path),
            "max_abs_numeric_diff": None,
        }
    original = pd.read_csv(original_path)
    key_cols = ["domain_id", "horizon", "split", "model_family", "model_id", "config_id", "metric_scope"]
    numeric_cols = ["n_windows", "n_points", "mae", "rmse", "wape", "smape", "r2", "bias"]
    merged = original.merge(
        reproduced,
        on=key_cols,
        how="outer",
        suffixes=("_original", "_reproduced"),
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        return {
            "status": "key_mismatch",
            "original_path": str(original_path),
            "row_count_original": int(len(original)),
            "row_count_reproduced": int(len(reproduced)),
            "merge_counts": merged["_merge"].value_counts().to_dict(),
            "max_abs_numeric_diff": None,
        }
    max_diff = 0.0
    per_metric: dict[str, float] = {}
    for col in numeric_cols:
        left_col = f"{col}_original"
        right_col = f"{col}_reproduced"
        if left_col not in merged.columns or right_col not in merged.columns:
            continue
        left = pd.to_numeric(merged[left_col], errors="coerce")
        right = pd.to_numeric(merged[right_col], errors="coerce")
        diff = (left - right).abs()
        metric_max = float(diff.max(skipna=True)) if diff.notna().any() else 0.0
        per_metric[col] = metric_max
        max_diff = max(max_diff, metric_max)
    return {
        "status": "ok",
        "original_path": str(original_path),
        "row_count_original": int(len(original)),
        "row_count_reproduced": int(len(reproduced)),
        "max_abs_numeric_diff": float(max_diff),
        "per_metric_max_abs_diff": per_metric,
    }


def audit_reproduced_metrics(
    *,
    original_validation_metrics_path: Path,
    reproduced_validation_metrics: pd.DataFrame,
    original_test_metrics_path: Path,
    reproduced_test_metrics: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "validation": audit_metric_reproduction(
            original_validation_metrics_path,
            reproduced_validation_metrics,
        ),
        "test": audit_metric_reproduction(
            original_test_metrics_path,
            reproduced_test_metrics,
        ),
        "note": "Small nonzero differences can occur from CUDA floating-point kernels; the audit records the exact tolerance observed.",
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = resolve_device(args.accelerator)
    paths = make_run_paths(
        output_root=args.output_root,
        model_id=MODEL_ID,
        domain_id=args.domain,
        horizon=args.horizon,
        config_id=args.config_id,
        seed=args.seed,
    )
    ensure_formal_run_dirs(paths, resume=args.resume)
    start_time = time.time()

    log_stdout(paths, f"[{MODEL_ID}] formal fixed-base run: {args.domain}/{args.horizon} seed={args.seed}")
    log_stdout(paths, f"run_dir={paths.run_dir}")
    log_stdout(paths, f"device={device}")

    try:
        eval_splits = tuple(args.eval_split or ("validation", "test"))
        if "validation" not in eval_splits:
            raise ValueError("validation must be included in --eval-split")
        train_base, val_base, test_base, subsets = get_or_create_window_subsets(
            paths,
            domain=args.domain,
            horizon=args.horizon,
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
        train_loader, val_loader, test_loader = make_data_loaders(
            train_base=train_base,
            val_base=val_base,
            test_base=test_base,
            subsets=subsets,
            hist_exog_cols=hist_exog_cols,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )

        first_batch = train_base.get(int(subsets["train"]["positions"][0]))
        lookback_steps = int(first_batch.metadata["context_steps"])
        horizon_steps = int(first_batch.metadata["horizon_steps"])
        n_variates = 1 + len(hist_exog_cols) + len(CALENDAR_FEATURES)
        model = make_model(
            lookback_steps=lookback_steps,
            horizon_steps=horizon_steps,
            n_variates=n_variates,
            args=args,
            device=device,
        )
        loss_fn = make_loss_fn(args.loss, args.huber_delta)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )

        config = build_run_config(args, hist_exog_cols=hist_exog_cols)
        config["shape_config"] = {
            "lookback_steps": int(lookback_steps),
            "horizon_steps": int(horizon_steps),
            "n_variates": int(n_variates),
            "hist_exog_count": int(len(hist_exog_cols)),
            "calendar_feature_count": int(len(CALENDAR_FEATURES)),
        }
        config["parameter_count"] = count_parameters(model)
        config["package_versions"] = package_versions()
        write_json_atomic(paths.config, config)
        write_inference_entrypoint(paths, runner_script=Path(__file__).resolve())

        manifest: dict[str, Any] = {
            "status": "running",
            "run_dir": str(paths.run_dir),
            "model_family": MODEL_FAMILY,
            "model_id": MODEL_ID,
            "domain_id": args.domain,
            "horizon": args.horizon,
            "config_id": args.config_id,
            "seed": int(args.seed),
            "config_hash": config["config_hash"],
            "parameter_count": config["parameter_count"],
            "train_windows": int(subsets["train"]["count"]),
            "validation_windows": int(subsets["validation"]["count"]),
            "test_windows": int(subsets["test"]["count"]),
            "hist_exog_cols": hist_exog_cols,
            "normalizer_policy": "per-window target/context scaling; historical covariates normalized from context only",
            "loss_function": args.loss,
            "validation_selection_metric": "validation WAPE",
            "validation_metric_scope": args.validation_metric_scope,
            "p1c_window_source": str(PROJECT / "data" / "energy_tsfm_windows_p1c" / args.domain / f"window_index_{args.horizon}.parquet"),
            "window_subset_manifest": str(args.window_subset_manifest) if args.window_subset_manifest else None,
            "runner": str(Path(__file__).resolve()),
            "started_at_unix": start_time,
            "device": str(device),
            "cuda_metadata_start": cuda_metadata(device),
            "package_versions": package_versions(),
            "args": vars(args),
        }
        write_json_atomic(paths.manifest, manifest)

        start_epoch = 0
        global_step = 0
        best_metric = math.inf
        best_epoch: int | None = None
        bad_epochs = 0
        if args.resume:
            if not paths.last_checkpoint.exists():
                raise FileNotFoundError(f"--resume requested but missing {paths.last_checkpoint}")
            state = load_checkpoint(paths.last_checkpoint, device=device)
            restored = restore_training_state(
                state,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                restore_rng=True,
            )
            start_epoch = int(restored["epoch"]) + 1
            global_step = int(restored["global_step"])
            best_metric = float(restored["best_validation_metric"])
            best_epoch = restored["best_epoch"]
            bad_epochs = int(restored["bad_epochs"])
            log_stdout(paths, f"resumed from epoch={start_epoch - 1}, global_step={global_step}")

        train_rows = load_jsonl(paths.train_log_jsonl)
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            train_loss, train_seen = train_one_epoch(
                model,
                train_loader,
                device=device,
                loss_fn=loss_fn,
                optimizer=optimizer,
                gradient_clip=args.gradient_clip,
                max_train_batches=args.max_train_batches,
            )
            global_step += int(math.ceil(train_seen / max(1, args.batch_size)))
            validation_record = evaluate_loader(
                model,
                val_loader,
                device=device,
                loss_fn=loss_fn,
                metric_scope=args.validation_metric_scope,
            )
            val_metric = float(validation_record["validation_primary_wape"])
            scheduler.step(val_metric)
            improved = val_metric < (best_metric - args.min_delta)
            if improved:
                best_metric = val_metric
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1

            payload = make_checkpoint_payload(
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_metric=best_metric,
                best_epoch=best_epoch,
                bad_epochs=bad_epochs,
                config=config,
                hist_exog_cols=hist_exog_cols,
                subsets=subsets,
            )
            save_last_and_previous(paths, payload)
            if improved or not paths.best_checkpoint.exists():
                save_checkpoint_atomic(paths.best_checkpoint, payload)

            row = {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "train_loss": float(train_loss),
                "train_windows_seen": int(train_seen),
                **validation_record,
                "learning_rate": current_lr(optimizer),
                "best_validation_primary_wape": float(best_metric),
                "best_epoch": best_epoch,
                "bad_epochs": int(bad_epochs),
                "is_best": bool(improved),
                "epoch_runtime_sec": float(time.time() - epoch_start),
            }
            append_jsonl(paths.train_log_jsonl, row)
            train_rows.append(row)
            update_curve_artifacts(paths, train_rows, primary_metric_col="validation_primary_wape")
            log_stdout(
                paths,
                (
                    f"epoch={epoch} train_loss={train_loss:.6g} "
                    f"val_loss={validation_record['validation_loss']:.6g} "
                    f"val_wape={val_metric:.6g} best={best_metric:.6g} "
                    f"lr={current_lr(optimizer):.3g}"
                ),
            )

            if bad_epochs >= args.patience:
                log_stdout(paths, f"early stopping at epoch={epoch}; bad_epochs={bad_epochs}")
                break

        if not paths.best_checkpoint.exists():
            raise FileNotFoundError("best checkpoint was not created")
        best_state = load_checkpoint(paths.best_checkpoint, device=device)
        restore_training_state(best_state, model=model)
        output_info = write_split_outputs(
            paths=paths,
            model=model,
            val_loader=val_loader,
            test_loader=test_loader,
            domain=args.domain,
            horizon=args.horizon,
            device=device,
            config_id=args.config_id,
            seed=args.seed,
            hist_exog_cols=hist_exog_cols,
            eval_splits=eval_splits,
        )

        manifest.update(
            {
                "status": "ok",
                "completed_at_unix": time.time(),
                "runtime_sec": float(time.time() - start_time),
                "best_validation_primary_wape": float(best_metric),
                "best_epoch": best_epoch,
                "best_checkpoint": str(paths.best_checkpoint),
                "last_checkpoint": str(paths.last_checkpoint),
                "previous_checkpoint": str(paths.previous_checkpoint),
                "train_log_jsonl": str(paths.train_log_jsonl),
                "train_log_csv": str(paths.train_log_csv),
                "curves": {
                    "loss_curve_csv": str(paths.loss_curve_csv),
                    "loss_curve_png": str(paths.loss_curve_png),
                    "metric_curve_png": str(paths.metric_curve_png),
                },
                "outputs": output_info,
                "one_command_inference": str(paths.inference_script),
                "cuda_metadata_end": cuda_metadata(device),
                "evaluator_validation_status": "validate_prediction_against_windows passed before metrics",
                "interpretation": "formal fixed-base runner artifact; dry-run limits, if used, are not paper results",
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
    run_dir = args.run_dir
    if run_dir is None:
        raise ValueError("--inference-only requires --run-dir")
    paths = FormalRunPaths(run_dir=run_dir)
    config = read_json(paths.config)
    subsets = read_json(paths.window_subsets)
    device = resolve_device(args.accelerator)

    domain = str(config["domain_id"])
    horizon = str(config["horizon"])
    seed = int(config["seed"])
    hist_exog_cols = [str(col) for col in config["hist_exog_cols"]]
    eval_splits = tuple(args.eval_split or config.get("eval_splits") or ("validation", "test"))

    train_base = P1cWindowDataset(domain, horizon, split="train")
    val_base = P1cWindowDataset(domain, horizon, split="validation")
    test_base = P1cWindowDataset(domain, horizon, split="test")
    validate_window_subset_record(train_base, subsets["train"], split_name="train")
    validate_window_subset_record(val_base, subsets["validation"], split_name="validation")
    validate_window_subset_record(test_base, subsets["test"], split_name="test")

    _, val_loader, test_loader = make_data_loaders(
        train_base=train_base,
        val_base=val_base,
        test_base=test_base,
        subsets=subsets,
        hist_exog_cols=hist_exog_cols,
        batch_size=int(config["training_config"]["batch_size"]),
        eval_batch_size=int(config["training_config"]["eval_batch_size"]),
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=0,
    )
    first_batch = train_base.get(int(subsets["train"]["positions"][0]))
    model_args = argparse.Namespace(
        d_model=int(config["model_config"]["d_model"]),
        n_heads=int(config["model_config"]["n_heads"]),
        e_layers=int(config["model_config"]["e_layers"]),
        d_ff=int(config["model_config"]["d_ff"]),
        dropout=float(config["model_config"]["dropout"]),
        disable_future_calendar=not bool(config["data_contract"]["future_calendar_only"]),
    )
    model = make_model(
        lookback_steps=int(first_batch.metadata["context_steps"]),
        horizon_steps=int(first_batch.metadata["horizon_steps"]),
        n_variates=1 + len(hist_exog_cols) + len(CALENDAR_FEATURES),
        args=model_args,
        device=device,
    )
    state = load_checkpoint(paths.best_checkpoint, device=device)
    restore_training_state(state, model=model)

    full_index = load_window_index(domain, horizon)
    validation_predictions = predict_split(
        model,
        val_loader,
        device=device,
        config_id=str(config["config_id"]),
        seed=seed,
        hist_exog_cols=hist_exog_cols,
    )
    validate_prediction_against_windows(validation_predictions, full_index)
    validation_path = paths.inference_dir / "validation_predictions_reproduced.parquet"
    validation_predictions.to_parquet(validation_path, index=False)
    validation_metrics = evaluate_prediction_frame(validation_predictions)
    validation_metric_paths = write_metrics(validation_metrics, paths.inference_dir, stem="validation_metrics_reproduced")

    manifest = {
        "status": "ok",
        "run_dir": str(paths.run_dir),
        "checkpoint": str(paths.best_checkpoint),
        "device": str(device),
        "validation_predictions": str(validation_path),
        "validation_metrics": validation_metric_paths,
        "validation_prediction_rows": int(len(validation_predictions)),
        "validation_metric_rows": int(len(validation_metrics)),
    }
    if "test" in eval_splits:
        test_predictions = predict_split(
            model,
            test_loader,
            device=device,
            config_id=str(config["config_id"]),
            seed=seed,
            hist_exog_cols=hist_exog_cols,
        )
        validate_prediction_against_windows(test_predictions, full_index)
        test_path = paths.inference_dir / "test_predictions_reproduced.parquet"
        test_predictions.to_parquet(test_path, index=False)
        test_metrics = evaluate_prediction_frame(test_predictions)
        test_metric_paths = write_metrics(test_metrics, paths.inference_dir, stem="test_metrics_reproduced")
        reproducibility_audit = audit_reproduced_metrics(
            original_validation_metrics_path=paths.validation_metrics_csv,
            reproduced_validation_metrics=validation_metrics,
            original_test_metrics_path=paths.test_metrics_csv,
            reproduced_test_metrics=test_metrics,
        )
        manifest.update(
            {
                "test_predictions": str(test_path),
                "test_metrics": test_metric_paths,
                "test_prediction_rows": int(len(test_predictions)),
                "test_metric_rows": int(len(test_metrics)),
                "reproducibility_audit": reproducibility_audit,
            }
        )
    write_json_atomic(paths.inference_dir / "inference_manifest.json", manifest)
    print(dumps_json(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="aluminum_load")
    parser.add_argument("--horizon", choices=list(MAIN_HORIZONS), default="4h")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--config-id", default="fixed_base_v0")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-train-windows", type=int, default=0, help="0 means full train split.")
    parser.add_argument("--max-val-windows", type=int, default=0, help="0 means full validation split.")
    parser.add_argument("--max-test-windows", type=int, default=0, help="0 means full test split.")
    parser.add_argument("--window-subset-manifest", type=Path, default=None)
    parser.add_argument("--eval-split", action="append", choices=["validation", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prefetch-factor", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["huber", "l1"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--validation-metric-scope", default="full_day")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--e-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-hist-exog", type=int, default=16)
    parser.add_argument("--min-hist-exog-coverage", type=float, default=0.95)
    parser.add_argument("--disable-future-calendar", action="store_true")
    parser.add_argument("--accelerator", choices=["cuda", "auto"], default="auto")
    parser.add_argument("--devices", default="1", help="Recorded for compatibility; this runner uses the first visible device.")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=False)
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inference_only:
        run_inference_only(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
