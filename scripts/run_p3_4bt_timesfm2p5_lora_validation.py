#!/usr/bin/env python3
"""Run P3-4bt TimesFM 2.5 Transformers LoRA validation screen.

This validation-only runner extends the P3-4bq engineering preflight into a
small multi-cell supervised LoRA screen. It compares the same PEFT-wrapped
TimesFM 2.5 Transformers checkpoint with adapters disabled (frozen target-only)
against a target-only LoRA fine-tuned route on identical validation windows.

No test split is read. This is not a replacement for TimesFM XReg covariate
selection; it specifically tests whether the public Transformers checkpoint can
support supervised LoRA adaptation under the project window contract.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
try:
    from transformers import TimesFm2_5ModelForPrediction
except ImportError:
    from transformers import TimesFmModelForPrediction as TimesFm2_5ModelForPrediction

from energy_tsfm_p2_core import (
    P1cWindowDataset,
    build_prediction_stub,
    load_window_index,
    validate_prediction_against_windows,
)
from evaluate_energy_tsfm_predictions import evaluate_prediction_frame, write_metrics


PROJECT = Path(__file__).resolve().parents[1]
PLAN_ID = "p3_4bt_timesfm2p5_lora_validation_v0_codex_20260517"
DEFAULT_OUT_ROOT = PROJECT / "results" / "energy_tsfm_tuning" / PLAN_ID
DEFAULT_SUBSET_MANIFEST = (
    PROJECT
    / "data"
    / "energy_tsfm_tuning"
    / "p3_4k_stride_tuning_policy_v0_codex_20260515"
    / "stride4"
    / "subset_manifest.json"
)
DEFAULT_RUN_ID = "p3_4bt_timesfm2p5_lora_val_5domain_2h_train32_val32_epochs2_codex_20260517"
DEFAULT_MODEL_NAME = "google/timesfm-2.5-200m-transformers"
DEFAULT_SEED = 20260517
MODEL_FAMILY = "tsfm"
MODEL_ID = "timesfm2p5"
BASE_CONFIG_ID = "timesfm2p5_transformers_frozen_target_only_p3_4bt_v0_codex_20260517"
LORA_CONFIG_ID = "timesfm2p5_transformers_lora_target_only_p3_4bt_v0_codex_20260517"
ROUTE_BASE = "timesfm2p5_transformers_frozen_target_only_p3_4bt"
ROUTE_LORA = "timesfm2p5_transformers_lora_target_only_p3_4bt"
SELECTION_ID = "p3_4bt_timesfm2p5_lora_shared_windows"
TIMESFM_TRANSFORMERS_PATCH_LEN = 32
TIMESFM_TRANSFORMERS_NATIVE_OUTPUT_STEPS = 128
CONTEXT_ALIGNMENT = "right_truncate_to_largest_patch_multiple"


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


def require_cuda() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; TimesFM 2.5 LoRA validation is GPU-only")
    index = torch.cuda.current_device()
    return {
        "cuda_required": True,
        "cuda_available": True,
        "device": "cuda",
        "cuda_device_index": int(index),
        "cuda_device_name": torch.cuda.get_device_name(index),
    }


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")


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


def batches_from_positions(domain: str, horizon: str, split: str, positions: list[int]) -> list[Any]:
    dataset = P1cWindowDataset(domain, horizon, split=split)
    return [dataset.get(pos) for pos in positions]


def clean_numeric(values: pd.Series) -> torch.Tensor:
    numeric = pd.to_numeric(values, errors="coerce").astype(float).ffill().bfill().fillna(0.0)
    return torch.tensor(numeric.to_numpy(dtype=np.float32), dtype=torch.float32)


def align_context_for_timesfm_transformers(values: torch.Tensor) -> torch.Tensor:
    """Keep recent context while satisfying TimesFM 2.5 patch-size contract."""
    length = int(values.numel())
    if length % TIMESFM_TRANSFORMERS_PATCH_LEN == 0:
        return values
    usable = (length // TIMESFM_TRANSFORMERS_PATCH_LEN) * TIMESFM_TRANSFORMERS_PATCH_LEN
    if usable <= 0:
        raise ValueError(f"context length {length} is shorter than TimesFM patch length {TIMESFM_TRANSFORMERS_PATCH_LEN}")
    return values[-usable:]


def context_alignment_summary(batches: list[Any]) -> dict[str, Any]:
    raw = [int(len(batch.context)) for batch in batches]
    used = [
        int(align_context_for_timesfm_transformers(clean_numeric(batch.context["target"])).numel())
        for batch in batches
    ]
    return {
        "context_alignment": CONTEXT_ALIGNMENT,
        "patch_len": TIMESFM_TRANSFORMERS_PATCH_LEN,
        "raw_context_lengths": sorted(set(raw)),
        "used_context_lengths": sorted(set(used)),
        "any_context_truncated": bool(any(r != u for r, u in zip(raw, used, strict=True))),
    }


def iter_chunks(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def context_list(batches: list[Any], device: torch.device) -> list[torch.Tensor]:
    return [align_context_for_timesfm_transformers(clean_numeric(batch.context["target"])).to(device) for batch in batches]


def target_tensor(batches: list[Any], device: torch.device) -> torch.Tensor:
    values = [clean_numeric(batch.target["target"]) for batch in batches]
    lengths = {int(v.numel()) for v in values}
    if len(lengths) != 1:
        raise ValueError(f"mixed target lengths in one training batch: {sorted(lengths)}")
    return torch.stack(values).to(device)


def horizon_steps_for_batches(batches: list[Any]) -> int:
    lengths = {int(batch.metadata["horizon_steps"]) for batch in batches}
    if len(lengths) != 1:
        raise ValueError(f"mixed horizon steps in one batch: {sorted(lengths)}")
    return int(next(iter(lengths)))


def parameter_summary(model: torch.nn.Module) -> dict[str, int | float]:
    total = int(sum(param.numel() for param in model.parameters()))
    trainable = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
    return {
        "parameter_count_total": total,
        "parameter_count_trainable": trainable,
        "parameter_trainable_fraction": float(trainable / total) if total else 0.0,
    }


def make_lora_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    base = TimesFm2_5ModelForPrediction.from_pretrained(
        args.model_name,
        local_files_only=not bool(args.allow_download),
    )
    base.to(device)
    lora_config = LoraConfig(
        r=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=None,
    )
    model = get_peft_model(base, lora_config)
    model.to(device)
    return model


@contextlib.contextmanager
def adapters_disabled(model: torch.nn.Module) -> Iterator[None]:
    disable = getattr(model, "disable_adapter", None)
    if disable is None:
        yield
    else:
        with disable():
            yield


def model_forward_mean(model: torch.nn.Module, batches: list[Any], device: torch.device) -> torch.Tensor:
    past_values = context_list(batches, device)
    out = model(
        past_values=past_values,
        future_values=None,
        forecast_context_len=max(int(x.numel()) for x in past_values),
        truncate_negative=False,
        force_flip_invariance=False,
    )
    return out.mean_predictions


def forecast_mean_to_horizon(
    model: torch.nn.Module,
    batches: list[Any],
    *,
    device: torch.device,
    horizon_steps: int,
) -> torch.Tensor:
    contexts = context_list(batches, device)
    generated: list[torch.Tensor] = []
    remaining = int(horizon_steps)
    while remaining > 0:
        out = model(
            past_values=contexts,
            future_values=None,
            forecast_context_len=max(int(x.numel()) for x in contexts),
            truncate_negative=False,
            force_flip_invariance=False,
        )
        step = out.mean_predictions[:, : min(remaining, int(out.mean_predictions.shape[1]))]
        generated.append(step)
        remaining -= int(step.shape[1])
        if remaining > 0:
            contexts = [
                align_context_for_timesfm_transformers(torch.cat([context, step[idx].detach()], dim=0))
                for idx, context in enumerate(contexts)
            ]
    return torch.cat(generated, dim=1)


def train_lora(
    model: torch.nn.Module,
    train_batches: list[Any],
    *,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    rng = random.Random(seed)
    history: list[dict[str, Any]] = []
    for epoch in range(int(args.epochs)):
        model.train()
        order = list(range(len(train_batches)))
        rng.shuffle(order)
        losses: list[float] = []
        grad_norms: list[float] = []
        for chunk_indexes in iter_chunks(order, int(args.batch_size)):
            chunk = [train_batches[idx] for idx in chunk_indexes]
            future = target_tensor(chunk, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model_forward_mean(model, chunk, device)
            loss_steps = min(int(pred.shape[1]), int(future.shape[1]))
            loss = torch.nn.functional.mse_loss(pred[:, :loss_steps], future[:, :loss_steps])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite TimesFM LoRA loss: {float(loss.detach().cpu())}")
            loss.backward()
            grad_sq = 0.0
            nonzero_grad_tensors = 0
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    grad_sq += float(param.grad.detach().float().pow(2).sum().item())
                    if float(param.grad.detach().float().abs().sum().item()) > 0.0:
                        nonzero_grad_tensors += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.gradient_clip))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            grad_norms.append(float(math.sqrt(grad_sq)))
        history.append(
            {
                "epoch": int(epoch),
                "batches": int(len(losses)),
                "train_loss": float(np.mean(losses)) if losses else None,
                "adapter_grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else None,
                "nonzero_grad_parameter_tensors_last_batch": int(nonzero_grad_tensors) if losses else 0,
                "loss_steps_last_batch": int(loss_steps) if losses else 0,
            }
        )
    return history


@torch.no_grad()
def predict_route(
    model: torch.nn.Module,
    batches: list[Any],
    *,
    device: torch.device,
    batch_size: int,
    config_id: str,
    route_label: str,
    seed: int,
    split: str,
    notes: str,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    for chunk in iter_chunks(batches, int(batch_size)):
        horizon_steps = horizon_steps_for_batches(chunk)
        pred = forecast_mean_to_horizon(model, chunk, device=device, horizon_steps=horizon_steps).detach().float().cpu().numpy()
        for idx, batch in enumerate(chunk):
            horizon_steps = int(batch.metadata["horizon_steps"])
            row = build_prediction_stub(
                batch,
                model_family=MODEL_FAMILY,
                model_id=MODEL_ID,
                config_id=config_id,
                seed=seed,
                y_pred=pd.Series(pred[idx, :horizon_steps]),
                notes=notes,
            )
            row["split"] = split
            row["route_label"] = route_label
            row["h1_branch"] = PLAN_ID
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_full_day(metrics: pd.DataFrame) -> dict[str, float]:
    mapping = {BASE_CONFIG_ID: ROUTE_BASE, LORA_CONFIG_ID: ROUTE_LORA}
    out: dict[str, float] = {}
    scoped = metrics[metrics["metric_scope"].astype(str) == "full_day"]
    for row in scoped.to_dict(orient="records"):
        if pd.notna(row.get("wape")):
            out[mapping.get(str(row["config_id"]), str(row["config_id"]))] = float(row["wape"])
    return out


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
        limit=int(args.max_train_windows),
        selection_id=f"{args.selection_id}:train",
    )
    split_limits = {
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
    train_batches = batches_from_positions(domain, horizon, "train", train_sel["positions"])
    eval_batches = {
        split: batches_from_positions(domain, horizon, split, selection["positions"])
        for split, selection in eval_selections.items()
    }
    if not train_batches or any(not batches for batches in eval_batches.values()):
        raise ValueError(f"{domain}/{horizon}: empty train or eval selection")

    cell_dir = run_dir / "cells" / f"{domain}_{horizon}"
    pred_dir = cell_dir / "predictions"
    metric_dir = cell_dir / "metrics"
    adapter_dir = cell_dir / "adapter"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    model = make_lora_model(args, device)
    params = parameter_summary(model)

    train_history = train_lora(model, train_batches, device=device, args=args, seed=seed)
    model.save_pretrained(adapter_dir)
    split_outputs: dict[str, Any] = {}
    split_prediction_frames: list[pd.DataFrame] = []
    for split, batches in eval_batches.items():
        with adapters_disabled(model):
            base_pred = predict_route(
                model,
                batches,
                device=device,
                batch_size=int(args.eval_batch_size),
                config_id=BASE_CONFIG_ID,
                route_label=ROUTE_BASE,
                seed=seed,
                split=split,
                notes=f"P3-4bt TimesFM 2.5 Transformers frozen target-only {split} prediction; adapters disabled.",
            )
        lora_pred = predict_route(
            model,
            batches,
            device=device,
            batch_size=int(args.eval_batch_size),
            config_id=LORA_CONFIG_ID,
            route_label=ROUTE_LORA,
            seed=seed,
            split=split,
            notes=(
                f"P3-4bt TimesFM 2.5 Transformers target-only LoRA {split} prediction; "
                "custom prefix loss on true project horizon."
            ),
        )
        split_predictions = pd.concat([base_pred, lora_pred], ignore_index=True)
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
        "arms": [ROUTE_BASE, ROUTE_LORA],
        "base_config_id": BASE_CONFIG_ID,
        "lora_config_id": LORA_CONFIG_ID,
        "model_name": args.model_name,
        "checkpoint_is_full_power_for_transformers_lora_route": True,
        "larger_timesfm_family_500m_not_used": True,
        "larger_timesfm_family_500m_reason": "No public Transformers/PEFT LoRA checkpoint route verified in this project yet.",
        "target_only_lora": True,
        "covariates_used": False,
        "full_train_used": bool(int(args.max_train_windows) <= 0),
        "full_validation_used": bool(int(args.max_validation_windows) <= 0),
        "full_test_used": bool("test" in eval_splits and int(args.max_test_windows) <= 0),
        "max_train_windows": int(args.max_train_windows),
        "max_validation_windows": int(args.max_validation_windows),
        "max_test_windows": int(args.max_test_windows),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "custom_prefix_loss_on_project_horizon": True,
        "native_model_output_steps": TIMESFM_TRANSFORMERS_NATIVE_OUTPUT_STEPS,
        "horizon_steps": int(train_batches[0].metadata["horizon_steps"]),
        "recursive_prediction_extension_used": bool(int(train_batches[0].metadata["horizon_steps"]) > TIMESFM_TRANSFORMERS_NATIVE_OUTPUT_STEPS),
        "training_loss_full_horizon_used": bool(int(train_batches[0].metadata["horizon_steps"]) <= TIMESFM_TRANSFORMERS_NATIVE_OUTPUT_STEPS),
        "training_loss_max_steps": int(min(int(train_batches[0].metadata["horizon_steps"]), TIMESFM_TRANSFORMERS_NATIVE_OUTPUT_STEPS)),
        "context_alignment_train": context_alignment_summary(train_batches),
        "context_alignment_eval": {split: context_alignment_summary(batches) for split, batches in eval_batches.items()},
        "train_selection": train_sel,
        "eval_splits": list(eval_splits),
        "eval_outputs": split_outputs,
        "validation_selection": split_outputs.get("validation", {}).get("selection"),
        "test_selection": split_outputs.get("test", {}).get("selection"),
        "train_history": train_history,
        "parameter_summary": params,
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "full_day_wape_by_route": wape,
        "prediction_rows": int(len(predictions)),
        "metric_rows": int(len(metrics)),
        "adapter_dir": str(adapter_dir),
        "predictions": str(pred_path),
        "metrics": metric_paths,
        "test_predictions_generated": "test" in eval_splits,
        "test_artifacts_created": "test" in eval_splits,
        "important_boundary": "TimesFM 2.5 target-only LoRA runner; test split is only evaluated when explicitly requested by P5 executor.",
        "elapsed_sec": round(time.time() - started, 3),
    }
    (cell_dir / "cell_manifest.json").write_text(dumps(cell_manifest) + "\n", encoding="utf-8")
    del model
    release_cuda()
    return predictions, cell_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", action="append", default=None)
    parser.add_argument("--horizon", action="append", default=None)
    parser.add_argument("--subset-manifest", type=Path, default=DEFAULT_SUBSET_MANIFEST)
    parser.add_argument("--max-train-windows", type=int, default=32)
    parser.add_argument("--max-validation-windows", type=int, default=32)
    parser.add_argument("--max-test-windows", type=int, default=0)
    parser.add_argument("--eval-split", action="append", choices=["validation", "test"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--accelerator", choices=["cuda"], default="cuda")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--selection-id", default=SELECTION_ID)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domains = args.domain or ["provincial_load", "aluminum_load", "microgrid_load", "arena_pv", "aidc_power_optional"]
    horizons = args.horizon or ["4h", "24h"]
    eval_splits = tuple(args.eval_split or ("validation",))
    started = time.time()
    cuda = require_cuda()
    torch.cuda.reset_peak_memory_stats()
    subset_manifest = read_json(args.subset_manifest)
    if not allowed_subset_manifest(subset_manifest):
        raise ValueError("P3-4bt expects an approved P3-4k stride4 or P3 target-pure subset manifest")

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
        "run_dir": str(run_dir),
        "subset_manifest": str(args.subset_manifest),
        "subset_id": subset_manifest.get("subset_id"),
        "selection_id": args.selection_id,
        "domains": domains,
        "horizons": horizons,
        "eval_splits": list(eval_splits),
        "arms": [ROUTE_BASE, ROUTE_LORA],
        "target_only_lora": True,
        "covariates_used": False,
        "full_train_used": bool(int(args.max_train_windows) <= 0),
        "full_validation_used": bool(int(args.max_validation_windows) <= 0),
        "full_test_used": bool("test" in eval_splits and int(args.max_test_windows) <= 0),
        "max_train_windows": int(args.max_train_windows),
        "max_validation_windows": int(args.max_validation_windows),
        "max_test_windows": int(args.max_test_windows),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
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
        "fine_tuning_kind": "peft_lora_target_only_custom_prefix_loss",
        "context_alignment": CONTEXT_ALIGNMENT,
        "context_patch_len": TIMESFM_TRANSFORMERS_PATCH_LEN,
        "native_model_output_steps": TIMESFM_TRANSFORMERS_NATIVE_OUTPUT_STEPS,
        "inference_launched": True,
        "forecast_metrics_computed": True,
        "prediction_artifact_saved": True,
        "adapter_artifact_saved": True,
        "test_split_read": "test" in eval_splits,
        "test_predictions_generated": "test" in eval_splits,
        "test_artifacts_created": "test" in eval_splits,
        "important_boundary": "TimesFM 2.5 target-only LoRA runner; test split is only evaluated when explicitly requested by P5 executor and no covariate/XReg claim is made.",
        "elapsed_sec": round(time.time() - started, 3),
    }
    (run_dir / "manifest.json").write_text(dumps(manifest) + "\n", encoding="utf-8")
    print(dumps(manifest))


if __name__ == "__main__":
    main()
