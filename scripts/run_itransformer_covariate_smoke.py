#!/usr/bin/env python3
"""Run a covariate-aware iTransformer smoke test on P1c windows.

The NeuralForecast iTransformer backend available in the project environment is
target-only in practice. This runner provides a local, auditable implementation
of the iTransformer inverted-variate idea for the P1c task contract:

- target history, historical exogenous variables and historical calendar
  features are represented as variate tokens;
- known-future calendar features are used only through a separate conditioning
  head;
- future measured weather, irradiance, PV generation, load, demand or power
  values are never used at validation/test time.

The architecture follows the inverted-token design of the official THUML
iTransformer repository. This file is a project implementation for smoke and
preflight validation only; it is not yet a formal paper-result runner.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from energy_tsfm_p2_core import (
    MAIN_HORIZONS,
    P1cWindowDataset,
    WindowBatch,
    build_prediction_stub,
    list_domains,
    load_canonical,
    load_window_index,
    validate_prediction_against_windows,
)
from evaluate_energy_tsfm_predictions import evaluate_prediction_frame, write_metrics


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = PROJECT / "results" / "energy_tsfm_p2_smoke" / "itransformer_covariate"
DEFAULT_SEED = 20260514
MODEL_FAMILY = "dl"
MODEL_ID = "itransformer"

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


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default)


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


def calendar_features(timestamp: Any) -> list[float]:
    ts = pd.Timestamp(timestamp)
    hour = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    dow = ts.dayofweek
    month = ts.month
    return [
        math.sin(2.0 * math.pi * hour / 24.0),
        math.cos(2.0 * math.pi * hour / 24.0),
        math.sin(2.0 * math.pi * dow / 7.0),
        math.cos(2.0 * math.pi * dow / 7.0),
        math.sin(2.0 * math.pi * (month - 1) / 12.0),
        math.cos(2.0 * math.pi * (month - 1) / 12.0),
        float(dow >= 5),
    ]


def calendar_frame(timestamps: pd.Series) -> np.ndarray:
    return np.asarray([calendar_features(ts) for ts in timestamps], dtype=np.float32)


def finite_fill(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=np.float32)
    arr[~np.isfinite(arr)] = np.nan
    if np.isnan(arr).all():
        return np.zeros_like(arr, dtype=np.float32)
    fill = float(np.nanmedian(arr))
    arr = np.where(np.isnan(arr), fill, arr)
    return arr.astype(np.float32)


def normalize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    center = float(np.mean(values))
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return ((values - center) / scale).astype(np.float32), center, scale


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


@dataclass(frozen=True)
class TensorWindow:
    batch: WindowBatch
    context_vars: torch.Tensor
    future_calendar: torch.Tensor
    y_scaled: torch.Tensor
    target_center: float
    target_scale: float


def batch_to_tensor_window(batch: WindowBatch, hist_exog_cols: list[str]) -> TensorWindow:
    context = batch.context.reset_index(drop=True)
    target = batch.target.reset_index(drop=True)

    target_context = finite_fill(context["target"])
    target_input, target_center, target_scale = normalize(target_context)
    target_future_raw = finite_fill(target["target"])
    target_future_scaled = ((target_future_raw - target_center) / target_scale).astype(np.float32)

    variates: list[np.ndarray] = [target_input]

    for col in hist_exog_cols:
        values = finite_fill(context[col])
        values_scaled, _, _ = normalize(values)
        variates.append(values_scaled)

    context_calendar = calendar_frame(context["timestamp"])
    for idx in range(context_calendar.shape[1]):
        variates.append(context_calendar[:, idx])

    context_vars = np.stack(variates, axis=1).astype(np.float32)
    future_calendar = calendar_frame(target["timestamp"])

    return TensorWindow(
        batch=batch,
        context_vars=torch.from_numpy(context_vars),
        future_calendar=torch.from_numpy(future_calendar),
        y_scaled=torch.from_numpy(target_future_scaled),
        target_center=target_center,
        target_scale=target_scale,
    )


class ITransformerWindowDataset(Dataset[TensorWindow]):
    def __init__(
        self,
        dataset: P1cWindowDataset,
        positions: list[int],
        *,
        hist_exog_cols: list[str],
    ) -> None:
        self.dataset = dataset
        self.positions = positions
        self.hist_exog_cols = hist_exog_cols

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> TensorWindow:
        return batch_to_tensor_window(self.dataset.get(self.positions[index]), self.hist_exog_cols)


def collate_tensor_windows(items: list[TensorWindow]) -> dict[str, Any]:
    return {
        "batch_objects": [item.batch for item in items],
        "context_vars": torch.stack([item.context_vars for item in items], dim=0),
        "future_calendar": torch.stack([item.future_calendar for item in items], dim=0),
        "y_scaled": torch.stack([item.y_scaled for item in items], dim=0),
        "target_center": torch.tensor([item.target_center for item in items], dtype=torch.float32),
        "target_scale": torch.tensor([item.target_scale for item in items], dtype=torch.float32),
    }


class LocalCovariateITransformer(nn.Module):
    """Minimal covariate-aware iTransformer-style encoder.

    Inputs are shaped as B x L x V, where V includes the target, historical
    exogenous variables and historical calendar variables. Attention is applied
    across inverted variate tokens. The target token is decoded to the forecast
    horizon, optionally conditioned on known-future calendar features.
    """

    def __init__(
        self,
        *,
        lookback_steps: int,
        horizon_steps: int,
        n_variates: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_ff: int,
        dropout: float,
        use_future_calendar: bool,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.lookback_steps = lookback_steps
        self.horizon_steps = horizon_steps
        self.n_variates = n_variates
        self.use_future_calendar = use_future_calendar

        self.temporal_embedding = nn.Linear(lookback_steps, d_model)
        self.variate_embedding = nn.Parameter(torch.zeros(1, n_variates, d_model))
        nn.init.normal_(self.variate_embedding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        if use_future_calendar:
            self.future_calendar_encoder = nn.Sequential(
                nn.Linear(horizon_steps * len(CALENDAR_FEATURES), d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            self.future_calendar_encoder = None

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, horizon_steps),
        )

    def forward(self, context_vars: torch.Tensor, future_calendar: torch.Tensor | None = None) -> torch.Tensor:
        if context_vars.ndim != 3:
            raise ValueError(f"context_vars must be B x L x V, got {tuple(context_vars.shape)}")
        if context_vars.shape[1] != self.lookback_steps:
            raise ValueError(f"lookback mismatch: {context_vars.shape[1]} != {self.lookback_steps}")
        if context_vars.shape[2] != self.n_variates:
            raise ValueError(f"variate mismatch: {context_vars.shape[2]} != {self.n_variates}")

        tokens = context_vars.transpose(1, 2)
        tokens = self.temporal_embedding(tokens) + self.variate_embedding
        encoded = self.encoder_norm(self.encoder(tokens))
        target_token = encoded[:, 0, :]

        if self.future_calendar_encoder is not None:
            if future_calendar is None:
                raise ValueError("future_calendar is required when use_future_calendar=True")
            target_token = target_token + self.future_calendar_encoder(future_calendar.flatten(start_dim=1))

        return self.head(target_token)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def resolve_device(accelerator: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/GPU is required for DL/TSFM tests in this project")
    if accelerator == "auto":
        accelerator = "cuda"
    if accelerator == "cuda":
        return torch.device("cuda")
    if accelerator == "cpu":
        raise ValueError("CPU execution is forbidden for DL/TSFM tests in this project")
    raise ValueError(f"unsupported accelerator: {accelerator}")


def train_model(
    model: LocalCovariateITransformer,
    train_loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    max_train_batches: int | None,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, Any]] = []
    model.train()
    for epoch in range(epochs):
        losses: list[float] = []
        for step, batch in enumerate(train_loader):
            if max_train_batches is not None and step >= max_train_batches:
                break
            context_vars = batch["context_vars"].to(device)
            future_calendar = batch["future_calendar"].to(device)
            y_scaled = batch["y_scaled"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred_scaled = model(context_vars, future_calendar)
            loss = F.mse_loss(pred_scaled, y_scaled)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        history.append(
            {
                "epoch": epoch,
                "train_batches": len(losses),
                "train_mse": None if not losses else float(np.mean(losses)),
            }
        )
    return history


@torch.no_grad()
def predict_split(
    model: LocalCovariateITransformer,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    config_id: str,
    seed: int,
    hist_exog_cols: list[str],
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    notes = (
        "local_covariate_itransformer;"
        f"hist_exog_count={len(hist_exog_cols)};"
        "future_calendar_only=true;"
        "future_measured_exog=false"
    )
    for batch in loader:
        context_vars = batch["context_vars"].to(device)
        future_calendar = batch["future_calendar"].to(device)
        pred_scaled = model(context_vars, future_calendar).detach().cpu()
        centers = batch["target_center"].view(-1, 1)
        scales = batch["target_scale"].view(-1, 1)
        pred_raw = pred_scaled * scales + centers
        for batch_obj, one_pred in zip(batch["batch_objects"], pred_raw.numpy(), strict=True):
            rows.append(
                build_prediction_stub(
                    batch_obj,
                    model_family=MODEL_FAMILY,
                    model_id=MODEL_ID,
                    config_id=config_id,
                    seed=seed,
                    y_pred=pd.Series(one_pred),
                    notes=notes,
                )
            )
    return pd.DataFrame(rows)


def run_one(
    *,
    domain: str,
    horizon: str,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    train_base = P1cWindowDataset(domain, horizon, split="train")
    val_base = P1cWindowDataset(domain, horizon, split="validation")
    test_base = P1cWindowDataset(domain, horizon, split="test")

    train_positions = sample_positions(
        len(train_base),
        args.max_train_windows,
        f"{args.seed}:{domain}:{horizon}:train",
    )
    val_positions = sample_positions(
        len(val_base),
        args.max_eval_windows,
        f"{args.seed}:{domain}:{horizon}:validation",
    )
    test_positions = sample_positions(
        len(test_base),
        args.max_eval_windows,
        f"{args.seed}:{domain}:{horizon}:test",
    )
    if not train_positions or not val_positions or not test_positions:
        raise ValueError(f"{domain}/{horizon}: empty sampled train/validation/test positions")

    hist_exog_cols = select_hist_exog_columns(
        domain,
        max_hist_exog=args.max_hist_exog,
        min_coverage=args.min_hist_exog_coverage,
    )

    first_batch = train_base.get(train_positions[0])
    lookback_steps = int(first_batch.metadata["context_steps"])
    horizon_steps = int(first_batch.metadata["horizon_steps"])
    n_variates = 1 + len(hist_exog_cols) + len(CALENDAR_FEATURES)

    train_set = ITransformerWindowDataset(train_base, train_positions, hist_exog_cols=hist_exog_cols)
    val_set = ITransformerWindowDataset(val_base, val_positions, hist_exog_cols=hist_exog_cols)
    test_set = ITransformerWindowDataset(test_base, test_positions, hist_exog_cols=hist_exog_cols)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_tensor_windows,
        num_workers=0,
    )
    eval_batch_size = max(1, min(args.eval_batch_size, args.max_eval_windows if args.max_eval_windows > 0 else args.eval_batch_size))
    val_loader = DataLoader(
        val_set,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_tensor_windows,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_tensor_windows,
        num_workers=0,
    )

    model = LocalCovariateITransformer(
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

    start = time.time()
    history = train_model(
        model,
        train_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_train_batches=args.max_train_batches,
    )
    runtime_sec = time.time() - start

    config_id = (
        f"local_cov_itransformer_smoke_v1_"
        f"d{args.d_model}_h{args.n_heads}_L{args.e_layers}_ff{args.d_ff}_"
        f"hist{len(hist_exog_cols)}"
    )

    predictions = pd.concat(
        [
            predict_split(
                model,
                val_loader,
                device=device,
                config_id=config_id,
                seed=args.seed,
                hist_exog_cols=hist_exog_cols,
            ),
            predict_split(
                model,
                test_loader,
                device=device,
                config_id=config_id,
                seed=args.seed,
                hist_exog_cols=hist_exog_cols,
            ),
        ],
        ignore_index=True,
    )

    full_index = load_window_index(domain, horizon)
    validate_prediction_against_windows(predictions, full_index)
    metrics = evaluate_prediction_frame(predictions)

    safe_horizon = horizon.replace("/", "_")
    prediction_path = run_dir / f"predictions_{domain}_{safe_horizon}.parquet"
    metrics_dir = run_dir / "per_domain_metrics" / domain / safe_horizon
    predictions.to_parquet(prediction_path, index=False)
    write_metrics(metrics, metrics_dir, stem="metrics")

    return {
        "domain": domain,
        "horizon": horizon,
        "status": "ok",
        "prediction_path": str(prediction_path),
        "prediction_rows": int(len(predictions)),
        "metric_rows": int(len(metrics)),
        "train_windows": int(len(train_positions)),
        "validation_windows": int(len(val_positions)),
        "test_windows": int(len(test_positions)),
        "lookback_steps": lookback_steps,
        "horizon_steps": horizon_steps,
        "hist_exog_cols": hist_exog_cols,
        "calendar_feature_count": len(CALENDAR_FEATURES),
        "future_calendar_used": not args.disable_future_calendar,
        "future_measured_exog_used": False,
        "n_variates": n_variates,
        "config_id": config_id,
        "parameter_count": count_parameters(model),
        "train_history": history,
        "runtime_sec": runtime_sec,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", default=None, help="Domain IDs. Defaults to all P1c domains.")
    parser.add_argument("--horizons", nargs="+", default=list(MAIN_HORIZONS), choices=list(MAIN_HORIZONS))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-train-windows", type=int, default=8)
    parser.add_argument("--max-eval-windows", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-hist-exog", type=int, default=8)
    parser.add_argument("--min-hist-exog-coverage", type=float, default=0.95)
    parser.add_argument("--disable-future-calendar", action="store_true")
    parser.add_argument("--accelerator", choices=["cuda", "auto"], default="auto")
    parser.add_argument("--devices", default="1", help="Recorded for compatibility; this runner uses the first visible device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    device = resolve_device(args.accelerator)
    if device.type == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU device")

    domains = args.domains or list_domains()
    run_name = (
        f"itransformer_covariate_smoke_seed{args.seed}_"
        f"tr{args.max_train_windows}_ev{args.max_eval_windows}_ep{args.epochs}_"
        f"hist{args.max_hist_exog}_d{args.d_model}_L{args.e_layers}_"
        f"acc{args.accelerator}_dev{args.devices}"
    )
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for domain in domains:
        for horizon in args.horizons:
            print(f"[{MODEL_ID}] {domain}/{horizon}")
            record = run_one(domain=domain, horizon=horizon, args=args, run_dir=run_dir, device=device)
            runs.append(record)
            prediction_frames.append(pd.read_parquet(record["prediction_path"]))

    predictions_all = pd.concat(prediction_frames, ignore_index=True)
    metrics_all = evaluate_prediction_frame(predictions_all)
    predictions_all_path = run_dir / "predictions_all.parquet"
    predictions_all.to_parquet(predictions_all_path, index=False)
    metric_paths = write_metrics(metrics_all, run_dir, stem="metrics_all")

    manifest = {
        "status": "ok",
        "script": str(Path(__file__).resolve()),
        "purpose": "covariate-aware iTransformer smoke runner; not formal paper results",
        "official_reference": "https://github.com/thuml/iTransformer",
        "backend": "local_project_implementation_inverted_variate_tokens",
        "model_family": MODEL_FAMILY,
        "model_id": MODEL_ID,
        "args": vars(args),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "calendar_features": CALENDAR_FEATURES,
        "future_measured_exog_used": False,
        "prediction_rows": int(len(predictions_all)),
        "metric_rows": int(len(metrics_all)),
        "predictions_all": str(predictions_all_path),
        "metrics": metric_paths,
        "runs": runs,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(dumps(manifest) + "\n", encoding="utf-8")
    print(dumps({"status": "ok", "run_dir": str(run_dir), "prediction_rows": len(predictions_all), "metric_rows": len(metrics_all)}))


if __name__ == "__main__":
    main()
