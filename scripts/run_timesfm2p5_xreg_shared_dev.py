#!/usr/bin/env python3
"""Run TimesFM 2.5 200M XReg on the shared P2 development subset.

This is a TimesFM-family covariate-adaptation development route, not the
full-power headline route. It keeps the 2.5 200M XReg result separate from the
TimesFM 2.0 500M target-only zero-shot result.

Inputs use only information available at origin time plus known-future calendar
features. Future measured weather, irradiance, load, PV, demand, and power
values remain forbidden.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import random
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import timesfm
import torch

from energy_tsfm_p2_core import (
    MAIN_HORIZONS,
    P1cWindowDataset,
    build_prediction_stub,
    list_domains,
    load_window_index,
    serialize_series,
    validate_prediction_against_windows,
)
from evaluate_energy_tsfm_predictions import evaluate_prediction_frame, write_metrics


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SUBSET_MANIFEST = (
    PROJECT
    / "data"
    / "energy_tsfm_dev_subsets"
    / "p2_fixed_base_dev_v0_codex_20260514"
    / "subset_manifest.json"
)
DEFAULT_OUT_ROOT = PROJECT / "results" / "energy_tsfm_formal" / "timesfm2p5_xreg_shared_dev"
DEFAULT_SEED = 20260514
MODEL_ID = "timesfm2p5"
MODEL_FAMILY = "tsfm"
CONFIG_ID = "timesfm2p5_xreg_shared_dev_v0_codex_20260514"
DEFAULT_MODEL_NAME = "google/timesfm-2.5-200m-pytorch"


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def load_timesfm2p5(
    model_name: str,
    *,
    allow_download: bool,
    torch_compile: bool,
) -> Any:
    """Load TimesFM 2.5 while avoiding a hub-mixin kwargs incompatibility."""
    return timesfm.TimesFM_2p5_200M_torch._from_pretrained(
        model_id=model_name,
        revision=None,
        cache_dir=None,
        force_download=False,
        local_files_only=not allow_download,
        token=None,
        torch_compile=torch_compile,
    )


def load_subset_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ok":
        raise ValueError(f"subset manifest is not ok: {path}")
    if "subsets" not in manifest:
        raise ValueError(f"subset manifest missing 'subsets': {path}")
    return manifest


def manifest_positions(manifest: dict[str, Any], domain: str, horizon: str, split: str) -> list[int]:
    try:
        values = manifest["subsets"][domain][horizon][split]["positions"]
    except KeyError as exc:
        raise KeyError(f"missing subset positions for {domain}/{horizon}/{split}") from exc
    return [int(v) for v in values]


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


def select_positions(
    positions: list[int],
    *,
    limit: int,
    seed_key: str,
) -> tuple[list[int], list[int]]:
    if limit <= 0 or limit >= len(positions):
        return list(positions), list(range(len(positions)))
    selected_manifest_indexes = sample_positions(len(positions), limit, seed_key)
    return [positions[i] for i in selected_manifest_indexes], selected_manifest_indexes


def batches_from_positions(domain: str, horizon: str, split: str, positions: list[int]) -> list[Any]:
    dataset = P1cWindowDataset(domain, horizon, split=split)
    return [dataset.get(pos) for pos in positions]


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


def numeric_target_history(context: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(context["target"], errors="coerce").astype(float)
    values = values.ffill().bfill().fillna(0.0)
    return values.to_numpy(dtype=float)


def build_calendar_covariates(batch: Any) -> pd.DataFrame:
    timestamps = pd.concat(
        [
            pd.to_datetime(batch.context["timestamp"], errors="raise").reset_index(drop=True),
            pd.to_datetime(batch.target["timestamp"], errors="raise").reset_index(drop=True),
        ],
        ignore_index=True,
    )
    return calendar_features(timestamps)


def build_dynamic_calendar_covariates(batches: list[Any]) -> dict[str, list[list[float]]]:
    by_col: dict[str, list[list[float]]] = {}
    for batch in batches:
        features = build_calendar_covariates(batch)
        for col in features.columns:
            by_col.setdefault(col, []).append(
                pd.to_numeric(features[col], errors="raise").astype(float).to_list()
            )
    return by_col


def quantile_columns(quantile_forecast: np.ndarray) -> dict[str, str]:
    if quantile_forecast.ndim != 2 or quantile_forecast.shape[1] < 10:
        return {}
    return {
        "q10": serialize_series(pd.Series(quantile_forecast[:, 1])),
        "q50": serialize_series(pd.Series(quantile_forecast[:, 5])),
        "q90": serialize_series(pd.Series(quantile_forecast[:, 9])),
    }


def predict_batches(
    model: Any,
    batches: list[Any],
    *,
    split: str,
    xreg_mode: str,
    ridge: float,
    max_rows_per_col: int,
    force_xreg_cpu: bool,
    batch_size: int,
) -> pd.DataFrame:
    if not batches:
        return pd.DataFrame()
    horizon_steps = int(batches[0].metadata["horizon_steps"])
    for batch in batches:
        if int(batch.metadata["horizon_steps"]) != horizon_steps:
            raise ValueError(f"{batch.window_id}: inconsistent horizon length")

    rows: list[dict[str, Any]] = []
    for start in range(0, len(batches), int(batch_size)):
        chunk = batches[start : start + int(batch_size)]
        point_forecast, quantile_forecast = model.forecast_with_covariates(
            inputs=[numeric_target_history(batch.context).tolist() for batch in chunk],
            dynamic_numerical_covariates=build_dynamic_calendar_covariates(chunk),
            xreg_mode=xreg_mode,
            normalize_xreg_target_per_input=True,
            ridge=ridge,
            max_rows_per_col=max_rows_per_col,
            force_on_cpu=force_xreg_cpu,
        )
        points = np.asarray(point_forecast, dtype=float)
        quantiles = np.asarray(quantile_forecast, dtype=float)
        for batch, y_pred, q_pred in zip(chunk, points, quantiles, strict=True):
            y_pred = np.asarray(y_pred, dtype=float)[:horizon_steps]
            q_pred = np.asarray(q_pred, dtype=float)[:horizon_steps]
            if len(y_pred) != horizon_steps:
                raise ValueError(
                    f"{batch.window_id}: TimesFM returned {len(y_pred)} steps, expected {horizon_steps}"
                )
            row = build_prediction_stub(
                batch,
                model_family=MODEL_FAMILY,
                model_id=MODEL_ID,
                config_id=CONFIG_ID,
                seed=DEFAULT_SEED,
                y_pred=pd.Series(y_pred),
                notes=(
                    "timesfm2p5_xreg_shared_dev; 200M checkpoint; context target plus "
                    "known-future calendar XReg; "
                    f"xreg_mode={xreg_mode}; ridge={ridge:g}; no future measured exogenous covariates"
                ),
            )
            row["split"] = split
            row.update(quantile_columns(q_pred))
            rows.append(row)
    return pd.DataFrame(rows)


def selected_batches(
    manifest: dict[str, Any],
    *,
    domain: str,
    horizon: str,
    split: str,
    limit: int,
    selection_id: str,
) -> tuple[list[Any], dict[str, Any]]:
    source_positions = manifest_positions(manifest, domain, horizon, split)
    positions, manifest_indexes = select_positions(
        source_positions,
        limit=int(limit),
        seed_key=f"{domain}:{horizon}:{split}:{selection_id}:{DEFAULT_SEED}",
    )
    if not positions:
        raise ValueError(f"{domain}/{horizon}/{split}: no selected positions")
    batches = batches_from_positions(domain, horizon, split, positions)
    return batches, {
        "source_count": int(len(source_positions)),
        "selected_count": int(len(positions)),
        "selected_manifest_indexes": manifest_indexes,
        "positions": positions,
    }


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def require_cuda_environment() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/GPU is required for TimesFM XReg shared-dev inference")


def cuda_metadata() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False, "device": "unavailable"}
    device_index = torch.cuda.current_device()
    return {
        "cuda_available": True,
        "device": "cuda",
        "cuda_device_index": int(device_index),
        "cuda_device_name": torch.cuda.get_device_name(device_index),
        "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(device_index)),
        "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
    }


def requested_eval_splits(args: argparse.Namespace) -> tuple[str, ...]:
    splits = tuple(args.eval_split or ("validation", "test"))
    if len(set(splits)) != len(splits):
        raise ValueError(f"duplicate eval splits requested: {splits}")
    return splits


def selection_seed_label(args: argparse.Namespace) -> str:
    return str(args.selection_id or args.run_id)


def run_cell(
    args: argparse.Namespace,
    *,
    model: Any,
    domain: str,
    horizon: str,
    cell_index: int,
    manifest: dict[str, Any],
    run_dir: Path,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    started = time.time()
    random.seed(DEFAULT_SEED + cell_index)
    np.random.seed(DEFAULT_SEED + cell_index)
    torch.manual_seed(DEFAULT_SEED + cell_index)
    eval_splits = requested_eval_splits(args)
    selection_id = selection_seed_label(args)
    cell_name = f"{domain}_{horizon}"
    cell_dir = run_dir / "cells" / cell_name
    pred_dir = cell_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    split_limits = {
        "validation": int(args.max_validation_windows),
        "test": int(args.max_test_windows),
    }
    all_predictions: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}

    for split in eval_splits:
        split_started = time.time()
        batches, selected_summary = selected_batches(
            manifest,
            domain=domain,
            horizon=horizon,
            split=split,
            limit=split_limits[split],
            selection_id=selection_id,
        )
        predictions = predict_batches(
            model,
            batches,
            split=split,
            xreg_mode=args.xreg_mode,
            ridge=float(args.ridge),
            max_rows_per_col=int(args.max_rows_per_col),
            force_xreg_cpu=bool(args.force_xreg_cpu),
            batch_size=int(args.batch_size),
        )
        if not predictions.empty:
            window_index = load_window_index(domain, horizon, split=split)
            validate_prediction_against_windows(predictions.drop(columns=["split"], errors="ignore"), window_index)
        pred_path = pred_dir / f"{split}_{CONFIG_ID}_predictions.parquet"
        predictions.to_parquet(pred_path, index=False)
        all_predictions.append(predictions)
        selected[split] = selected_summary
        summaries.append(
            {
                "split": split,
                "config_id": CONFIG_ID,
                "prediction_rows": int(len(predictions)),
                "prediction_path": str(pred_path),
                "elapsed_sec": round(time.time() - split_started, 3),
            }
        )

    cell_predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    metric_paths: dict[str, str] = {}
    metric_rows = 0
    if not cell_predictions.empty:
        metrics = evaluate_prediction_frame(cell_predictions)
        metric_paths = write_metrics(metrics, cell_dir / "metrics", stem="metrics")
        metric_rows = int(len(metrics))
    cell_manifest = {
        "status": "ok",
        "domain": domain,
        "horizon": horizon,
        "cell_index": int(cell_index),
        "seed": int(DEFAULT_SEED + cell_index),
        "model_id": MODEL_ID,
        "model_family": MODEL_FAMILY,
        "config_id": CONFIG_ID,
        "model_name": args.model_name,
        "checkpoint_is_full_power": False,
        "selection_id": selection_id,
        "selected": selected,
        "prediction_rows": int(len(cell_predictions)),
        "metric_rows": metric_rows,
        "metrics": metric_paths,
        "summaries": summaries,
        "elapsed_sec": round(time.time() - started, 3),
        "important_boundary": "Shared P2 development subset; TimesFM 2.5 200M XReg adaptation route; not the 500M full-power route.",
    }
    (cell_dir / "cell_manifest.json").write_text(dumps(cell_manifest) + "\n", encoding="utf-8")
    return all_predictions, cell_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", action="append", choices=list_domains(), help="Domain(s) to run. Defaults to all.")
    parser.add_argument("--horizon", action="append", choices=list(MAIN_HORIZONS), help="Horizon(s) to run. Defaults to all.")
    parser.add_argument("--eval-split", action="append", choices=["validation", "test"], help="Eval split(s). Defaults to validation and test.")
    parser.add_argument("--subset-manifest", type=Path, default=DEFAULT_SUBSET_MANIFEST)
    parser.add_argument("--max-validation-windows", type=int, default=64)
    parser.add_argument("--max-test-windows", type=int, default=64)
    parser.add_argument("--selection-id", default="timesfm2p5_xreg_shared_dev_v1")
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument("--max-horizon", type=int, default=256)
    parser.add_argument("--per-core-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accelerator", choices=["cuda"], default="cuda")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face download if weights are not cached.")
    parser.add_argument("--torch-compile", action="store_true", help="Use torch.compile during weight loading.")
    parser.add_argument("--xreg-mode", choices=["xreg + timesfm", "timesfm + xreg"], default="xreg + timesfm")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--max-rows-per-col", type=int, default=0)
    parser.add_argument(
        "--force-xreg-cpu",
        action="store_true",
        help="Forbidden by project policy; retained only to fail fast on stale commands.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default="timesfm2p5_xreg_shared_dev_seed20260514_eval64")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domains = args.domain or list_domains()
    horizons = args.horizon or list(MAIN_HORIZONS)
    run_dir = args.out_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    pred_dir = run_dir / "predictions"
    metric_dir = run_dir / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    torch.set_float32_matmul_precision("high")
    require_cuda_environment()
    torch.cuda.reset_peak_memory_stats()
    cuda_start = cuda_metadata()
    if args.force_xreg_cpu:
        raise ValueError("CPU XReg execution is forbidden for DL/TSFM tests in this project")
    manifest = load_subset_manifest(args.subset_manifest)
    model = load_timesfm2p5(
        args.model_name,
        allow_download=bool(args.allow_download),
        torch_compile=bool(args.torch_compile),
    )
    forecast_config = timesfm.ForecastConfig(
        max_context=args.max_context,
        max_horizon=args.max_horizon,
        normalize_inputs=True,
        per_core_batch_size=args.per_core_batch_size,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
        return_backcast=True,
    )
    model.compile(forecast_config)

    cell_manifests: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    cell_index = 0
    for domain in domains:
        for horizon in horizons:
            try:
                predictions, cell_manifest = run_cell(
                    args,
                    model=model,
                    domain=domain,
                    horizon=horizon,
                    cell_index=cell_index,
                    manifest=manifest,
                    run_dir=run_dir,
                )
                all_predictions.extend(predictions)
                cell_manifests.append(cell_manifest)
                print(dumps({"status": "cell_ok", "domain": domain, "horizon": horizon}))
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
                if not args.continue_on_error:
                    raise
            finally:
                cell_index += 1
                release_cuda()

    predictions_all = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    combined_path = pred_dir / "predictions_all.parquet"
    predictions_all.to_parquet(combined_path, index=False)
    metrics = evaluate_prediction_frame(predictions_all) if not predictions_all.empty else pd.DataFrame()
    metric_paths = write_metrics(metrics, metric_dir, stem="metrics") if not metrics.empty else {}

    manifest_out = {
        "status": "ok" if not failures else "partial",
        "model_id": MODEL_ID,
        "model_family": MODEL_FAMILY,
        "config_id": CONFIG_ID,
        "model_name": args.model_name,
        "run_dir": str(run_dir),
        "device": "cuda",
        "accelerator": args.accelerator,
        "args": {"accelerator": args.accelerator},
        "checkpoint_is_full_power": False,
        "route_role": "smaller_newer_xreg_adaptation_variant",
        "cuda_required": True,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_metadata_start": cuda_start,
        "cuda_metadata_end": cuda_metadata(),
        "allow_download": bool(args.allow_download),
        "torch_compile": bool(args.torch_compile),
        "domains": domains,
        "horizons": horizons,
        "eval_splits": requested_eval_splits(args),
        "max_validation_windows": int(args.max_validation_windows),
        "max_test_windows": int(args.max_test_windows),
        "selection_id": selection_seed_label(args),
        "subset_manifest": str(args.subset_manifest),
        "subset_id": manifest.get("subset_id"),
        "forecast_config": dataclasses.asdict(model.forecast_config),
        "batch_size": int(args.batch_size),
        "use_covariates": True,
        "covariate_policy": "known-future calendar only; no future measured exogenous covariates",
        "xreg_mode": args.xreg_mode,
        "ridge": float(args.ridge),
        "max_rows_per_col": int(args.max_rows_per_col),
        "force_xreg_cpu": bool(args.force_xreg_cpu),
        "cells_requested": int(len(domains) * len(horizons)),
        "cells_completed": int(len(cell_manifests)),
        "cells_failed": int(len(failures)),
        "prediction_rows": int(len(predictions_all)),
        "metric_rows": int(len(metrics)),
        "predictions_all": str(combined_path),
        "metrics": metric_paths,
        "cell_manifests": cell_manifests,
        "failures": failures,
        "elapsed_sec": round(time.time() - started, 3),
        "important_boundary": "Shared P2 development subset; TimesFM 2.5 200M XReg adaptation route; keep separate from TimesFM 2.0 500M full-power zero-shot.",
    }
    (run_dir / "manifest.json").write_text(dumps(manifest_out) + "\n", encoding="utf-8")
    print(dumps(manifest_out))


if __name__ == "__main__":
    main()
