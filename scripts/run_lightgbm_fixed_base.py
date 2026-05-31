#!/usr/bin/env python3
"""Formal fixed-base LightGBM runner for P1c energy-TSFM windows."""

from __future__ import annotations

import argparse
import pickle
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor

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
from energy_tsfm_p2_core import MAIN_HORIZONS, P1cWindowDataset, load_window_index, validate_prediction_against_windows
from evaluate_energy_tsfm_predictions import evaluate_prediction_frame, write_metrics
from run_itransformer_covariate_fixed_base import (
    audit_reproduced_metrics,
    load_shared_window_subset_record,
    validate_window_subset_record,
    window_subset_record,
)
from run_lightgbm_smoke import (
    MODEL_FAMILY,
    MODEL_ID,
    build_supervised_rows,
    prediction_rows_from_model,
    sample_positions,
)


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 20260514


def save_pickle_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(value, handle)
    tmp_path.replace(path)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def model_paths(paths: FormalRunPaths) -> dict[str, Path]:
    return {
        "best": paths.checkpoints_dir / "best.pkl",
        "last": paths.checkpoints_dir / "last.pkl",
        "previous": paths.checkpoints_dir / "previous.pkl",
    }


def fit_lightgbm_fixed(x_train: pd.DataFrame, y_train: pd.Series, args: argparse.Namespace) -> LGBMRegressor:
    model = LGBMRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        objective="regression",
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    return model


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
        record["model_id"] = MODEL_ID
        write_json_atomic(paths.window_subsets, record)
        return train_base, val_base, test_base, record

    train_positions = sample_positions(len(train_base), max_train_windows, f"{seed}:{domain}:{horizon}:formal:train")
    val_positions = sample_positions(len(val_base), max_val_windows, f"{seed}:{domain}:{horizon}:formal:validation")
    test_positions = sample_positions(len(test_base), max_test_windows, f"{seed}:{domain}:{horizon}:formal:test")
    if not train_positions or not val_positions or not test_positions:
        raise ValueError(f"{domain}/{horizon}: empty train/validation/test subset")
    record = {
        "domain_id": domain,
        "horizon": horizon,
        "model_id": MODEL_ID,
        "seed": int(seed),
        "sampling_rule": "sample_positions; 0 means full split; includes first/middle/last anchors",
        "train": window_subset_record(train_base, train_positions),
        "validation": window_subset_record(val_base, val_positions),
        "test": window_subset_record(test_base, test_positions),
    }
    write_json_atomic(paths.window_subsets, record)
    return train_base, val_base, test_base, record


def write_split_outputs(
    *,
    paths: FormalRunPaths,
    model: LGBMRegressor,
    val_dataset: P1cWindowDataset,
    test_dataset: P1cWindowDataset,
    subsets: dict[str, Any],
    config_id: str,
    seed: int,
    horizon: str,
    domain: str,
    output_dir: Path,
    eval_splits: tuple[str, ...],
) -> dict[str, Any]:
    full_index = load_window_index(domain, horizon)
    outputs: dict[str, Any] = {}
    for split, dataset, pred_path, metric_stem in [
        ("validation", val_dataset, output_dir / "validation_predictions.parquet", "validation_metrics"),
        ("test", test_dataset, output_dir / "test_predictions.parquet", "test_metrics"),
    ]:
        if split not in eval_splits:
            continue
        preds, feature_rows = prediction_rows_from_model(
            model,
            dataset,
            [int(pos) for pos in subsets[split]["positions"]],
            seed=seed,
            notes=(
                "LightGBM fixed-base run using P1c frozen windows and context-only tabular features; "
                f"config_id={config_id}"
            ),
        )
        preds["config_id"] = config_id
        validate_prediction_against_windows(preds, full_index)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(pred_path, index=False)
        metrics = evaluate_prediction_frame(preds)
        metric_dir = paths.metrics_dir if output_dir == paths.predictions_dir else output_dir
        metric_paths = write_metrics(metrics, metric_dir, stem=metric_stem)
        outputs[f"{split}_feature_rows"] = int(feature_rows)
        outputs[f"{split}_prediction_rows"] = int(len(preds))
        outputs[f"{split}_metric_rows"] = int(len(metrics))
        outputs[f"{split}_predictions"] = str(pred_path)
        outputs[f"{split}_metrics"] = metric_paths
    return outputs


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    paths = make_run_paths(
        output_root=args.output_root,
        model_id=MODEL_ID,
        domain_id=args.domain,
        horizon=args.horizon,
        config_id=args.config_id,
        seed=args.seed,
    )
    ensure_formal_run_dirs(paths, resume=args.resume)
    start = time.time()
    log_stdout(paths, f"[{MODEL_ID}] fixed-base run: {args.domain}/{args.horizon}")
    log_stdout(paths, f"run_dir={paths.run_dir}")
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
        train_positions = [int(pos) for pos in subsets["train"]["positions"]]
        x_train, y_train, _, _ = build_supervised_rows(train_base, train_positions)
        if x_train.empty:
            raise ValueError(f"{args.domain}/{args.horizon}: no training rows")

        config = {
            "runner": str(Path(__file__).resolve()),
            "model_family": MODEL_FAMILY,
            "model_id": MODEL_ID,
            "implementation": "lightgbm.LGBMRegressor",
            "domain_id": args.domain,
            "horizon": args.horizon,
            "config_id": args.config_id,
            "seed": int(args.seed),
                "window_subset_manifest": str(args.window_subset_manifest) if args.window_subset_manifest else None,
                "eval_splits": list(eval_splits),
                "data_contract": {
                "window_source": "P1c frozen window index",
                "horizons_allowed": list(MAIN_HORIZONS),
                "feature_boundary": "context-only measured features plus known calendar features",
                "split_rule": "forecast endpoint split from P1c",
            },
            "training_config": {
                "n_estimators": int(args.n_estimators),
                "learning_rate": float(args.learning_rate),
                "num_leaves": int(args.num_leaves),
                "max_depth": int(args.max_depth),
                "min_child_samples": int(args.min_child_samples),
                "subsample": float(args.subsample),
                "colsample_bytree": float(args.colsample_bytree),
                "reg_alpha": float(args.reg_alpha),
                "reg_lambda": float(args.reg_lambda),
                "n_jobs": int(args.n_jobs),
            },
            "n_features": int(x_train.shape[1]),
            "train_rows": int(len(x_train)),
            "package_versions": package_versions(),
            "args": vars(args),
        }
        write_json_atomic(paths.config, config)
        write_inference_entrypoint(paths, runner_script=Path(__file__).resolve())

        manifest = {
            "status": "running",
            "model_family": MODEL_FAMILY,
            "model_id": MODEL_ID,
            "domain_id": args.domain,
            "horizon": args.horizon,
            "config_id": args.config_id,
            "seed": int(args.seed),
            "run_dir": str(paths.run_dir),
            "p1c_window_source": str(PROJECT / "data" / "energy_tsfm_windows_p1c" / args.domain / f"window_index_{args.horizon}.parquet"),
            "window_subset_manifest": str(args.window_subset_manifest) if args.window_subset_manifest else None,
            "train_windows": int(subsets["train"]["count"]),
            "validation_windows": int(subsets["validation"]["count"]),
            "test_windows": int(subsets["test"]["count"]),
            "train_rows": int(len(x_train)),
            "n_features": int(x_train.shape[1]),
            "validation_selection_metric": "validation WAPE from common evaluator; one-shot tree fit has best=last",
            "started_at_unix": start,
            "package_versions": package_versions(),
            "args": vars(args),
        }
        write_json_atomic(paths.manifest, manifest)

        model = fit_lightgbm_fixed(x_train, y_train, args)
        checkpoints = model_paths(paths)
        if args.resume and checkpoints["last"].exists() and not checkpoints["previous"].exists():
            shutil.copy2(checkpoints["last"], checkpoints["previous"])
        save_pickle_atomic(checkpoints["best"], model)
        save_pickle_atomic(checkpoints["last"], model)

        output_info = write_split_outputs(
            paths=paths,
            model=model,
            val_dataset=val_base,
            test_dataset=test_base,
            subsets=subsets,
            config_id=args.config_id,
            seed=args.seed,
            horizon=args.horizon,
            domain=args.domain,
            output_dir=paths.predictions_dir,
            eval_splits=eval_splits,
        )
        val_metrics = pd.read_csv(paths.validation_metrics_csv)
        best_wape = float(val_metrics["wape"].iloc[0]) if "wape" in val_metrics else np.nan
        train_rows = [
            {
                "epoch": 0,
                "global_step": 0,
                "train_loss": np.nan,
                "validation_loss": np.nan,
                "validation_primary_wape": best_wape,
                "learning_rate": float(args.learning_rate),
                "is_best": True,
            }
        ]
        update_curve_artifacts(paths, train_rows, primary_metric_col="validation_primary_wape")
        manifest.update(
            {
                "status": "ok",
                "completed_at_unix": time.time(),
                "runtime_sec": float(time.time() - start),
                "best_validation_primary_wape": best_wape,
                "best_checkpoint": str(checkpoints["best"]),
                "last_checkpoint": str(checkpoints["last"]),
                "previous_checkpoint": str(checkpoints["previous"]) if checkpoints["previous"].exists() else None,
                "train_log_csv": str(paths.train_log_csv),
                "curves": {
                    "loss_curve_csv": str(paths.loss_curve_csv),
                    "loss_curve_png": str(paths.loss_curve_png),
                    "metric_curve_png": str(paths.metric_curve_png),
                },
                "outputs": output_info,
                "one_command_inference": str(paths.inference_script),
                "cuda_metadata_end": cuda_metadata(torch.device("cpu")),
                "evaluator_validation_status": "validate_prediction_against_windows passed before metrics",
                "interpretation": "fixed-base LightGBM scaffold; sampled development limits, if used, are not paper results",
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
    domain = str(config["domain_id"])
    horizon = str(config["horizon"])
    seed = int(config["seed"])
    eval_splits = tuple(args.eval_split or config.get("eval_splits") or ("validation", "test"))
    model = load_pickle(model_paths(paths)["best"])
    val_base = P1cWindowDataset(domain, horizon, split="validation")
    test_base = P1cWindowDataset(domain, horizon, split="test")
    inference_dir = paths.inference_dir
    output_info = write_split_outputs(
        paths=paths,
        model=model,
        val_dataset=val_base,
        test_dataset=test_base,
        subsets=subsets,
        config_id=str(config["config_id"]),
        seed=seed,
        horizon=horizon,
        domain=domain,
        output_dir=inference_dir,
        eval_splits=eval_splits,
    )
    audit = None
    if "test" in eval_splits:
        metric_frames = {
            "validation": pd.read_csv(inference_dir / "validation_metrics.csv"),
            "test": pd.read_csv(inference_dir / "test_metrics.csv"),
        }
        audit = audit_reproduced_metrics(
            original_validation_metrics_path=paths.validation_metrics_csv,
            reproduced_validation_metrics=metric_frames["validation"],
            original_test_metrics_path=paths.test_metrics_csv,
            reproduced_test_metrics=metric_frames["test"],
        )
    result = {
        "status": "ok",
        "run_dir": str(paths.run_dir),
        "checkpoint": str(model_paths(paths)["best"]),
        "domain_id": domain,
        "horizon": horizon,
        "model_id": MODEL_ID,
        **output_info,
        "reproducibility_audit": audit,
    }
    write_json_atomic(inference_dir / "inference_manifest.json", result)
    print(dumps_json(result))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="aluminum_load")
    parser.add_argument("--horizon", choices=list(MAIN_HORIZONS), default="4h")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--config-id", default="fixed_base_lightgbm_v0")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--max-test-windows", type=int, default=0)
    parser.add_argument("--window-subset-manifest", type=Path, default=None)
    parser.add_argument("--eval-split", action="append", choices=["validation", "test"])
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=0.0)
    parser.add_argument("--n-jobs", type=int, default=8)
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
