#!/usr/bin/env python3
"""Guarded P5 test-once entrypoint.

The dry-run path validates the locked P5 command contract without reading test
windows, training models, or writing predictions. The real execution path is
guarded by an explicit approval token and should be driven by the audited P5
queue executor so every locked unit has a durable log and status record.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
VALID_DOMAINS = {
    "aidc_power_optional",
    "aluminum_load",
    "arena_pv",
    "microgrid_load",
    "provincial_load",
}
VALID_HORIZONS = {"4h", "24h"}
NON_LIGHTGBM_MODELS = {"nbeatsx", "itransformer", "chronos2", "timesfm2p5"}
VALID_MODELS = {"lightgbm"} | NON_LIGHTGBM_MODELS
CHRONOS_HIDDEN_ROUTE = "chronos2_covariate_aware_hidden_adapter_p3_4bs"
TIMESFM_XREG_ROUTE = "timesfm2p5_xreg_cellwise_parity_ablation_p3_4bp"
TIMESFM_LORA_ROUTE = "timesfm2p5_transformers_lora_target_only_p3_4bt"
AH_PAT_PYTHON = sys.executable
TIMESFM_LORA_PYTHON = sys.executable  # adjust if using separate venv for TimesFM LoRA
P5_MAIN_ROOT = PROJECT / "results" / "energy_tsfm_p5_main" / "p5_main_test_once_v0_codex_20260517"
APPROVAL_TOKEN = "P5_TEST_ONCE_APPROVED_20260517"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def resolve_project_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT / path
    return path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def append_option(argv: list[str], option: str, value: Any) -> None:
    if value is None:
        return
    argv.extend([option, str(value)])


def append_bool_flag(argv: list[str], option: str, value: Any) -> None:
    if value is True:
        argv.append(option)
    elif value is False:
        argv.append("--no-" + option.removeprefix("--"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config_source(args: argparse.Namespace) -> dict[str, Any]:
    path = resolve_project_path(args.config_source)
    if path is None:
        return {}
    return load_json(path)


def p5_output_root(run_id: str) -> str:
    return str(P5_MAIN_ROOT / run_id / "runner_outputs")


def common_formal_args(args: argparse.Namespace, *, config: dict[str, Any], script: str) -> list[str]:
    seed = int(config.get("seed", 20260517))
    return [
        AH_PAT_PYTHON,
        script,
        "--domain",
        args.domain,
        "--horizon",
        args.horizon,
        "--config-id",
        args.run_id,
        "--window-subset-manifest",
        args.subset_manifest,
        "--eval-split",
        "validation",
        "--eval-split",
        "test",
        "--max-train-windows",
        "0",
        "--max-val-windows",
        "0",
        "--max-test-windows",
        "0",
        "--seed",
        str(seed),
        "--output-root",
        p5_output_root(args.run_id),
        "--no-resume",
    ]


def lightgbm_command(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    params = config.get("best_params") or {}
    argv = common_formal_args(args, config=config, script="scripts/run_lightgbm_fixed_base.py")
    mapping = {
        "n_estimators": "--n-estimators",
        "learning_rate": "--learning-rate",
        "num_leaves": "--num-leaves",
        "max_depth": "--max-depth",
        "min_child_samples": "--min-child-samples",
        "subsample": "--subsample",
        "colsample_bytree": "--colsample-bytree",
        "reg_alpha": "--reg-alpha",
        "reg_lambda": "--reg-lambda",
    }
    for key, option in mapping.items():
        append_option(argv, option, params.get(key))
    append_option(argv, "--n-jobs", 36)
    return argv


def nbeatsx_command(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    params = config.get("best_params") or {}
    argv = common_formal_args(args, config=config, script="scripts/run_neuralforecast_dl_fixed_base.py")
    argv.extend(["--model", "nbeatsx", "--accelerator", "cuda", "--devices", "1"])
    mapping = {
        "max_steps": "--max-steps",
        "val_check_steps": "--val-check-steps",
        "early_stop_patience_steps": "--early-stop-patience-steps",
        "batch_size": "--batch-size",
        "valid_batch_size": "--valid-batch-size",
        "windows_batch_size": "--windows-batch-size",
        "inference_windows_batch_size": "--inference-windows-batch-size",
        "learning_rate": "--learning-rate",
        "weight_decay": "--weight-decay",
        "min_lr": "--min-lr",
        "huber_delta": "--huber-delta",
        "dropout": "--dropout",
        "scaler_type": "--scaler-type",
        "nbeatsx_blocks": "--nbeatsx-blocks",
        "nbeatsx_width": "--nbeatsx-width",
    }
    for key, option in mapping.items():
        append_option(argv, option, params.get(key))
    append_option(argv, "--preprocess-workers", 36)
    append_option(argv, "--preprocess-chunk-size", 0)
    return argv


def itransformer_command(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    params = config.get("best_params") or {}
    argv = common_formal_args(args, config=config, script="scripts/run_itransformer_covariate_fixed_base.py")
    argv.extend(["--accelerator", "cuda", "--devices", "1"])
    mapping = {
        "batch_size": "--batch-size",
        "eval_batch_size": "--eval-batch-size",
        "num_workers": "--num-workers",
        "prefetch_factor": "--prefetch-factor",
        "epochs": "--epochs",
        "patience": "--patience",
        "learning_rate": "--learning-rate",
        "weight_decay": "--weight-decay",
        "gradient_clip": "--gradient-clip",
        "lr_patience": "--lr-patience",
        "d_model": "--d-model",
        "n_heads": "--n-heads",
        "e_layers": "--e-layers",
        "d_ff": "--d-ff",
        "dropout": "--dropout",
        "max_hist_exog": "--max-hist-exog",
        "min_hist_exog_coverage": "--min-hist-exog-coverage",
    }
    for key, option in mapping.items():
        append_option(argv, option, params.get(key))
    append_bool_flag(argv, "--pin-memory", params.get("pin_memory"))
    append_bool_flag(argv, "--persistent-workers", params.get("persistent_workers"))
    return argv


def chronos_plan_command(args: argparse.Namespace) -> list[str]:
    argv = [
        AH_PAT_PYTHON,
        "scripts/run_p3_4bs_chronos2_covariate_adapter_validation.py",
        "--domain",
        args.domain,
        "--horizon",
        args.horizon,
        "--subset-manifest",
        args.subset_manifest,
        "--max-train-windows",
        "0",
        "--max-validation-windows",
        "0",
        "--max-test-windows",
        "0",
        "--eval-split",
        "validation",
        "--eval-split",
        "test",
        "--epochs",
        str(args.epochs or 5),
        "--batch-size",
        str(args.batch_size or 64),
        "--eval-batch-size",
        str(args.eval_batch_size or 64),
    ]
    append_option(argv, "--adapter-bottleneck", args.adapter_bottleneck)
    append_option(argv, "--adapter-dropout", args.adapter_dropout)
    append_bool_flag(argv, "--adapter-position-bias", args.adapter_position_bias)
    append_bool_flag(argv, "--adapter-future-patch-condition", args.adapter_future_patch_condition)
    append_option(argv, "--adapter-loss", args.adapter_loss)
    append_option(argv, "--learning-rate", args.learning_rate)
    append_option(argv, "--weight-decay", args.weight_decay)
    append_option(argv, "--gradient-clip", args.gradient_clip)
    append_option(argv, "--preprocess-workers", args.preprocess_workers)
    append_bool_flag(argv, "--array-preprocess", args.array_preprocess)
    append_bool_flag(argv, "--zero-init-adapter", args.zero_init_adapter)
    append_option(argv, "--condition-standardization", args.condition_standardization)
    append_bool_flag(argv, "--extended-calendar-condition", args.extended_calendar_condition)
    argv.extend(
        [
            "--condition-feature-mode",
            args.condition_feature_mode,
            "--device-map",
            "cuda",
            "--out-root",
            str(P5_MAIN_ROOT / args.run_id / "runner_outputs"),
            "--run-id",
            args.run_id,
        ]
    )
    return argv


def timesfm_xreg_plan_command(args: argparse.Namespace) -> list[str]:
    return [
        AH_PAT_PYTHON,
        "scripts/run_p3_4bp_timesfm2p5_parity_ablation_validation.py",
        "--domain",
        args.domain,
        "--horizon",
        args.horizon,
        "--variant",
        args.variant,
        "--subset-manifest",
        args.subset_manifest,
        "--max-train-windows",
        "0",
        "--max-validation-windows",
        "0",
        "--max-test-windows",
        "0",
        "--eval-split",
        "validation",
        "--eval-split",
        "test",
        "--accelerator",
        "cuda",
        "--batch-size",
        "64",
        "--per-core-batch-size",
        "64",
        "--out-root",
        str(P5_MAIN_ROOT / args.run_id / "runner_outputs"),
        "--run-id",
        args.run_id,
    ]


def timesfm_lora_plan_command(args: argparse.Namespace) -> list[str]:
    argv = [
        TIMESFM_LORA_PYTHON,
        "scripts/run_p3_4bt_timesfm2p5_lora_validation.py",
        "--domain",
        args.domain,
        "--horizon",
        args.horizon,
        "--subset-manifest",
        args.subset_manifest,
        "--max-train-windows",
        "0",
        "--max-validation-windows",
        "0",
        "--max-test-windows",
        "0",
        "--eval-split",
        "validation",
        "--eval-split",
        "test",
        "--accelerator",
        "cuda",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
    ]
    append_option(argv, "--lora-rank", args.lora_rank)
    append_option(argv, "--lora-alpha", args.lora_alpha)
    append_option(argv, "--lora-dropout", args.lora_dropout)
    append_option(argv, "--learning-rate", args.learning_rate)
    argv.extend(
        [
            "--out-root",
            str(P5_MAIN_ROOT / args.run_id / "runner_outputs"),
            "--run-id",
            args.run_id,
        ]
    )
    return argv


def build_execution_plan(args: argparse.Namespace, validation: dict[str, Any]) -> dict[str, Any]:
    if validation["errors"]:
        return {
            "status": "not_built",
            "execution_state": "invalid_args",
            "underlying_command": [],
            "underlying_command_shell": "",
            "notes": ["argument validation failed"],
        }

    notes: list[str] = []
    executable_now = True
    execution_state = "ready_for_real_test_once_if_explicitly_approved"
    config = load_config_source(args) if args.config_source else {}
    if config and config.get("test_predictions_generated") is not False:
        notes.append("config-source does not explicitly record test_predictions_generated=false")
        executable_now = False

    if args.model == "lightgbm":
        command = lightgbm_command(args, config)
        route_executor = "scripts/run_lightgbm_fixed_base.py"
    elif args.model == "nbeatsx":
        command = nbeatsx_command(args, config)
        route_executor = "scripts/run_neuralforecast_dl_fixed_base.py"
    elif args.model == "itransformer":
        command = itransformer_command(args, config)
        route_executor = "scripts/run_itransformer_covariate_fixed_base.py"
    elif args.model == "chronos2":
        command = chronos_plan_command(args)
        route_executor = "scripts/run_p3_4bs_chronos2_covariate_adapter_validation.py"
    elif args.model == "timesfm2p5" and args.route == TIMESFM_XREG_ROUTE:
        command = timesfm_xreg_plan_command(args)
        route_executor = "scripts/run_p3_4bp_timesfm2p5_parity_ablation_validation.py"
    elif args.model == "timesfm2p5" and args.route == TIMESFM_LORA_ROUTE:
        command = timesfm_lora_plan_command(args)
        route_executor = "scripts/run_p3_4bt_timesfm2p5_lora_validation.py"
    else:
        command = []
        route_executor = ""
        executable_now = False
        execution_state = "unsupported_route"
        notes.append("unsupported route")

    if executable_now and args.model in NON_LIGHTGBM_MODELS:
        joined = " ".join(command)
        if "--accelerator cuda" not in joined and "--accelerator gpu" not in joined and "--device-map cuda" not in joined:
            executable_now = False
            execution_state = "resource_contract_violation"
            notes.append("non-LightGBM underlying command does not force CUDA")
    if executable_now and args.model == "lightgbm" and "--n-jobs 36" not in " ".join(command):
        executable_now = False
        execution_state = "resource_contract_violation"
        notes.append("LightGBM underlying command does not force --n-jobs 36")

    return {
        "status": "ok",
        "execution_state": execution_state,
        "executable_now": executable_now,
        "route_executor": route_executor,
        "underlying_command": command,
        "underlying_command_shell": shlex.join(command),
        "output_root": p5_output_root(args.run_id) if command else "",
        "reads_test_if_executed": bool(executable_now),
        "writes_test_predictions_if_executed": bool(executable_now),
        "requires_explicit_approval_token": APPROVAL_TOKEN,
        "notes": notes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(VALID_MODELS))
    parser.add_argument("--route", default="")
    parser.add_argument("--domain", required=True, choices=sorted(VALID_DOMAINS))
    parser.add_argument("--horizon", required=True, choices=sorted(VALID_HORIZONS))
    parser.add_argument("--config-source", default="")
    parser.add_argument("--subset-manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--full-train-refit", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--lightgbm-n-jobs", type=int)
    parser.add_argument("--condition-feature-mode", default="")
    parser.add_argument("--variant", default="")
    parser.add_argument("--epochs", type=positive_int)
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--eval-batch-size", type=positive_int)
    parser.add_argument("--lora-rank", type=positive_int)
    parser.add_argument("--lora-alpha", type=positive_int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--adapter-bottleneck", type=positive_int)
    parser.add_argument("--adapter-dropout", type=float)
    parser.add_argument("--adapter-position-bias", action=argparse.BooleanOptionalAction)
    parser.add_argument("--adapter-future-patch-condition", action=argparse.BooleanOptionalAction)
    parser.add_argument("--adapter-loss", choices=["chronos_internal", "mae_q50", "huber_q50"])
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--gradient-clip", type=float)
    parser.add_argument("--preprocess-workers", type=positive_int)
    parser.add_argument("--array-preprocess", action=argparse.BooleanOptionalAction)
    parser.add_argument("--zero-init-adapter", action=argparse.BooleanOptionalAction)
    parser.add_argument("--condition-standardization", choices=["train_zscore", "none"])
    parser.add_argument("--extended-calendar-condition", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-output", default="")
    parser.add_argument("--execution-plan-output", default="")
    parser.add_argument(
        "--execute-test-once",
        action="store_true",
        help="Run the guarded real P5 command after explicit approval.",
    )
    parser.add_argument("--approval-token", default="")
    return parser


def validate_args(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if args.dry_run and args.execute_test_once:
        errors.append("--dry-run and --execute-test-once are mutually exclusive")
    if not args.dry_run and not args.execute_test_once:
        errors.append("use --dry-run for contract checks or --execute-test-once with explicit approval")
    if args.execute_test_once and args.approval_token != APPROVAL_TOKEN:
        errors.append("real P5 test-once execution requires the explicit approval token")
    if args.split != "test":
        errors.append(f"P5 locked commands must use --split test, got {args.split!r}")
    if not args.full_train_refit:
        errors.append("P5 locked commands must include --full-train-refit")
    if not args.run_id.startswith("p5_"):
        errors.append(f"--run-id must be a P5 run id, got {args.run_id!r}")

    subset_manifest = resolve_project_path(args.subset_manifest)
    if subset_manifest is None or not subset_manifest.exists():
        errors.append(f"--subset-manifest does not exist: {args.subset_manifest}")

    config_source = resolve_project_path(args.config_source)
    if args.config_source and (config_source is None or not config_source.exists()):
        errors.append(f"--config-source does not exist: {args.config_source}")

    route = str(args.route or "")
    if args.model == "lightgbm":
        if args.lightgbm_n_jobs != 36:
            errors.append("LightGBM P5 single-run command must use --lightgbm-n-jobs 36")
        if route not in {"", "lightgbm"}:
            errors.append(f"lightgbm route must be empty or 'lightgbm', got {route!r}")
        if not args.config_source:
            errors.append("lightgbm P5 command must include --config-source")
        if args.device and args.device.lower() != "cpu":
            warnings.append("LightGBM ignores --device; CPU is the only allowed exception")
    else:
        if str(args.device).lower() != "cuda":
            errors.append(f"{args.model} P5 command must include --device cuda")
        if args.lightgbm_n_jobs is not None:
            errors.append("non-LightGBM P5 commands must not set --lightgbm-n-jobs")

    if args.model in {"nbeatsx", "itransformer"}:
        if route not in {"", args.model}:
            errors.append(f"{args.model} route must be empty or {args.model!r}, got {route!r}")
        if not args.config_source:
            errors.append(f"{args.model} P5 command must include --config-source")
    elif args.model == "chronos2":
        if route != CHRONOS_HIDDEN_ROUTE:
            errors.append(f"chronos2 P5 route must be {CHRONOS_HIDDEN_ROUTE!r}")
        if not args.condition_feature_mode:
            errors.append("chronos2 hidden-adapter command must include --condition-feature-mode")
        if args.config_source:
            warnings.append("chronos2 hidden-adapter dry-run does not consume --config-source")
    elif args.model == "timesfm2p5":
        if route == TIMESFM_XREG_ROUTE:
            if not args.variant:
                errors.append("TimesFM 2.5 XReg command must include --variant")
            for name in (
                "epochs",
                "batch_size",
                "eval_batch_size",
                "lora_rank",
                "lora_alpha",
                "lora_dropout",
                "adapter_bottleneck",
                "adapter_dropout",
                "learning_rate",
                "weight_decay",
                "gradient_clip",
            ):
                if getattr(args, name) is not None:
                    errors.append(f"TimesFM 2.5 XReg command must not set --{name.replace('_', '-')}")
        elif route == TIMESFM_LORA_ROUTE:
            missing = [
                option
                for option, value in [
                    ("--epochs", args.epochs),
                    ("--batch-size", args.batch_size),
                    ("--eval-batch-size", args.eval_batch_size),
                ]
                if value is None
            ]
            if missing:
                errors.append("TimesFM 2.5 LoRA command missing " + ", ".join(missing))
            if args.variant:
                errors.append("TimesFM 2.5 LoRA command must not set --variant")
        else:
            errors.append(
                "timesfm2p5 P5 route must be one of "
                f"{TIMESFM_XREG_ROUTE!r} or {TIMESFM_LORA_ROUTE!r}"
            )

    return {
        "errors": errors,
        "warnings": warnings,
        "resolved_subset_manifest": str(subset_manifest) if subset_manifest else None,
        "resolved_config_source": str(config_source) if config_source else None,
    }


def manifest_from_args(
    args: argparse.Namespace,
    validation: dict[str, Any],
    execution_plan: dict[str, Any],
    execution_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed_args = vars(args).copy()
    dry_run_output = resolve_project_path(args.dry_run_output)
    real_execution_started = bool(args.execute_test_once and execution_result is not None)
    real_execution_succeeded = bool(
        real_execution_started and execution_result.get("status") == "ok"
    )
    return {
        "status": "ok" if not validation["errors"] else "failed",
        "created_at_utc": utc_now(),
        "dry_run": bool(args.dry_run),
        "test_data_read": real_execution_started,
        "test_predictions_generated": real_execution_succeeded,
        "model_execution_started": real_execution_started,
        "training_or_finetuning_started": real_execution_started,
        "execution_plan": execution_plan,
        "execution_result": execution_result,
        "project_root": str(PROJECT),
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "argv": sys.argv,
        "parsed_args": parsed_args,
        "resolved_paths": {
            "subset_manifest": validation["resolved_subset_manifest"],
            "config_source": validation["resolved_config_source"],
            "dry_run_output": str(dry_run_output) if dry_run_output else None,
        },
        "resource_contract": {
            "lightgbm_total_concurrent_n_jobs_budget": 36,
            "non_lightgbm_cuda_required": True,
            "validated_device": args.device or ("cpu" if args.model == "lightgbm" else ""),
        },
        "test_boundary": {
            "split_argument": args.split,
            "test_window_manifest_was_read": real_execution_started,
            "test_prediction_file_was_written": real_execution_succeeded,
            "real_test_execution_is_guarded": True,
            "real_test_execution_attempted": real_execution_started,
        },
        "validation": {
            "errors": validation["errors"],
            "warnings": validation["warnings"],
        },
    }


def write_manifest(path_value: str, manifest: dict[str, Any]) -> None:
    path = resolve_project_path(path_value)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(manifest) + "\n", encoding="utf-8")


def run_underlying_command(execution_plan: dict[str, Any]) -> dict[str, Any]:
    if not execution_plan.get("executable_now"):
        return {
            "status": "refused",
            "returncode": None,
            "reason": execution_plan.get("execution_state"),
        }
    command = list(execution_plan.get("underlying_command") or [])
    completed = subprocess.run(
        command,
        cwd=PROJECT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validation = validate_args(args)
    execution_plan = build_execution_plan(args, validation)
    if args.execution_plan_output:
        write_manifest(args.execution_plan_output, execution_plan)
    execution_result = None
    if args.execute_test_once and not validation["errors"]:
        if not execution_plan.get("executable_now"):
            validation["errors"].append(f"execution refused: {execution_plan.get('execution_state')}")
        else:
            execution_result = run_underlying_command(execution_plan)
            if execution_result.get("status") != "ok":
                validation["errors"].append("underlying P5 command failed")
    manifest = manifest_from_args(args, validation, execution_plan, execution_result)
    if args.dry_run_output:
        write_manifest(args.dry_run_output, manifest)
    if validation["errors"]:
        print(dumps(manifest), file=sys.stderr)
        return 2
    print(dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
