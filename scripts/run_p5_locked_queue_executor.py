#!/usr/bin/env python3
"""Execute the locked P5 test-once queue with per-row audit logs.

This script is intentionally conservative:

- it requires the explicit P5 approval token;
- it re-runs the guarded P5 dry-run gate for each locked row immediately before
  execution;
- it executes only the audited underlying command emitted by that gate;
- it writes one durable status JSON and one combined stdout/stderr log per row;
- it skips rows that already completed successfully unless explicitly told to
  rerun them.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK_CSV = (
    PROJECT
    / "data/energy_tsfm_tuning"
    / "p3_4du_chronos2_train4096_application_lock_v0_codex_20260519"
    / "p3_4du_p5_pre_executable_lock_after_chronos4096_application.csv"
)
DEFAULT_OUT_ROOT = (
    PROJECT
    / "results/energy_tsfm_p5_main"
    / "p5_main_test_once_v0_codex_20260517_queue_codex_20260518"
)
APPROVAL_TOKEN = "P5_TEST_ONCE_APPROVED_20260517"
NON_LIGHTGBM_MODELS = {"chronos2", "itransformer", "nbeatsx", "timesfm2p5"}
MODEL_ORDER = {
    "chronos2": 0,
    "timesfm2p5": 1,
    "itransformer": 2,
    "nbeatsx": 3,
    "lightgbm": 4,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def resolve_project_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT / path
    return path


def split_csv(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(value) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-csv", default=rel(DEFAULT_LOCK_CSV))
    parser.add_argument("--out-root", default=rel(DEFAULT_OUT_ROOT))
    parser.add_argument("--approval-token", required=True)
    parser.add_argument("--models", default="", help="Comma-separated model_id filter.")
    parser.add_argument("--domains", default="", help="Comma-separated domain_id filter.")
    parser.add_argument("--horizons", default="", help="Comma-separated horizon filter.")
    parser.add_argument("--lock-ids", default="", help="Comma-separated lock_id filter.")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--order",
        choices=["tsfm-first", "locked", "lightgbm-first"],
        default="tsfm-first",
    )
    parser.add_argument("--gate-timeout-seconds", type=int, default=120)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--cpu-threads", type=int, default=36)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser


def filter_queue(lock_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = lock_df.copy()
    for column, value in (
        ("model_id", args.models),
        ("domain_id", args.domains),
        ("horizon", args.horizons),
        ("lock_id", args.lock_ids),
    ):
        wanted = split_csv(value)
        if wanted:
            df = df[df[column].astype(str).isin(wanted)]
    if args.order == "tsfm-first":
        df = df.assign(_model_order=df["model_id"].astype(str).map(MODEL_ORDER).fillna(99))
        df = df.sort_values(["_model_order", "domain_id", "horizon", "lock_id"])
    elif args.order == "lightgbm-first":
        reverse_order = {
            model: (0 if model == "lightgbm" else MODEL_ORDER.get(model, 99) + 1)
            for model in MODEL_ORDER
        }
        df = df.assign(_model_order=df["model_id"].astype(str).map(reverse_order).fillna(99))
        df = df.sort_values(["_model_order", "domain_id", "horizon", "lock_id"])
    else:
        df = df.reset_index(names="_locked_order").sort_values("_locked_order")
    df = df.drop(columns=[column for column in ["_model_order", "_locked_order"] if column in df.columns])
    if args.max_rows is not None:
        df = df.head(args.max_rows)
    return df.reset_index(drop=True)


def row_paths(out_root: Path, lock_id: str) -> dict[str, Path]:
    row_dir = out_root / "rows" / lock_id
    return {
        "row_dir": row_dir,
        "gate_manifest": row_dir / "gate_manifest.json",
        "execution_plan": row_dir / "execution_plan.json",
        "status": row_dir / "status.json",
        "log": row_dir / "run.log",
    }


def build_gate_command(row: dict[str, Any], paths: dict[str, Path]) -> list[str]:
    argv = shlex.split(str(row["p5_command_template"]))
    argv.extend(
        [
            "--dry-run",
            "--dry-run-output",
            rel(paths["gate_manifest"]),
            "--execution-plan-output",
            rel(paths["execution_plan"]),
        ]
    )
    return argv


def base_env(args: argparse.Namespace, model_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = str(args.cpu_threads)
    env["OPENBLAS_NUM_THREADS"] = str(args.cpu_threads)
    env["MKL_NUM_THREADS"] = str(args.cpu_threads)
    if model_id != "lightgbm":
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    return env


def validate_plan(
    row: dict[str, Any],
    gate_returncode: int,
    gate_manifest: dict[str, Any],
    plan: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    model_id = str(row["model_id"])
    command_shell = str(plan.get("underlying_command_shell", ""))
    if gate_returncode != 0:
        errors.append(f"gate dry-run returncode={gate_returncode}")
    if gate_manifest.get("status") != "ok":
        errors.append(f"gate manifest status={gate_manifest.get('status')!r}")
    if not plan.get("executable_now"):
        errors.append(f"execution plan not executable: {plan.get('execution_state')}")
    if "--eval-split test" not in command_shell:
        errors.append("underlying command lacks --eval-split test")
    if "--eval-split validation" not in command_shell:
        errors.append("underlying command lacks --eval-split validation")
    if "--max-train-windows 0" not in command_shell:
        errors.append("underlying command is not full-train-window")
    if model_id == "lightgbm":
        if "--n-jobs 36" not in command_shell:
            errors.append("LightGBM underlying command lacks --n-jobs 36")
    elif model_id in NON_LIGHTGBM_MODELS:
        if (
            "--accelerator cuda" not in command_shell
            and "--accelerator gpu" not in command_shell
            and "--device-map cuda" not in command_shell
        ):
            errors.append("non-LightGBM underlying command does not force CUDA")
    return errors


def run_gate(
    row: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> tuple[int, str, str, dict[str, Any], dict[str, Any]]:
    argv = build_gate_command(row, paths)
    completed = subprocess.run(
        argv,
        cwd=PROJECT,
        env=base_env(args, str(row["model_id"])),
        text=True,
        capture_output=True,
        timeout=args.gate_timeout_seconds,
        check=False,
    )
    gate_manifest: dict[str, Any] = {}
    plan: dict[str, Any] = {}
    if paths["gate_manifest"].exists():
        gate_manifest = read_json(paths["gate_manifest"])
    if paths["execution_plan"].exists():
        plan = read_json(paths["execution_plan"])
    return completed.returncode, completed.stdout, completed.stderr, gate_manifest, plan


def stream_command(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n===== started_at_utc={utc_now()} =====\n")
        handle.write("command: " + shlex.join(command) + "\n\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
        returncode = process.wait()
        handle.write(f"\n===== finished_at_utc={utc_now()} returncode={returncode} =====\n")
        handle.flush()
    return int(returncode)


def write_queue_summary(out_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for status_path in sorted((out_root / "rows").glob("*/status.json")):
        try:
            rows.append(read_json(status_path))
        except Exception:
            continue
    if not rows:
        return
    summary_csv = out_root / "queue_status.csv"
    pd.DataFrame(rows).sort_values(["queue_index", "lock_id"]).to_csv(summary_csv, index=False)
    manifest = {
        "status": "ok"
        if all(row.get("status") in {"ok", "skipped_ok", "planned"} for row in rows)
        else "running_or_failed",
        "updated_at_utc": utc_now(),
        "rows_recorded": len(rows),
        "counts": pd.Series([row.get("status", "") for row in rows]).value_counts().to_dict(),
        "queue_status_csv": rel(summary_csv),
    }
    write_json(out_root / "queue_manifest.json", manifest)


def already_completed(status_path: Path) -> bool:
    if not status_path.exists():
        return False
    try:
        return read_json(status_path).get("status") == "ok"
    except Exception:
        return False


def execute_row(
    row: dict[str, Any],
    queue_index: int,
    args: argparse.Namespace,
    out_root: Path,
) -> dict[str, Any]:
    lock_id = str(row["lock_id"])
    model_id = str(row["model_id"])
    paths = row_paths(out_root, lock_id)
    paths["row_dir"].mkdir(parents=True, exist_ok=True)
    status_base = {
        "lock_id": lock_id,
        "queue_index": queue_index,
        "model_id": model_id,
        "route_id": str(row["route_id"]),
        "domain_id": str(row["domain_id"]),
        "horizon": str(row["horizon"]),
        "status_path": rel(paths["status"]),
        "log_path": rel(paths["log"]),
        "gate_manifest": rel(paths["gate_manifest"]),
        "execution_plan": rel(paths["execution_plan"]),
    }
    if args.resume and already_completed(paths["status"]):
        prior = read_json(paths["status"])
        return {**prior, "resume_action": "skipped_ok", "updated_at_utc": utc_now()}
    if paths["status"].exists() and not args.rerun_failed and not args.dry_run:
        prior = read_json(paths["status"])
        if prior.get("status") == "failed":
            return {**prior, "resume_action": "skipped_failed", "updated_at_utc": utc_now()}

    started = utc_now()
    write_json(paths["status"], {**status_base, "status": "gating", "started_at_utc": started})
    gate_returncode, gate_stdout, gate_stderr, gate_manifest, plan = run_gate(row, paths, args)
    gate_errors = validate_plan(row, gate_returncode, gate_manifest, plan)
    if gate_errors:
        status = {
            **status_base,
            "status": "failed",
            "stage": "gate",
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "returncode": gate_returncode,
            "gate_errors": gate_errors,
            "stdout_tail": gate_stdout[-2000:],
            "stderr_tail": gate_stderr[-2000:],
        }
        write_json(paths["status"], status)
        return status

    command = list(plan.get("underlying_command") or [])
    if args.dry_run:
        status = {
            **status_base,
            "status": "planned",
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "underlying_command_shell": plan.get("underlying_command_shell", ""),
            "output_root": plan.get("output_root", ""),
        }
        write_json(paths["status"], status)
        return status

    write_json(
        paths["status"],
        {
            **status_base,
            "status": "running",
            "started_at_utc": started,
            "underlying_command_shell": plan.get("underlying_command_shell", ""),
            "output_root": plan.get("output_root", ""),
        },
    )
    t0 = time.monotonic()
    returncode = stream_command(command, paths["log"], base_env(args, model_id))
    elapsed = round(time.monotonic() - t0, 3)
    status = {
        **status_base,
        "status": "ok" if returncode == 0 else "failed",
        "stage": "execute",
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "returncode": returncode,
        "underlying_command_shell": plan.get("underlying_command_shell", ""),
        "output_root": plan.get("output_root", ""),
        "test_data_read": True,
        "test_predictions_generated": returncode == 0,
        "model_execution_started": True,
        "training_or_finetuning_started": True,
    }
    write_json(paths["status"], status)
    return status


def main() -> int:
    args = build_parser().parse_args()
    if args.approval_token != APPROVAL_TOKEN:
        print("P5 queue execution requires the explicit approval token.", file=sys.stderr)
        return 2
    lock_csv = resolve_project_path(args.lock_csv)
    out_root = resolve_project_path(args.out_root)
    if lock_csv is None or not lock_csv.exists():
        print(f"lock csv not found: {args.lock_csv}", file=sys.stderr)
        return 2
    if out_root is None:
        print("out-root is required", file=sys.stderr)
        return 2
    out_root.mkdir(parents=True, exist_ok=True)

    lock_df = pd.read_csv(lock_csv)
    queue_df = filter_queue(lock_df, args)
    queue_csv = out_root / "queue_selected.csv"
    queue_df.to_csv(queue_csv, index=False)
    write_json(
        out_root / "queue_config.json",
        {
            "created_at_utc": utc_now(),
            "lock_csv": rel(lock_csv),
            "out_root": rel(out_root),
            "queue_selected_csv": rel(queue_csv),
            "selected_rows": int(len(queue_df)),
            "dry_run": bool(args.dry_run),
            "order": args.order,
            "models": args.models,
            "domains": args.domains,
            "horizons": args.horizons,
            "lock_ids": args.lock_ids,
            "max_rows": args.max_rows,
            "resource_contract": {
                "lightgbm_total_concurrent_n_jobs_budget": 36,
                "cpu_threads_env": args.cpu_threads,
                "non_lightgbm_cuda_visible_devices": args.cuda_visible_devices,
            },
        },
    )

    failures = 0
    for queue_index, row in enumerate(queue_df.to_dict(orient="records"), start=1):
        status = execute_row(row, queue_index, args, out_root)
        print(dumps(status), flush=True)
        write_queue_summary(out_root)
        if status.get("status") == "failed":
            failures += 1
            if not args.continue_on_failure:
                break
    write_queue_summary(out_root)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
