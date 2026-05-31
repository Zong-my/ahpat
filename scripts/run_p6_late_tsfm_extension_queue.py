#!/usr/bin/env python3
"""Queue driver for the P6 late TSFM route extension.

The driver executes cell-level jobs sequentially on one CUDA device, writes
per-row status files, and can run the guarded test-once phase only after the
corresponding optimize lock exists.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
PLAN_ID = "p6_late_tsfm_route_extension_v0_codex_20260528"
SCRIPT = PROJECT / "scripts" / "run_p6_late_tsfm_route_extension.py"
DEFAULT_OUT_ROOT = PROJECT / "results" / "energy_tsfm_late_extension" / PLAN_ID
DEFAULT_SUBSET_MANIFEST = (
    PROJECT
    / "data"
    / "energy_tsfm_formal_windows"
    / "p3_target_pure_v0_codex_20260514"
    / "subset_manifest.json"
)
LABELS = ("tirex", "sundial_base_128m")
DOMAINS = (
    "aidc_power_optional",
    "aluminum_load",
    "arena_pv",
    "microgrid_load",
    "provincial_load",
)
HORIZONS = ("4h", "24h")
TEST_TOKEN = "P6_LATE_TSFM_TEST_ONCE_APPROVED_20260528"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(dumps(payload) + "\n")


def run_command(cmd: list[str], *, log_path: Path, env: dict[str, str]) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT),
        text=True,
        capture_output=True,
        env=env,
    )
    row = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "elapsed_sec": round(time.time() - started, 3),
        "stdout_tail": proc.stdout[-12000:],
        "stderr_tail": proc.stderr[-12000:],
    }
    append_log(log_path, row)
    return row


def lock_path(out_root: Path, run_id: str, label: str, domain: str, horizon: str) -> Path:
    return out_root / run_id / label / f"{domain}_{horizon}" / "optimize" / "late_extension_lock.json"


def test_manifest_path(out_root: Path, run_id: str, label: str, domain: str, horizon: str) -> Path:
    return out_root / run_id / label / f"{domain}_{horizon}" / "test_once" / "manifest.json"


def optimize_cmd(args: argparse.Namespace, label: str, domain: str, horizon: str) -> list[str]:
    return [
        str(PYTHON),
        str(SCRIPT),
        "--stage",
        "optimize",
        "--label",
        label,
        "--domain",
        domain,
        "--horizon",
        horizon,
        "--subset-manifest",
        str(args.subset_manifest),
        "--out-root",
        str(args.out_root),
        "--run-id",
        args.run_id,
        "--max-train-windows",
        str(args.max_train_windows),
        "--max-validation-windows",
        str(args.max_validation_windows),
        "--max-ridge-fit-points",
        str(args.max_ridge_fit_points),
        "--batch-size",
        str(args.batch_size),
        "--resume",
    ]


def test_cmd(args: argparse.Namespace, label: str, domain: str, horizon: str) -> list[str]:
    lock = lock_path(args.out_root, args.run_id, label, domain, horizon)
    return [
        str(PYTHON),
        str(SCRIPT),
        "--stage",
        "test-once",
        "--label",
        label,
        "--domain",
        domain,
        "--horizon",
        horizon,
        "--subset-manifest",
        str(args.subset_manifest),
        "--out-root",
        str(args.out_root),
        "--run-id",
        args.run_id,
        "--max-test-windows",
        str(args.max_test_windows),
        "--batch-size",
        str(args.batch_size),
        "--lock-json",
        str(lock),
        "--approval-token",
        TEST_TOKEN,
        "--resume",
    ]


def queue_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label in args.label:
        for domain in args.domain:
            for horizon in args.horizon:
                rows.append({"label": label, "domain": domain, "horizon": horizon})
    return rows


def run_queue(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    queue_root = args.out_root / args.run_id / "queue_status"
    queue_root.mkdir(parents=True, exist_ok=True)
    log_path = queue_root / "queue_driver.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env.setdefault("HF_HUB_OFFLINE", "1")
    rows = queue_rows(args)
    status_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        label = row["label"]
        domain = row["domain"]
        horizon = row["horizon"]
        row_id = f"{index:03d}_{label}_{domain}_{horizon}"
        status_path = queue_root / f"{row_id}.json"
        status: dict[str, Any] = {
            "row_id": row_id,
            "label": label,
            "domain": domain,
            "horizon": horizon,
            "started_at_utc": utc_now(),
        }
        try:
            if args.stage in {"optimize", "optimize-then-test-once"}:
                lock = lock_path(args.out_root, args.run_id, label, domain, horizon)
                if lock.exists():
                    opt_result = {"ok": True, "skipped": True, "reason": "lock_exists", "lock_json": rel(lock)}
                else:
                    opt_result = run_command(optimize_cmd(args, label, domain, horizon), log_path=log_path, env=env)
                status["optimize"] = opt_result
                if not opt_result.get("ok"):
                    status["status"] = "failed_optimize"
                    status_path.write_text(dumps(status) + "\n", encoding="utf-8")
                    status_rows.append(status)
                    if not args.continue_on_error:
                        break
                    continue
                if not lock.exists():
                    status["status"] = "failed_missing_lock"
                    status_path.write_text(dumps(status) + "\n", encoding="utf-8")
                    status_rows.append(status)
                    if not args.continue_on_error:
                        break
                    continue
                status["lock_json"] = rel(lock)

            if args.stage in {"test-once", "optimize-then-test-once"}:
                lock = lock_path(args.out_root, args.run_id, label, domain, horizon)
                if not lock.exists():
                    status["status"] = "failed_missing_lock"
                    status_path.write_text(dumps(status) + "\n", encoding="utf-8")
                    status_rows.append(status)
                    if not args.continue_on_error:
                        break
                    continue
                manifest = test_manifest_path(args.out_root, args.run_id, label, domain, horizon)
                if manifest.exists():
                    test_result = {"ok": True, "skipped": True, "reason": "test_manifest_exists", "manifest": rel(manifest)}
                else:
                    test_result = run_command(test_cmd(args, label, domain, horizon), log_path=log_path, env=env)
                status["test_once"] = test_result
                if not test_result.get("ok"):
                    status["status"] = "failed_test_once"
                    status_path.write_text(dumps(status) + "\n", encoding="utf-8")
                    status_rows.append(status)
                    if not args.continue_on_error:
                        break
                    continue
                if manifest.exists():
                    status["test_manifest"] = rel(manifest)
            status["status"] = "ok"
        except Exception as exc:
            status["status"] = "failed_exception"
            status["error"] = repr(exc)
            if not args.continue_on_error:
                raise
        finally:
            status["finished_at_utc"] = utc_now()
            status_path.write_text(dumps(status) + "\n", encoding="utf-8")
            status_rows.append(status)

    summary = {
        "status": "ok" if all(row.get("status") == "ok" for row in status_rows) else "partial",
        "created_at_utc": utc_now(),
        "plan_id": PLAN_ID,
        "stage": args.stage,
        "run_id": args.run_id,
        "queue_root": rel(queue_root),
        "queue_driver_log": rel(log_path),
        "row_count": int(len(rows)),
        "completed_rows": int(len(status_rows)),
        "ok_rows": int(sum(1 for row in status_rows if row.get("status") == "ok")),
        "failed_rows": int(sum(1 for row in status_rows if row.get("status") != "ok")),
        "labels": list(args.label),
        "domains": list(args.domain),
        "horizons": list(args.horizon),
        "max_train_windows": int(args.max_train_windows),
        "max_validation_windows": int(args.max_validation_windows),
        "max_test_windows": int(args.max_test_windows),
        "batch_size": int(args.batch_size),
        "lightgbm_n_jobs_used": 0,
        "non_lightgbm_cuda_required": True,
        "elapsed_sec": round(time.time() - started, 3),
    }
    summary_path = queue_root / "queue_manifest.json"
    summary_path.write_text(dumps(summary) + "\n", encoding="utf-8")
    print(dumps(summary))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["optimize", "test-once", "optimize-then-test-once"], required=True)
    parser.add_argument("--run-id", default="p6_late_tsfm_route_extension_full_codex_20260528")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--subset-manifest", type=Path, default=DEFAULT_SUBSET_MANIFEST)
    parser.add_argument("--label", action="append", choices=LABELS, default=None)
    parser.add_argument("--domain", action="append", choices=DOMAINS, default=None)
    parser.add_argument("--horizon", action="append", choices=HORIZONS, default=None)
    parser.add_argument("--max-train-windows", type=int, default=4096)
    parser.add_argument("--max-validation-windows", type=int, default=0)
    parser.add_argument("--max-test-windows", type=int, default=0)
    parser.add_argument("--max-ridge-fit-points", type=int, default=250000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.label = args.label or list(LABELS)
    args.domain = args.domain or list(DOMAINS)
    args.horizon = args.horizon or list(HORIZONS)
    run_queue(args)


if __name__ == "__main__":
    main()
