#!/usr/bin/env python3
"""Build P5 test-once summary from per-route outputs and execution lock CSV.

This script bridges the gap between individual test-once route outputs
(produced by run_p5_locked_queue_executor.py) and the downstream H2/H3
and manuscript result packaging scripts.

It reads:
  - The execution lock CSV (route metadata)
  - Per-route metrics CSVs from test-once output directories

It writes:
  - p5_main_locked_primary_metrics.csv (merged metrics + metadata)
  - p5_main_result_summary.json (aggregate summary)
  - p5_main_tsfm_vs_non_tsfm_test_leaderboard.csv
  - p5_main_model_best_test_matrix.csv
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
LOCK_CSV = (
    PROJECT
    / "data/energy_tsfm_tuning"
    / "p3_4du_chronos2_train4096_application_lock_v0_codex_20260519"
    / "p3_4du_p5_pre_executable_lock_after_chronos4096_application.csv"
)
P5_ROOT = PROJECT / "results/energy_tsfm_p5_main/p5_main_test_once_v0_codex_20260517"
QUEUE_ROOT = PROJECT / "results/energy_tsfm_p5_main/p5_main_test_once_v0_codex_20260517_queue_codex_20260518"
OUT_DIR = PROJECT / "results/energy_tsfm_p5_main/p5_main_test_once_v0_codex_20260517_summary_codex_20260518"

TSFM_MODELS = {"chronos2", "timesfm2p5"}
METRIC_SCOPE = "full_day"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_run_id(lock_row: dict) -> str:
    """Extract run_id from lock row, falling back to command template."""
    run_id = lock_row.get("p5_run_id", "").strip()
    if not run_id:
        cmd = lock_row.get("p5_command_template", "")
        m = re.search(r'--run-id\s+(\S+)', cmd)
        if m:
            run_id = m.group(1)
    return run_id


def find_metrics_csvs(lock_row: dict, p5_root: Path) -> list[Path]:
    """Locate all metrics CSVs for a given route (test + validation)."""
    run_id = _resolve_run_id(lock_row)
    if not run_id:
        return []

    out_root = p5_root / run_id / "runner_outputs"
    if not out_root.exists():
        return []

    # Collect all metrics CSVs (test, validation, requested_split)
    found = set()
    for pattern in ["**/test_metrics.csv", "**/validation_metrics.csv",
                     "**/requested_split_metrics.csv"]:
        found.update(out_root.glob(pattern))
    return sorted(found)


def read_metrics(path: Path, split: str = "test") -> dict[str, Any]:
    """Read a metrics CSV and extract the primary metric row."""
    df = pd.read_csv(path)
    # Filter for the target split and scope
    mask = pd.Series([True] * len(df))
    if "split" in df.columns:
        mask &= df["split"] == split
    if "metric_scope" in df.columns:
        mask &= df["metric_scope"] == METRIC_SCOPE
    elif "scope" in df.columns:
        mask &= df["scope"] == METRIC_SCOPE

    filtered = df[mask]
    if filtered.empty:
        # Fall back to unfiltered
        filtered = df

    row = filtered.iloc[0] if not filtered.empty else pd.Series()
    return {
        "n_windows": int(row.get("n_windows", row.get("count", 0))),
        "n_points": int(row.get("n_points", 0)),
        "mae": float(row.get("mae", float("nan"))),
        "rmse": float(row.get("rmse", float("nan"))),
        "wape": float(row.get("wape", float("nan"))),
        "smape": float(row.get("smape", float("nan"))),
        "r2": float(row.get("r2", float("nan"))),
        "bias": float(row.get("bias", float("nan"))),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lock_rows = list(csv.DictReader(open(LOCK_CSV)))
    print(f"Processing {len(lock_rows)} locked routes...")

    # Also try to read queue status for timing info
    queue_status = {}
    queue_csv = QUEUE_ROOT / "queue_status.csv"
    if queue_csv.exists():
        for qr in csv.DictReader(open(queue_csv)):
            queue_status[qr["lock_id"]] = qr

    all_rows = []
    missing = []

    for i, lr in enumerate(lock_rows):
        lock_id = lr["lock_id"]
        model_id = lr["model_id"]
        route_id = lr.get("route_id", model_id)
        domain = lr["domain_id"]
        horizon = lr["horizon"]
        config_id = _resolve_run_id(lr)

        metrics_files = find_metrics_csvs(lr, P5_ROOT)
        if not metrics_files:
            missing.append(lock_id)
            continue

        qs = queue_status.get(lock_id, {})
        model_family = "tsfm" if model_id in TSFM_MODELS else ("dl" if model_id in {"itransformer", "nbeatsx"} else "ml")

        # Emit one row per (split, scope) found across all metrics files
        splits_emitted = set()
        for mf in metrics_files:
            mf_df = pd.read_csv(mf)
            for split_val in ["validation", "test"]:
                if split_val in splits_emitted:
                    continue
                metrics = read_metrics(mf, split=split_val)
                if np.isnan(metrics["wape"]):
                    continue
                splits_emitted.add(split_val)

                row = {
                    "lock_id": lock_id,
                    "queue_index": i + 1,
                    "lock_portfolio_group": lr.get("portfolio_group", ""),
                    "lock_model_id": model_id,
                    "lock_route_id": route_id,
                    "domain_id": domain,
                    "horizon": horizon,
                    "split": split_val,
                    "metric_scope": METRIC_SCOPE,
                    "config_id": config_id,
                    "lock_selected_config_or_variant": lr.get("selected_config_or_variant", ""),
                    **metrics,
                    "lock_validation_wape": lr.get("validation_wape", ""),
                    "source_metrics_file": str(mf),
                    "output_root": str(mf.parent.parent),
                    "status_elapsed_seconds": qs.get("elapsed_seconds", ""),
                    "status_started_at_utc": qs.get("started_at_utc", ""),
                    "status_finished_at_utc": qs.get("finished_at_utc", ""),
                    "training_or_finetuning_started": qs.get("training_or_finetuning_started", True),
                    "test_data_read": qs.get("test_data_read", True),
                    "test_predictions_generated": qs.get("test_predictions_generated", True),
                    "model_execution_started": qs.get("model_execution_started", True),
                    "model_family": model_family,
                    "model_id": model_id,
                    "route_id": route_id,
                    "selected_config_or_variant": lr.get("selected_config_or_variant", ""),
                    "validation_metric_scope_locked": METRIC_SCOPE,
                    "validation_wape_locked": lr.get("validation_wape", ""),
                    "lock_validation_metric_scope": METRIC_SCOPE,
                    "underlying_command_shell": lr.get("p5_command_template", ""),
                }
                all_rows.append(row)

    if missing:
        print(f"WARNING: {len(missing)} routes have no metrics (not yet trained):")
        for m in missing[:5]:
            print(f"  {m}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")

    if not all_rows:
        print("ERROR: No routes with metrics found. Run model training first (Step 2).")
        return

    df = pd.DataFrame(all_rows)
    primary_csv = OUT_DIR / "p5_main_locked_primary_metrics.csv"
    df.to_csv(primary_csv, index=False)
    print(f"Wrote {len(df)} rows to {primary_csv}")

    # Build leaderboard: best TSFM vs best non-TSFM per cell
    leaderboard_rows = []
    for (domain, horizon), group in df.groupby(["domain_id", "horizon"]):
        tsfm = group[group["model_family"] == "tsfm"]
        nontsfm = group[group["model_family"] != "tsfm"]
        best_tsfm = tsfm.loc[tsfm["wape"].idxmin()] if not tsfm.empty else None
        best_nontsfm = nontsfm.loc[nontsfm["wape"].idxmin()] if not nontsfm.empty else None
        leaderboard_rows.append({
            "domain_id": domain,
            "horizon": horizon,
            "best_tsfm_model": best_tsfm["model_id"] if best_tsfm is not None else "",
            "best_tsfm_route": best_tsfm["route_id"] if best_tsfm is not None else "",
            "best_tsfm_wape": float(best_tsfm["wape"]) if best_tsfm is not None else float("nan"),
            "best_nontsfm_model": best_nontsfm["model_id"] if best_nontsfm is not None else "",
            "best_nontsfm_route": best_nontsfm["route_id"] if best_nontsfm is not None else "",
            "best_nontsfm_wape": float(best_nontsfm["wape"]) if best_nontsfm is not None else float("nan"),
            "gap": (float(best_tsfm["wape"]) - float(best_nontsfm["wape"]))
            if best_tsfm is not None and best_nontsfm is not None
            else float("nan"),
        })
    lb = pd.DataFrame(leaderboard_rows)
    lb_csv = OUT_DIR / "p5_main_tsfm_vs_non_tsfm_test_leaderboard.csv"
    lb.to_csv(lb_csv, index=False)

    # Build model-best matrix: best config per model per cell
    matrix_rows = []
    for (model, domain, horizon), group in df.groupby(["model_id", "domain_id", "horizon"]):
        best = group.loc[group["wape"].idxmin()]
        matrix_rows.append({
            "model_id": model,
            "domain_id": domain,
            "horizon": horizon,
            "best_route_id": best["route_id"],
            "best_config": best["config_id"],
            "wape": float(best["wape"]),
            "mae": float(best["mae"]),
            "rmse": float(best["rmse"]),
        })
    mx = pd.DataFrame(matrix_rows)
    mx_csv = OUT_DIR / "p5_main_model_best_test_matrix.csv"
    mx.to_csv(mx_csv, index=False)

    # Build summary JSON
    summary = {
        "plan_id": "p5_main_test_once_v0_codex_20260517_summary",
        "created_at_utc": utc_now(),
        "total_routes": len(df),
        "total_routes_in_lock": len(lock_rows),
        "missing_routes": len(missing),
        "domains": sorted(df["domain_id"].unique().tolist()),
        "horizons": sorted(df["horizon"].unique().tolist()),
        "models": sorted(df["model_id"].unique().tolist()),
        "overall_mean_wape": float(df["wape"].mean()),
        "tsfm_mean_wape": float(df[df["model_family"] == "tsfm"]["wape"].mean()),
        "nontsfm_mean_wape": float(df[df["model_family"] != "tsfm"]["wape"].mean()),
    }
    summary_json = OUT_DIR / "p5_main_result_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    # Write raw metrics for downstream
    raw_csv = OUT_DIR / "p5_main_all_metric_rows_raw.csv"
    df.to_csv(raw_csv, index=False)

    # Write test-only primary metrics
    test_csv = OUT_DIR / "p5_main_test_primary_metrics.csv"
    df[df["split"] == "test"].to_csv(test_csv, index=False)

    print(f"\nSummary written to {OUT_DIR}")
    print(f"  {primary_csv.name}: {len(df)} rows")
    print(f"  {lb_csv.name}: {len(lb)} rows")
    print(f"  {mx_csv.name}: {len(mx)} rows")
    print(f"  {summary_json.name}")


if __name__ == "__main__":
    main()
