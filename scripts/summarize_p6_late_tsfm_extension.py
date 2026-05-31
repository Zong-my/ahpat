#!/usr/bin/env python3
"""Summarize P6 late TSFM optimize locks and guarded test-once manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
PLAN_ID = "p6_late_tsfm_route_extension_v0_codex_20260528"
DEFAULT_OUT_ROOT = PROJECT / "results" / "energy_tsfm_late_extension" / PLAN_ID


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def collect_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lock_path in sorted(run_root.glob("*/**/optimize/late_extension_lock.json")):
        lock = read_json(lock_path)
        label = str(lock["label"])
        domain = str(lock["domain"])
        horizon = str(lock["horizon"])
        test_manifest_path = run_root / label / f"{domain}_{horizon}" / "test_once" / "manifest.json"
        test_manifest = read_json(test_manifest_path) if test_manifest_path.exists() else {}
        rows.append(
            {
                "label": label,
                "domain": domain,
                "horizon": horizon,
                "lock_json": rel(lock_path),
                "selected_variant_id": lock.get("selected_variant_id"),
                "selected_transform_type": lock.get("selected_transform_type"),
                "selected_validation_wape": lock.get("selected_validation_wape"),
                "train_windows": lock.get("train_selection", {}).get("selected_count"),
                "validation_windows": lock.get("validation_selection", {}).get("selected_count"),
                "test_manifest": rel(test_manifest_path) if test_manifest_path.exists() else "",
                "selected_test_wape": test_manifest.get("selected_test_wape"),
                "test_windows": test_manifest.get("test_windows"),
                "test_data_read": bool(test_manifest.get("test_data_read", False)),
                "test_predictions_generated": bool(test_manifest.get("test_predictions_generated", False)),
                "license_note": "CC-BY-NC-4.0 model weights" if label == "moirai_moe_base" else "",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="p6_late_tsfm_route_extension_full_codex_20260528")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()
    run_root = args.out_root / args.run_id
    rows = collect_rows(run_root)
    frame = pd.DataFrame(rows).sort_values(["label", "domain", "horizon"]).reset_index(drop=True)
    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "p6_late_tsfm_extension_summary.csv"
    json_path = summary_dir / "p6_late_tsfm_extension_summary.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = {
        "status": "ok",
        "run_id": args.run_id,
        "row_count": int(len(frame)),
        "labels": sorted(frame["label"].dropna().unique().tolist()) if len(frame) else [],
        "test_rows": int(frame["test_data_read"].sum()) if len(frame) else 0,
        "csv": rel(csv_path),
        "json": rel(json_path),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
