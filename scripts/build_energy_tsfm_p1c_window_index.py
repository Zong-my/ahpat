#!/usr/bin/env python3
"""Freeze P1c source/loader/metric decisions for energy-TSFM experiments.

The script builds a reusable rolling-window index from the P1b segment-safe
canonical datasets. It enforces the project rule that main experiments use only
4h and 24h elapsed-duration horizons, with per-domain horizon steps derived
from each dataset's native resolution.

No target values are changed here. The output is metadata that downstream
baseline and TSFM runners must consume instead of recreating ad hoc windows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
P1B_ROOT = PROJECT / "data" / "energy_tsfm_canonical_p1b"
OUT_ROOT = PROJECT / "data" / "energy_tsfm_windows_p1c"
REPORT_PATH = PROJECT / "results" / "p1c_freeze_report.md"
SUMMARY_PATH = PROJECT / "results" / "p1c_window_index_summary.json"

MAIN_HORIZONS = ("4h", "24h")


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_daylight_columns(df: pd.DataFrame) -> list[str]:
    terms = ("irradiance", "radiation", "solar", "ghi", "dni", "shortwave", "pyranometer", "pyrano", "pyroup")
    return [
        c
        for c in df.columns
        if c not in {"target", "target_raw"}
        and pd.api.types.is_numeric_dtype(df[c])
        and any(term in c.lower() for term in terms)
    ]


def build_window_index(domain_dir: Path) -> dict[str, Any]:
    domain = domain_dir.name
    data_path = domain_dir / "canonical_segmented.parquet"
    validation_path = domain_dir / "validation.json"
    df = pd.read_parquet(data_path)
    validation = load_json(validation_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    horizons = validation["horizon_steps"]
    context_steps = int(validation["context_steps"]["default"])
    missing = sorted(set(MAIN_HORIZONS) - set(horizons))
    if missing:
        raise ValueError(f"{domain} missing horizons: {missing}")

    out_dir = OUT_ROOT / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    daylight_cols = candidate_daylight_columns(df) if domain == "arena_pv" else []
    domain_summary: dict[str, Any] = {
        "domain": domain,
        "input_path": str(data_path),
        "context_steps": context_steps,
        "native_step_minutes": float(df["native_step_minutes"].dropna().iloc[0]),
        "horizons": {},
        "metric_scopes": ["full_day"],
        "daylight_columns_used": daylight_cols[:5],
    }

    if domain == "arena_pv":
        domain_summary["metric_scopes"] = ["full_day", "positive_generation"]
        if daylight_cols:
            domain_summary["metric_scopes"].append("irradiance_positive")

    for horizon_name in MAIN_HORIZONS:
        horizon_steps = int(horizons[horizon_name])
        records: list[dict[str, Any]] = []
        for segment_id, seg in df.sort_values(["segment_id", "segment_row_index"]).groupby("segment_id", sort=False):
            seg = seg.reset_index(drop=True)
            n = len(seg)
            if n < context_steps + horizon_steps:
                continue
            for origin in range(context_steps - 1, n - horizon_steps):
                context_start = origin - context_steps + 1
                forecast_start = origin + 1
                forecast_end = origin + horizon_steps
                forecast = seg.iloc[forecast_start : forecast_end + 1]
                endpoint = seg.iloc[forecast_end]
                target = pd.to_numeric(forecast["target"], errors="coerce")
                rec: dict[str, Any] = {
                    "window_id": f"{domain}__{horizon_name}__{segment_id}__{origin:08d}",
                    "domain_id": str(endpoint["domain_id"]),
                    "series_id": str(endpoint["series_id"]),
                    "segment_id": str(segment_id),
                    "horizon": horizon_name,
                    "context_steps": context_steps,
                    "horizon_steps": horizon_steps,
                    "native_step_minutes": float(endpoint["native_step_minutes"]),
                    "context_start_row_index": int(context_start),
                    "context_end_row_index": int(origin),
                    "forecast_start_row_index": int(forecast_start),
                    "forecast_end_row_index": int(forecast_end),
                    "context_start_timestamp": str(seg.iloc[context_start]["timestamp"]),
                    "context_end_timestamp": str(seg.iloc[origin]["timestamp"]),
                    "forecast_start_timestamp": str(seg.iloc[forecast_start]["timestamp"]),
                    "forecast_end_timestamp": str(endpoint["timestamp"]),
                    "split": str(endpoint["split"]),
                    "target_mean": float(target.mean()),
                    "target_min": float(target.min()),
                    "target_max": float(target.max()),
                    "target_sum": float(target.sum()),
                    "positive_target_count": int(target.gt(0).sum()),
                    "positive_target_share": float(target.gt(0).mean()),
                    "all_zero_target": bool(target.eq(0).all()),
                }
                if domain == "arena_pv":
                    rec["metric_full_day"] = True
                    rec["metric_positive_generation"] = bool(target.gt(0).any())
                    for col in daylight_cols[:5]:
                        values = pd.to_numeric(forecast[col], errors="coerce")
                        rec[f"metric_{col}_positive"] = bool(values.gt(0).any())
                records.append(rec)

        index = pd.DataFrame(records)
        if not index.empty:
            index.to_parquet(out_dir / f"window_index_{horizon_name}.parquet", index=False)
        else:
            pd.DataFrame().to_parquet(out_dir / f"window_index_{horizon_name}.parquet", index=False)

        split_counts = (
            {str(k): int(v) for k, v in index["split"].value_counts().sort_index().items()}
            if not index.empty
            else {}
        )
        horizon_summary: dict[str, Any] = {
            "horizon_steps": horizon_steps,
            "context_steps": context_steps,
            "window_count": int(len(index)),
            "split_counts": split_counts,
            "path": str(out_dir / f"window_index_{horizon_name}.parquet"),
        }
        if domain == "arena_pv" and not index.empty:
            horizon_summary["positive_generation_windows"] = int(index["metric_positive_generation"].sum())
            horizon_summary["positive_generation_window_share"] = float(index["metric_positive_generation"].mean())
            horizon_summary["all_zero_windows"] = int(index["all_zero_target"].sum())
        domain_summary["horizons"][horizon_name] = horizon_summary

    return domain_summary


def write_report(summary: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    domains = summary["domains"]
    lines = [
        "# P1c Freeze Report",
        "",
        "P1c freezes the source-breadth, loader-edge and metric-scope decisions required before P2 baseline/zero-shot experiments.",
        "",
        "## Frozen Rules",
        "",
        "- Main horizons are exactly 4h and 24h. The 1h horizon is not part of the main experiment matrix.",
        "- Horizon steps are derived from elapsed time and native resolution, not forced to a common step count.",
        "- Downstream runners must consume `data/energy_tsfm_windows_p1c/*/window_index_{4h,24h}.parquet` rather than rebuilding windows ad hoc.",
        "- Each window stays within one `segment_id`; split is assigned by the forecast endpoint.",
        "- The default context is one native day for every domain.",
        "",
        "## Source-Breadth Decisions",
        "",
        "| Item | Decision | Rationale |",
        "|---|---|---|",
        "| Provincial load tail | Keep the main dataset at the load/weather timestamp intersection. Do not add the 2022-04-29 to 2022-06-17 load-only tail to the main P2 matrix. | The main matrix remains covariate-complete. A load-only tail can be a later univariate robustness dataset with a separate ID. |",
        "| Aluminum CSVs | Include the five `aluminum_load_line_*.csv` files as separate series, not as one aggregated plant-level load. | This increases industrial-load breadth without making an unsupported plant-wide aggregation claim. |",
        "| ARENA PV metrics | Report full-day metrics and a positive-generation view. | The clipped PV target contains many valid zero-generation points; full-day-only metrics can hide daytime behavior. |",
        "| Microgrid resolution | Keep native 10-minute resolution. | Comparisons are by 4h/24h elapsed duration, not by equal native step count. |",
        "",
        "## Window Index Summary",
        "",
        "| domain | native step min | context steps | 4h windows | 24h windows | metric scopes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in domains:
        lines.append(
            "| `{domain}` | {step:g} | {context} | {w4} | {w24} | {scopes} |".format(
                domain=item["domain"],
                step=item["native_step_minutes"],
                context=item["context_steps"],
                w4=item["horizons"]["4h"]["window_count"],
                w24=item["horizons"]["24h"]["window_count"],
                scopes=", ".join(item["metric_scopes"]),
            )
        )
    lines.extend(
        [
            "",
            "## P2 Entry Condition",
            "",
            "P2 baseline and zero-shot runs may start only after their loaders read this P1c window index and reproduce the split/window counts above.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    domains = [build_window_index(path) for path in sorted(P1B_ROOT.iterdir()) if path.is_dir()]
    summary = {
        "main_horizons": list(MAIN_HORIZONS),
        "output_root": str(OUT_ROOT),
        "domains": domains,
    }
    SUMMARY_PATH.write_text(dumps(summary) + "\n", encoding="utf-8")
    write_report(summary)
    print(dumps({"output_root": str(OUT_ROOT), "domains": [d["domain"] for d in domains]}))


if __name__ == "__main__":
    main()
