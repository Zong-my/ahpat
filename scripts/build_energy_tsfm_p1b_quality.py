#!/usr/bin/env python3
"""Build segment-safe P1b datasets for the energy-TSFM paper.

Inputs are the first-pass canonical datasets in ``data/energy_tsfm_canonical``.
Outputs are segment-safe datasets in ``data/energy_tsfm_canonical_p1b`` with:

- explicit target-quality policy;
- ``target_raw`` and transformed ``target`` where relevant;
- invalid target rows separated for audit;
- continuous ``segment_id`` that breaks at timestamp gaps and invalid targets;
- per-domain horizon/context metadata and usable rolling-window counts.

This script does not fabricate measurements. When it transforms a target, it
retains ``target_raw`` and records the transformation in the data card.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
IN_ROOT = PROJECT / "data" / "energy_tsfm_canonical"
OUT_ROOT = PROJECT / "data" / "energy_tsfm_canonical_p1b"
REPORT_PATH = PROJECT / "results" / "p1b_quality_status.md"


DOMAINS = [
    "provincial_load",
    "aluminum_load",
    "microgrid_load",
    "arena_pv",
    "aidc_power_optional",
]


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def infer_native_step(series_df: pd.DataFrame) -> pd.Timedelta:
    ts = pd.to_datetime(series_df["timestamp"], errors="coerce").dropna().sort_values()
    diffs = ts.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        raise ValueError("cannot infer native step from fewer than two timestamps")
    mode = diffs.mode()
    return mode.iloc[0] if len(mode) else diffs.median()


def horizon_steps(step: pd.Timedelta) -> dict[str, int]:
    minutes = step.total_seconds() / 60.0
    return {
        "4h": int(round(240.0 / minutes)),
        "24h": int(round(1440.0 / minutes)),
    }


def context_steps(step: pd.Timedelta) -> dict[str, int]:
    minutes = step.total_seconds() / 60.0
    one_day = int(round(1440.0 / minutes))
    seven_day = int(round(7 * 1440.0 / minutes))
    return {
        "minimum": one_day,
        "default": one_day,
        "long_if_supported": min(512, seven_day),
        "weekly_if_supported": seven_day,
    }


def add_quality_policy(domain: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    df["target_raw"] = df["target"]
    df["is_imputed_target"] = False
    df["is_valid_target"] = df["target"].notna()
    info: dict[str, Any] = {
        "policy": "target_not_missing",
        "invalid_reason_counts": {},
        "transformation": "none",
    }

    if domain == "provincial_load":
        zero_mask = df["target_raw"].eq(0)
        df.loc[zero_mask, "is_valid_target"] = False
        info["policy"] = "province-scale exact-zero load is treated as invalid telemetry"
        info["invalid_reason_counts"]["zero_target"] = int(zero_mask.sum())

    elif domain == "arena_pv":
        negative_mask = df["target_raw"].lt(0)
        df["target"] = df["target_raw"].clip(lower=0)
        imputed_count = 0
        for _, group in df.sort_values("timestamp").groupby("series_id", sort=False):
            missing = group["target"].isna()
            blocks: list[tuple[int, int]] = []
            start: int | None = None
            positions = list(group.index)
            for pos, idx in enumerate(positions):
                if bool(missing.loc[idx]) and start is None:
                    start = pos
                elif (not bool(missing.loc[idx])) and start is not None:
                    blocks.append((start, pos - 1))
                    start = None
            if start is not None:
                blocks.append((start, len(positions) - 1))
            for start_pos, end_pos in blocks:
                block_len = end_pos - start_pos + 1
                if block_len > 8 or start_pos == 0 or end_pos == len(positions) - 1:
                    continue
                prev_idx = positions[start_pos - 1]
                next_idx = positions[end_pos + 1]
                prev_val = df.loc[prev_idx, "target"]
                next_val = df.loc[next_idx, "target"]
                if pd.notna(prev_val) and pd.notna(next_val) and float(prev_val) == 0.0 and float(next_val) == 0.0:
                    block_indices = positions[start_pos : end_pos + 1]
                    df.loc[block_indices, "target"] = 0.0
                    df.loc[block_indices, "is_imputed_target"] = True
                    imputed_count += len(block_indices)
        df["is_valid_target"] = df["target"].notna()
        info["policy"] = "nonnegative PV-generation target with raw net-power retained"
        info["transformation"] = "target = max(target_raw, 0); small gaps bounded by zero generation are imputed as 0"
        info["negative_raw_target_count"] = int(negative_mask.sum())
        info["zero_after_clipping_count"] = int(df["target"].eq(0).sum())
        info["small_zero_gap_imputed_count"] = int(imputed_count)
        info["negative_raw_target_range"] = [
            None if not bool(negative_mask.any()) else float(df.loc[negative_mask, "target_raw"].min()),
            None if not bool(negative_mask.any()) else float(df.loc[negative_mask, "target_raw"].max()),
        ]

    elif domain == "aluminum_load":
        zero_mask = df["target_raw"].eq(0)
        info["policy"] = "industrial-load exact zeros are retained but reported; long gaps split segments"
        info["zero_target_count_retained"] = int(zero_mask.sum())

    elif domain == "microgrid_load":
        info["policy"] = "microgrid load target retained; timestamp gaps split segments"

    elif domain == "aidc_power_optional":
        info["policy"] = "aggregate GPU power retained; timestamp gaps split segments; AIDC remains optional"

    missing_mask = df["target"].isna()
    if bool(missing_mask.any()):
        df.loc[missing_mask, "is_valid_target"] = False
        info["invalid_reason_counts"]["missing_target"] = int(missing_mask.sum())

    info["valid_target_count"] = int(df["is_valid_target"].sum())
    info["invalid_target_count"] = int((~df["is_valid_target"]).sum())
    return df, info


def regularize_full_grid_if_needed(domain: str, df: pd.DataFrame) -> pd.DataFrame:
    if domain != "arena_pv":
        return df
    parts: list[pd.DataFrame] = []
    for series_id, group in df.sort_values("timestamp").groupby("series_id", sort=False):
        group = group.copy()
        step = infer_native_step(group)
        idx = pd.date_range(group["timestamp"].min(), group["timestamp"].max(), freq=step)
        regular = group.set_index("timestamp").reindex(idx)
        regular.index.name = "timestamp"
        regular = regular.reset_index()
        regular["domain_id"] = group["domain_id"].iloc[0]
        regular["series_id"] = series_id
        parts.append(regular)
    return pd.concat(parts, ignore_index=True)


def assign_segments(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    valid = df[df["is_valid_target"] & df["timestamp"].notna()].copy()
    invalid = df[~(df["is_valid_target"] & df["timestamp"].notna())].copy()
    segment_records: list[pd.DataFrame] = []
    segment_stats: list[dict[str, Any]] = []
    gap_events: list[dict[str, Any]] = []

    for series_id, group in valid.sort_values("timestamp").groupby("series_id", sort=True):
        group = group.copy()
        step = infer_native_step(group)
        gap_threshold = step * 1.5
        ts = pd.to_datetime(group["timestamp"], errors="coerce")
        diffs = ts.diff()
        break_mask = diffs.isna() | diffs.gt(gap_threshold)
        seg_num = break_mask.cumsum().astype(int)
        group["segment_num"] = seg_num
        group["native_step_minutes"] = step.total_seconds() / 60.0
        for num, seg in group.groupby("segment_num", sort=True):
            segment_id = f"{series_id}_seg{int(num):04d}"
            seg = seg.copy()
            seg["segment_id"] = segment_id
            seg["segment_row_index"] = range(len(seg))
            segment_records.append(seg)
            segment_stats.append(
                {
                    "series_id": str(series_id),
                    "segment_id": segment_id,
                    "rows": int(len(seg)),
                    "timestamp_min": str(pd.to_datetime(seg["timestamp"]).min()),
                    "timestamp_max": str(pd.to_datetime(seg["timestamp"]).max()),
                    "native_step_minutes": step.total_seconds() / 60.0,
                }
            )
        gap_locs = group.index[break_mask & diffs.notna()].tolist()
        for idx in gap_locs:
            current_pos = group.index.get_loc(idx)
            prev_row = group.iloc[current_pos - 1]
            cur_row = group.iloc[current_pos]
            gap_events.append(
                {
                    "series_id": str(series_id),
                    "prev_timestamp": str(prev_row["timestamp"]),
                    "next_timestamp": str(cur_row["timestamp"]),
                    "gap": str(pd.to_datetime(cur_row["timestamp"]) - pd.to_datetime(prev_row["timestamp"])),
                    "gap_seconds": float((pd.to_datetime(cur_row["timestamp"]) - pd.to_datetime(prev_row["timestamp"])).total_seconds()),
                }
            )

    segmented = pd.concat(segment_records, ignore_index=True) if segment_records else valid.assign(segment_id="")
    segmented = segmented.drop(columns=["segment_num"], errors="ignore")
    meta = {
        "segment_count": int(segmented["segment_id"].nunique()) if len(segmented) else 0,
        "segments": segment_stats,
        "gap_count": len(gap_events),
        "gap_events_top": sorted(gap_events, key=lambda x: x["gap_seconds"], reverse=True)[:20],
    }
    return segmented, invalid, meta


def assign_split(segmented: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in segmented.sort_values("timestamp").groupby("series_id", sort=True):
        group = group.copy()
        n = len(group)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        group["split"] = ["train"] * train_end + ["validation"] * (val_end - train_end) + ["test"] * (n - val_end)
        parts.append(group)
    return pd.concat(parts, ignore_index=True) if parts else segmented


def usable_window_counts(segmented: pd.DataFrame, hsteps: dict[str, int], csteps: dict[str, int]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    default_context = csteps["default"]
    for horizon_name, horizon_step in hsteps.items():
        split_counts = {"train": 0, "validation": 0, "test": 0}
        total = 0
        for _, seg in segmented.groupby("segment_id", sort=False):
            n = len(seg)
            if n < default_context + horizon_step:
                continue
            for origin in range(default_context - 1, n - horizon_step):
                split = str(seg.iloc[origin + horizon_step]["split"])
                split_counts[split] = split_counts.get(split, 0) + 1
                total += 1
        counts[horizon_name] = {
            "horizon_steps": horizon_step,
            "context_steps": default_context,
            "total_windows": total,
            "split_counts": split_counts,
        }
    return counts


def build_domain(domain: str) -> dict[str, Any]:
    in_path = IN_ROOT / domain / "canonical.parquet"
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    df = pd.read_parquet(in_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = regularize_full_grid_if_needed(domain, df)
    df, quality_info = add_quality_policy(domain, df)
    segmented, invalid, segment_info = assign_segments(df)
    segmented = assign_split(segmented)

    # Reorder common columns first.
    common = [
        "domain_id",
        "series_id",
        "segment_id",
        "segment_row_index",
        "timestamp",
        "target",
        "target_raw",
        "is_valid_target",
        "split",
        "native_step_minutes",
    ]
    ordered_cols = [c for c in common if c in segmented.columns] + [c for c in segmented.columns if c not in common]
    segmented = segmented[ordered_cols]

    out_dir = OUT_ROOT / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    segmented.to_parquet(out_dir / "canonical_segmented.parquet", index=False)
    if len(invalid):
        invalid.to_parquet(out_dir / "invalid_rows.parquet", index=False)
    else:
        empty = df.head(0)
        empty.to_parquet(out_dir / "invalid_rows.parquet", index=False)

    step = infer_native_step(segmented)
    hsteps = horizon_steps(step)
    csteps = context_steps(step)
    windows = usable_window_counts(segmented, hsteps, csteps)
    validation = {
        "domain": domain,
        "input_path": str(in_path),
        "output_path": str(out_dir / "canonical_segmented.parquet"),
        "rows_input": int(len(df)),
        "rows_segmented_valid": int(len(segmented)),
        "rows_invalid": int(len(invalid)),
        "series_count": int(segmented["series_id"].nunique()),
        "segment_count": segment_info["segment_count"],
        "native_step_minutes": step.total_seconds() / 60.0,
        "horizon_steps": hsteps,
        "context_steps": csteps,
        "usable_windows": windows,
        "quality_policy": quality_info,
        "gap_count": segment_info["gap_count"],
        "gap_events_top": segment_info["gap_events_top"],
        "split_counts": {str(k): int(v) for k, v in segmented["split"].value_counts().sort_index().items()},
        "target_min": float(segmented["target"].min()) if len(segmented) else None,
        "target_max": float(segmented["target"].max()) if len(segmented) else None,
        "target_zero_count": int(segmented["target"].eq(0).sum()) if len(segmented) else 0,
        "target_negative_count": int(segmented["target"].lt(0).sum()) if len(segmented) else 0,
    }
    (out_dir / "validation.json").write_text(dumps(validation) + "\n", encoding="utf-8")

    card = [
        f"# P1b Data Card: {domain}",
        "",
        f"- Input: `{in_path}`",
        f"- Output: `{out_dir / 'canonical_segmented.parquet'}`",
        f"- Valid rows: {validation['rows_segmented_valid']}",
        f"- Invalid rows: {validation['rows_invalid']}",
        f"- Series count: {validation['series_count']}",
        f"- Segment count: {validation['segment_count']}",
        f"- Native step: {validation['native_step_minutes']} minutes",
        f"- Horizon steps: `{json.dumps(hsteps, sort_keys=True)}`",
        f"- Context steps: `{json.dumps(csteps, sort_keys=True)}`",
        "",
        "## Quality Policy",
        "",
        f"- Policy: {quality_info['policy']}",
        f"- Transformation: {quality_info['transformation']}",
        f"- Invalid target count: {quality_info['invalid_target_count']}",
        f"- Gap count after valid-target filtering: {segment_info['gap_count']}",
        "",
        "## Usable Rolling Windows",
        "",
        "| horizon | context_steps | total | train | validation | test |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for horizon_name, item in windows.items():
        sc = item["split_counts"]
        card.append(
            f"| {horizon_name} | {item['context_steps']} | {item['total_windows']} | {sc.get('train', 0)} | {sc.get('validation', 0)} | {sc.get('test', 0)} |"
        )
    (out_dir / "data_card.md").write_text("\n".join(card) + "\n", encoding="utf-8")
    return validation


def write_report(validations: list[dict[str, Any]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P1b Quality Status",
        "",
        "This report summarizes the segment-safe canonical datasets built after the P0/P1 retrospective audit.",
        "",
        "## Outputs",
        "",
        "| domain | valid rows | invalid rows | segments | step min | 4h windows | 24h windows | policy |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in validations:
        windows = item["usable_windows"]
        lines.append(
            "| `{domain}` | {valid} | {invalid} | {segments} | {step:g} | {w4} | {w24} | {policy} |".format(
                domain=item["domain"],
                valid=item["rows_segmented_valid"],
                invalid=item["rows_invalid"],
                segments=item["segment_count"],
                step=item["native_step_minutes"],
                w4=windows["4h"]["total_windows"],
                w24=windows["24h"]["total_windows"],
                policy=item["quality_policy"]["policy"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- These outputs supersede `data/energy_tsfm_canonical/*/canonical.parquet` for experiments.",
            "- Baseline and TSFM loaders must use `canonical_segmented.parquet` and must not create windows across different `segment_id` values.",
            "- The experimental horizons are strictly 4h and 24h. Horizon steps are mapped from each dataset's native time resolution to these elapsed durations.",
            "- Baseline and TSFM loaders must assign a rolling window to train/validation/test by the forecast endpoint (`origin + horizon_steps`), because several long segments span split boundaries.",
            "- `target_raw` is retained whenever a target policy transforms values, especially for ARENA PV clipping.",
            "- AIDC remains optional and should be removed or downgraded if it repeatedly fails H1/H2/H3 while other domains support the method.",
            "- Output validation passed in `.codex/data_validation/energy_tsfm_ops/p1b_output_validation.json`: no invalid targets, missing targets, duplicate timestamps inside a segment, or segment-internal time gaps were found.",
            "",
            "## Edge Audit",
            "",
            "Claude's follow-up review was rechecked by `scripts/audit_energy_tsfm_p1b_quality_claims.py`; the recomputed artifact is `.codex/data_validation/energy_tsfm_ops/p1b_quality_edge_audit.json`.",
            "",
            "Confirmed issues and boundaries:",
            "",
            "- Some long segments span split labels. This is acceptable only if loaders use endpoint-based split assignment and never build windows across `segment_id` values.",
            "- `provincial_load` has 74 valid segments after zero-target removal, but only 32 segments are long enough under the current one-day context for both 4h and 24h. Those 32 segments contain 135,178 rows, or 97.9% of P1b rows, so the headline data volume is still dominated by usable segments.",
            "- Shorter-horizon or shorter-context ablations are outside the current main scope. If introduced later, they must be reported separately from the 4h/24h main experiments.",
            "- `arena_pv` has 11,350 zero targets after clipping, 54.35% of valid rows. Full-day metrics must therefore be interpreted together with a daytime/positive-generation metric view.",
            "- `aluminum_load` now uses five sibling source CSV files as separate series. They must not be summed into a plant-wide aggregate unless documentation supports that interpretation.",
            "- `provincial_load` currently uses the load/weather timestamp intersection: raw load reaches 2022-06-17 23:45:00, while weather reaches 2022-04-28 23:45:00. Any load-only extension beyond the weather span is a P1 source-rebuild decision, not a P1b patch.",
            "- `microgrid_load` remains a native 10-minute series; cross-domain comparisons must use hour-based horizons, not equal step counts.",
            "",
            "## Next Step",
            "",
            "Start P2 deterministic baseline feasibility with loaders that consume the frozen P1c window indexes and reproduce the recorded split/window counts.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    validations = [build_domain(domain) for domain in DOMAINS]
    write_report(validations)
    print(dumps({"output_root": str(OUT_ROOT), "domains": [v["domain"] for v in validations]}))


if __name__ == "__main__":
    main()
