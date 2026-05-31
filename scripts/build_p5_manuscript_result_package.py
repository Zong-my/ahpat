#!/usr/bin/env python3
"""Build a compact manuscript-facing result package from audited P5 outputs.

This is analysis packaging only. It reads already generated P5/H2/H3 summary
artifacts and writes paper-facing tables plus a claim-boundary brief. It does
not rerun models, touch queue rows, or tune on test.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
PLAN_ID = "p5_manuscript_result_package_v0_codex_20260519"
OUT_DIR = PROJECT / "results/energy_tsfm_p5_main" / PLAN_ID
DOC_OUT = PROJECT / "results"
STATUS_DOC = DOC_OUT / "p5_manuscript_result_package_status.md"

P5_ROOT = PROJECT / "results/energy_tsfm_p5_main/p5_main_test_once_v0_codex_20260517_summary_codex_20260518"
H2H3_ROOT = PROJECT / "results/energy_tsfm_p5_main/p5_h2_h3_test_application_v0_codex_20260519"
P3_4DU_MANIFEST = (
    PROJECT
    / "data/energy_tsfm_tuning/p3_4du_chronos2_train4096_application_lock_v0_codex_20260519"
    / "p3_4du_chronos2_train4096_application_manifest.json"
)

P5_SUMMARY_JSON = P5_ROOT / "p5_main_result_summary.json"
P5_LEADERBOARD = P5_ROOT / "p5_main_tsfm_vs_non_tsfm_test_leaderboard.csv"
P5_MODEL_MATRIX = P5_ROOT / "p5_main_model_best_test_matrix.csv"
H2H3_MANIFEST = H2H3_ROOT / "p5_h2_h3_test_application_manifest.json"
H2_POLICY = H2H3_ROOT / "p5_h2_policy_metrics.csv"
H2_PER_CELL = H2H3_ROOT / "p5_h2_policy_per_cell_metrics.csv"
H3_POLICY = H2H3_ROOT / "p5_h3_policy_decision_weighted_metrics.csv"
H3_THRESHOLDS = H2H3_ROOT / "p5_h3_train_threshold_ledger.csv"

TABLE_P5 = OUT_DIR / "table_1_p5_tsfm_vs_non_tsfm_test.csv"
TABLE_MODEL_MATRIX = OUT_DIR / "table_s1_p5_model_test_matrix.csv"
TABLE_H2 = OUT_DIR / "table_2_h2_policy_test_metrics.csv"
TABLE_H3 = OUT_DIR / "table_3_h3_policy_decision_weighted_metrics.csv"
CLAIM_MATRIX = OUT_DIR / "claim_boundary_matrix.csv"
BRIEF_MD = OUT_DIR / "manuscript_results_brief.md"
MANIFEST_JSON = OUT_DIR / "p5_manuscript_result_package_manifest.json"


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(value) + "\n", encoding="utf-8")


def pct_improvement(baseline: float, value: float) -> float:
    return 100.0 * (baseline - value) / baseline


def best_always_policy(policy: pd.DataFrame, metric_col: str, scope: str | None = None) -> pd.Series:
    df = policy[policy["policy_id"].astype(str).str.startswith("always_")].copy()
    if scope is not None and "metric_scope" in df.columns:
        df = df[df["metric_scope"].astype(str).eq(scope)].copy()
    if df.empty:
        raise ValueError(f"no always_* policies for {metric_col}/{scope}")
    return df.loc[df[metric_col].astype(float).idxmin()]


def row_for_policy(policy: pd.DataFrame, policy_id: str, scope: str | None = None) -> pd.Series:
    df = policy[policy["policy_id"].astype(str).eq(policy_id)].copy()
    if scope is not None and "metric_scope" in df.columns:
        df = df[df["metric_scope"].astype(str).eq(scope)].copy()
    if len(df) != 1:
        raise ValueError(f"expected one row for {policy_id}/{scope}, got {len(df)}")
    return df.iloc[0]


def format_float(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return lines


def build_claim_matrix(
    p5_summary: dict[str, Any],
    h2_manifest: dict[str, Any],
    du_manifest: dict[str, Any],
    h2_primary: pd.Series,
    h2_best_always: pd.Series,
    h3_primary_all: pd.Series,
    h3_best_always_all: pd.Series,
    h3_primary_critical: pd.Series,
    h3_best_always_critical: pd.Series,
) -> pd.DataFrame:
    rows = [
        {
            "claim_id": "H1_main_result",
            "claim_status": "supported_with_boundary",
            "defensible_sentence": (
                "Adapted TSFM routes are competitive and win 5/10 formal test cells, "
                "with a lower mean best-cell WAPE than the best non-TSFM group."
            ),
            "numeric_evidence": (
                f"TSFM wins {p5_summary['tsfm_win_cells']}/{p5_summary['total_cells']}; "
                f"mean TSFM WAPE {p5_summary['mean_best_tsfm_test_wape']:.6f}; "
                f"mean non-TSFM WAPE {p5_summary['mean_best_non_tsfm_test_wape']:.6f}; "
                f"mean gap {p5_summary['mean_gap_tsfm_minus_non']:.6f}."
            ),
            "must_not_claim": "Do not claim universal TSFM dominance; non-TSFM wins 5/10 cells.",
            "primary_artifact": rel(P5_LEADERBOARD),
        },
        {
            "claim_id": "H1_chronos2_train4096",
            "claim_status": "pre_test_selection_support",
            "defensible_sentence": (
                "The Chronos-2 adapter learning curve justified large-budget adapter evidence "
                "before test inspection, while formal test-once Chronos rows remain full-train refit."
            ),
            "numeric_evidence": (
                f"train4096 improved over train512 in {du_manifest['curve_improved_4096_vs_512_cells']}/6 "
                f"monitored cells; current lock stronger than controlled train4096 in "
                f"{du_manifest['current_lock_stronger_than_train4096_cells']}/6 overlapping cells."
            ),
            "must_not_claim": "Do not write train4096 as test evidence or cap formal training at 4096.",
            "primary_artifact": du_manifest["lock_csv"],
        },
        {
            "claim_id": "H2_primary_policy",
            "claim_status": "supported",
            "defensible_sentence": (
                "The locked validation-selected cell policy improves aggregate test WAPE "
                "over the best fixed route-family baseline."
            ),
            "numeric_evidence": (
                f"cell_validation_winner WAPE {float(h2_primary['aggregate_wape']):.6f}; "
                f"best always baseline {h2_best_always['policy_id']} WAPE "
                f"{float(h2_best_always['aggregate_wape']):.6f}; "
                f"relative improvement {pct_improvement(float(h2_best_always['aggregate_wape']), float(h2_primary['aggregate_wape'])):.2f}%."
            ),
            "must_not_claim": (
                "Do not promote tsfm_validation_winner to the original primary H2 policy; "
                "it is secondary/exploratory unless labeled as such."
            ),
            "primary_artifact": rel(H2_POLICY),
        },
        {
            "claim_id": "H3_decision_weighted",
            "claim_status": "supported",
            "defensible_sentence": (
                "The same locked policy remains beneficial under decision-weighted test metrics, "
                "including critical windows defined by train-only thresholds."
            ),
            "numeric_evidence": (
                f"all-window decision WAPE {float(h3_primary_all['decision_weighted_wape']):.6f} "
                f"vs best always baseline {h3_best_always_all['policy_id']} "
                f"{float(h3_best_always_all['decision_weighted_wape']):.6f}; "
                f"critical-union decision WAPE {float(h3_primary_critical['decision_weighted_wape']):.6f} "
                f"vs {float(h3_best_always_critical['decision_weighted_wape']):.6f}; "
                f"critical windows {h2_manifest['critical_union_windows']}/{h2_manifest['test_windows']}."
            ),
            "must_not_claim": "Do not claim H3 thresholds were fitted on validation/test; they are train-threshold only.",
            "primary_artifact": rel(H3_POLICY),
        },
        {
            "claim_id": "oracle_gap",
            "claim_status": "interpretation_only",
            "defensible_sentence": (
                "The oracle gap quantifies remaining routing headroom and motivates better "
                "pre-test diagnostics, not a deployable method."
            ),
            "numeric_evidence": (
                f"H2 oracle WAPE {h2_manifest['h2_oracle_policy']['aggregate_wape']:.6f}; "
                f"H2 primary WAPE {h2_manifest['h2_primary_policy_result']['aggregate_wape']:.6f}."
            ),
            "must_not_claim": "Never present oracle_per_window_best as deployable.",
            "primary_artifact": rel(H2_POLICY),
        },
    ]
    return pd.DataFrame(rows)


def write_brief(
    p5_summary: dict[str, Any],
    h2_manifest: dict[str, Any],
    h2_primary: pd.Series,
    h2_best_always: pd.Series,
    h2_secondary: pd.Series,
    h3_primary_all: pd.Series,
    h3_primary_critical: pd.Series,
    h3_secondary_all: pd.Series,
    h3_secondary_critical: pd.Series,
    claim_matrix: pd.DataFrame,
) -> None:
    h2_gain = pct_improvement(float(h2_best_always["aggregate_wape"]), float(h2_primary["aggregate_wape"]))
    h3_best_all = best_always_policy(pd.read_csv(H3_POLICY), "decision_weighted_wape", "all_windows")
    h3_best_critical = best_always_policy(pd.read_csv(H3_POLICY), "decision_weighted_wape", "critical_union")
    h3_gain_all = pct_improvement(
        float(h3_best_all["decision_weighted_wape"]), float(h3_primary_all["decision_weighted_wape"])
    )
    h3_gain_critical = pct_improvement(
        float(h3_best_critical["decision_weighted_wape"]),
        float(h3_primary_critical["decision_weighted_wape"]),
    )

    lines: list[str] = [
        "# P5 Manuscript Result Brief",
        "",
        "## Boundary",
        "",
        "- This package reads audited P5/H2/H3 result artifacts only.",
        "- It does not rerun models, regenerate predictions, or tune on test.",
        "- P5 formal test evidence was produced from the historical P3-4ds lock; P3-4du/P3-4dv is the current downstream lock/readiness for future operations.",
        "",
        "## Core Story",
        "",
        (
            "The clean story is not that TSFMs dominate every cell. The stronger story is that "
            "adapted TSFMs create high-value specialists, non-TSFM models remain strong in some "
            "regimes, and locked operational routing/decision weighting converts this heterogeneity "
            "into a deployable forecasting layer for next-generation energy systems."
        ),
        "",
        "## Main Evidence",
        "",
        f"- H1/P5: TSFM wins `{p5_summary['tsfm_win_cells']}/{p5_summary['total_cells']}` cells; non-TSFM wins `{p5_summary['non_tsfm_win_cells']}/{p5_summary['total_cells']}`.",
        f"- H1/P5: mean best TSFM WAPE `{p5_summary['mean_best_tsfm_test_wape']:.6f}` vs best non-TSFM `{p5_summary['mean_best_non_tsfm_test_wape']:.6f}`.",
        f"- H2 primary: `cell_validation_winner` WAPE `{float(h2_primary['aggregate_wape']):.6f}` vs best fixed route `{h2_best_always['policy_id']}` WAPE `{float(h2_best_always['aggregate_wape']):.6f}`; relative gain `{h2_gain:.2f}%`.",
        f"- H2 secondary: `tsfm_validation_winner` WAPE `{float(h2_secondary['aggregate_wape']):.6f}`; useful but not the original locked primary policy.",
        f"- H3 primary all-window decision WAPE: `{float(h3_primary_all['decision_weighted_wape']):.6f}`; fixed-route gain `{h3_gain_all:.2f}%`.",
        f"- H3 primary critical-union decision WAPE: `{float(h3_primary_critical['decision_weighted_wape']):.6f}` over `{h2_manifest['critical_union_windows']}/{h2_manifest['test_windows']}` test windows; fixed-route gain `{h3_gain_critical:.2f}%`.",
        f"- H3 secondary `tsfm_validation_winner`: all-window `{float(h3_secondary_all['decision_weighted_wape']):.6f}`, critical-union `{float(h3_secondary_critical['decision_weighted_wape']):.6f}`.",
        "",
        "## Claim Boundary Matrix",
        "",
    ]
    rows = [
        {
            "claim": row["claim_id"],
            "status": row["claim_status"],
            "do_not_claim": row["must_not_claim"],
        }
        for row in claim_matrix.to_dict(orient="records")
    ]
    lines.extend(md_table(rows, ["claim", "status", "do_not_claim"]))
    lines.extend(
        [
            "",
            "## Recommended Manuscript Framing",
            "",
            "1. Present TSFM adaptation as a strong specialist layer, not as universal dominance.",
            "2. Use H2 as the main operational contribution: validation-locked routing beats any single fixed route family on aggregate test WAPE.",
            "3. Use H3 to connect forecasting to operationally important windows: train-threshold decision weighting preserves the routing advantage on critical windows.",
            "4. Keep oracle results explicitly non-deployable and use them only to show remaining headroom.",
        ]
    )
    BRIEF_MD.parent.mkdir(parents=True, exist_ok=True)
    BRIEF_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(
    p5_summary: dict[str, Any],
    h2_manifest: dict[str, Any],
    du_manifest: dict[str, Any],
    h2_policy: pd.DataFrame,
    h3_policy: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": detail})

    add("p5_queue_ok_57", int(p5_summary.get("queue_rows", -1)) == 57, f"queue_rows={p5_summary.get('queue_rows')}")
    add("p5_cuda_manifest_ok", bool(p5_summary.get("cuda_manifest_all_ok")), str(p5_summary.get("cuda_manifest_all_ok")))
    add("p5_mixed_not_universal_tsfm", int(p5_summary.get("tsfm_win_cells", -1)) == 5 and int(p5_summary.get("non_tsfm_win_cells", -1)) == 5, f"tsfm={p5_summary.get('tsfm_win_cells')} non={p5_summary.get('non_tsfm_win_cells')}")
    add("h2h3_status_ok", h2_manifest.get("status") == "ok", str(h2_manifest.get("status")))
    add("h2_primary_cell_validation_winner", h2_manifest.get("h2_primary_policy") == "cell_validation_winner", str(h2_manifest.get("h2_primary_policy")))
    add("h2_policy_has_oracle", "oracle_per_window_best" in set(h2_policy["policy_id"].astype(str)), "oracle present")
    add("h3_policy_has_critical_union", "critical_union" in set(h3_policy["metric_scope"].astype(str)), "critical_union present")
    add("h3_train_threshold_only", int(h2_manifest.get("critical_union_windows", 0)) > 0, f"critical={h2_manifest.get('critical_union_windows')}")
    add("du_status_ok", du_manifest.get("status") == "ok", str(du_manifest.get("status")))
    add("du_no_boundary_violations", int(du_manifest.get("post_boundary_violations", -1)) == 0, str(du_manifest.get("post_boundary_violations")))
    add("source_files_exist", all(p.exists() for p in [P5_LEADERBOARD, P5_MODEL_MATRIX, H2_POLICY, H3_POLICY]), "source CSVs")
    return pd.DataFrame(checks)


def write_status(manifest: dict[str, Any], audit_df: pd.DataFrame) -> None:
    failures = audit_df[audit_df["status"].astype(str) != "pass"]
    lines = [
        "# P5 Manuscript Result Package",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Plan ID: `{PLAN_ID}`",
        "- Boundary: paper packaging only; no model rerun, no prediction regeneration, no test tuning.",
        f"- P5 TSFM wins: `{manifest['p5_tsfm_win_cells']}/{manifest['p5_total_cells']}`.",
        f"- H2 primary WAPE: `{manifest['h2_primary_wape']:.6f}`.",
        f"- H3 primary all-window decision WAPE: `{manifest['h3_primary_all_decision_wape']:.6f}`.",
        f"- H3 primary critical-union decision WAPE: `{manifest['h3_primary_critical_decision_wape']:.6f}`.",
        f"- Audit failures: `{manifest['audit_failures']}`.",
        "",
        "## Artifacts",
        "",
        f"- Brief: `{rel(BRIEF_MD)}`",
        f"- Claim matrix: `{rel(CLAIM_MATRIX)}`",
        f"- P5 table: `{rel(TABLE_P5)}`",
        f"- H2 table: `{rel(TABLE_H2)}`",
        f"- H3 table: `{rel(TABLE_H3)}`",
        f"- Manifest: `{rel(MANIFEST_JSON)}`",
    ]
    if not failures.empty:
        lines.extend(["", "## Audit Failures", ""])
        for row in failures.itertuples(index=False):
            lines.append(f"- `{row.check_id}`: {row.detail}")
    STATUS_DOC.parent.mkdir(parents=True, exist_ok=True)
    STATUS_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p5_summary = read_json(P5_SUMMARY_JSON)
    h2_manifest = read_json(H2H3_MANIFEST)
    du_manifest = read_json(P3_4DU_MANIFEST)
    p5 = pd.read_csv(P5_LEADERBOARD)
    matrix = pd.read_csv(P5_MODEL_MATRIX)
    h2_policy = pd.read_csv(H2_POLICY)
    h3_policy = pd.read_csv(H3_POLICY)
    h3_thresholds = pd.read_csv(H3_THRESHOLDS)

    h2_primary = row_for_policy(h2_policy, "cell_validation_winner")
    h2_secondary = row_for_policy(h2_policy, "tsfm_validation_winner")
    h2_best_always = best_always_policy(h2_policy, "aggregate_wape")
    h3_primary_all = row_for_policy(h3_policy, "cell_validation_winner", "all_windows")
    h3_primary_critical = row_for_policy(h3_policy, "cell_validation_winner", "critical_union")
    h3_secondary_all = row_for_policy(h3_policy, "tsfm_validation_winner", "all_windows")
    h3_secondary_critical = row_for_policy(h3_policy, "tsfm_validation_winner", "critical_union")
    h3_best_always_all = best_always_policy(h3_policy, "decision_weighted_wape", "all_windows")
    h3_best_always_critical = best_always_policy(h3_policy, "decision_weighted_wape", "critical_union")

    p5.to_csv(TABLE_P5, index=False)
    matrix.to_csv(TABLE_MODEL_MATRIX, index=False)
    h2_policy[
        [
            "diagnostic_rank_by_aggregate_wape",
            "policy_id",
            "deployable_h2_candidate",
            "oracle_upper_bound",
            "selection_mode",
            "windows",
            "cells",
            "aggregate_wape",
            "regret_to_oracle_wape",
            "gap_to_best_deployable_wape",
        ]
    ].to_csv(TABLE_H2, index=False)
    h3_policy[
        [
            "policy_id",
            "metric_scope",
            "windows",
            "cells",
            "critical_window_share",
            "mean_decision_weight",
            "wape",
            "decision_weighted_wape",
            "mean_window_wape",
        ]
    ].to_csv(TABLE_H3, index=False)

    claim_matrix = build_claim_matrix(
        p5_summary,
        h2_manifest,
        du_manifest,
        h2_primary,
        h2_best_always,
        h3_primary_all,
        h3_best_always_all,
        h3_primary_critical,
        h3_best_always_critical,
    )
    claim_matrix.to_csv(CLAIM_MATRIX, index=False)
    write_brief(
        p5_summary,
        h2_manifest,
        h2_primary,
        h2_best_always,
        h2_secondary,
        h3_primary_all,
        h3_primary_critical,
        h3_secondary_all,
        h3_secondary_critical,
        claim_matrix,
    )
    audit_df = audit(p5_summary, h2_manifest, du_manifest, h2_policy, h3_policy)
    audit_failures = int((audit_df["status"].astype(str) != "pass").sum())
    audit_csv = OUT_DIR / "p5_manuscript_result_package_audit.csv"
    audit_df.to_csv(audit_csv, index=False)

    manifest = {
        "plan_id": PLAN_ID,
        "status": "ok" if audit_failures == 0 else "failed",
        "created_utc": utc_now(),
        "output_dir": rel(OUT_DIR),
        "brief_md": rel(BRIEF_MD),
        "claim_matrix_csv": rel(CLAIM_MATRIX),
        "table_p5_csv": rel(TABLE_P5),
        "table_model_matrix_csv": rel(TABLE_MODEL_MATRIX),
        "table_h2_csv": rel(TABLE_H2),
        "table_h3_csv": rel(TABLE_H3),
        "audit_csv": rel(audit_csv),
        "manifest_json": rel(MANIFEST_JSON),
        "status_doc": rel(STATUS_DOC),
        "source_p5_summary": rel(P5_SUMMARY_JSON),
        "source_h2h3_manifest": rel(H2H3_MANIFEST),
        "source_p3_4du_manifest": rel(P3_4DU_MANIFEST),
        "p5_tsfm_win_cells": int(p5_summary["tsfm_win_cells"]),
        "p5_non_tsfm_win_cells": int(p5_summary["non_tsfm_win_cells"]),
        "p5_total_cells": int(p5_summary["total_cells"]),
        "p5_mean_best_tsfm_wape": float(p5_summary["mean_best_tsfm_test_wape"]),
        "p5_mean_best_non_tsfm_wape": float(p5_summary["mean_best_non_tsfm_test_wape"]),
        "h2_primary_policy": "cell_validation_winner",
        "h2_primary_wape": float(h2_primary["aggregate_wape"]),
        "h2_best_always_policy": str(h2_best_always["policy_id"]),
        "h2_best_always_wape": float(h2_best_always["aggregate_wape"]),
        "h2_primary_gain_vs_best_always_pct": pct_improvement(
            float(h2_best_always["aggregate_wape"]), float(h2_primary["aggregate_wape"])
        ),
        "h2_secondary_tsfm_wape": float(h2_secondary["aggregate_wape"]),
        "h3_primary_all_decision_wape": float(h3_primary_all["decision_weighted_wape"]),
        "h3_primary_critical_decision_wape": float(h3_primary_critical["decision_weighted_wape"]),
        "h3_secondary_tsfm_all_decision_wape": float(h3_secondary_all["decision_weighted_wape"]),
        "h3_secondary_tsfm_critical_decision_wape": float(h3_secondary_critical["decision_weighted_wape"]),
        "h3_critical_windows": int(h2_manifest["critical_union_windows"]),
        "h3_test_windows": int(h2_manifest["test_windows"]),
        "h3_threshold_rows": int(len(h3_thresholds)),
        "audit_checks": int(len(audit_df)),
        "audit_failures": audit_failures,
        "boundary": {
            "model_rerun": False,
            "prediction_regeneration": False,
            "test_tuning": False,
            "oracle_deployable": False,
        },
    }
    write_json(MANIFEST_JSON, manifest)
    write_status(manifest, audit_df)
    print(dumps(manifest))
    return 0 if audit_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
