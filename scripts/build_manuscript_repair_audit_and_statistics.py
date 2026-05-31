#!/usr/bin/env python3
"""Build manuscript-repair audit tables and statistical checks.

This script reads existing frozen-window, formal-test, and H2/H3 artifacts.
It does not train models, rerun completed test rows, or tune on test.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/energy_tsfm_manuscript_repair_v1_20260521"

P5_SUMMARY = ROOT / (
    "results/energy_tsfm_p5_main/"
    "p5_main_test_once_v0_codex_20260517_summary_codex_20260518"
)
H2H3_ROOT = ROOT / (
    "results/energy_tsfm_p5_main/"
    "p5_h2_h3_test_application_v0_codex_20260519"
)
QUEUE_ROOT = ROOT / (
    "results/energy_tsfm_p5_main/"
    "p5_main_test_once_v0_codex_20260517_queue_codex_20260518"
)

STANDARD_COLUMNS = {
    "domain_id",
    "domain",
    "series_id",
    "segment_id",
    "segment_row_index",
    "timestamp",
    "target",
    "target_raw",
    "is_valid_target",
    "split",
    "native_step_minutes",
    "is_imputed_target",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def ensure_out(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def is_tsfm_family(value: object) -> bool:
    text = str(value).lower()
    return any(key in text for key in ["chronos", "timesfm", "tirex", "sundial"])


def aggregate_wape(df: pd.DataFrame, weight_col: str | None = None) -> float:
    if weight_col is None:
        err = df["abs_error_sum"].sum()
        truth = df["abs_true_sum"].sum()
    else:
        err = (df["abs_error_sum"] * df[weight_col]).sum()
        truth = (df["abs_true_sum"] * df[weight_col]).sum()
    return float(err / truth) if truth else float("nan")


def horizon_to_hours(horizon: str) -> float:
    if str(horizon).endswith("h"):
        return float(str(horizon)[:-1])
    return float("nan")


@dataclass
class PairedArrays:
    cell_id: str
    window_id: np.ndarray
    y_true_len: np.ndarray
    a_error: np.ndarray
    b_error: np.ndarray
    truth: np.ndarray
    weight: np.ndarray
    a_loss: np.ndarray
    b_loss: np.ndarray


def make_paired_arrays(
    a: pd.DataFrame,
    b: pd.DataFrame,
    a_name: str,
    b_name: str,
    weight_col: str | None = None,
) -> list[PairedArrays]:
    keys = ["cell_id", "window_id"]
    use_cols = [
        "cell_id",
        "window_id",
        "target_start_time",
        "y_true_len",
        "abs_error_sum",
        "abs_true_sum",
        "wape",
    ]
    if weight_col and weight_col in a.columns:
        use_cols.append(weight_col)
    a2 = a[use_cols].rename(
        columns={
            "abs_error_sum": f"{a_name}_error",
            "abs_true_sum": "truth_a",
            "wape": f"{a_name}_loss",
        }
    )
    b2 = b[use_cols].rename(
        columns={
            "abs_error_sum": f"{b_name}_error",
            "abs_true_sum": "truth_b",
            "wape": f"{b_name}_loss",
        }
    )
    merged = a2.merge(b2, on=keys, suffixes=("_a", "_b"), how="inner")
    if merged.empty:
        return []
    merged["target_start_time"] = pd.to_datetime(
        merged.get("target_start_time_a", merged.get("target_start_time")),
        errors="coerce",
        utc=True,
    )
    merged = merged.sort_values(["cell_id", "target_start_time", "window_id"])
    if weight_col:
        col = f"{weight_col}_a" if f"{weight_col}_a" in merged.columns else weight_col
        merged["_weight"] = merged[col].fillna(1.0).astype(float)
    else:
        merged["_weight"] = 1.0
    y_col = "y_true_len_a" if "y_true_len_a" in merged.columns else "y_true_len"
    arrays: list[PairedArrays] = []
    for cell_id, group in merged.groupby("cell_id", sort=False):
        truth = group["truth_a"].to_numpy(float)
        arrays.append(
            PairedArrays(
                cell_id=str(cell_id),
                window_id=group["window_id"].to_numpy(str),
                y_true_len=group[y_col].to_numpy(float),
                a_error=group[f"{a_name}_error"].to_numpy(float),
                b_error=group[f"{b_name}_error"].to_numpy(float),
                truth=truth,
                weight=group["_weight"].to_numpy(float),
                a_loss=group[f"{a_name}_loss"].to_numpy(float),
                b_loss=group[f"{b_name}_loss"].to_numpy(float),
            )
        )
    return arrays


def block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.int64)
    block_size = max(1, min(int(block_size), n))
    n_blocks = int(math.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks)
    parts = [(start + np.arange(block_size)) % n for start in starts]
    return np.concatenate(parts)[:n]


def block_bootstrap_diff(
    arrays: list[PairedArrays],
    n_bootstrap: int,
    seed: int,
    weighted: bool = False,
) -> dict[str, float]:
    if not arrays:
        return {
            "paired_windows": 0,
            "observed_diff": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "bootstrap_p_two_sided": float("nan"),
        }
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        a_err = 0.0
        b_err = 0.0
        truth = 0.0
        for arr in arrays:
            n = len(arr.window_id)
            block_size = int(np.nanmedian(arr.y_true_len) * 2)
            idx = block_indices(n, block_size, rng)
            weight = arr.weight[idx] if weighted else 1.0
            a_err += float(np.sum(arr.a_error[idx] * weight))
            b_err += float(np.sum(arr.b_error[idx] * weight))
            truth += float(np.sum(arr.truth[idx] * weight))
        diffs[b] = (a_err / truth) - (b_err / truth) if truth else np.nan
    a_total = 0.0
    b_total = 0.0
    truth_total = 0.0
    for arr in arrays:
        weight = arr.weight if weighted else 1.0
        a_total += float(np.sum(arr.a_error * weight))
        b_total += float(np.sum(arr.b_error * weight))
        truth_total += float(np.sum(arr.truth * weight))
    observed = (a_total / truth_total) - (b_total / truth_total)
    finite = diffs[np.isfinite(diffs)]
    if finite.size:
        p_two = 2.0 * min(np.mean(finite <= 0.0), np.mean(finite >= 0.0))
        p_two = min(1.0, float(p_two))
        ci_low, ci_high = np.quantile(finite, [0.025, 0.975])
    else:
        p_two = float("nan")
        ci_low = ci_high = float("nan")
    return {
        "paired_windows": int(sum(len(arr.window_id) for arr in arrays)),
        "observed_diff": float(observed),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_p_two_sided": p_two,
    }


def dm_like_pvalue(arrays: list[PairedArrays], max_lag: int | None = None) -> dict[str, float]:
    losses = []
    for arr in arrays:
        losses.append(arr.a_loss - arr.b_loss)
    if not losses:
        return {"dm_like_z": float("nan"), "dm_like_p": float("nan"), "loss_windows": 0}
    d = np.concatenate(losses)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return {"dm_like_z": float("nan"), "dm_like_p": float("nan"), "loss_windows": n}
    if max_lag is None:
        max_lag = int(min(200, max(1, round(np.sqrt(n)))))
    mean = float(np.mean(d))
    centered = d - mean
    gamma0 = float(np.dot(centered, centered) / n)
    var = gamma0
    for lag in range(1, min(max_lag, n - 1) + 1):
        cov = float(np.dot(centered[lag:], centered[:-lag]) / n)
        var += 2.0 * (1.0 - lag / (max_lag + 1.0)) * cov
    se = math.sqrt(max(var / n, 1e-30))
    z = mean / se
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"dm_like_z": float(z), "dm_like_p": float(p), "loss_windows": int(n)}


def build_dataset_audits(out_dir: Path) -> None:
    provenance_rows = []
    covariate_rows = []
    target_rows = []
    preprocessing_rows = []

    for path in sorted((ROOT / "data/energy_tsfm_canonical_p1b").glob("*/canonical_segmented.parquet")):
        domain = path.parent.name
        df = pd.read_parquet(path)
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        target = pd.to_numeric(df["target"], errors="coerce")
        cov_cols = [c for c in df.columns if c not in STANDARD_COLUMNS]
        provenance_rows.append(
            {
                "domain": domain,
                "canonical_segmented_path": str(path.relative_to(ROOT)),
                "rows": len(df),
                "series_count": df["series_id"].nunique(),
                "segment_count": df["segment_id"].nunique(),
                "timestamp_min": ts.min(),
                "timestamp_max": ts.max(),
                "native_step_minutes_median": pd.to_numeric(
                    df["native_step_minutes"], errors="coerce"
                ).median(),
                "target_nonnull_rate": float(target.notna().mean()),
                "valid_target_rate": float(df["is_valid_target"].mean())
                if "is_valid_target" in df
                else float("nan"),
                "imputed_target_rate": float(df["is_imputed_target"].mean())
                if "is_imputed_target" in df
                else float("nan"),
                "covariate_count": len(cov_cols),
                "split_counts": json.dumps(df["split"].value_counts().to_dict(), sort_keys=True),
            }
        )
        target_rows.append(
            {
                "domain": domain,
                "target_unit": infer_target_unit(domain),
                "rows": len(df),
                "target_mean": float(target.mean()),
                "target_median": float(target.median()),
                "target_iqr": float(target.quantile(0.75) - target.quantile(0.25)),
                "target_abs_mean": float(target.abs().mean()),
                "target_abs_median": float(target.abs().median()),
                "target_min": float(target.min()),
                "target_max": float(target.max()),
                "zero_target_share": float((target == 0).mean()),
                "positive_target_share": float((target > 0).mean()),
            }
        )
        preprocessing_rows.append(
            {
                "domain": domain,
                "invalid_rows_path": str((path.parent / "invalid_rows.parquet").relative_to(ROOT)),
                "invalid_rows": read_invalid_rows(path.parent / "invalid_rows.parquet"),
                "has_imputed_target_flag": "is_imputed_target" in df.columns,
                "imputed_target_count": int(df["is_imputed_target"].sum())
                if "is_imputed_target" in df
                else 0,
                "standardized_target_column": "target",
                "target_raw_column_present": "target_raw" in df.columns,
                "split_column_present": "split" in df.columns,
            }
        )
        for col in cov_cols:
            s = df[col]
            covariate_rows.append(
                {
                    "domain": domain,
                    "covariate": col,
                    "dtype": str(s.dtype),
                    "nonnull_rate": float(s.notna().mean()),
                    "train_nonnull_rate": float(s[df["split"] == "train"].notna().mean())
                    if (df["split"] == "train").any()
                    else float("nan"),
                    "validation_nonnull_rate": float(
                        s[df["split"] == "validation"].notna().mean()
                    )
                    if (df["split"] == "validation").any()
                    else float("nan"),
                    "test_nonnull_rate": float(s[df["split"] == "test"].notna().mean())
                    if (df["split"] == "test").any()
                    else float("nan"),
                }
            )

    window_rows = []
    for path in sorted((ROOT / "data/energy_tsfm_windows_p1c").glob("*/window_index_*.parquet")):
        df = pd.read_parquet(path)
        domain = str(df["domain_id"].iloc[0])
        horizon = str(df["horizon"].iloc[0])
        counts = df["split"].value_counts().to_dict()
        window_rows.append(
            {
                "domain": domain,
                "horizon": horizon,
                "window_index_path": str(path.relative_to(ROOT)),
                "windows_total": len(df),
                "windows_train": counts.get("train", 0),
                "windows_validation": counts.get("validation", 0),
                "windows_test": counts.get("test", 0),
                "series_count": df["series_id"].nunique(),
                "segment_count": df["segment_id"].nunique(),
                "context_steps_median": float(df["context_steps"].median()),
                "horizon_steps_median": float(df["horizon_steps"].median()),
                "native_step_minutes_median": float(df["native_step_minutes"].median()),
                "all_zero_target_share": float(df["all_zero_target"].mean()),
                "positive_target_share_mean": float(df["positive_target_share"].mean()),
                "forecast_start_min": pd.to_datetime(
                    df["forecast_start_timestamp"], errors="coerce", utc=True
                ).min(),
                "forecast_end_max": pd.to_datetime(
                    df["forecast_end_timestamp"], errors="coerce", utc=True
                ).max(),
            }
        )

    pd.DataFrame(provenance_rows).to_csv(out_dir / "dataset_provenance_table.csv", index=False)
    pd.DataFrame(window_rows).to_csv(out_dir / "split_window_audit_table.csv", index=False)
    pd.DataFrame(covariate_rows).to_csv(out_dir / "covariate_availability_ledger.csv", index=False)
    pd.DataFrame(preprocessing_rows).to_csv(out_dir / "preprocessing_scope_audit.csv", index=False)
    pd.DataFrame(target_rows).to_csv(out_dir / "domain_target_units_and_magnitude.csv", index=False)


def infer_target_unit(domain: str) -> str:
    mapping = {
        "provincial_load": "MW-scale load proxy from anonymized northern China provincial load data",
        "aluminum_load": "industrial aluminum-load active-power target in project-normalized engineering units",
        "microgrid_load": "microgrid load in dataset-native kW-like units",
        "arena_pv": "PV active-power/export target in dataset-native power units",
        "aidc_power_optional": "aggregate GPU power in W",
    }
    return mapping.get(domain, "dataset-native target unit")


def read_invalid_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_parquet(path)))
    except Exception:
        return -1


def parse_vector(text: object) -> np.ndarray:
    if isinstance(text, np.ndarray):
        return text.astype(float)
    if isinstance(text, list):
        return np.asarray(text, dtype=float)
    value = str(text).strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if not value:
        return np.asarray([], dtype=float)
    return np.fromstring(value, sep=",", dtype=float)


def formal_prediction_path(source_metrics_file: str) -> Path:
    metrics_path = Path(source_metrics_file)
    if not metrics_path.is_absolute():
        metrics_path = ROOT / metrics_path
    pred_dir = metrics_path.parent.parent / "predictions"
    candidates = [
        pred_dir / "test_predictions_all_arms.parquet",
        pred_dir / "test_predictions_all_variants.parquet",
        pred_dir / "test_predictions.parquet",
        pred_dir / "requested_split_predictions_all_arms.parquet",
        pred_dir / "requested_split_predictions_all_variants.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No formal test prediction parquet found. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def load_formal_lock_window_metrics(
    all_metric_rows: pd.DataFrame,
    lock_id: str,
    expected_wape: float,
) -> pd.DataFrame:
    row = all_metric_rows[
        (all_metric_rows["lock_id"] == lock_id)
        & (all_metric_rows["split"] == "test")
        & (all_metric_rows["metric_scope"] == "full_day")
    ].copy()
    if row.empty:
        raise ValueError(f"No formal metric row for lock_id={lock_id}")
    row["_wape_distance"] = (pd.to_numeric(row["wape"], errors="coerce") - expected_wape).abs()
    row0 = row.sort_values("_wape_distance").iloc[0]
    pred_path = formal_prediction_path(str(row0["source_metrics_file"]))
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    pred = pd.read_parquet(pred_path)
    if "config_id" in pred.columns:
        pred = pred[pred["config_id"] == row0["config_id"]].copy()
    if pred.empty:
        raise ValueError(f"No prediction rows for lock_id={lock_id}, config={row0['config_id']}")
    if "split" in pred.columns:
        pred = pred[pred["split"].astype(str) == "test"].copy()
    records = []
    for rec in pred.itertuples(index=False):
        y_true = parse_vector(getattr(rec, "y_true"))
        y_pred = parse_vector(getattr(rec, "y_pred"))
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Length mismatch in {lock_id} {getattr(rec, 'window_id')}: "
                f"{len(y_true)} vs {len(y_pred)}"
            )
        abs_true = float(np.sum(np.abs(y_true)))
        abs_error = float(np.sum(np.abs(y_true - y_pred)))
        records.append(
            {
                "cell_id": f"{getattr(rec, 'domain_id')}::{getattr(rec, 'horizon')}",
                "domain": getattr(rec, "domain_id"),
                "horizon": getattr(rec, "horizon"),
                "window_id": getattr(rec, "window_id"),
                "target_start_time": getattr(rec, "target_start_time"),
                "y_true_len": len(y_true),
                "abs_error_sum": abs_error,
                "abs_true_sum": abs_true,
                "wape": abs_error / abs_true if abs_true else float("nan"),
                "candidate_id": lock_id,
                "route_family": getattr(rec, "model_id"),
                "model_id": getattr(rec, "model_id"),
                "source_prediction_path": str(pred_path.relative_to(ROOT)),
            }
        )
    out = pd.DataFrame(records)
    observed_wape = aggregate_wape(out)
    expected_wape = float(row0["wape"])
    if not math.isclose(observed_wape, expected_wape, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(
            f"Recomputed WAPE mismatch for {lock_id}: "
            f"{observed_wape} vs formal {expected_wape}"
        )
    return out


def build_h1_statistics(out_dir: Path, n_boot: int, seed: int) -> pd.DataFrame:
    leaderboard = safe_read_csv(P5_SUMMARY / "p5_main_tsfm_vs_non_tsfm_test_leaderboard.csv")
    all_metric_rows = safe_read_csv(P5_SUMMARY / "p5_main_all_metric_rows_raw.csv")
    rows = []
    dm_rows = []
    for i, rec in leaderboard.sort_values(["domain_id", "horizon"]).iterrows():
        a = load_formal_lock_window_metrics(
            all_metric_rows,
            str(rec["best_tsfm_lock_id"]),
            expected_wape=float(rec["best_tsfm_wape"]),
        )
        b = load_formal_lock_window_metrics(
            all_metric_rows,
            str(rec["best_non_tsfm_lock_id"]),
            expected_wape=float(rec["best_non_tsfm_wape"]),
        )
        arrays = make_paired_arrays(a, b, "tsfm", "non_tsfm")
        boot = block_bootstrap_diff(arrays, n_boot, seed + i, weighted=False)
        dm = dm_like_pvalue(arrays)
        cell_id = f"{rec['domain_id']}::{rec['horizon']}"
        rows.append(
            {
                "cell_id": cell_id,
                "domain": rec["domain_id"],
                "horizon": rec["horizon"],
                "best_tsfm_candidate_id": rec["best_tsfm_lock_id"],
                "best_tsfm_route_family": rec["best_tsfm_model"],
                "best_tsfm_wape": rec["best_tsfm_wape"],
                "best_non_tsfm_candidate_id": rec["best_non_tsfm_lock_id"],
                "best_non_tsfm_route_family": rec["best_non_tsfm_model"],
                "best_non_tsfm_wape": rec["best_non_tsfm_wape"],
                "wape_diff_tsfm_minus_non_tsfm": rec["wape_gap_tsfm_minus_non"],
                "winner_group": rec["winner_group"],
                "paired_windows": boot["paired_windows"],
                "block_bootstrap_ci_low": boot["ci_low"],
                "block_bootstrap_ci_high": boot["ci_high"],
                "bootstrap_p_two_sided": boot["bootstrap_p_two_sided"],
                "dm_like_p_window_wape": dm["dm_like_p"],
                "dm_like_z_window_wape": dm["dm_like_z"],
            }
        )
        dm_rows.append(
            {
                "comparison_scope": "H1_cell",
                "comparison_id": cell_id,
                "a": "best_tsfm",
                "b": "best_non_tsfm",
                **dm,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "h1_cell_bootstrap_ci.csv", index=False)
    pd.DataFrame(dm_rows).to_csv(out_dir / "_h1_dm_like_pvalues.csv", index=False)
    return out


def policy_frame(policy_rows: pd.DataFrame, policy_id: str) -> pd.DataFrame:
    df = policy_rows[policy_rows["policy_id"] == policy_id].copy()
    if df.empty:
        raise ValueError(f"Missing policy rows: {policy_id}")
    return df


def build_h2_statistics(out_dir: Path, n_boot: int, seed: int) -> pd.DataFrame:
    policies = safe_read_csv(H2H3_ROOT / "p5_h2_policy_selections.csv")
    comparisons = [
        ("cell_validation_winner", "always_itransformer", "primary_vs_best_fixed"),
        ("cell_validation_winner", "always_lightgbm", "primary_vs_lightgbm"),
        ("cell_validation_winner", "always_nbeatsx", "primary_vs_nbeatsx"),
        ("tsfm_validation_winner", "cell_validation_winner", "secondary_tsfm_only_vs_primary"),
        ("tsfm_validation_winner", "always_itransformer", "secondary_tsfm_only_vs_best_fixed"),
    ]
    rows = []
    dm_rows = []
    for i, (a_id, b_id, label) in enumerate(comparisons):
        a = policy_frame(policies, a_id)
        b = policy_frame(policies, b_id)
        arrays = make_paired_arrays(a, b, "a", "b")
        boot = block_bootstrap_diff(arrays, n_boot, seed + 100 + i, weighted=False)
        dm = dm_like_pvalue(arrays)
        rows.append(
            {
                "comparison_id": label,
                "policy_a": a_id,
                "policy_b": b_id,
                "policy_a_wape": aggregate_wape(a),
                "policy_b_wape": aggregate_wape(b),
                "wape_diff_a_minus_b": aggregate_wape(a) - aggregate_wape(b),
                "paired_windows": boot["paired_windows"],
                "block_bootstrap_ci_low": boot["ci_low"],
                "block_bootstrap_ci_high": boot["ci_high"],
                "bootstrap_p_two_sided": boot["bootstrap_p_two_sided"],
                "dm_like_p_window_wape": dm["dm_like_p"],
                "dm_like_z_window_wape": dm["dm_like_z"],
            }
        )
        dm_rows.append(
            {
                "comparison_scope": "H2_policy",
                "comparison_id": label,
                "a": a_id,
                "b": b_id,
                **dm,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "h2_policy_bootstrap_ci.csv", index=False)
    pd.DataFrame(dm_rows).to_csv(out_dir / "_h2_dm_like_pvalues.csv", index=False)
    return out


def build_h3_statistics(out_dir: Path, n_boot: int, seed: int) -> pd.DataFrame:
    policies = safe_read_csv(H2H3_ROOT / "p5_h2_policy_selections.csv")
    labels = safe_read_csv(H2H3_ROOT / "p5_h3_test_window_labels.csv")
    keep = ["cell_id", "window_id", "critical_union", "decision_weight"]
    policies = policies.merge(labels[keep], on=["cell_id", "window_id"], how="left")
    comparisons = [
        ("cell_validation_winner", "always_itransformer", "primary_vs_best_fixed"),
        ("cell_validation_winner", "always_lightgbm", "primary_vs_lightgbm"),
        ("cell_validation_winner", "always_nbeatsx", "primary_vs_nbeatsx"),
        ("tsfm_validation_winner", "cell_validation_winner", "secondary_tsfm_only_vs_primary"),
    ]
    rows = []
    dm_rows = []
    for scope, scope_df in [
        ("all_windows", policies),
        ("critical_windows", policies[policies["critical_union"].fillna(False)].copy()),
    ]:
        for i, (a_id, b_id, label) in enumerate(comparisons):
            a = policy_frame(scope_df, a_id)
            b = policy_frame(scope_df, b_id)
            arrays = make_paired_arrays(a, b, "a", "b", weight_col="decision_weight")
            boot = block_bootstrap_diff(arrays, n_boot, seed + 200 + i, weighted=True)
            dm = dm_like_pvalue(arrays)
            rows.append(
                {
                    "metric_scope": scope,
                    "comparison_id": label,
                    "policy_a": a_id,
                    "policy_b": b_id,
                    "policy_a_dwape": aggregate_wape(a, "decision_weight"),
                    "policy_b_dwape": aggregate_wape(b, "decision_weight"),
                    "dwape_diff_a_minus_b": aggregate_wape(a, "decision_weight")
                    - aggregate_wape(b, "decision_weight"),
                    "paired_windows": boot["paired_windows"],
                    "block_bootstrap_ci_low": boot["ci_low"],
                    "block_bootstrap_ci_high": boot["ci_high"],
                    "bootstrap_p_two_sided": boot["bootstrap_p_two_sided"],
                    "dm_like_p_window_wape": dm["dm_like_p"],
                    "dm_like_z_window_wape": dm["dm_like_z"],
                }
            )
            dm_rows.append(
                {
                    "comparison_scope": f"H3_{scope}",
                    "comparison_id": label,
                    "a": a_id,
                    "b": b_id,
                    **dm,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "h3_stress_bootstrap_ci.csv", index=False)
    pd.DataFrame(dm_rows).to_csv(out_dir / "_h3_dm_like_pvalues.csv", index=False)
    return out


def with_h3_labels(df: pd.DataFrame) -> pd.DataFrame:
    labels = safe_read_csv(H2H3_ROOT / "p5_h3_test_window_labels.csv")
    use_cols = ["cell_id", "window_id", "critical_union", "decision_weight"]
    out = df.merge(labels[use_cols], on=["cell_id", "window_id"], how="left")
    out["critical_union"] = out["critical_union"].fillna(False).astype(bool)
    out["decision_weight"] = pd.to_numeric(out["decision_weight"], errors="coerce").fillna(1.0)
    return out


def aggregate_policy_row(policy_id: str, df: pd.DataFrame, selection_mode: str) -> dict[str, object]:
    df = with_h3_labels(df)
    critical = df[df["critical_union"]].copy()
    selected = df["selected_candidate_id"] if "selected_candidate_id" in df.columns else df["candidate_id"]
    route_family = (
        df["selected_route_family"] if "selected_route_family" in df.columns else df["route_family"]
    )
    tsfm_mask = route_family.map(is_tsfm_family)
    return {
        "policy_id": policy_id,
        "selection_mode": selection_mode,
        "windows": int(len(df)),
        "cells": int(df["cell_id"].nunique()),
        "selected_route_count": int(selected.nunique()),
        "tsfm_window_share": float(tsfm_mask.mean()) if len(df) else float("nan"),
        "ordinary_wape_all": aggregate_wape(df),
        "decision_weighted_wape_all": aggregate_wape(df, "decision_weight"),
        "ordinary_wape_critical": aggregate_wape(critical) if len(critical) else float("nan"),
        "decision_weighted_wape_critical": aggregate_wape(critical, "decision_weight")
        if len(critical)
        else float("nan"),
        "critical_windows": int(len(critical)),
        "abs_error_sum": float(df["abs_error_sum"].sum()),
        "abs_true_sum": float(df["abs_true_sum"].sum()),
    }


def choose_validation_winners(
    candidates: pd.DataFrame,
    policy_id: str,
    eligible_mask: pd.Series,
) -> pd.DataFrame:
    eligible = candidates[eligible_mask].copy()
    if eligible.empty:
        raise ValueError(f"No eligible candidates for {policy_id}")
    selector = (
        eligible.groupby(["cell_id", "candidate_id"], as_index=False)
        .agg(
            selection_loss=("selection_loss", "mean"),
            route_family=("route_family", "first"),
            model_id=("model_id", "first"),
        )
        .sort_values(["cell_id", "selection_loss", "candidate_id"])
    )
    selected = selector.groupby("cell_id", as_index=False).first()[
        ["cell_id", "candidate_id", "route_family"]
    ].rename(
        columns={
            "candidate_id": "selected_candidate_id",
            "route_family": "selected_route_family",
        }
    )
    out = eligible.merge(
        selected,
        left_on=["cell_id", "candidate_id"],
        right_on=["cell_id", "selected_candidate_id"],
        how="inner",
    ).copy()
    out["policy_id"] = policy_id
    out["selection_mode"] = "validation_cell_best_reconstructed_from_candidate_artifacts"
    return out


def build_route_artifact_inventory(out_dir: Path) -> pd.DataFrame:
    queue = safe_read_csv(QUEUE_ROOT / "queue_selected.csv")
    metrics = safe_read_csv(P5_SUMMARY / "p5_main_all_metric_rows_raw.csv")
    rows = []
    test_full = metrics[(metrics["split"] == "test") & (metrics["metric_scope"] == "full_day")]
    val_full = metrics[
        (metrics["split"] == "validation") & (metrics["metric_scope"] == "full_day")
    ]
    for rec in queue.itertuples(index=False):
        lock_id = getattr(rec, "lock_id")
        test_rows = test_full[test_full["lock_id"] == lock_id]
        val_rows = val_full[val_full["lock_id"] == lock_id]
        test_metrics_file = str(test_rows["source_metrics_file"].iloc[0]) if len(test_rows) else ""
        test_prediction_path = ""
        test_prediction_exists = False
        if test_metrics_file:
            try:
                pred_path = formal_prediction_path(test_metrics_file)
                test_prediction_path = str(pred_path.relative_to(ROOT))
                test_prediction_exists = pred_path.exists()
            except FileNotFoundError:
                test_prediction_path = ""
                test_prediction_exists = False
        validation_metrics_file = getattr(rec, "source_validation_metrics_csv")
        validation_metrics_path = ROOT / str(validation_metrics_file)
        summary_manifest_file = getattr(rec, "source_summary_manifest")
        summary_manifest_path = ROOT / str(summary_manifest_file)
        rows.append(
            {
                "lock_id": lock_id,
                "portfolio_group": getattr(rec, "portfolio_group"),
                "model_id": getattr(rec, "model_id"),
                "route_id": getattr(rec, "route_id"),
                "domain": getattr(rec, "domain_id"),
                "horizon": getattr(rec, "horizon"),
                "candidate_role": getattr(rec, "p5_candidate_role"),
                "evidence_source_stage": getattr(rec, "evidence_source_stage"),
                "selected_config_or_variant": getattr(rec, "selected_config_or_variant"),
                "selected_feature_policy": getattr(rec, "selected_feature_policy"),
                "adaptation_kind": getattr(rec, "adaptation_kind"),
                "uses_cuda": bool(getattr(rec, "uses_cuda")),
                "uses_lightgbm": bool(getattr(rec, "uses_lightgbm")),
                "lightgbm_n_jobs_single_run": getattr(rec, "lightgbm_n_jobs_single_run"),
                "p5_epochs": getattr(rec, "p5_epochs"),
                "p5_batch_size": getattr(rec, "p5_batch_size"),
                "p5_eval_batch_size": getattr(rec, "p5_eval_batch_size"),
                "parameter_count_total": getattr(rec, "parameter_count_total"),
                "parameter_count_trainable": getattr(rec, "parameter_count_trainable"),
                "validation_wape_locked": getattr(rec, "validation_wape"),
                "test_wape_full_day": float(test_rows["wape"].iloc[0]) if len(test_rows) else float("nan"),
                "validation_metrics_path": str(validation_metrics_file),
                "validation_metrics_exists": validation_metrics_path.exists(),
                "test_metrics_path": test_metrics_file,
                "test_metrics_exists": bool(test_metrics_file and Path(test_metrics_file).exists()),
                "test_prediction_path": test_prediction_path,
                "test_prediction_exists": test_prediction_exists,
                "summary_manifest_path": str(summary_manifest_file),
                "summary_manifest_exists": summary_manifest_path.exists(),
                "valid_for_p5_formal_comparison": bool(
                    getattr(rec, "enter_p5_test_once_main")
                    and len(test_rows)
                    and test_prediction_exists
                ),
                "formal_status": "formal_test_completed"
                if len(test_rows) and test_prediction_exists
                else "formal_artifact_incomplete",
                "boundary": getattr(rec, "boundary"),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "route_prediction_artifact_inventory.csv", index=False)
    return out


def build_p4_ablation_outputs(out_dir: Path) -> dict[str, int]:
    inventory = build_route_artifact_inventory(out_dir)
    candidates = safe_read_csv(H2H3_ROOT / "p5_h2_candidate_window_metrics.csv")
    policies = safe_read_csv(H2H3_ROOT / "p5_h2_policy_selections.csv")
    base_windows = policies[policies["policy_id"] == "cell_validation_winner"][
        ["cell_id", "window_id"]
    ].drop_duplicates()
    candidates = candidates.merge(base_windows, on=["cell_id", "window_id"], how="inner")
    policy_rows: list[dict[str, object]] = []

    existing_policy_ids = [
        "cell_validation_winner",
        "tsfm_validation_winner",
        "timesfm2p5_validation_winner",
        "always_itransformer",
        "always_lightgbm",
        "always_nbeatsx",
        "always_chronos2_hidden_adapter",
        "always_timesfm2p5_xreg",
        "context_diagnostic_nn_loow",
        "context_diagnostic_nn_leave_cell_out",
        "oracle_per_window_best",
    ]
    for policy_id in existing_policy_ids:
        frame = policies[policies["policy_id"] == policy_id].copy()
        if frame.empty:
            continue
        mode = str(frame["selection_mode"].iloc[0]) if "selection_mode" in frame else "existing_policy"
        policy_rows.append(aggregate_policy_row(policy_id, frame, mode))

    tsfm_mask = candidates["route_family"].map(is_tsfm_family)
    non_tsfm = choose_validation_winners(candidates, "non_tsfm_validation_winner", ~tsfm_mask)
    policy_rows.append(
        aggregate_policy_row(
            "non_tsfm_validation_winner",
            non_tsfm,
            "validation_cell_best_non_tsfm_candidate_reconstructed",
        )
    )
    routing = pd.DataFrame(policy_rows).sort_values("ordinary_wape_all")
    best_fixed = routing[routing["policy_id"].isin(["always_itransformer", "always_lightgbm", "always_nbeatsx"])]
    if len(best_fixed):
        best_fixed_wape = float(best_fixed["ordinary_wape_all"].min())
        routing["relative_reduction_vs_best_fixed"] = (
            best_fixed_wape - routing["ordinary_wape_all"]
        ) / best_fixed_wape
    else:
        routing["relative_reduction_vs_best_fixed"] = float("nan")
    routing.to_csv(out_dir / "routing_composition_ablation.csv", index=False)

    labels = safe_read_csv(H2H3_ROOT / "p5_h3_test_window_labels.csv")[
        ["cell_id", "window_id", "critical_union", "decision_weight"]
    ]
    stress = []
    for policy_id in [
        "cell_validation_winner",
        "tsfm_validation_winner",
        "non_tsfm_validation_winner",
        "always_itransformer",
        "always_lightgbm",
        "always_nbeatsx",
    ]:
        frame = non_tsfm if policy_id == "non_tsfm_validation_winner" else policies[
            policies["policy_id"] == policy_id
        ].copy()
        if frame.empty:
            continue
        frame = frame.merge(labels, on=["cell_id", "window_id"], how="left")
        frame["decision_weight"] = pd.to_numeric(frame["decision_weight"], errors="coerce").fillna(1)
        frame["critical_union"] = frame["critical_union"].fillna(False).astype(bool)
        for window_scope, scoped in [
            ("all_windows", frame),
            ("critical_windows", frame[frame["critical_union"]]),
        ]:
            if scoped.empty:
                continue
            stress.append(
                {
                    "policy_id": policy_id,
                    "window_scope": window_scope,
                    "ordinary_wape": aggregate_wape(scoped),
                    "decision_weighted_wape": aggregate_wape(scoped, "decision_weight"),
                    "windows": int(len(scoped)),
                    "mean_decision_weight": float(scoped["decision_weight"].mean()),
                }
            )
    pd.DataFrame(stress).to_csv(out_dir / "stress_metric_ablation.csv", index=False)

    route_meta = inventory.rename(columns={"lock_id": "candidate_id"})
    route_group_cols = [
        "candidate_id",
        "route_family",
        "model_id",
        "route_id",
        "config_id",
        "domain",
        "horizon",
    ]
    route_metrics = (
        candidates.groupby(route_group_cols, as_index=False)
        .agg(
            windows=("window_id", "nunique"),
            abs_error_sum=("abs_error_sum", "sum"),
            abs_true_sum=("abs_true_sum", "sum"),
            mean_window_wape=("wape", "mean"),
            validation_selection_loss=("selection_loss", "mean"),
        )
        .merge(
            route_meta[
                [
                    "candidate_id",
                    "portfolio_group",
                    "selected_feature_policy",
                    "adaptation_kind",
                    "selected_config_or_variant",
                    "uses_cuda",
                    "uses_lightgbm",
                    "p5_epochs",
                    "p5_batch_size",
                    "parameter_count_total",
                    "parameter_count_trainable",
                    "valid_for_p5_formal_comparison",
                ]
            ],
            on="candidate_id",
            how="left",
        )
    )
    route_metrics["aggregate_wape"] = route_metrics["abs_error_sum"] / route_metrics["abs_true_sum"]
    route_metrics["is_tsfm"] = route_metrics["route_family"].map(is_tsfm_family)
    adaptation = route_metrics[route_metrics["is_tsfm"]].copy()

    expected_modes = pd.DataFrame(
        [
            {"model_id": "chronos2", "adaptation_kind": "zero_shot", "availability_status": "not_available_under_formal_artifact_audit"},
            {"model_id": "chronos2", "adaptation_kind": "frozen_backbone_hidden_adapter", "availability_status": "available"},
            {"model_id": "timesfm2p5", "adaptation_kind": "target_only_zero_shot", "availability_status": "not_available_under_formal_artifact_audit"},
            {"model_id": "timesfm2p5", "adaptation_kind": "xreg_covariate_adaptation_no_parameter_finetune", "availability_status": "available"},
            {"model_id": "timesfm2p5", "adaptation_kind": "peft_lora_target_only_custom_prefix_loss", "availability_status": "available"},
            {"model_id": "timesfm2p5", "adaptation_kind": "xreg_plus_lora", "availability_status": "not_available_under_formal_artifact_audit"},
            {"model_id": "tirex", "adaptation_kind": "any_formal_p5_route", "availability_status": "not_in_current_formal_p5_pool"},
            {"model_id": "sundial", "adaptation_kind": "any_formal_p5_route", "availability_status": "not_in_current_formal_p5_pool"},
        ]
    )
    adaptation["availability_status"] = "available_formal_candidate"
    adaptation = pd.concat([adaptation, expected_modes], ignore_index=True, sort=False)
    adaptation.to_csv(out_dir / "adaptation_mode_ablation.csv", index=False)

    covariate = route_metrics.copy()
    covariate["feature_policy_group"] = covariate["selected_feature_policy"].fillna("model_native")
    covariate.to_csv(out_dir / "covariate_ablation.csv", index=False)

    oracle = policies[policies["policy_id"] == "oracle_per_window_best"].copy()
    oracle_rows = []
    if len(oracle):
        group_cols = ["cell_id", "selected_candidate_id", "selected_route_family"]
        if "model_id" in oracle.columns:
            group_cols.append("model_id")
        total = len(oracle)
        for keys, group in oracle.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            rec = dict(zip(group_cols, keys))
            rec.update(
                {
                    "windows": int(len(group)),
                    "window_share_overall": float(len(group) / total),
                    "cell_window_share": float(len(group) / len(oracle[oracle["cell_id"] == rec["cell_id"]])),
                    "oracle_selected_wape": aggregate_wape(group),
                }
            )
            oracle_rows.append(rec)
    pd.DataFrame(oracle_rows).sort_values(
        ["cell_id", "windows"], ascending=[True, False]
    ).to_csv(out_dir / "oracle_route_distribution.csv", index=False)

    return {
        "inventory_rows": len(inventory),
        "routing_rows": len(routing),
        "adaptation_rows": len(adaptation),
        "covariate_rows": len(covariate),
        "oracle_rows": len(oracle_rows),
    }


def manifest_path_from_metrics(metrics_file: object) -> Path | None:
    if pd.isna(metrics_file):
        return None
    path = Path(str(metrics_file))
    if not path.is_absolute():
        path = ROOT / path
    parts = list(path.parts)
    if "cells" in parts:
        idx = parts.index("cells")
        return Path(*parts[:idx]) / "manifest.json"
    if path.name.endswith(".csv"):
        return path.parent.parent / "manifest.json"
    return None


def read_json_or_empty(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_p5_reproducibility_tables(out_dir: Path) -> dict[str, int]:
    inventory = pd.read_csv(out_dir / "route_prediction_artifact_inventory.csv")
    queue = safe_read_csv(QUEUE_ROOT / "queue_selected.csv")
    metrics = safe_read_csv(P5_SUMMARY / "p5_main_all_metric_rows_raw.csv")
    python_env_by_lock = queue.set_index("lock_id")["python_env"].to_dict()

    route_inventory = inventory[
        [
            "lock_id",
            "portfolio_group",
            "model_id",
            "route_id",
            "domain",
            "horizon",
            "selected_config_or_variant",
            "selected_feature_policy",
            "adaptation_kind",
            "parameter_count_total",
            "parameter_count_trainable",
            "uses_cuda",
            "uses_lightgbm",
            "p5_epochs",
            "p5_batch_size",
            "validation_wape_locked",
            "test_wape_full_day",
            "formal_status",
            "test_prediction_path",
        ]
    ].copy()
    route_inventory.to_csv(out_dir / "route_configuration_inventory_for_manuscript.csv", index=False)

    detail_cols = [
        "lock_id",
        "model_id",
        "domain_id",
        "horizon",
        "validation_metric_scope",
        "validation_wape",
        "selected_config_or_variant",
        "selected_feature_policy",
        "adaptation_kind",
        "p5_train_sample_policy",
        "source_seed",
        "p5_epochs",
        "p5_batch_size",
        "p5_eval_batch_size",
        "p5_learning_rate",
        "p5_weight_decay",
        "p5_gradient_clip",
        "p5_lora_rank",
        "p5_lora_alpha",
        "p5_lora_dropout",
        "p5_adapter_bottleneck",
        "p5_adapter_dropout",
        "lightgbm_n_jobs_single_run",
        "lightgbm_global_concurrent_n_jobs_budget",
        "full_train_or_refit_required_for_p5",
        "python_env",
        "p5_command_template",
    ]
    hyper = queue[[c for c in detail_cols if c in queue.columns]].copy()
    hyper["search_method"] = "validation_locked_pre_test_selection"
    hyper["search_budget"] = "see source_summary_manifest/source_validation_metrics_csv"
    hyper["test_once_status"] = "formal_test_completed_without_post_test_tuning"
    hyper.to_csv(out_dir / "hyperparameter_training_detail_table.csv", index=False)

    fairness_rows = []
    split_audit = pd.read_csv(out_dir / "split_window_audit_table.csv")
    cov_ledger = pd.read_csv(out_dir / "covariate_availability_ledger.csv")
    cov_counts = cov_ledger.groupby("domain")["covariate"].nunique().to_dict()
    for rec in route_inventory.itertuples(index=False):
        cell = split_audit[
            (split_audit["domain"].astype(str) == str(rec.domain))
            & (split_audit["horizon"].astype(str) == str(rec.horizon))
        ]
        context_steps = float(cell["context_steps_median"].iloc[0]) if len(cell) else float("nan")
        fairness_rows.append(
            {
                "domain": rec.domain,
                "horizon": rec.horizon,
                "model_id": rec.model_id,
                "route_id": rec.route_id,
                "target_history_context_steps": context_steps,
                "covariate_count_available_in_canonical_data": cov_counts.get(rec.domain, 0),
                "selected_feature_policy": rec.selected_feature_policy,
                "future_unknown_variables_excluded": True,
                "preprocessing_fit_scope": "train_only_for_scalers_imputers_feature_selection_and_thresholds",
                "test_window_contract": "same_frozen_test_windows_per_cell",
                "covariate_parity_note": (
                    "model_family_native_interface_under_common_origin_time_boundary"
                ),
            }
        )
    pd.DataFrame(fairness_rows).to_csv(out_dir / "baseline_fairness_feature_access_table.csv", index=False)

    test_full = metrics[(metrics["split"] == "test") & (metrics["metric_scope"] == "full_day")]
    env_rows = []
    for rec in route_inventory.itertuples(index=False):
        metric_row = test_full[test_full["lock_id"] == rec.lock_id]
        metrics_file = str(metric_row["source_metrics_file"].iloc[0]) if len(metric_row) else ""
        manifest_path = manifest_path_from_metrics(metrics_file)
        manifest = read_json_or_empty(manifest_path)
        cuda_meta = manifest.get("cuda_metadata_end") or {}
        package_versions = manifest.get("package_versions") or {}
        env_rows.append(
            {
                "lock_id": rec.lock_id,
                "model_id": rec.model_id,
                "route_id": rec.route_id,
                "domain": rec.domain,
                "horizon": rec.horizon,
                "python_env": python_env_by_lock.get(rec.lock_id, ""),
                "device": manifest.get("device", "cpu" if rec.uses_lightgbm else "cuda"),
                "cuda_available": manifest.get("cuda_available", cuda_meta.get("cuda_available")),
                "cuda_device_name": manifest.get("cuda_device_name", cuda_meta.get("cuda_device_name")),
                "seed": manifest.get("seed", ""),
                "manifest_path": str(manifest_path.relative_to(ROOT))
                if manifest_path and manifest_path.exists()
                else "",
                "metrics_path": metrics_file,
                "status": manifest.get("status", "unknown"),
                "package_versions_json": json.dumps(package_versions, sort_keys=True),
            }
        )
    pd.DataFrame(env_rows).to_csv(out_dir / "execution_environment_table.csv", index=False)

    artifact_map = pd.DataFrame(
        [
            {
                "manuscript_component": "Data provenance and split audit",
                "source_artifact": "dataset_provenance_table.csv; split_window_audit_table.csv",
            },
            {
                "manuscript_component": "H1 significance",
                "source_artifact": "h1_cell_bootstrap_ci.csv; dm_like_pvalues.csv",
            },
            {
                "manuscript_component": "H2/H3 policy and ablation",
                "source_artifact": "routing_composition_ablation.csv; stress_metric_ablation.csv",
            },
            {
                "manuscript_component": "Route reproducibility",
                "source_artifact": "route_configuration_inventory_for_manuscript.csv; hyperparameter_training_detail_table.csv",
            },
            {
                "manuscript_component": "Cost-accounting analysis",
                "source_artifact": "computational_cost_benchmark.csv; cost_accuracy_tradeoff.pdf",
            },
            {
                "manuscript_component": "H3 stress-weight sensitivity",
                "source_artifact": "h3_weight_formula.md; h3_threshold_sensitivity.csv",
            },
        ]
    )
    artifact_map.to_csv(out_dir / "artifact_to_manuscript_component_map.csv", index=False)

    release_manifest = {
        "status": "draft_release_plan_before_public_url",
        "release_boundary": (
            "The user has committed to opening complete code and processed data. "
            "Final GitHub/archival URLs must be inserted before submission."
        ),
        "required_release_items": [
            "code",
            "processed_canonical_datasets",
            "frozen_window_identifiers",
            "prediction_artifacts",
            "metric_scripts",
            "bootstrap_and_ablation_outputs",
            "figure_generation_scripts",
            "environment_files",
            "license_and_privacy_notes",
        ],
        "current_repair_artifact_root": str(out_dir.relative_to(ROOT)),
    }
    (out_dir / "open_source_release_package_manifest_draft.json").write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "route_inventory_rows": len(route_inventory),
        "hyperparameter_rows": len(hyper),
        "fairness_rows": len(fairness_rows),
        "environment_rows": len(env_rows),
    }


def build_p6_cost_outputs(out_dir: Path) -> dict[str, int]:
    inventory = pd.read_csv(out_dir / "route_prediction_artifact_inventory.csv")
    metrics = safe_read_csv(P5_SUMMARY / "p5_main_all_metric_rows_raw.csv")
    test_full = metrics[(metrics["split"] == "test") & (metrics["metric_scope"] == "full_day")]
    rows = []
    for rec in inventory.itertuples(index=False):
        metric_row = test_full[test_full["lock_id"] == rec.lock_id]
        if metric_row.empty:
            continue
        metric0 = metric_row.iloc[0]
        manifest_path = manifest_path_from_metrics(metric0["source_metrics_file"])
        manifest = read_json_or_empty(manifest_path)
        elapsed = manifest.get("elapsed_sec", manifest.get("runtime_sec", float("nan")))
        try:
            elapsed = float(elapsed)
        except Exception:
            elapsed = float("nan")
        n_windows = float(metric0.get("n_windows", float("nan")))
        vram = (
            manifest.get("max_cuda_memory_allocated_bytes")
            or manifest.get("cuda_max_memory_allocated_bytes")
            or (manifest.get("cuda_metadata_end") or {}).get("cuda_max_memory_allocated_bytes")
        )
        rows.append(
            {
                "lock_id": rec.lock_id,
                "model_id": rec.model_id,
                "portfolio_group": rec.portfolio_group,
                "domain": rec.domain,
                "horizon": rec.horizon,
                "measurement_type": "formal_run_manifest_accounting_not_pure_latency_benchmark",
                "device": manifest.get("device", "cpu" if rec.uses_lightgbm else "cuda"),
                "cuda_device_name": manifest.get(
                    "cuda_device_name",
                    (manifest.get("cuda_metadata_end") or {}).get("cuda_device_name"),
                ),
                "parameter_count_total": rec.parameter_count_total,
                "parameter_count_trainable": rec.parameter_count_trainable,
                "peak_vram_bytes": vram,
                "end_to_end_elapsed_sec": elapsed,
                "test_windows": n_windows,
                "throughput_windows_per_sec_proxy": n_windows / elapsed
                if elapsed and math.isfinite(elapsed)
                else float("nan"),
                "latency_ms_per_window_proxy": elapsed * 1000.0 / n_windows
                if n_windows and elapsed and math.isfinite(elapsed)
                else float("nan"),
                "lightgbm_cpu_threads": rec.lightgbm_n_jobs_single_run,
                "test_wape_full_day": rec.test_wape_full_day,
                "manifest_path": str(manifest_path.relative_to(ROOT))
                if manifest_path and manifest_path.exists()
                else "",
            }
        )
    cost = pd.DataFrame(rows)
    cost.to_csv(out_dir / "computational_cost_benchmark.csv", index=False)

    family = (
        cost.groupby(["model_id", "portfolio_group"], as_index=False)
        .agg(
            median_latency_ms_per_window_proxy=("latency_ms_per_window_proxy", "median"),
            median_elapsed_sec=("end_to_end_elapsed_sec", "median"),
            median_wape=("test_wape_full_day", "median"),
            median_peak_vram_bytes=("peak_vram_bytes", "median"),
            route_count=("lock_id", "nunique"),
        )
        .sort_values("median_wape")
    )
    family.to_csv(out_dir / "computational_cost_family_summary.csv", index=False)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        color_map = {
            "chronos2": "#4C78A8",
            "timesfm2p5": "#72B7B2",
            "itransformer": "#F58518",
            "nbeatsx": "#E45756",
            "lightgbm": "#54A24B",
        }
        for row in family.itertuples(index=False):
            ax.scatter(
                row.median_latency_ms_per_window_proxy,
                row.median_wape,
                s=90,
                color=color_map.get(row.model_id, "#777777"),
                edgecolor="black",
                linewidth=0.6,
                label=row.model_id,
            )
            ax.annotate(
                row.model_id,
                (row.median_latency_ms_per_window_proxy, row.median_wape),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Formal-run latency proxy (ms/window, log scale)")
        ax.set_ylabel("Median formal test WAPE")
        ax.set_title("Cost-accuracy accounting by model family")
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "cost_accuracy_tradeoff.pdf")
        fig.savefig(out_dir / "cost_accuracy_tradeoff.png", dpi=300)
        plt.close(fig)
    except Exception as exc:
        (out_dir / "cost_accuracy_tradeoff_plot_error.txt").write_text(str(exc), encoding="utf-8")
    return {"cost_rows": len(cost), "family_rows": len(family)}


def compute_sensitivity_labels(high_q: float, low_q: float, base_windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain in sorted(base_windows["domain"].unique()):
        for horizon in sorted(base_windows[base_windows["domain"].eq(domain)]["horizon"].unique()):
            path = ROOT / "data/energy_tsfm_windows_p1c" / str(domain) / f"window_index_{horizon}.parquet"
            index = pd.read_parquet(path)
            index["target_range"] = pd.to_numeric(index["target_max"], errors="coerce") - pd.to_numeric(
                index["target_min"], errors="coerce"
            )
            train = index[index["split"].astype(str).eq("train")].copy()
            wanted = base_windows[
                base_windows["domain"].eq(domain) & base_windows["horizon"].eq(horizon)
            ]["window_id"].astype(str)
            test = index[index["window_id"].astype(str).isin(set(wanted))].copy()
            th_peak = float(train["target_max"].quantile(high_q))
            th_ramp = float(train["target_range"].quantile(high_q))
            th_energy = float(train["target_sum"].quantile(high_q))
            th_low_pv = float(train["target_sum"].quantile(low_q)) if domain == "arena_pv" else float("nan")
            for item in test.itertuples(index=False):
                peak = float(item.target_max) >= th_peak
                ramp = float(item.target_range) >= th_ramp
                high_energy = float(item.target_sum) >= th_energy
                pv_low = bool(domain == "arena_pv" and float(item.target_sum) <= th_low_pv)
                score = int(peak) + int(ramp) + int(high_energy) + int(pv_low)
                rows.append(
                    {
                        "domain": domain,
                        "horizon": horizon,
                        "cell_id": f"{domain}::{horizon}",
                        "window_id": str(item.window_id),
                        "critical_union": bool(score > 0),
                        "decision_weight": float(1.0 + 2.0 * score),
                    }
                )
    return pd.DataFrame(rows)


def build_p7_h3_formula_and_sensitivity(out_dir: Path) -> dict[str, int]:
    formula = """# H3 train-defined stress-weight formula

The H3 metric uses train-window metadata only to define stress indicators. For
each domain-horizon cell, the thresholds are fitted on train windows, not on
validation or test windows.

For test window i, define four binary indicators when applicable:

- peak_i = 1[target_max_i >= train quantile q_high of target_max]
- ramp_i = 1[(target_max_i - target_min_i) >= train quantile q_high of target_range]
- energy_i = 1[target_sum_i >= train quantile q_high of target_sum]
- lowPV_i = 1[target_sum_i <= train quantile q_low of target_sum] for arena_pv only, otherwise 0

The stress score and weight are:

s_i = peak_i + ramp_i + energy_i + lowPV_i

w_i = 1 + 2 s_i

The stress-weighted WAPE is:

DWAPE(p, S) = sum_{i in S} w_i |y_i - yhat_i|_1 / sum_{i in S} w_i |y_i|_1

The default manuscript setting uses q_high = 0.90 and q_low = 0.20. This is an
operational stress proxy, not a downstream dispatch-cost model.
"""
    (out_dir / "h3_weight_formula.md").write_text(formula, encoding="utf-8")

    policies = safe_read_csv(H2H3_ROOT / "p5_h2_policy_selections.csv")
    base = policies[policies["policy_id"].eq("cell_validation_winner")][
        ["domain", "horizon", "cell_id", "window_id"]
    ].drop_duplicates()
    compare_ids = ["cell_validation_winner", "always_itransformer", "tsfm_validation_winner"]
    policy_base = policies[policies["policy_id"].isin(compare_ids)].copy()
    rows = []
    for high_q in [0.80, 0.85, 0.90, 0.95]:
        for low_q in [0.10, 0.15, 0.20, 0.25]:
            labels = compute_sensitivity_labels(high_q, low_q, base)
            labeled = policy_base.merge(labels, on=["cell_id", "window_id"], how="left")
            if labeled["decision_weight"].isna().any():
                raise ValueError("missing threshold-sensitivity H3 labels")
            for policy_id, group in labeled.groupby("policy_id", sort=False):
                critical = group[group["critical_union"].astype(bool)]
                rows.append(
                    {
                        "high_quantile": high_q,
                        "low_pv_quantile": low_q,
                        "policy_id": policy_id,
                        "windows": int(len(group)),
                        "critical_windows": int(len(critical)),
                        "critical_window_share": float(len(critical) / len(group)) if len(group) else float("nan"),
                        "dwape_all": aggregate_wape(group, "decision_weight"),
                        "dwape_critical": aggregate_wape(critical, "decision_weight")
                        if len(critical)
                        else float("nan"),
                    }
                )
    sens = pd.DataFrame(rows)
    primary = sens[sens["policy_id"].eq("cell_validation_winner")].copy()
    best_fixed = sens[sens["policy_id"].eq("always_itransformer")].copy()
    merged = primary.merge(
        best_fixed,
        on=["high_quantile", "low_pv_quantile"],
        suffixes=("_primary", "_best_fixed"),
    )
    merged["relative_reduction_dwape_all"] = (
        merged["dwape_all_best_fixed"] - merged["dwape_all_primary"]
    ) / merged["dwape_all_best_fixed"]
    merged["relative_reduction_dwape_critical"] = (
        merged["dwape_critical_best_fixed"] - merged["dwape_critical_primary"]
    ) / merged["dwape_critical_best_fixed"]
    sens.to_csv(out_dir / "h3_threshold_sensitivity_long.csv", index=False)
    merged.to_csv(out_dir / "h3_threshold_sensitivity.csv", index=False)
    try:
        import matplotlib.pyplot as plt

        pivot = merged.pivot(
            index="high_quantile",
            columns="low_pv_quantile",
            values="relative_reduction_dwape_critical",
        )
        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{r:.2f}" for r in pivot.index])
        ax.set_xlabel("Low-PV quantile")
        ax.set_ylabel("High-stress quantile")
        ax.set_title("H3 critical-window DWAPE reduction sensitivity")
        for y in range(pivot.shape[0]):
            for x in range(pivot.shape[1]):
                ax.text(x, y, f"{pivot.values[y, x]*100:.1f}%", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, label="Relative reduction vs iTransformer")
        fig.tight_layout()
        fig.savefig(out_dir / "h3_threshold_sensitivity_figure.pdf")
        fig.savefig(out_dir / "h3_threshold_sensitivity_figure.png", dpi=300)
        plt.close(fig)
    except Exception as exc:
        (out_dir / "h3_threshold_sensitivity_plot_error.txt").write_text(str(exc), encoding="utf-8")
    return {"sensitivity_rows": len(merged), "long_rows": len(sens)}


def combine_dm_outputs(out_dir: Path) -> None:
    frames = []
    for path in sorted(out_dir.glob("_*_dm_like_pvalues.csv")):
        frames.append(pd.read_csv(path))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out_dir / "dm_like_pvalues.csv", index=False)


def build_manifest(out_dir: Path, args: argparse.Namespace) -> None:
    files = sorted(p.name for p in out_dir.glob("*.csv"))
    manifest = {
        "created_by": "scripts/build_manuscript_repair_audit_and_statistics.py",
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "input_artifacts": {
            "candidate_window_metrics": str(
                (H2H3_ROOT / "p5_h2_candidate_window_metrics.csv").relative_to(ROOT)
            ),
            "policy_selections": str(
                (H2H3_ROOT / "p5_h2_policy_selections.csv").relative_to(ROOT)
            ),
            "h3_test_window_labels": str(
                (H2H3_ROOT / "p5_h3_test_window_labels.csv").relative_to(ROOT)
            ),
            "queue_selected": str((QUEUE_ROOT / "queue_selected.csv").relative_to(ROOT)),
        },
        "outputs": files,
        "test_once_boundary": (
            "This script uses existing formal test-once artifacts only; it does not "
            "rerun completed rows or tune any model on test outcomes."
        ),
    }
    (out_dir / "audit_and_statistics_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    ensure_out(out_dir)
    build_dataset_audits(out_dir)
    h1 = build_h1_statistics(out_dir, args.n_bootstrap, args.seed)
    h2 = build_h2_statistics(out_dir, args.n_bootstrap, args.seed)
    h3 = build_h3_statistics(out_dir, args.n_bootstrap, args.seed)
    p4 = build_p4_ablation_outputs(out_dir)
    p5 = build_p5_reproducibility_tables(out_dir)
    p6 = build_p6_cost_outputs(out_dir)
    p7 = build_p7_h3_formula_and_sensitivity(out_dir)
    combine_dm_outputs(out_dir)
    build_manifest(out_dir, args)
    print(
        json.dumps(
            {
                "status": "ok",
                "out_dir": str(out_dir.relative_to(ROOT)),
                "h1_rows": len(h1),
                "h2_rows": len(h2),
                "h3_rows": len(h3),
                "p4_rows": p4,
                "p5_rows": p5,
                "p6_rows": p6,
                "p7_rows": p7,
                "n_bootstrap": args.n_bootstrap,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
