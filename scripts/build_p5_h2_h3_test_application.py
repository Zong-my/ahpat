#!/usr/bin/env python3
"""Apply locked H2/H3 protocols to the P5 formal test-once predictions.

This script is post-P5 analysis only. It reads generated P5 validation/test
predictions and frozen P1c window metadata, but it does not train, fine-tune,
rerun, or tune any base forecasting model.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from energy_tsfm_p2_core import load_canonical  # noqa: E402


PLAN_ID = "p5_h2_h3_test_application_v0_codex_20260519"
OUT_DIR = PROJECT / "results" / "energy_tsfm_p5_main" / PLAN_ID
DOC_OUT = PROJECT / "results"
STATUS_DOC = DOC_OUT / "p5_h2_h3_test_application_status.md"

P5_SUMMARY_ROOT = (
    PROJECT
    / "results"
    / "energy_tsfm_p5_main"
    / "p5_main_test_once_v0_codex_20260517_summary_codex_20260518"
)
P5_PRIMARY_METRICS = P5_SUMMARY_ROOT / "p5_main_locked_primary_metrics.csv"
P5_MAIN_SUMMARY = P5_SUMMARY_ROOT / "p5_main_result_summary.json"
P1C_ROOT = PROJECT / "data" / "energy_tsfm_windows_p1c"

H2_LOCK = (
    PROJECT
    / "data"
    / "energy_tsfm_tuning"
    / "p3_4ba_p5_pre_final_lock_manifest_v0_codex_20260516"
    / "p3_4ba_h2_policy_lock_table.csv"
)
H3_LOCK = (
    PROJECT
    / "data"
    / "energy_tsfm_tuning"
    / "p3_4ba_p5_pre_final_lock_manifest_v0_codex_20260516"
    / "p3_4ba_h3_metric_lock_table.csv"
)

CANDIDATE_WINDOW_METRICS = OUT_DIR / "p5_h2_candidate_window_metrics.csv"
ROUTE_FAMILY_WINDOW_METRICS = OUT_DIR / "p5_h2_full_coverage_route_family_window_metrics.csv"
WINDOW_FEATURES = OUT_DIR / "p5_h2_selector_window_features.csv"
POLICY_SELECTIONS = OUT_DIR / "p5_h2_policy_selections.csv"
POLICY_METRICS = OUT_DIR / "p5_h2_policy_metrics.csv"
PER_CELL_METRICS = OUT_DIR / "p5_h2_policy_per_cell_metrics.csv"
THRESHOLD_LEDGER = OUT_DIR / "p5_h3_train_threshold_ledger.csv"
TEST_WINDOW_LABELS = OUT_DIR / "p5_h3_test_window_labels.csv"
CANDIDATE_H3_METRICS = OUT_DIR / "p5_h3_candidate_decision_weighted_metrics.csv"
POLICY_H3_METRICS = OUT_DIR / "p5_h3_policy_decision_weighted_metrics.csv"
MANIFEST = OUT_DIR / "p5_h2_h3_test_application_manifest.json"

EXPECTED_DOMAINS = [
    "provincial_load",
    "aluminum_load",
    "microgrid_load",
    "arena_pv",
    "aidc_power_optional",
]
EXPECTED_HORIZONS = ["4h", "24h"]
FULL_COVERAGE_ROUTE_FAMILIES = [
    "lightgbm",
    "nbeatsx",
    "itransformer",
    "chronos2_hidden_adapter",
    "timesfm2p5_xreg",
]
SCOPES = [
    "all_windows",
    "critical_union",
    "peak_q90",
    "ramp_range_q90",
    "high_energy_sum_q90",
    "pv_low_generation_q20",
]
BASE_FEATURE_COLUMNS = [
    "log_abs_context_mean",
    "context_cv",
    "context_zero_fraction",
    "context_recent_ramp_ratio",
    "context_peak_concentration",
    "context_last_ratio",
    "origin_hour_sin",
    "origin_hour_cos",
    "origin_dow_sin",
    "origin_dow_cos",
    "horizon_hours",
    "native_step_minutes",
]
EPS = 1e-9
TRUTH_ABS_TOL = 1.0
TRUTH_REL_TOL = 1e-7


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def as_array(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D array, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("array contains non-finite values")
    return arr


def truth_hash(values: np.ndarray) -> str:
    rounded = [round(float(v), 10) for v in values]
    payload = json.dumps(rounded, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def route_family(row: pd.Series) -> str:
    model = str(row["lock_model_id"])
    route = str(row["lock_route_id"])
    if model in {"lightgbm", "nbeatsx", "itransformer"}:
        return model
    if model == "chronos2":
        return "chronos2_hidden_adapter"
    if model == "timesfm2p5" and "lora" in route:
        return "timesfm2p5_lora"
    if model == "timesfm2p5" and "xreg" in route:
        return "timesfm2p5_xreg"
    return f"{model}:{route}"


def find_prediction_file(output_root: Path, split: str, family: str) -> Path:
    if family in {"lightgbm", "nbeatsx", "itransformer"}:
        pattern = f"**/predictions/{split}_predictions.parquet"
    elif family == "timesfm2p5_xreg":
        pattern = f"**/predictions/{split}_predictions_all_variants.parquet"
    else:
        pattern = f"**/predictions/{split}_predictions_all_arms.parquet"
    matches = sorted(p for p in output_root.glob(pattern) if p.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {split} prediction for {output_root} / {family}, got {len(matches)}")
    return matches[0]


def load_selected_prediction_rows(metric_row: pd.Series) -> pd.DataFrame:
    split = str(metric_row["split"])
    family = str(metric_row["route_family"])
    path = find_prediction_file(Path(metric_row["output_root"]), split, family)
    df = pd.read_parquet(path)
    df = df[
        df["domain_id"].astype(str).eq(str(metric_row["domain_id"]))
        & df["horizon"].astype(str).eq(str(metric_row["horizon"]))
        & df["config_id"].astype(str).eq(str(metric_row["config_id"]))
    ].copy()
    if "split" in df.columns:
        df = df[df["split"].astype(str).eq(split)].copy()
    if len(df) != int(metric_row["n_windows"]):
        raise ValueError(
            f"{metric_row['lock_id']} {split} selected prediction row count {len(df)} "
            f"!= metric n_windows {metric_row['n_windows']}"
        )
    df["prediction_path"] = rel(path)
    return df


def build_candidate_window_metrics(primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = primary[primary["split"].isin(["validation", "test"])].copy()
    for item in selected.itertuples(index=False):
        pred = load_selected_prediction_rows(pd.Series(item._asdict()))
        for prow in pred.itertuples(index=False):
            y_true = as_array(prow.y_true)
            y_pred = as_array(prow.y_pred)
            if len(y_true) != len(y_pred):
                raise ValueError(f"{item.lock_id}/{prow.window_id} prediction length mismatch")
            err = y_pred - y_true
            abs_error_sum = float(np.abs(err).sum())
            abs_true_sum = float(np.abs(y_true).sum())
            zero_truth_window = abs_true_sum <= EPS
            rows.append(
                {
                    "candidate_id": str(item.lock_id),
                    "route_family": str(item.route_family),
                    "model_id": str(item.lock_model_id),
                    "route_id": str(item.lock_route_id),
                    "config_id": str(item.config_id),
                    "domain": str(item.domain_id),
                    "horizon": str(item.horizon),
                    "cell_id": f"{item.domain_id}::{item.horizon}",
                    "window_id": str(prow.window_id),
                    "series_id": str(prow.series_id),
                    "segment_id": str(prow.segment_id),
                    "origin_time": str(prow.origin_time),
                    "target_start_time": str(prow.target_start_time),
                    "target_end_time": str(prow.target_end_time),
                    "split": str(item.split),
                    "y_true_len": int(len(y_true)),
                    "truth_hash": truth_hash(y_true),
                    "abs_error_sum": abs_error_sum,
                    "abs_true_sum": abs_true_sum,
                    "zero_truth_window": bool(zero_truth_window),
                    "wape": np.nan if zero_truth_window else abs_error_sum / abs_true_sum,
                    "selection_loss": abs_error_sum if zero_truth_window else abs_error_sum / abs_true_sum,
                    "mae": float(np.abs(err).mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(err)))),
                    "bias": float(err.mean()),
                    "source_prediction_path": str(pred["prediction_path"].iloc[0]),
                }
            )
    return pd.DataFrame(rows)


def full_coverage_family_metrics(candidate: pd.DataFrame) -> pd.DataFrame:
    full = candidate[candidate["route_family"].isin(FULL_COVERAGE_ROUTE_FAMILIES)].copy()
    counts = full.groupby(["split", "domain", "horizon", "route_family"])["window_id"].nunique().reset_index()
    expected = full.groupby(["split", "domain", "horizon"])["window_id"].nunique().reset_index(name="expected")
    audit = counts.merge(expected, on=["split", "domain", "horizon"], how="left")
    bad = audit[audit["window_id"] != audit["expected"]]
    if not bad.empty:
        raise ValueError(f"full coverage route family mismatch:\n{bad.head(20)}")
    return full


def build_thresholds() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for domain in EXPECTED_DOMAINS:
        for horizon in EXPECTED_HORIZONS:
            df = pd.read_parquet(P1C_ROOT / domain / f"window_index_{horizon}.parquet")
            train = df[df["split"].astype(str).eq("train")].copy()
            if train.empty:
                raise ValueError(f"empty train window index for {domain}/{horizon}")
            train["target_range"] = pd.to_numeric(train["target_max"], errors="raise") - pd.to_numeric(
                train["target_min"], errors="raise"
            )
            rows.append(
                {
                    "domain": domain,
                    "horizon": horizon,
                    "threshold_split": "train",
                    "train_window_count": int(len(train)),
                    "peak_target_max_q90": float(train["target_max"].quantile(0.90)),
                    "ramp_target_range_q90": float(train["target_range"].quantile(0.90)),
                    "high_energy_target_sum_q90": float(train["target_sum"].quantile(0.90)),
                    "low_generation_target_sum_q20": float(train["target_sum"].quantile(0.20))
                    if domain == "arena_pv"
                    else np.nan,
                    "threshold_protocol": "train_quantiles_only_no_validation_or_test_threshold_fit",
                }
            )
    return pd.DataFrame(rows)


def build_window_labels(thresholds: pd.DataFrame, candidate: pd.DataFrame, split: str) -> pd.DataFrame:
    target_windows = candidate[candidate["split"].eq(split)][["domain", "horizon", "window_id"]].drop_duplicates()
    threshold_lookup = {
        (str(row.domain), str(row.horizon)): row for row in thresholds.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for domain in EXPECTED_DOMAINS:
        for horizon in EXPECTED_HORIZONS:
            wanted = set(
                target_windows[
                    target_windows["domain"].astype(str).eq(domain)
                    & target_windows["horizon"].astype(str).eq(horizon)
                ]["window_id"].astype(str)
            )
            index = pd.read_parquet(P1C_ROOT / domain / f"window_index_{horizon}.parquet")
            index = index[index["window_id"].astype(str).isin(wanted)].copy()
            if len(index) != len(wanted):
                raise ValueError(f"{domain}/{horizon}/{split} label count mismatch")
            th = threshold_lookup[(domain, horizon)]
            index["target_range"] = pd.to_numeric(index["target_max"], errors="raise") - pd.to_numeric(
                index["target_min"], errors="raise"
            )
            for item in index.sort_values("window_id", kind="mergesort").itertuples(index=False):
                peak = float(item.target_max) >= float(th.peak_target_max_q90)
                ramp = float(item.target_range) >= float(th.ramp_target_range_q90)
                high_energy = float(item.target_sum) >= float(th.high_energy_target_sum_q90)
                pv_low = bool(domain == "arena_pv" and float(item.target_sum) <= float(th.low_generation_target_sum_q20))
                critical = bool(peak or ramp or high_energy or pv_low)
                decision_weight = 1.0 + (2.0 if peak else 0.0) + (2.0 if ramp else 0.0) + (
                    2.0 if high_energy else 0.0
                ) + (2.0 if pv_low else 0.0)
                rows.append(
                    {
                        "domain": domain,
                        "horizon": horizon,
                        "cell_id": f"{domain}::{horizon}",
                        "window_id": str(item.window_id),
                        "split": split,
                        "target_max": float(item.target_max),
                        "target_min": float(item.target_min),
                        "target_sum": float(item.target_sum),
                        "target_range": float(item.target_range),
                        "positive_target_share": float(item.positive_target_share),
                        "peak_q90": bool(peak),
                        "ramp_range_q90": bool(ramp),
                        "high_energy_sum_q90": bool(high_energy),
                        "pv_low_generation_q20": bool(pv_low),
                        "critical_union": critical,
                        "decision_weight": float(decision_weight),
                        "threshold_source": "train_window_metadata_only",
                    }
                )
    return pd.DataFrame(rows)


def compute_context_features(candidate: pd.DataFrame) -> pd.DataFrame:
    target = candidate[["domain", "horizon", "window_id", "split"]].drop_duplicates()
    rows: list[pd.DataFrame] = []
    for domain in EXPECTED_DOMAINS:
        canonical = load_canonical(domain)
        segments = {
            str(seg_id): seg.sort_values("segment_row_index", kind="mergesort").reset_index(drop=True)
            for seg_id, seg in canonical.groupby("segment_id", sort=False)
        }
        for horizon in EXPECTED_HORIZONS:
            wanted = target[
                target["domain"].astype(str).eq(domain) & target["horizon"].astype(str).eq(horizon)
            ]["window_id"].astype(str)
            if wanted.empty:
                continue
            index = pd.read_parquet(P1C_ROOT / domain / f"window_index_{horizon}.parquet")
            index = index[index["window_id"].astype(str).isin(set(wanted))].copy()
            if len(index) != len(set(wanted)):
                raise ValueError(f"{domain}/{horizon} context feature count mismatch")
            parts: list[pd.DataFrame] = []
            for segment_id, win in index.groupby("segment_id", sort=False):
                seg = segments[str(segment_id)]
                values = pd.to_numeric(seg["target"], errors="raise").to_numpy(dtype=float)
                abs_values = np.abs(values)
                zero_values = (values == 0.0).astype(float)
                cs = np.concatenate([[0.0], np.cumsum(values)])
                css = np.concatenate([[0.0], np.cumsum(values * values)])
                cabs = np.concatenate([[0.0], np.cumsum(abs_values)])
                czero = np.concatenate([[0.0], np.cumsum(zero_values)])
                block = win.copy()
                start = block["context_start_row_index"].to_numpy(dtype=int)
                end = block["context_end_row_index"].to_numpy(dtype=int)
                n = block["context_steps"].to_numpy(dtype=float)
                sums = cs[end + 1] - cs[start]
                sq_sums = css[end + 1] - css[start]
                abs_sums = cabs[end + 1] - cabs[start]
                zero_sums = czero[end + 1] - czero[start]
                mean = sums / np.maximum(n, 1.0)
                var = np.maximum(sq_sums / np.maximum(n, 1.0) - mean * mean, 0.0)
                std = np.sqrt(var)
                first = values[start]
                last = values[end]
                max_abs = np.empty(len(block), dtype=float)
                for steps, idxs in block.groupby("context_steps").groups.items():
                    rolling = pd.Series(abs_values).rolling(window=int(steps), min_periods=int(steps)).max().to_numpy()
                    pos = block.index.get_indexer(idxs)
                    max_abs[pos] = rolling[end[pos]]
                origin = pd.to_datetime(block["context_end_timestamp"], errors="raise")
                hour = origin.dt.hour + origin.dt.minute / 60.0
                dow = origin.dt.dayofweek
                out = pd.DataFrame(
                    {
                        "domain": domain,
                        "horizon": horizon,
                        "cell_id": f"{domain}::{horizon}",
                        "window_id": block["window_id"].astype(str).to_numpy(),
                        "split": block["split"].astype(str).to_numpy(),
                        "series_id": block["series_id"].astype(str).to_numpy(),
                        "segment_id": block["segment_id"].astype(str).to_numpy(),
                        "origin_time": block["context_end_timestamp"].astype(str).to_numpy(),
                        "horizon_hours": float(str(horizon).removesuffix("h")),
                        "native_step_minutes": pd.to_numeric(block["native_step_minutes"], errors="raise").to_numpy(
                            dtype=float
                        ),
                        "context_mean": mean,
                        "context_std": std,
                        "log_abs_context_mean": np.log1p(np.abs(mean)),
                        "context_cv": std / np.maximum(np.abs(mean), EPS),
                        "context_zero_fraction": zero_sums / np.maximum(n, 1.0),
                        "context_recent_ramp_ratio": (last - first) / np.maximum(np.abs(mean), EPS),
                        "context_peak_concentration": max_abs / np.maximum(abs_sums, EPS),
                        "context_last_ratio": last / np.maximum(np.abs(mean), EPS),
                        "origin_hour_sin": np.sin(2.0 * math.pi * hour / 24.0),
                        "origin_hour_cos": np.cos(2.0 * math.pi * hour / 24.0),
                        "origin_dow_sin": np.sin(2.0 * math.pi * dow / 7.0),
                        "origin_dow_cos": np.cos(2.0 * math.pi * dow / 7.0),
                        "feature_availability": "origin_time_context_only",
                        "future_target_columns_used": False,
                    }
                )
                parts.append(out)
            rows.append(pd.concat(parts, ignore_index=True))
    features = pd.concat(rows, ignore_index=True)
    return features.sort_values(["split", "domain", "horizon", "window_id"], kind="mergesort").reset_index(drop=True)


def best_by_cell(metrics: pd.DataFrame, *, id_col: str, split: str, families: set[str] | None = None) -> dict[tuple[str, str], str]:
    data = metrics[metrics["split"].eq(split)].copy()
    if families is not None:
        data = data[data["route_family"].isin(families)].copy()
    grouped = data.groupby(["domain", "horizon", id_col], as_index=False).agg(
        abs_error_sum=("abs_error_sum", "sum"),
        abs_true_sum=("abs_true_sum", "sum"),
    )
    grouped["cell_wape"] = grouped["abs_error_sum"] / grouped["abs_true_sum"].clip(lower=EPS)
    return {
        (str(row.domain), str(row.horizon)): str(getattr(row, id_col))
        for row in grouped.sort_values(["domain", "horizon", "cell_wape", id_col], kind="mergesort")
        .groupby(["domain", "horizon"], as_index=False)
        .first()
        .itertuples(index=False)
    }


def best_by_window(metrics: pd.DataFrame, *, id_col: str, split: str, families: set[str] | None = None) -> pd.DataFrame:
    data = metrics[metrics["split"].eq(split)].copy()
    if families is not None:
        data = data[data["route_family"].isin(families)].copy()
    return (
        data.sort_values(["domain", "horizon", "window_id", "selection_loss", id_col], kind="mergesort")
        .groupby(["domain", "horizon", "window_id"], as_index=False)
        .first()[["domain", "horizon", "window_id", id_col, "selection_loss"]]
        .rename(columns={id_col: f"best_{id_col}", "selection_loss": "best_selection_loss"})
    )


def feature_matrix(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    base = features[["domain", "horizon", "window_id", "cell_id", "split", *BASE_FEATURE_COLUMNS]].copy()
    domain_dummies = pd.get_dummies(features["domain"].astype(str), prefix="domain", dtype=float)
    horizon_dummies = pd.get_dummies(features["horizon"].astype(str), prefix="horizon", dtype=float)
    out = pd.concat([base, domain_dummies, horizon_dummies], axis=1)
    return out, BASE_FEATURE_COLUMNS + list(domain_dummies.columns) + list(horizon_dummies.columns)


def train_validation_nn_policy(
    features: pd.DataFrame,
    validation_labels: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    *,
    leave_same_cell_out: bool,
    k: int = 7,
) -> dict[tuple[str, str, str], str]:
    matrix, feature_cols = feature_matrix(features)
    train_all = matrix[matrix["split"].eq("validation")].copy()
    test_all = matrix[matrix["split"].eq("test")].copy()
    label_map = {
        (str(row.domain), str(row.horizon), str(row.window_id)): str(row.best_route_family)
        for row in validation_labels.itertuples(index=False)
    }
    loss_lookup = {
        (str(row.domain), str(row.horizon), str(row.window_id), str(row.route_family)): float(row.selection_loss)
        for row in validation_metrics.itertuples(index=False)
    }
    predictions: dict[tuple[str, str, str], str] = {}
    for cell_id, test_cell in test_all.groupby("cell_id", sort=False):
        if leave_same_cell_out:
            train = train_all[train_all["cell_id"].astype(str).ne(str(cell_id))].copy()
        else:
            train = train_all
        if train.empty:
            for row in test_cell.itertuples(index=False):
                predictions[(str(row.domain), str(row.horizon), str(row.window_id))] = FULL_COVERAGE_ROUTE_FAMILIES[0]
            continue
        train_x = train[feature_cols].to_numpy(dtype=float)
        test_x = test_cell[feature_cols].to_numpy(dtype=float)
        mean = train_x.mean(axis=0)
        std = train_x.std(axis=0)
        std[std < EPS] = 1.0
        train_z = (train_x - mean) / std
        test_z = (test_x - mean) / std
        nn = NearestNeighbors(n_neighbors=min(k, len(train)), algorithm="auto")
        nn.fit(train_z)
        _, indices = nn.kneighbors(test_z)
        train_records = train.reset_index(drop=True)
        for row_idx, row in enumerate(test_cell.itertuples(index=False)):
            votes: dict[str, int] = {}
            mean_loss: dict[str, float] = {}
            for train_idx in indices[row_idx]:
                nrow = train_records.iloc[int(train_idx)]
                key = (str(nrow["domain"]), str(nrow["horizon"]), str(nrow["window_id"]))
                route = label_map[key]
                votes[route] = votes.get(route, 0) + 1
                mean_loss[route] = mean_loss.get(route, 0.0) + loss_lookup[(key[0], key[1], key[2], route)]
            for route in list(mean_loss):
                mean_loss[route] /= votes[route]
            selected = sorted(
                votes,
                key=lambda route: (
                    -votes[route],
                    mean_loss[route],
                    FULL_COVERAGE_ROUTE_FAMILIES.index(route)
                    if route in FULL_COVERAGE_ROUTE_FAMILIES
                    else len(FULL_COVERAGE_ROUTE_FAMILIES),
                ),
            )[0]
            predictions[(str(row.domain), str(row.horizon), str(row.window_id))] = selected
    return predictions


def build_policy_selections(candidate: pd.DataFrame, full_family: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    val_all = candidate[candidate["split"].eq("validation")].copy()
    test_all = candidate[candidate["split"].eq("test")].copy()
    val_full = full_family[full_family["split"].eq("validation")].copy()
    test_windows = test_all[["domain", "horizon", "cell_id", "window_id"]].drop_duplicates()

    cell_best_candidate = best_by_cell(candidate, id_col="candidate_id", split="validation")
    tsfm_best_candidate = best_by_cell(
        candidate,
        id_col="candidate_id",
        split="validation",
        families={"chronos2_hidden_adapter", "timesfm2p5_xreg", "timesfm2p5_lora"},
    )
    timesfm_best_candidate = best_by_cell(
        candidate,
        id_col="candidate_id",
        split="validation",
        families={"timesfm2p5_xreg", "timesfm2p5_lora"},
    )
    full_family_best = best_by_window(full_family, id_col="route_family", split="validation")
    nn_all = train_validation_nn_policy(
        features,
        full_family_best.rename(columns={"best_route_family": "best_route_family"}),
        val_full,
        leave_same_cell_out=False,
    )
    nn_lco = train_validation_nn_policy(
        features,
        full_family_best.rename(columns={"best_route_family": "best_route_family"}),
        val_full,
        leave_same_cell_out=True,
    )
    test_oracle = best_by_window(candidate, id_col="candidate_id", split="test")

    by_candidate = {
        (str(row.domain), str(row.horizon), str(row.window_id), str(row.candidate_id)): row
        for row in test_all.itertuples(index=False)
    }
    by_family = {
        (str(row.domain), str(row.horizon), str(row.window_id), str(row.route_family)): row
        for row in full_family[full_family["split"].eq("test")].itertuples(index=False)
    }
    oracle_lookup = {
        (str(row.domain), str(row.horizon), str(row.window_id)): str(row.best_candidate_id)
        for row in test_oracle.itertuples(index=False)
    }

    rows: list[dict[str, Any]] = []

    def append_selected(base: dict[str, Any], policy_id: str, selected_id: str, *, id_kind: str, deployable: bool, oracle: bool, mode: str) -> None:
        if id_kind == "candidate":
            metric = by_candidate[(base["domain"], base["horizon"], base["window_id"], selected_id)]
            selected_candidate = selected_id
            selected_family = str(metric.route_family)
        else:
            metric = by_family[(base["domain"], base["horizon"], base["window_id"], selected_id)]
            selected_candidate = str(metric.candidate_id)
            selected_family = selected_id
        rows.append(
            {
                **base,
                "policy_id": policy_id,
                "selected_candidate_id": selected_candidate,
                "selected_route_family": selected_family,
                "deployable_h2_candidate": deployable,
                "oracle_upper_bound": oracle,
                "selection_mode": mode,
                "candidate_id": str(metric.candidate_id),
                "route_family": str(metric.route_family),
                "model_id": str(metric.model_id),
                "route_id": str(metric.route_id),
                "config_id": str(metric.config_id),
                "series_id": str(metric.series_id),
                "segment_id": str(metric.segment_id),
                "origin_time": str(metric.origin_time),
                "target_start_time": str(metric.target_start_time),
                "target_end_time": str(metric.target_end_time),
                "y_true_len": int(metric.y_true_len),
                "truth_hash": str(metric.truth_hash),
                "abs_error_sum": float(metric.abs_error_sum),
                "abs_true_sum": float(metric.abs_true_sum),
                "zero_truth_window": bool(metric.zero_truth_window),
                "wape": float(metric.wape) if pd.notna(metric.wape) else np.nan,
                "selection_loss": float(metric.selection_loss),
                "mae": float(metric.mae),
                "rmse": float(metric.rmse),
                "bias": float(metric.bias),
            }
        )

    for item in test_windows.sort_values(["domain", "horizon", "window_id"], kind="mergesort").itertuples(index=False):
        base = {
            "domain": str(item.domain),
            "horizon": str(item.horizon),
            "cell_id": str(item.cell_id),
            "window_id": str(item.window_id),
            "split": "test",
        }
        cell_key = (base["domain"], base["horizon"])
        win_key = (base["domain"], base["horizon"], base["window_id"])
        for family in FULL_COVERAGE_ROUTE_FAMILIES:
            append_selected(
                base,
                f"always_{family}",
                family,
                id_kind="family",
                deployable=True,
                oracle=False,
                mode="fixed_full_coverage_route_family",
            )
        append_selected(
            base,
            "timesfm2p5_validation_winner",
            timesfm_best_candidate[cell_key],
            id_kind="candidate",
            deployable=True,
            oracle=False,
            mode="validation_cell_best_timesfm2p5_candidate",
        )
        append_selected(
            base,
            "tsfm_validation_winner",
            tsfm_best_candidate[cell_key],
            id_kind="candidate",
            deployable=True,
            oracle=False,
            mode="validation_cell_best_tsfm_candidate",
        )
        append_selected(
            base,
            "cell_validation_winner",
            cell_best_candidate[cell_key],
            id_kind="candidate",
            deployable=True,
            oracle=False,
            mode="validation_cell_best_all_locked_candidates",
        )
        append_selected(
            base,
            "context_diagnostic_nn_loow",
            nn_all[win_key],
            id_kind="family",
            deployable=True,
            oracle=False,
            mode="validation_trained_knn_full_coverage_families",
        )
        append_selected(
            base,
            "context_diagnostic_nn_leave_cell_out",
            nn_lco[win_key],
            id_kind="family",
            deployable=True,
            oracle=False,
            mode="validation_trained_knn_leave_same_cell_out_full_coverage_families",
        )
        append_selected(
            base,
            "oracle_per_window_best",
            oracle_lookup[win_key],
            id_kind="candidate",
            deployable=False,
            oracle=True,
            mode="realized_test_error_oracle_upper_bound",
        )
    return pd.DataFrame(rows)


def aggregate_policy_metrics(selections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = selections.groupby(
        ["policy_id", "deployable_h2_candidate", "oracle_upper_bound", "selection_mode"], as_index=False
    ).agg(
        windows=("window_id", "nunique"),
        cells=("cell_id", "nunique"),
        abs_error_sum=("abs_error_sum", "sum"),
        abs_true_sum=("abs_true_sum", "sum"),
        mean_window_wape=("wape", "mean"),
        median_window_wape=("wape", "median"),
        mean_selection_loss=("selection_loss", "mean"),
        mean_mae=("mae", "mean"),
        mean_rmse=("rmse", "mean"),
    )
    grouped["aggregate_wape"] = grouped["abs_error_sum"] / grouped["abs_true_sum"].clip(lower=EPS)
    oracle_wape = float(grouped.loc[grouped["policy_id"].eq("oracle_per_window_best"), "aggregate_wape"].iloc[0])
    deployable = grouped[grouped["deployable_h2_candidate"].astype(bool)]
    best_deployable = float(deployable["aggregate_wape"].min())
    grouped["regret_to_oracle_wape"] = grouped["aggregate_wape"] - oracle_wape
    grouped["gap_to_best_deployable_wape"] = grouped["aggregate_wape"] - best_deployable
    grouped = grouped.sort_values(["aggregate_wape", "policy_id"], kind="mergesort").reset_index(drop=True)
    grouped.insert(0, "diagnostic_rank_by_aggregate_wape", np.arange(1, len(grouped) + 1))

    per_cell = selections.groupby(["policy_id", "domain", "horizon", "cell_id"], as_index=False).agg(
        windows=("window_id", "nunique"),
        abs_error_sum=("abs_error_sum", "sum"),
        abs_true_sum=("abs_true_sum", "sum"),
        mean_window_wape=("wape", "mean"),
    )
    per_cell["aggregate_wape"] = per_cell["abs_error_sum"] / per_cell["abs_true_sum"].clip(lower=EPS)
    return grouped, per_cell


def scope_mask(df: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "all_windows":
        return pd.Series(True, index=df.index)
    if scope not in df.columns:
        raise ValueError(scope)
    return df[scope].astype(bool)


def aggregate_h3(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        scoped = df[scope_mask(df, scope)].copy()
        if scoped.empty:
            continue
        for key, group in scoped.groupby(group_col, sort=False):
            weighted_abs_error = float((group["decision_weight"] * group["abs_error_sum"]).sum())
            weighted_abs_true = float((group["decision_weight"] * group["abs_true_sum"]).sum())
            abs_error = float(group["abs_error_sum"].sum())
            abs_true = float(group["abs_true_sum"].sum())
            rows.append(
                {
                    group_col: str(key),
                    "metric_scope": scope,
                    "windows": int(group["window_id"].nunique()),
                    "cells": int(group["cell_id"].nunique()),
                    "critical_window_share": float(group["critical_union"].astype(bool).mean()),
                    "mean_decision_weight": float(group["decision_weight"].mean()),
                    "abs_error_sum": abs_error,
                    "abs_true_sum": abs_true,
                    "wape": abs_error / abs_true if abs_true > EPS else np.nan,
                    "decision_weighted_abs_error_sum": weighted_abs_error,
                    "decision_weighted_abs_true_sum": weighted_abs_true,
                    "decision_weighted_wape": weighted_abs_error / weighted_abs_true if weighted_abs_true > EPS else np.nan,
                    "mean_window_wape": float(group["wape"].mean()),
                    "zero_truth_windows": int(group["zero_truth_window"].astype(bool).sum()),
                }
            )
    return pd.DataFrame(rows)


def rank_summary(metrics: pd.DataFrame, id_col: str) -> dict[str, Any]:
    all_scope = metrics[metrics["metric_scope"].astype(str).eq("all_windows")].copy()
    avg = all_scope.sort_values(["wape", id_col], kind="mergesort").iloc[0]
    weighted = all_scope.sort_values(["decision_weighted_wape", id_col], kind="mergesort").iloc[0]
    critical = metrics[metrics["metric_scope"].astype(str).eq("critical_union")].copy()
    critical_best = critical.sort_values(["decision_weighted_wape", id_col], kind="mergesort").iloc[0]
    return {
        "best_by_average_wape": {
            id_col: str(avg[id_col]),
            "wape": float(avg["wape"]),
            "decision_weighted_wape": float(avg["decision_weighted_wape"]),
        },
        "best_by_decision_weighted_wape": {
            id_col: str(weighted[id_col]),
            "wape": float(weighted["wape"]),
            "decision_weighted_wape": float(weighted["decision_weighted_wape"]),
        },
        "best_on_critical_union": {
            id_col: str(critical_best[id_col]),
            "decision_weighted_wape": float(critical_best["decision_weighted_wape"]),
            "windows": int(critical_best["windows"]),
        },
        "ranking_changes_under_decision_weight": str(avg[id_col]) != str(weighted[id_col]),
    }


def write_status(payload: dict[str, Any]) -> None:
    h2_primary = payload["h2_primary_policy_result"]
    h2_best = payload["h2_best_deployable_policy"]
    h2_best_locked = payload["h2_best_p3_4ba_locked_deployable_policy"]
    h3_policy = payload["h3_policy_rank_summary"]["best_by_decision_weighted_wape"]
    h3_critical = payload["h3_policy_rank_summary"]["best_on_critical_union"]
    STATUS_DOC.write_text(
        "\n".join(
            [
                "# P5 H2/H3 Test Application",
                "",
                f"- Status: `{payload['status']}`",
                "- Boundary: reads completed P5 predictions and frozen P1c metadata only; no model rerun, no post-test tuning.",
                f"- Plan ID: `{PLAN_ID}`",
                f"- Output root: `{rel(OUT_DIR)}`",
                "",
                "## H2 Test Application",
                "",
                f"- Primary deployable policy: `cell_validation_winner`.",
                f"- Locked primary aggregate test WAPE: `cell_validation_winner` = `{h2_primary['aggregate_wape']:.6f}`.",
                f"- Best P3-4ba locked deployable policy: `{h2_best_locked['policy_id']}` = `{h2_best_locked['aggregate_wape']:.6f}`.",
                f"- Best secondary/exploratory deployable policy: `{h2_best['policy_id']}` = `{h2_best['aggregate_wape']:.6f}`.",
                f"- Oracle upper bound WAPE: `{payload['h2_oracle_policy']['aggregate_wape']:.6f}`.",
                "- Diagnostic NN policies are reported as sensitivity and use validation-trained full-coverage route-family labels.",
                "",
                "## H3 Test Application",
                "",
                f"- Critical-union test windows: `{payload['critical_union_windows']}` / `{payload['test_windows']}`.",
                f"- Best policy by all-window decision-weighted WAPE: `{h3_policy['policy_id']}` = `{h3_policy['decision_weighted_wape']:.6f}`.",
                f"- Best policy on critical-union decision-weighted WAPE: `{h3_critical['policy_id']}` = `{h3_critical['decision_weighted_wape']:.6f}`.",
                f"- Policy ranking changes under decision weighting: `{str(payload['h3_policy_rank_summary']['ranking_changes_under_decision_weight']).lower()}`.",
                "",
                "## Artifacts",
                "",
                f"- manifest: `{rel(MANIFEST)}`",
                f"- H2 candidate window metrics: `{rel(CANDIDATE_WINDOW_METRICS)}`",
                f"- H2 policy selections: `{rel(POLICY_SELECTIONS)}`",
                f"- H2 policy metrics: `{rel(POLICY_METRICS)}`",
                f"- H3 train threshold ledger: `{rel(THRESHOLD_LEDGER)}`",
                f"- H3 test labels: `{rel(TEST_WINDOW_LABELS)}`",
                f"- H3 candidate weighted metrics: `{rel(CANDIDATE_H3_METRICS)}`",
                f"- H3 policy weighted metrics: `{rel(POLICY_H3_METRICS)}`",
                "",
                "## Guardrail",
                "",
                "Use these outputs for manuscript analysis only. They do not license additional test-based model selection or rerunning completed P5 rows.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    primary = pd.read_csv(P5_PRIMARY_METRICS)
    primary["route_family"] = primary.apply(route_family, axis=1)

    candidate = build_candidate_window_metrics(primary)
    candidate.to_csv(CANDIDATE_WINDOW_METRICS, index=False)

    truth_audit = candidate.groupby(["split", "domain", "horizon", "window_id"]).agg(
        truth_hash_count=("truth_hash", "nunique"),
        y_true_len_count=("y_true_len", "nunique"),
        abs_true_min=("abs_true_sum", "min"),
        abs_true_max=("abs_true_sum", "max"),
    )
    truth_audit["abs_true_diff"] = truth_audit["abs_true_max"] - truth_audit["abs_true_min"]
    truth_audit["abs_true_rel_diff"] = truth_audit["abs_true_diff"] / truth_audit["abs_true_max"].clip(lower=EPS)
    bad_truth = truth_audit[
        (truth_audit["y_true_len_count"] != 1)
        | (
            (truth_audit["abs_true_diff"] > TRUTH_ABS_TOL)
            & (truth_audit["abs_true_rel_diff"] > TRUTH_REL_TOL)
        )
    ]
    if not bad_truth.empty:
        raise ValueError(f"truth values are not tolerance-aligned across candidates:\n{bad_truth.head(20)}")

    full_family = full_coverage_family_metrics(candidate)
    full_family.to_csv(ROUTE_FAMILY_WINDOW_METRICS, index=False)

    thresholds = build_thresholds()
    thresholds.to_csv(THRESHOLD_LEDGER, index=False)
    test_labels = build_window_labels(thresholds, candidate, split="test")
    test_labels.to_csv(TEST_WINDOW_LABELS, index=False)

    features = compute_context_features(candidate)
    features.to_csv(WINDOW_FEATURES, index=False)

    selections = build_policy_selections(candidate, full_family, features)
    selections.to_csv(POLICY_SELECTIONS, index=False)
    h2_policy, h2_per_cell = aggregate_policy_metrics(selections)
    h2_policy.to_csv(POLICY_METRICS, index=False)
    h2_per_cell.to_csv(PER_CELL_METRICS, index=False)

    candidate_h3_input = candidate[candidate["split"].eq("test")].merge(
        test_labels,
        on=["domain", "horizon", "cell_id", "window_id", "split"],
        how="left",
        validate="many_to_one",
    )
    if candidate_h3_input["decision_weight"].isna().any():
        raise ValueError("candidate H3 label merge failed")
    candidate_h3 = aggregate_h3(candidate_h3_input, "candidate_id")
    candidate_h3.to_csv(CANDIDATE_H3_METRICS, index=False)

    policy_h3_input = selections.merge(
        test_labels,
        on=["domain", "horizon", "cell_id", "window_id", "split"],
        how="left",
        validate="many_to_one",
    )
    if policy_h3_input["decision_weight"].isna().any():
        raise ValueError("policy H3 label merge failed")
    policy_h3 = aggregate_h3(policy_h3_input, "policy_id")
    policy_h3.to_csv(POLICY_H3_METRICS, index=False)

    h2_best = (
        h2_policy[h2_policy["deployable_h2_candidate"].astype(bool)]
        .sort_values(["aggregate_wape", "policy_id"], kind="mergesort")
        .iloc[0]
        .to_dict()
    )
    p3_4ba_locked_policy_ids = {
        "cell_validation_winner",
        "context_diagnostic_nn_loow",
        "context_diagnostic_nn_leave_cell_out",
        "always_timesfm2p5_xreg",
    }
    h2_best_locked = (
        h2_policy[
            h2_policy["deployable_h2_candidate"].astype(bool)
            & h2_policy["policy_id"].astype(str).isin(p3_4ba_locked_policy_ids)
        ]
        .sort_values(["aggregate_wape", "policy_id"], kind="mergesort")
        .iloc[0]
        .to_dict()
    )
    h2_primary = h2_policy[h2_policy["policy_id"].eq("cell_validation_winner")].iloc[0].to_dict()
    h2_oracle = h2_policy[h2_policy["policy_id"].eq("oracle_per_window_best")].iloc[0].to_dict()
    payload = {
        "status": "ok",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_id": PLAN_ID,
        "output_root": rel(OUT_DIR),
        "p5_main_summary": rel(P5_MAIN_SUMMARY),
        "p5_primary_metrics": rel(P5_PRIMARY_METRICS),
        "h2_lock": rel(H2_LOCK),
        "h3_lock": rel(H3_LOCK),
        "boundary": "post-P5 analysis only; no base-model training, fine-tuning, inference rerun, or test tuning",
        "candidate_window_metric_rows": int(len(candidate)),
        "test_windows": int(test_labels["window_id"].nunique()),
        "validation_windows": int(features[features["split"].eq("validation")]["window_id"].nunique()),
        "truth_hashes_byte_identical_across_candidates": bool(
            int(truth_audit["truth_hash_count"].max()) == 1
        ),
        "truth_values_tolerance_aligned_across_candidates": True,
        "truth_alignment_abs_tolerance": TRUTH_ABS_TOL,
        "truth_alignment_rel_tolerance": TRUTH_REL_TOL,
        "truth_alignment_max_abs_true_diff": float(truth_audit["abs_true_diff"].max()),
        "truth_alignment_max_abs_true_rel_diff": float(truth_audit["abs_true_rel_diff"].max()),
        "full_coverage_route_families": FULL_COVERAGE_ROUTE_FAMILIES,
        "h2_primary_policy": "cell_validation_winner",
        "h2_primary_policy_result": {
            "policy_id": str(h2_primary["policy_id"]),
            "aggregate_wape": float(h2_primary["aggregate_wape"]),
            "regret_to_oracle_wape": float(h2_primary["regret_to_oracle_wape"]),
            "gap_to_best_deployable_wape": float(h2_primary["gap_to_best_deployable_wape"]),
        },
        "h2_best_deployable_policy": {
            "policy_id": str(h2_best["policy_id"]),
            "aggregate_wape": float(h2_best["aggregate_wape"]),
            "boundary": "includes secondary/exploratory validation-defined policies beyond the original P3-4ba H2 lock table",
        },
        "h2_best_p3_4ba_locked_deployable_policy": {
            "policy_id": str(h2_best_locked["policy_id"]),
            "aggregate_wape": float(h2_best_locked["aggregate_wape"]),
            "locked_policy_ids_considered": sorted(p3_4ba_locked_policy_ids),
        },
        "h2_oracle_policy": {
            "policy_id": str(h2_oracle["policy_id"]),
            "aggregate_wape": float(h2_oracle["aggregate_wape"]),
        },
        "critical_union_windows": int(test_labels["critical_union"].astype(bool).sum()),
        "peak_windows": int(test_labels["peak_q90"].astype(bool).sum()),
        "ramp_windows": int(test_labels["ramp_range_q90"].astype(bool).sum()),
        "high_energy_windows": int(test_labels["high_energy_sum_q90"].astype(bool).sum()),
        "pv_low_generation_windows": int(test_labels["pv_low_generation_q20"].astype(bool).sum()),
        "h3_candidate_rank_summary": rank_summary(candidate_h3, "candidate_id"),
        "h3_policy_rank_summary": rank_summary(policy_h3, "policy_id"),
        "artifacts": {
            "candidate_window_metrics": rel(CANDIDATE_WINDOW_METRICS),
            "full_coverage_route_family_window_metrics": rel(ROUTE_FAMILY_WINDOW_METRICS),
            "selector_window_features": rel(WINDOW_FEATURES),
            "policy_selections": rel(POLICY_SELECTIONS),
            "h2_policy_metrics": rel(POLICY_METRICS),
            "h2_per_cell_metrics": rel(PER_CELL_METRICS),
            "threshold_ledger": rel(THRESHOLD_LEDGER),
            "test_window_labels": rel(TEST_WINDOW_LABELS),
            "h3_candidate_metrics": rel(CANDIDATE_H3_METRICS),
            "h3_policy_metrics": rel(POLICY_H3_METRICS),
            "status_doc": rel(STATUS_DOC),
        },
        "notes": [
            "cell_validation_winner uses validation aggregate performance from generated P5 validation predictions, not test metrics.",
            "context diagnostic NN policies are sensitivity outputs over full-coverage route families.",
            "oracle_per_window_best is a non-deployable upper bound computed from realized test errors.",
            "H3 thresholds are fitted from train window metadata only and applied to test labels.",
        ],
    }
    MANIFEST.write_text(dumps(payload), encoding="utf-8")
    write_status(payload)
    print(dumps(payload))


if __name__ == "__main__":
    main()
