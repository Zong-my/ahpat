#!/usr/bin/env python3
"""Evaluate common-schema predictions for the energy-TSFM experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from energy_tsfm_p2_core import load_window_index, validate_prediction_against_windows


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT / "results" / "energy_tsfm_p2_smoke" / "evaluation"
EPS = 1e-12


def parse_json_array(value: Any) -> np.ndarray:
    if isinstance(value, str):
        parsed = json.loads(value)
    elif isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        parsed = value
    else:
        raise TypeError(f"unsupported serialized array value: {type(value)!r}")
    return np.array([np.nan if v is None else float(v) for v in parsed], dtype=float)


def collect_prediction_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            combined = path / "predictions_all.parquet"
            if combined.exists():
                out.append(combined)
            else:
                out.extend(sorted(path.rglob("*.parquet")))
                out.extend(sorted(path.rglob("*.csv")))
        elif path.exists():
            out.append(path)
        else:
            raise FileNotFoundError(path)
    if not out:
        raise ValueError("no prediction files found")
    return out


def read_prediction_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported prediction file suffix: {path}")
    df = df.copy()
    df["prediction_source"] = str(path)
    return df


def read_predictions(paths: list[Path]) -> pd.DataFrame:
    frames = [read_prediction_table(path) for path in collect_prediction_paths(paths)]
    return pd.concat(frames, ignore_index=True)


def attach_window_metadata(predictions: pd.DataFrame) -> pd.DataFrame:
    groups: list[pd.DataFrame] = []
    for (domain, horizon), group in predictions.groupby(["domain_id", "horizon"], sort=False):
        index = load_window_index(str(domain), str(horizon))
        validate_prediction_against_windows(
            group.drop(columns=["prediction_source", "split"], errors="ignore"),
            index,
        )
        metric_cols = [c for c in index.columns if c.startswith("metric_")]
        keep_cols = ["window_id", "split", *metric_cols]
        merge_group = group.rename(columns={"split": "prediction_split"}) if "split" in group.columns else group
        merged = merge_group.merge(index[keep_cols], on="window_id", how="left", validate="many_to_one")
        if merged["split"].isna().any():
            raise ValueError(f"{domain}/{horizon}: failed to attach split for some predictions")
        if "prediction_split" in merged.columns:
            mismatch = (
                merged["prediction_split"].notna()
                & merged["prediction_split"].astype(str).ne(merged["split"].astype(str))
            )
            if mismatch.any():
                bad = merged.loc[mismatch, ["window_id", "prediction_split", "split"]].head()
                raise ValueError(
                    f"{domain}/{horizon}: prediction split disagrees with window index:\n"
                    + bad.to_string(index=False)
                )
            merged = merged.drop(columns=["prediction_split"])
        groups.append(merged)
    return pd.concat(groups, ignore_index=True)


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | None]:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    if len(y_true) == 0:
        return {
            "n_points": 0,
            "mae": None,
            "rmse": None,
            "wape": None,
            "smape": None,
            "r2": None,
            "bias": None,
        }

    err = y_pred - y_true
    abs_err = np.abs(err)
    denom_abs = float(np.sum(np.abs(y_true)))
    sse = float(np.sum(err**2))
    centered = y_true - float(np.mean(y_true))
    sst = float(np.sum(centered**2))
    smape_denom = np.abs(y_true) + np.abs(y_pred)
    smape_terms = np.where(smape_denom > EPS, 2.0 * abs_err / smape_denom, 0.0)

    return {
        "n_points": int(len(y_true)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(math.sqrt(np.mean(err**2))),
        "wape": None if denom_abs <= EPS else float(np.sum(abs_err) / denom_abs),
        "smape": float(np.mean(smape_terms)),
        "r2": None if sst <= EPS else float(1.0 - sse / sst),
        "bias": None if denom_abs <= EPS else float(np.sum(err) / denom_abs),
    }


def available_metric_scopes(group: pd.DataFrame) -> dict[str, pd.Series]:
    scopes = {"full_day": pd.Series(True, index=group.index)}
    for col in sorted(c for c in group.columns if c.startswith("metric_")):
        scope = col.removeprefix("metric_")
        if scope == "full_day":
            continue
        values = group[col].fillna(False).astype(bool)
        if values.any():
            scopes[scope] = values
    return scopes


def evaluate_prediction_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    attached = attach_window_metadata(predictions)
    rows: list[dict[str, Any]] = []
    group_cols = ["domain_id", "horizon", "split", "model_family", "model_id", "config_id"]
    for keys, group in attached.groupby(group_cols, dropna=False, sort=True):
        key_record = dict(zip(group_cols, keys, strict=True))
        for scope, mask in available_metric_scopes(group).items():
            scoped = group.loc[mask]
            if scoped.empty:
                continue
            true_arrays = [parse_json_array(v) for v in scoped["y_true"]]
            pred_arrays = [parse_json_array(v) for v in scoped["y_pred"]]
            for window_id, y_true, y_pred in zip(scoped["window_id"], true_arrays, pred_arrays, strict=True):
                if len(y_true) != len(y_pred):
                    raise ValueError(f"{window_id}: y_true/y_pred length mismatch")
            y_true_all = np.concatenate(true_arrays)
            y_pred_all = np.concatenate(pred_arrays)
            rows.append(
                {
                    **key_record,
                    "metric_scope": scope,
                    "n_windows": int(len(scoped)),
                    **metric_values(y_true_all, y_pred_all),
                }
            )
    return pd.DataFrame(rows)


def write_metrics(metrics: pd.DataFrame, output_dir: Path, stem: str = "metrics") -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    metrics.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(metrics.to_dict(orient="records"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"csv": str(csv_path), "json": str(json_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+", type=Path, help="Prediction parquet/csv files or directories.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = read_predictions(args.predictions)
    metrics = evaluate_prediction_frame(predictions)
    paths = write_metrics(metrics, args.output_dir, stem=args.stem)
    print(json.dumps({"status": "ok", "rows": int(len(metrics)), **paths}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
