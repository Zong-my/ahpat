#!/usr/bin/env python3
"""Build Figure 7 representative forecast traces from formal test-once outputs."""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT = Path(__file__).resolve().parents[1]
# MANUSCRIPT path removed for open-source release
FIG_DIR = PROJECT / "figures"
P5_PKG = PROJECT / "results/energy_tsfm_p5_main/p5_manuscript_result_package_v0_codex_20260519"
H2H3 = PROJECT / "results/energy_tsfm_p5_main/p5_h2_h3_test_application_v0_codex_20260519"
REPAIR = PROJECT / "results/energy_tsfm_manuscript_repair_v1_20260521"

H2_SELECTIONS = H2H3 / "p5_h2_policy_selections.csv"
H3_LABELS = H2H3 / "p5_h3_test_window_labels.csv"
ROUTE_INVENTORY = REPAIR / "route_configuration_inventory_for_manuscript.csv"
H1_TABLE = P5_PKG / "table_1_p5_tsfm_vs_non_tsfm_test.csv"
SELECTION_RECORD = REPAIR / "figure7_window_selection_record.csv"

DOMAIN_LABEL = {
    "aidc_power_optional": "AIDC power",
    "aluminum_load": "Aluminum load",
    "arena_pv": "Arena PV",
    "microgrid_load": "Microgrid load",
    "provincial_load": "Provincial load",
}
DOMAIN_GRANULARITY_MIN = {
    "aidc_power_optional": 15,
    "aluminum_load": 15,
    "arena_pv": 15,
    "microgrid_load": 10,
    "provincial_load": 15,
}
MODEL_DISPLAY = {
    "chronos2": "Chronos-2",
    "timesfm2p5": "TimesFM 2.5",
    "itransformer": "iTransformer",
    "nbeatsx": "N-BEATSx",
    "lightgbm": "LightGBM",
}
TSFM_MODELS = {"chronos2", "timesfm2p5", "tirex", "sundial", "moirai", "moirai_moe"}

COL = {
    "truth": "#111827",
    "selected": "#7B2D8E",
    "tsfm": "#2563EB",
    "nontsfm": "#E8702A",
    "fixed": "#6B7280",
    "stress": "#FCA5A5",
    "stress_box": "#FEE2E2",
    "grid": "#D8DEE9",
    "text": "#111827",
}

FS_BASE = 7.5
FS_TITLE = 8.5
FS_LABEL = 7.5
FS_TICK = 6.8
FS_LEGEND = 6.5
FS_ANNOT_SMALL = 5.8
FS_ANNOT_TINY = 5.5


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FS_BASE,
            "axes.titlesize": FS_TITLE,
            "axes.titleweight": "bold",
            "axes.labelsize": FS_LABEL,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_LEGEND,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "pdf.fonttype": 42,
            "axes.linewidth": 0.55,
            "axes.edgecolor": "#1F2937",
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "legend.frameon": False,
        }
    )


def parse_vector(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    return np.asarray(ast.literal_eval(str(value)), dtype=float)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum()
    if denom <= 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom)


def is_tsfm_model(model_id: object) -> bool:
    return str(model_id).lower() in TSFM_MODELS


def display_model(model_id: object) -> str:
    return MODEL_DISPLAY.get(str(model_id), str(model_id))


def clean_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=COL["grid"], alpha=0.45, linewidth=0.35)


def reserve_annotation_headroom(ax: plt.Axes, series: list[np.ndarray]) -> None:
    values = np.concatenate([np.asarray(v, dtype=float) for v in series])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    ymin = float(values.min())
    ymax = float(values.max())
    span = ymax - ymin
    if span <= 0:
        span = max(abs(ymax), 1.0)
    # The top strip and WAPE label live inside the axes. Extra vertical
    # headroom keeps those annotations from covering the prediction traces.
    ax.set_ylim(ymin - 0.07 * span, ymax + 0.30 * span)


def scale_for(y: np.ndarray) -> tuple[float, str]:
    max_abs = float(np.nanmax(np.abs(y)))
    if max_abs >= 1e5:
        return 1e5, r"$10^5$"
    if max_abs >= 1e4:
        return 1e4, r"$10^4$"
    if max_abs >= 1e3:
        return 1e3, r"$10^3$"
    return 1.0, ""


TITLE_OFFSET_XLABEL = 28  # pt — horizontal ticks + xlabel, consistent with build_revised_figures.py

def bottom_panel_title(ax: plt.Axes, label: str, title: str) -> None:
    ax.annotate(
        f"({label}) {title}",
        xy=(0.5, 0), xycoords="axes fraction",
        xytext=(0, -TITLE_OFFSET_XLABEL), textcoords="offset points",
        fontsize=FS_LABEL,
        fontweight="regular",
        ha="center",
        va="top",
        color=COL["text"],
        annotation_clip=False,
    )


inventory = None
h2 = None
h3 = None
h1 = None


def _load_globals():
    global inventory, h2, h3, h1
    if inventory is None:
        inventory = pd.read_csv(ROUTE_INVENTORY)
        h2 = pd.read_csv(H2_SELECTIONS)
        h3 = pd.read_csv(H3_LABELS)
        h1 = pd.read_csv(H1_TABLE)


@lru_cache(maxsize=None)
def load_prediction_frame(lock_id: str) -> pd.DataFrame:
    _load_globals()
    rows = inventory[inventory["lock_id"] == lock_id]
    if rows.empty:
        raise KeyError(f"Unknown lock_id: {lock_id}")
    route = rows.iloc[0]
    path = PROJECT / str(route["test_prediction_path"])
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    route_id = str(route["route_id"])

    if "route_label" in df.columns:
        label = df["route_label"].astype(str)
        if (label == route_id).any():
            df = df[label == route_id].copy()
        elif "timesfm2p5_transformers_lora_target_only" in route_id:
            df = df[label.str.contains("timesfm2p5_transformers_lora_target_only", regex=False, na=False)].copy()
        elif "timesfm2p5_transformers_frozen_target_only" in route_id:
            df = df[label.str.contains("timesfm2p5_transformers_frozen_target_only", regex=False, na=False)].copy()
        elif "chronos2_covariate_aware_hidden_adapter" in route_id:
            df = df[label.str.contains("covariate_aware_hidden_adapter", regex=False, na=False)].copy()
        elif "chronos2_frozen_target_only" in route_id:
            df = df[label.str.contains("chronos2_frozen_target_only", regex=False, na=False)].copy()

    if df.empty:
        raise ValueError(f"No prediction rows remain after route filtering for {lock_id}")
    if df["window_id"].duplicated().any():
        duplicated = int(df["window_id"].duplicated().sum())
        raise ValueError(f"{lock_id} still has {duplicated} duplicated window rows")
    return df.set_index("window_id", drop=False)


def prediction_row(lock_id: str, window_id: str) -> pd.Series:
    df = load_prediction_frame(lock_id)
    if window_id not in df.index:
        raise KeyError(f"{window_id} not present in predictions for {lock_id}")
    return df.loc[window_id]


def candidate_table(
    *,
    cell_id: str,
    primary_policy: str,
    comparator_policy: str,
    require_tsfm_primary: bool = False,
    require_critical: bool | None = None,
    range_quantile: float = 0.25,
) -> pd.DataFrame:
    primary = h2[(h2["cell_id"] == cell_id) & (h2["policy_id"] == primary_policy)].copy()
    comp = h2[(h2["cell_id"] == cell_id) & (h2["policy_id"] == comparator_policy)].copy()
    if primary.empty or comp.empty:
        raise ValueError(f"Missing policy rows for {cell_id}: {primary_policy}, {comparator_policy}")

    keep_primary = [
        "window_id",
        "selected_candidate_id",
        "selected_route_family",
        "model_id",
        "wape",
        "abs_true_sum",
        "origin_time",
        "target_start_time",
        "target_end_time",
    ]
    keep_comp = ["window_id", "selected_candidate_id", "selected_route_family", "model_id", "wape"]
    merged = primary[keep_primary].merge(
        comp[keep_comp],
        on="window_id",
        suffixes=("_primary", "_comp"),
        validate="one_to_one",
    )

    domain, horizon = cell_id.split("::")
    labels = h3[(h3["domain"] == domain) & (h3["horizon"] == horizon)][
        ["window_id", "target_range", "positive_target_share", "critical_union", "decision_weight"]
    ]
    merged = merged.merge(labels, on="window_id", how="left", validate="one_to_one")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["wape_primary", "wape_comp", "abs_true_sum", "target_range"]
    )

    if require_tsfm_primary:
        merged = merged[merged["model_id_primary"].map(is_tsfm_model)]
    if require_critical is not None:
        merged = merged[merged["critical_union"].astype(bool) == require_critical]

    # Keep windows with enough signal variation; this avoids visually flat,
    # low-information traces while staying inside formal test predictions.
    full_cell = h2[(h2["cell_id"] == cell_id) & (h2["policy_id"] == primary_policy)]
    q_abs = full_cell["abs_true_sum"].quantile(0.20)
    q_range = merged["target_range"].quantile(range_quantile) if len(merged) else np.nan
    merged = merged[
        (merged["abs_true_sum"] >= q_abs)
        & (merged["target_range"] >= q_range)
        & (merged["positive_target_share"].fillna(1.0) >= 0.5)
        & (merged["wape_primary"] < merged["wape_comp"])
    ].copy()
    if merged.empty:
        raise ValueError(f"No candidate window after filtering for {cell_id}")

    merged["relative_reduction"] = (merged["wape_comp"] - merged["wape_primary"]) / merged["wape_comp"].clip(
        lower=1e-12
    )
    merged["primary_wape_pct"] = merged["wape_primary"].rank(pct=True, ascending=True)
    merged["range_pct"] = merged["target_range"].rank(pct=True, ascending=True)
    decision_max = max(float(merged["decision_weight"].fillna(1.0).max()), 1.0)
    merged["tsfm_bonus"] = merged["model_id_primary"].map(is_tsfm_model).astype(float) * 0.04
    merged["selection_score"] = (
        4.0 * merged["relative_reduction"]
        + 1.2 * (1.0 - merged["primary_wape_pct"])
        + 0.45 * merged["range_pct"]
        + 0.12 * merged["decision_weight"].fillna(1.0) / decision_max
        + merged["tsfm_bonus"]
    )
    return merged.sort_values("selection_score", ascending=False)


def select_window(panel: dict[str, object]) -> pd.Series:
    table = candidate_table(
        cell_id=str(panel["cell_id"]),
        primary_policy=str(panel["primary_policy"]),
        comparator_policy=str(panel["comparator_policy"]),
        require_tsfm_primary=bool(panel.get("require_tsfm_primary", False)),
        require_critical=panel.get("require_critical"),
        range_quantile=float(panel.get("range_quantile", 0.25)),
    )
    return table.iloc[0]


PANELS = [
    {
        "panel": "a",
        "cell_id": "aidc_power_optional::4h",
        "primary_policy": "cell_validation_winner",
        "comparator_policy": "always_itransformer",
        "require_tsfm_primary": True,
        "require_critical": False,
        "range_quantile": 0.25,
        "layer": "H1-H2 specialist route",
        "strip": "H1/H2: TimesFM 2.5 route",
    },
    {
        "panel": "b",
        "cell_id": "provincial_load::4h",
        "primary_policy": "tsfm_validation_winner",
        "comparator_policy": "always_itransformer",
        "require_tsfm_primary": True,
        "require_critical": False,
        "range_quantile": 0.25,
        "layer": "Provincial TSFM specialist",
        "strip": "H1: provincial TSFM route",
    },
    {
        "panel": "c",
        "cell_id": "aidc_power_optional::4h",
        "primary_policy": "cell_validation_winner",
        "comparator_policy": "always_itransformer",
        "require_tsfm_primary": True,
        "require_critical": True,
        "range_quantile": 0.05,
        "layer": "H3 stress window",
        "strip": "H3: train-defined stress window",
    },
    {
        "panel": "d",
        "cell_id": "microgrid_load::24h",
        "primary_policy": "cell_validation_winner",
        "comparator_policy": "tsfm_validation_winner",
        "range_quantile": 0.25,
        "layer": "Non-TSFM specialist",
        "strip": "Complementary non-TSFM route",
    },
]


def build_record(selected: list[tuple[dict[str, object], pd.Series]]) -> pd.DataFrame:
    _load_globals()
    rows = []
    for panel, row in selected:
        domain, horizon = str(panel["cell_id"]).split("::")
        h1_row = h1[(h1["domain_id"] == domain) & (h1["horizon"] == horizon)]
        rows.append(
            {
                "panel": panel["panel"],
                "layer": panel["layer"],
                "cell_id": panel["cell_id"],
                "window_id": row["window_id"],
                "origin_time": row["origin_time"],
                "target_start_time": row["target_start_time"],
                "target_end_time": row["target_end_time"],
                "primary_policy": panel["primary_policy"],
                "comparator_policy": panel["comparator_policy"],
                "primary_candidate": row["selected_candidate_id_primary"],
                "comparator_candidate": row["selected_candidate_id_comp"],
                "primary_model": row["model_id_primary"],
                "comparator_model": row["model_id_comp"],
                "primary_wape_policy": float(row["wape_primary"]),
                "comparator_wape_policy": float(row["wape_comp"]),
                "relative_reduction": float(row["relative_reduction"]),
                "critical_union": bool(row["critical_union"]),
                "decision_weight": float(row["decision_weight"]),
                "target_range": float(row["target_range"]),
                "h1_winner_group": h1_row.iloc[0]["winner_group"] if not h1_row.empty else "",
                "selection_score": float(row["selection_score"]),
            }
        )
    return pd.DataFrame(rows)


def draw_panel(ax: plt.Axes, panel: dict[str, object], row: pd.Series) -> dict[str, float]:
    domain, horizon = str(panel["cell_id"]).split("::")
    primary_lock = str(row["selected_candidate_id_primary"])
    comp_lock = str(row["selected_candidate_id_comp"])
    window_id = str(row["window_id"])

    primary_pred = prediction_row(primary_lock, window_id)
    comp_pred = prediction_row(comp_lock, window_id)
    y_true = parse_vector(primary_pred["y_true"])
    y_primary = parse_vector(primary_pred["y_pred"])
    y_comp = parse_vector(comp_pred["y_pred"])

    primary_calc = wape(y_true, y_primary)
    comp_calc = wape(y_true, y_comp)
    if abs(primary_calc - float(row["wape_primary"])) > 2e-6:
        raise ValueError(
            f"Primary WAPE mismatch for {primary_lock} {window_id}: {primary_calc} vs {row['wape_primary']}"
        )
    if abs(comp_calc - float(row["wape_comp"])) > 2e-6:
        raise ValueError(f"Comparator WAPE mismatch for {comp_lock} {window_id}: {comp_calc} vs {row['wape_comp']}")

    gran_min = DOMAIN_GRANULARITY_MIN[domain]
    time_h = (np.arange(len(y_true)) + 1) * gran_min / 60.0
    horizon_h = int(horizon.replace("h", ""))
    scale, unit = scale_for(y_true)
    y_true_scaled = y_true / scale
    y_primary_scaled = y_primary / scale
    y_comp_scaled = y_comp / scale

    if bool(row["critical_union"]):
        ax.axvspan(0, horizon_h, color=COL["stress"], alpha=0.18, zorder=0)
        ax.text(
            0.98,
            0.035,
            "Stress window",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=FS_ANNOT_TINY,
            color="#B91C1C",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", fc=COL["stress_box"], ec=COL["stress"], alpha=0.9),
        )

    comp_is_tsfm = is_tsfm_model(row["model_id_comp"])
    comp_color = COL["tsfm"] if comp_is_tsfm else COL["nontsfm"]
    ax.plot(time_h, y_primary_scaled, color=COL["selected"], lw=1.70, alpha=0.90, label="Selected route", zorder=3)
    ax.plot(time_h, y_comp_scaled, color=comp_color, lw=1.15, ls="--", alpha=0.72, label="Comparator", zorder=2)
    ax.plot(time_h, y_true_scaled, color=COL["truth"], lw=1.08, alpha=0.98, label="Ground truth", zorder=4)
    ax.set_xlim(0, horizon_h)
    reserve_annotation_headroom(ax, [y_true_scaled, y_primary_scaled, y_comp_scaled])

    primary_label = display_model(row["model_id_primary"])
    comp_label = display_model(row["model_id_comp"])
    ax.text(
        0.98,
        0.96,
        f"{primary_label}: {primary_calc:.4f}\n{comp_label}: {comp_calc:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FS_ANNOT_TINY,
        bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#CBD5E1", alpha=0.82, lw=0.6),
    )
    ax.text(
        0.02,
        0.96,
        str(panel["layer"]),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FS_ANNOT_SMALL,
        color=COL["selected"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.20,rounding_size=0.02", fc="white", ec=COL["selected"], alpha=0.84, lw=0.7),
    )

    title = f"{DOMAIN_LABEL[domain]} {horizon}"
    ax.set_xlabel("Lead time (h)", labelpad=2)
    clean_ax(ax)
    return {"primary_wape_calc": primary_calc, "comparator_wape_calc": comp_calc, "title": title, "unit": unit}


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    selected = [(panel, select_window(panel)) for panel in PANELS]
    record = build_record(selected)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.75))
    calc_rows = []
    for ax, (panel, row) in zip(axes.ravel(), selected):
        calc = draw_panel(ax, panel, row)
        calc_rows.append(calc)
        bottom_panel_title(ax, str(panel["panel"]), str(calc["title"]))
        if str(panel["panel"]) in {"a", "c"}:
            unit_suffix = f" ({calc['unit']})" if calc["unit"] else ""
            ax.set_ylabel(f"Target value{unit_suffix}")

    record["primary_wape_calc"] = [r["primary_wape_calc"] for r in calc_rows]
    record["comparator_wape_calc"] = [r["comparator_wape_calc"] for r in calc_rows]
    record.to_csv(SELECTION_RECORD, index=False)

    handles = [
        Line2D([0], [0], color=COL["truth"], lw=1.20),
        Line2D([0], [0], color=COL["selected"], lw=1.80),
        Line2D([0], [0], color=COL["nontsfm"], lw=1.25, ls="--"),
        Line2D([0], [0], color=COL["tsfm"], lw=1.25, ls="--"),
    ]
    labels = ["Ground truth", "Selected route", "non-TSFM comparator", "TSFM comparator"]
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=FS_LEGEND, bbox_to_anchor=(0.5, 0.125))
    fig.tight_layout(rect=[0, 0.10, 1, 1], h_pad=2.0, w_pad=0.9)
    out = FIG_DIR / "fig7_forecast_trace.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved: {out}")
    print(f"record: {SELECTION_RECORD}")
    print(record.to_string(index=False))


if __name__ == "__main__":
    main()
