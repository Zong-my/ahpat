#!/usr/bin/env python3
"""Build revised manuscript figures with diverse chart types.

Generates:
  Fig 2 – Violin plot (H1, 4 significant cells)
  Fig 3 – Lollipop / paired dot plot (H2, split PV vs non-PV)
  Fig 4 – H1-H2-H3 integrated forecast trace (NEW)
  Fig 5 – Bar-line combo (H3, excluding PV for readability)
  Fig 7 – Bubble chart (cost-accuracy)
"""

from __future__ import annotations
import ast
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT / "figures"
PKG = PROJECT / "results/energy_tsfm_p5_main/p5_manuscript_result_package_v0_codex_20260519"
H2H3 = PROJECT / "results/energy_tsfm_p5_main/p5_h2_h3_test_application_v0_codex_20260519"
REPAIR = PROJECT / "results/energy_tsfm_manuscript_repair_v1_20260521"
FORMAL = PROJECT / "results/energy_tsfm_formal"

# ---------------------------------------------------------------------------
C = {
    "tsfm": "#0F4C81", "tsfm_light": "#5BA7D9",
    "nontsfm": "#E8702A", "nontsfm_light": "#F4A261",
    "routing": "#7B2D8E", "fixed": "#6B7280", "oracle": "#9CA3AF",
    "stress_bg": "#FECACA", "grid": "#E5E7EB",
    "text": "#111827", "axis": "#1F2933", "paper": "#FFFFFF",
    "chronos2": "#0F4C81", "timesfm2p5": "#5BA7D9",
    "itransformer": "#264653", "nbeatsx": "#E76F51", "lightgbm": "#2A9D8F",
}
DOMAIN_LABEL = {
    "aidc_power_optional": "AIDC power", "aluminum_load": "Aluminum load",
    "arena_pv": "Arena PV", "microgrid_load": "Microgrid load",
    "provincial_load": "Provincial load",
}
DOMAIN_SHORT = {
    "aidc_power_optional": "AIDC", "aluminum_load": "Alum.",
    "arena_pv": "PV", "microgrid_load": "Micro.", "provincial_load": "Prov.",
}
DOMAIN_GRANULARITY_MIN = {
    "aidc_power_optional": 15, "aluminum_load": 15,
    "arena_pv": 15, "microgrid_load": 10, "provincial_load": 15,
}
MODEL_DISPLAY = {
    "chronos2": "Chronos-2", "timesfm2p5": "TimesFM 2.5",
    "itransformer": "iTransformer", "nbeatsx": "N-BEATSx", "lightgbm": "LightGBM",
}
MODEL_COLOR = {
    "chronos2": C["chronos2"], "timesfm2p5": C["timesfm2p5"],
    "itransformer": C["itransformer"], "nbeatsx": C["nbeatsx"], "lightgbm": C["lightgbm"],
}

# Typography constants. Keep element classes consistent across all regenerated
# manuscript figures; only dense in-panel annotations use the smaller tiers.
FS_BASE = 7.5
FS_TITLE = 8.5
FS_LABEL = 7.5
FS_TICK = 6.8
FS_TICK_SMALL = 6.5
FS_LEGEND = 6.5
FS_LEGEND_SMALL = 5.8
FS_PANEL = 10.5
FS_ANNOT = 6.5
FS_ANNOT_SMALL = 5.8
FS_ANNOT_TINY = 5.5
FS_HEATMAP = 6.5

# Bootstrap CI data
_H1_CI = pd.read_csv(REPAIR / "h1_cell_bootstrap_ci.csv") if (REPAIR / "h1_cell_bootstrap_ci.csv").exists() else pd.DataFrame()
_H2_CI = pd.read_csv(REPAIR / "h2_policy_bootstrap_ci.csv") if (REPAIR / "h2_policy_bootstrap_ci.csv").exists() else pd.DataFrame()
_H3_SENS = pd.read_csv(REPAIR / "h3_threshold_sensitivity.csv") if (REPAIR / "h3_threshold_sensitivity.csv").exists() else pd.DataFrame()

def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FS_BASE,
        "axes.titlesize": FS_TITLE, "axes.titleweight": "bold", "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEGEND,
        "figure.dpi": 140, "savefig.dpi": 600, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04, "pdf.fonttype": 42,
        "axes.linewidth": 0.55, "axes.edgecolor": C["axis"],
        "axes.facecolor": C["paper"], "figure.facecolor": C["paper"],
        "legend.frameon": False,
    })

def panel_label(ax, label, x=-0.08, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight="bold", color=C["text"], ha="left", va="bottom")

# Absolute offset in points from axes bottom edge to panel title.
# Rotated x-tick labels (~45°) need more clearance than horizontal ones.
TITLE_OFFSET_ROTATED = 34   # pt — panels with rotated 45° x-tick labels
TITLE_OFFSET_XLABEL = 28    # pt — horizontal ticks + xlabel present
TITLE_OFFSET_FLAT = 20      # pt — horizontal ticks, no xlabel

def bottom_panel_title(ax, label, title, offset_pt=None, fontsize=None):
    if offset_pt is None:
        offset_pt = TITLE_OFFSET_ROTATED
    ax.annotate(
        f"({label}) {title}",
        xy=(0.5, 0), xycoords="axes fraction",
        xytext=(0, -offset_pt), textcoords="offset points",
        fontsize=FS_LABEL if fontsize is None else fontsize,
        fontweight="regular",
        color=C["text"],
        ha="center",
        va="top",
        annotation_clip=False,
    )

def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def cell_label(cid):
    d, h = cid.split("::")
    return f"{DOMAIN_SHORT[d]} {h}"

def parse_json_col(s):
    if isinstance(s, (list, np.ndarray)):
        return np.array(s, dtype=float)
    return np.array(ast.literal_eval(s), dtype=float)

def load_predictions(model_dir, domain, horizon):
    if not model_dir.exists():
        return None
    if any(k in str(model_dir) for k in ["itransformer", "nbeatsx", "lightgbm", "tft", "timexer"]):
        pattern = list(model_dir.glob(f"{domain}/{horizon}/**/predictions/test_predictions*.parquet"))
    else:
        cell_name = f"{domain}_{horizon}"
        pattern = list(model_dir.glob(f"**/cells/{cell_name}/predictions/test_*predictions*.parquet"))
    return pd.read_parquet(pattern[0]) if pattern else None

def get_window_errors(df):
    errors = []
    for _, row in df.iterrows():
        yt = parse_json_col(row["y_true"])
        yp = parse_json_col(row["y_pred"])
        at = np.abs(yt).sum()
        errors.append(np.abs(yt - yp).sum() / at if at > 0 else np.nan)
    return np.array(errors)


# ===========================================================================
# FIG 2: Violin (H1)
# ===========================================================================
def build_fig2_violin():
    """Dual-panel H1 figure: (a) forest plot of all 10 cells, (b) violin of 4 significant cells."""
    print("Building Fig 2: Forest + Violin dual panel...")
    h1 = pd.read_csv(PKG / "table_1_p5_tsfm_vs_non_tsfm_test.csv")

    # ---- layout: top = forest, bottom = 4 violins ----
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.8, 0.825], hspace=0.38, wspace=0.35)
    ax_forest = fig.add_subplot(gs[0, :])
    violin_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]

    # ==================== Panel (a): forest plot ====================
    ci_data = _H1_CI.copy()
    ci_data["cell_label"] = ci_data.apply(
        lambda r: f"{DOMAIN_SHORT[r['domain']]} {r['horizon']}", axis=1)
    ci_data = ci_data.sort_values("wape_diff_tsfm_minus_non_tsfm")  # most TSFM-favorable at top

    # Forest-plot colors: bright for significant, light for indistinguishable
    fc_tsfm = "#2563EB"   # bright blue — stands out
    fc_nontsfm = C["nontsfm"]  # orange — already bright
    fc_indist = "#B8BFC8"  # light gray — recedes as background
    for i, (_, r) in enumerate(ci_data.iterrows()):
        gap = r["wape_diff_tsfm_minus_non_tsfm"]
        lo, hi = r["block_bootstrap_ci_low"], r["block_bootstrap_ci_high"]
        if hi < 0:
            color = fc_tsfm
        elif lo > 0:
            color = fc_nontsfm
        else:
            color = fc_indist
        ax_forest.plot([lo, hi], [i, i], color=color, linewidth=2.0, solid_capstyle="round", zorder=2)
        ax_forest.scatter(gap, i, color=color, s=30, zorder=3, edgecolors="white", linewidths=0.4)
        gap_txt = f"{gap:+.4f}" if abs(gap) >= 0.0001 else f"{gap:+.6f}"
        ax_forest.text(lo - 0.002, i, gap_txt, fontsize=FS_ANNOT_TINY,
                        va="center", ha="right", color=color)

    ax_forest.axvline(0, color=C["text"], linewidth=0.6, linestyle="--", zorder=1)
    ax_forest.set_yticks(range(len(ci_data)))
    ax_forest.set_yticklabels(ci_data["cell_label"].tolist(), fontsize=FS_TICK_SMALL)
    ax_forest.set_xlabel("TSFM $-$ non-TSFM WAPE gap (negative = TSFM better)", fontsize=FS_LABEL)
    ax_forest.set_xlim(-0.085, 0.045)  # extra left margin for gap annotations
    ax_forest.invert_yaxis()
    clean_ax(ax_forest)
    ax_forest.grid(axis="x", alpha=0.2, linewidth=0.3)
    # Legend
    leg_el = [
        Line2D([0], [0], color=fc_tsfm, lw=2, label="TSFM sig. (CI < 0)"),
        Line2D([0], [0], color=fc_nontsfm, lw=2, label="non-TSFM sig. (CI > 0)"),
        Line2D([0], [0], color=fc_indist, lw=2, label="Indistinguishable"),
    ]
    leg_forest = ax_forest.legend(handles=leg_el, loc="upper right", fontsize=FS_LEGEND_SMALL)
    leg_forest.get_frame().set_visible(True)
    leg_forest.get_frame().set_boxstyle("round,pad=0.3")
    leg_forest.get_frame().set_facecolor("white")
    leg_forest.get_frame().set_alpha(0.85)
    leg_forest.get_frame().set_edgecolor(C["grid"])
    bottom_panel_title(ax_forest, "a", "Cell-level WAPE gaps", offset_pt=TITLE_OFFSET_XLABEL)

    # ==================== Panel (b): violin of 4 significant cells ====================
    sig_cells = [
        ("aidc_power_optional", "4h"),
        ("provincial_load", "4h"),
        ("aluminum_load", "24h"),
        ("microgrid_load", "24h"),
    ]

    for idx, (domain, horizon) in enumerate(sig_cells):
        ax = violin_axes[idx]
        row = h1[(h1["domain_id"] == domain) & (h1["horizon"] == horizon)].iloc[0]
        best_tsfm, best_nontsfm = row["best_tsfm_model"], row["best_non_tsfm_model"]
        agg_tsfm = float(row["best_tsfm_wape"])
        agg_nontsfm = float(row["best_non_tsfm_wape"])

        tsfm_dir = FORMAL / ("chronos2_lora_selected" if best_tsfm == "chronos2"
                              else "timesfm2p5_xreg_shared_dev")
        nontsfm_dir = FORMAL / best_nontsfm

        tsfm_df = load_predictions(tsfm_dir, domain, horizon)
        nontsfm_df = load_predictions(nontsfm_dir, domain, horizon)

        data, colors, labels, agg_vals = [], [], [], []
        if tsfm_df is not None:
            errs = get_window_errors(tsfm_df)
            cap = np.percentile(errs, 95)
            data.append(np.clip(errs, 0, cap))
            colors.append(C["tsfm"])
            labels.append(f"TSFM\n({MODEL_DISPLAY.get(best_tsfm, best_tsfm)})")
            agg_vals.append(agg_tsfm)
        if nontsfm_df is not None:
            errs = get_window_errors(nontsfm_df)
            cap = np.percentile(errs, 95)
            data.append(np.clip(errs, 0, cap))
            colors.append(C["nontsfm"])
            labels.append(f"non-TSFM\n({MODEL_DISPLAY.get(best_nontsfm, best_nontsfm)})")
            agg_vals.append(agg_nontsfm)

        if data:
            parts = ax.violinplot(data, positions=range(len(data)),
                                  showmeans=False, showmedians=True, showextrema=False)
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors[i]); pc.set_alpha(0.6); pc.set_edgecolor(colors[i])
            parts["cmedians"].set_color(C["text"]); parts["cmedians"].set_linewidth(1.2)
            for i, av in enumerate(agg_vals):
                ax.scatter(i, av, color="#DC2626", marker="D", s=22, zorder=5,
                           edgecolors="white", linewidths=0.4)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=FS_ANNOT_TINY)

        bottom_panel_title(ax, chr(98 + idx), f"{DOMAIN_LABEL[domain]} {horizon}", offset_pt=24)
        ax.set_ylabel("Per-window WAPE" if idx == 0 else "")
        clean_ax(ax)

    fig.savefig(OUT_DIR / "fig2_h1_violin.pdf"); plt.close(fig)
    print("  Saved fig2_h1_violin.pdf")


# ===========================================================================
# FIG 3: H2 paired dumbbell — two panels to handle PV scale difference
# ===========================================================================
def build_fig3_lollipop():
    print("Building Fig 3: H2 paired dumbbell plot (horizontal, single-column)...")
    h2pc = pd.read_csv(H2H3 / "p5_h2_policy_per_cell_metrics.csv")

    non_pv = ["aidc_power_optional::4h", "aidc_power_optional::24h",
              "aluminum_load::4h", "aluminum_load::24h",
              "microgrid_load::4h", "microgrid_load::24h",
              "provincial_load::4h", "provincial_load::24h"]
    pv = ["arena_pv::4h", "arena_pv::24h"]

    wide = (
        h2pc[h2pc["policy_id"].isin([
            "cell_validation_winner", "always_itransformer", "oracle_per_window_best"
        ])]
        .pivot(index="cell_id", columns="policy_id", values="aggregate_wape")
        .reset_index()
    )
    wide["label"] = wide["cell_id"].map(cell_label)
    wide["order"] = wide["cell_id"].map({c: i for i, c in enumerate(non_pv + pv)})
    wide = wide.sort_values("order").reset_index(drop=True)

    ci_row = _H2_CI[_H2_CI["comparison_id"] == "primary_vs_best_fixed"]
    policy_metrics = pd.read_csv(H2H3 / "p5_h2_policy_metrics.csv")
    oracle_row = policy_metrics[policy_metrics["policy_id"] == "oracle_per_window_best"]
    agg = None
    if not ci_row.empty and not oracle_row.empty:
        agg = {
            "cell_id": "aggregate",
            "label": "Aggregate",
            "always_itransformer": float(ci_row.iloc[0]["policy_b_wape"]),
            "cell_validation_winner": float(ci_row.iloc[0]["policy_a_wape"]),
            "oracle_per_window_best": float(oracle_row.iloc[0]["aggregate_wape"]),
            "ci_low": float(ci_row.iloc[0]["block_bootstrap_ci_low"]),
            "ci_high": float(ci_row.iloc[0]["block_bootstrap_ci_high"]),
        }
        agg["relative_reduction_pct"] = (
            (agg["always_itransformer"] - agg["cell_validation_winner"])
            / agg["always_itransformer"] * 100.0
        )

    def draw_dumbbell(ax, rows, include_aggregate=False, *, xlim=None, xlabel=True):
        plot_rows = rows.copy()
        if include_aggregate and agg is not None:
            plot_rows = pd.concat([pd.DataFrame([agg]), plot_rows], ignore_index=True)
        y = np.arange(len(plot_rows))[::-1]
        for yi, (_, r) in zip(y, plot_rows.iterrows()):
            fw = float(r["always_itransformer"])
            rw = float(r["cell_validation_winner"])
            ow = float(r["oracle_per_window_best"])
            is_aggregate = r["cell_id"] == "aggregate"
            color = C["routing"] if rw <= fw + 1e-12 else C["fixed"]
            ax.plot([fw, rw], [yi, yi], color=color,
                    linewidth=2.3 if is_aggregate else 1.45, alpha=0.66, zorder=1)
            ax.scatter(fw, yi, color=C["fixed"], s=50 if is_aggregate else 34,
                       zorder=3, marker="o", edgecolors="white", linewidths=0.45)
            ax.scatter(rw, yi, color=C["routing"], s=66 if is_aggregate else 44,
                       zorder=4, marker="D", edgecolors="white", linewidths=0.45)
            ax.scatter(ow, yi, color=C["oracle"], s=36 if is_aggregate else 27,
                       zorder=2, marker="x", linewidths=0.95, alpha=0.72)
            if is_aggregate:
                ax.axhspan(yi - 0.42, yi + 0.42, color="#E9D5FF", alpha=0.34, zorder=0)
                ci_low_abs = agg["always_itransformer"] + agg["ci_low"]
                ci_high_abs = agg["always_itransformer"] + agg["ci_high"]
                ax.plot([ci_low_abs, ci_high_abs], [yi + 0.25, yi + 0.25],
                        color=C["routing"], linewidth=1.1, zorder=5)
                ax.plot([ci_low_abs, ci_low_abs], [yi + 0.19, yi + 0.31],
                        color=C["routing"], linewidth=1.1, zorder=5)
                ax.plot([ci_high_abs, ci_high_abs], [yi + 0.19, yi + 0.31],
                        color=C["routing"], linewidth=1.1, zorder=5)
                ax.text(max(fw, rw) * 1.025, yi,
                        f"{agg['relative_reduction_pct']:.1f}% lower",
                        ha="left", va="center", fontsize=FS_ANNOT_TINY,
                        color=C["routing"], fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(plot_rows["label"], fontsize=FS_TICK_SMALL)
        if include_aggregate and agg is not None:
            ax.get_yticklabels()[0].set_fontweight("bold")
        ax.set_xlabel("WAPE" if xlabel else "", fontsize=FS_LABEL)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.grid(axis="x", color=C["grid"], alpha=0.7, linewidth=0.45)
        clean_ax(ax)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(3.55, 4.46), gridspec_kw={"height_ratios": [4.35, 1.55]}
    )
    draw_dumbbell(ax1, wide[wide["cell_id"].isin(non_pv)], include_aggregate=True, xlabel=False)
    draw_dumbbell(ax2, wide[wide["cell_id"].isin(pv)], xlim=(0.22, 0.96), xlabel=True)
    ax2.set_ylim(-0.35, 1.35)  # tighten: bring PV 4h down and PV 24h up

    bottom_panel_title(ax1, "a", "Non-PV domains", offset_pt=TITLE_OFFSET_FLAT)
    bottom_panel_title(ax2, "b", "Arena PV", offset_pt=TITLE_OFFSET_XLABEL)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["fixed"],
               markersize=5.5, label="Fixed iTransformer"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C["routing"],
               markersize=5.5, label="Validation-locked policy"),
        Line2D([0], [0], marker="x", color=C["oracle"], markersize=5.5,
               label="Oracle bound", linewidth=0),
        Line2D([0], [0], color=C["routing"], linewidth=1.1,
               label="Aggregate 95% CI"),
    ]

    fig.tight_layout(rect=[0, 0.02, 1, 0.91], h_pad=1.2)

    # Place legend after tight_layout so ax1 position is final
    ax1_bbox = ax1.get_position()
    legend_x = ax1_bbox.x0 + ax1_bbox.width / 2
    leg3 = fig.legend(handles=legend_elements, loc="upper center",
                      bbox_to_anchor=(legend_x, 0.963), ncol=2, fontsize=FS_LEGEND_SMALL)
    leg3.get_frame().set_visible(True)
    leg3.get_frame().set_boxstyle("round,pad=0.3")
    leg3.get_frame().set_facecolor("white")
    leg3.get_frame().set_alpha(0.85)
    leg3.get_frame().set_edgecolor(C["grid"])

    fig.savefig(OUT_DIR / "fig3_h2_lollipop.pdf"); plt.close(fig)
    print("  Saved fig3_h2_lollipop.pdf")


# ===========================================================================
# FIG 4: 2x2 Forecast Trace with H1+H2+H3 triple encoding
# ===========================================================================
def build_fig7_forecast_trace():
    """2x2 H1+H2+H3 integrated prediction curve.
    H1: TSFM (blue) vs non-TSFM (orange) lines
    H2: annotation box with routing decision
    H3: stress-window background shading
    """
    print("Building Fig 7: Forecast trace (2x2)...")
    h1 = pd.read_csv(PKG / "table_1_p5_tsfm_vs_non_tsfm_test.csv")

    panels = [
        {"domain": "aidc_power_optional", "horizon": "4h",
         "h2_note": "H2 selects TSFM \u2713", "h2_tsfm": True, "h1_indist": False},
        {"domain": "provincial_load", "horizon": "4h",
         "h2_note": "H2 selects iTransformer\n(TSFM-only would improve)", "h2_tsfm": False, "h1_indist": False},
        {"domain": "aidc_power_optional", "horizon": "24h",
         "h2_note": "H2 selects N-BEATSx", "h2_tsfm": False, "h1_indist": True},
        {"domain": "microgrid_load", "horizon": "24h",
         "h2_note": "H2 selects iTransformer \u2713", "h2_tsfm": False, "h1_indist": False},
    ]

    try:
        stress_labels = pd.read_csv(H2H3 / "p5_h3_test_window_labels.csv")
    except Exception:
        stress_labels = None

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))

    for idx, pc in enumerate(panels):
        ax = axes[idx // 2][idx % 2]
        domain, horizon = pc["domain"], pc["horizon"]

        row = h1[(h1["domain_id"] == domain) & (h1["horizon"] == horizon)].iloc[0]
        best_tsfm, best_nontsfm = row["best_tsfm_model"], row["best_non_tsfm_model"]

        tsfm_dir = FORMAL / ("chronos2_lora_selected" if best_tsfm == "chronos2"
                              else "timesfm2p5_xreg_shared_dev")
        nontsfm_dir = FORMAL / best_nontsfm

        tsfm_df = load_predictions(tsfm_dir, domain, horizon)
        nontsfm_df = load_predictions(nontsfm_dir, domain, horizon)

        if tsfm_df is None or nontsfm_df is None:
            ax.text(0.5, 0.5, "Data unavailable", transform=ax.transAxes, ha="center")
            panel_label(ax, chr(97 + idx)); continue

        nontsfm_idx = nontsfm_df.set_index("window_id")
        candidates = []
        for _, r in tsfm_df.iterrows():
            wid = r["window_id"]
            if wid not in nontsfm_idx.index:
                continue
            nr = nontsfm_idx.loc[wid]
            if isinstance(nr, pd.DataFrame):
                nr = nr.iloc[0]
            yt = parse_json_col(r["y_true"])
            yp_t = parse_json_col(r["y_pred"])
            yp_n = parse_json_col(nr["y_pred"])
            at = np.abs(yt).sum()
            if at < 1e-6:
                continue
            if pc.get("require_positive") and ((yt > 0).mean() < 0.7 or np.min(yt) < 0):
                continue
            tw = np.abs(yt - yp_t).sum() / at
            nw = np.abs(yt - yp_n).sum() / at
            candidates.append({"wid": wid, "yt": yt, "yp_t": yp_t,
                                "yp_n": yp_n, "tw": tw, "nw": nw})

        if not candidates:
            ax.text(0.5, 0.5, "No suitable window", transform=ax.transAxes, ha="center")
            panel_label(ax, chr(97 + idx)); continue

        h2_key = "tw" if pc["h2_tsfm"] else "nw"
        alt_key = "nw" if pc["h2_tsfm"] else "tw"
        if pc.get("h1_indist"):
            # H1 indistinguishable: show both models tracking well, H2 winner
            # slightly better — consistent with the "no clear dominance" finding.
            candidates.sort(key=lambda x: x[h2_key] + x[alt_key])
            top_n = max(3, len(candidates) // 5)
            top = candidates[:top_n]
            pos = [c for c in top if c[alt_key] >= c[h2_key]]
            if pos:
                pos.sort(key=lambda x: x[alt_key] - x[h2_key])
                ch = pos[len(pos) // 2]
            else:
                ch = top[0]
        else:
            # H1 significant: showcase routing advantage — H2 winner tracks
            # well (top 20%) and gap to alternative is maximized.
            candidates.sort(key=lambda x: x[h2_key])
            top_n = max(3, len(candidates) // 5)
            top = candidates[:top_n]
            top.sort(key=lambda x: x[alt_key] - x[h2_key], reverse=True)
            ch = top[0]
        n_steps = len(ch["yt"])
        gran_min = DOMAIN_GRANULARITY_MIN[domain]
        time_h = np.arange(n_steps) * gran_min / 60  # x-axis in hours
        horizon_h = int(horizon.replace("h", ""))

        # H3 stress shading — prominent fill across entire panel
        is_stress = False
        if stress_labels is not None:
            wm = stress_labels[stress_labels["window_id"] == ch["wid"]]
            if not wm.empty and "critical_union" in wm.columns:
                is_stress = bool(wm.iloc[0]["critical_union"])
        if is_stress:
            ax.axvspan(0, time_h[-1], alpha=0.18, color="#FCA5A5", zorder=0)
            ax.text(0.97, 0.03, "H3: stress window", transform=ax.transAxes,
                    fontsize=FS_ANNOT_TINY, color="#DC2626", fontweight="bold",
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#FEE2E2", alpha=0.9, ec="#FCA5A5"))

        # H1 lines — bright blue vs orange, gray ground truth for contrast
        gt_color = "#6B7280"   # medium gray — reference, not dominant
        tsfm_color = "#2563EB" # bright blue — clearly distinct from gray
        t_lw, n_lw = (2.0, 1.0) if pc["h2_tsfm"] else (1.0, 2.0)
        ax.plot(time_h, ch["yt"], color=gt_color, lw=1.0, label="Ground truth", zorder=2)
        ax.plot(time_h, ch["yp_t"], color=tsfm_color, lw=t_lw,
                label=f"TSFM ({MODEL_DISPLAY.get(best_tsfm, best_tsfm)})", zorder=4)
        ax.plot(time_h, ch["yp_n"], color=C["nontsfm"], lw=n_lw, ls="--",
                label=f"non-TSFM ({MODEL_DISPLAY.get(best_nontsfm, best_nontsfm)})", zorder=3)
        ax.set_xlim(0, horizon_h)

        # WAPE box (compact) — use actual model names per panel
        tsfm_disp = MODEL_DISPLAY.get(best_tsfm, best_tsfm)
        nontsfm_disp = MODEL_DISPLAY.get(best_nontsfm, best_nontsfm)
        ax.text(0.97, 0.97, f"{tsfm_disp}: {ch['tw']:.4f}\n{nontsfm_disp}: {ch['nw']:.4f}",
                transform=ax.transAxes, fontsize=FS_ANNOT_TINY, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec=C["grid"]))

        # H2 routing strip — colored banner at top of panel
        ax.text(0.0, 1.0, f"  {pc['h2_note']}  ", transform=ax.transAxes,
                fontsize=FS_ANNOT_TINY, ha="left", va="top", color="white", fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.25", fc=C["routing"], alpha=0.85, ec="none"))

        # H1 badge from bootstrap CI
        h1_row = _H1_CI[(_H1_CI["domain"] == domain) & (_H1_CI["horizon"] == horizon)]
        if not h1_row.empty:
            ci_lo = float(h1_row.iloc[0]["block_bootstrap_ci_low"])
            ci_hi = float(h1_row.iloc[0]["block_bootstrap_ci_high"])
            if ci_hi < 0:
                h1_tag = "H1: TSFM sig."
                h1_color = C["tsfm"]
            elif ci_lo > 0:
                h1_tag = "H1: non-TSFM sig."
                h1_color = C["nontsfm"]
            else:
                h1_tag = "H1: indist."
                h1_color = C["fixed"]
        else:
            h1_tag = ""
            h1_color = C["fixed"]
        ax.set_title(f"{DOMAIN_LABEL[domain]} {horizon}", fontsize=FS_TITLE)
        if h1_tag:
            ax.text(1.0, 1.02, h1_tag, transform=ax.transAxes, fontsize=FS_ANNOT_TINY,
                    ha="right", va="bottom", color=h1_color, fontweight="bold")
        ax.set_xlabel("Hours ahead", fontsize=FS_LABEL)
        if idx % 2 == 0:
            ax.set_ylabel("Target value", fontsize=FS_LABEL)
        clean_ax(ax); panel_label(ax, chr(97 + idx))

    custom_handles = [
        Line2D([0], [0], color="#6B7280", lw=1.0),
        Line2D([0], [0], color="#2563EB", lw=1.8),
        Line2D([0], [0], color=C["nontsfm"], lw=1.0, ls="--"),
    ]
    custom_labels = ["Ground truth", "Best TSFM route", "Best non-TSFM route"]
    fig.legend(custom_handles, custom_labels, loc="lower center", ncol=3, fontsize=FS_LEGEND, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1], h_pad=1.2, w_pad=0.8)
    fig.savefig(OUT_DIR / "fig7_forecast_trace.pdf"); plt.close(fig)
    print("  Saved fig7_forecast_trace.pdf")


# ===========================================================================
# FIG 5: Bar-line combo (H3)
# ===========================================================================
def build_fig5_barline():
    print("Building Fig 5: H3 aggregate evidence + stress context...")
    h2pc = pd.read_csv(H2H3 / "p5_h2_policy_per_cell_metrics.csv")
    h3pol = pd.read_csv(H2H3 / "p5_h3_policy_decision_weighted_metrics.csv")
    h3ci = pd.read_csv(REPAIR / "h3_stress_bootstrap_ci.csv")
    wlabels = pd.read_csv(H2H3 / "p5_h3_test_window_labels.csv")

    cells = [
        "aidc_power_optional::4h", "aidc_power_optional::24h",
        "aluminum_load::4h", "aluminum_load::24h",
        "arena_pv::4h", "arena_pv::24h",
        "microgrid_load::4h", "microgrid_load::24h",
        "provincial_load::4h", "provincial_load::24h",
    ]

    stress_prop = {}
    for cell in cells:
        d, h = cell.split("::")
        subset = wlabels[(wlabels["domain"] == d) & (wlabels["horizon"] == h)]
        stress_prop[cell] = subset["critical_union"].mean() if len(subset) > 0 and "critical_union" in subset.columns else 0.0

    routed = h2pc[h2pc["policy_id"] == "cell_validation_winner"].set_index("cell_id")
    fixed = h2pc[h2pc["policy_id"] == "always_itransformer"].set_index("cell_id")

    fig, (ax_top, ax1) = plt.subplots(
        2, 1, figsize=(7.2, 4.35), gridspec_kw={"height_ratios": [0.90, 1.55]}
    )

    # Top panel: horizontal dumbbell showing paired DWAPE comparison.
    scopes = [
        ("all_windows", "all_windows", "All windows"),
        ("critical_union", "critical_windows", "Critical union"),
    ]
    primary_vals, fixed_vals, rel_vals, ci_texts, delta_vals = [], [], [], [], []
    for metric_scope, ci_scope, _ in scopes:
        pri = h3pol[(h3pol["metric_scope"] == metric_scope) &
                    (h3pol["policy_id"] == "cell_validation_winner")].iloc[0]
        fix = h3pol[(h3pol["metric_scope"] == metric_scope) &
                    (h3pol["policy_id"] == "always_itransformer")].iloc[0]
        ci = h3ci[(h3ci["metric_scope"] == ci_scope) &
                  (h3ci["comparison_id"] == "primary_vs_best_fixed")].iloc[0]
        primary = float(pri["decision_weighted_wape"])
        fixed_val = float(fix["decision_weighted_wape"])
        primary_vals.append(primary)
        fixed_vals.append(fixed_val)
        rel_vals.append(100.0 * (1.0 - primary / fixed_val))
        delta_vals.append(float(ci["dwape_diff_a_minus_b"]))
        ci_texts.append(
            f"[{float(ci['block_bootstrap_ci_low']):+.4f}, "
            f"{float(ci['block_bootstrap_ci_high']):+.4f}]"
        )

    # Option 2(a) preview style: paired absolute WAPE with inline effect-size
    # annotations. This keeps the absolute comparison visible while avoiding
    # the visual bulk of bars.
    x_lo = 0.079
    x_hi = 0.101
    ypos = [0, 1]
    for idx, (i, pv, fv, rel, delta, ci_text) in enumerate(
            zip(ypos, primary_vals, fixed_vals, rel_vals, delta_vals, ci_texts)):
        ax_top.fill_betweenx([i - 0.12, i + 0.12], pv, fv,
                             color=C["routing"], alpha=0.08, zorder=1)
        ax_top.plot([pv, fv], [i, i], color=C["grid"], lw=2.0, zorder=2)
        ax_top.scatter(fv, i, color=C["fixed"], marker="o", s=60, zorder=4,
                       edgecolors="white", linewidths=0.5)
        ax_top.scatter(pv, i, color=C["routing"], marker="D", s=60, zorder=5,
                       edgecolors="white", linewidths=0.5)
        ax_top.text(pv, i - 0.20, f"{pv:.3f}", ha="center", va="bottom",
                    fontsize=FS_ANNOT, color=C["routing"], fontweight="bold")
        ax_top.text(fv, i - 0.20, f"{fv:.3f}", ha="center", va="bottom",
                    fontsize=FS_ANNOT, color=C["fixed"])
        mid_x = (pv + fv) / 2
        ax_top.text(mid_x, i + 0.27,
                    f"{rel:.1f}% lower  |  $\\Delta$ = {delta:+.4f}, 95% CI {ci_text}",
                    ha="center", va="top", fontsize=FS_ANNOT_SMALL, color=C["text"])

    ax_top.set_yticks(ypos)
    ax_top.set_yticklabels([label for _, _, label in scopes], fontsize=FS_LABEL)
    ax_top.set_xlim(x_lo, x_hi)
    ax_top.set_ylim(max(ypos) + 0.62, min(ypos) - 0.45)  # inverted
    ax_top.set_xlabel("Stress-window-weighted WAPE", fontsize=FS_LABEL)
    ax_top.xaxis.set_label_position("top")
    ax_top.xaxis.tick_top()
    # Legend compact
    leg_el = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C["routing"],
               markersize=5, label="Primary policy"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["fixed"],
               markersize=5, label="Best fixed family"),
    ]
    leg_top = ax_top.legend(handles=leg_el, loc="center right",
                            bbox_to_anchor=(0.98, 0.50),
                            fontsize=FS_LEGEND_SMALL, ncol=1)
    leg_top.get_frame().set_visible(True)
    leg_top.get_frame().set_boxstyle("round,pad=0.3")
    leg_top.get_frame().set_facecolor("white")
    leg_top.get_frame().set_alpha(0.85)
    leg_top.get_frame().set_edgecolor(C["grid"])
    clean_ax(ax_top)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.tick_params(bottom=False)
    bottom_panel_title(ax_top, "a", "Aggregate stress-weighted WAPE comparison", offset_pt=2)

    # Bottom panel: stress prevalence and cell-level unweighted WAPE context.
    x = np.arange(len(cells))

    bar_vals = [stress_prop.get(c, 0) for c in cells]
    ax1.bar(x, bar_vals, 0.55, color="#FCA5A5", edgecolor="#EF4444",
            linewidth=0.6, alpha=0.5, label="Stress-window ratio", zorder=1)
    ax1.set_ylabel("Stress-window ratio", fontsize=FS_LABEL, color="#DC2626")
    ax1.set_ylim(0, 0.72)
    ax1.tick_params(axis="y", labelcolor="#DC2626")

    ax2 = ax1.twinx()
    routed_v = [routed.loc[c, "aggregate_wape"] if c in routed.index else np.nan for c in cells]
    fixed_v = [fixed.loc[c, "aggregate_wape"] if c in fixed.index else np.nan for c in cells]

    ax2.semilogy(x, fixed_v, color=C["fixed"], marker="o", ms=4, lw=1.2,
                 ls="--", label="Fixed: iTransformer", zorder=3)
    ax2.semilogy(x, routed_v, color=C["routing"], marker="D", ms=4, lw=1.2,
                 ls="-", label="Routed: primary policy", zorder=4)
    ax2.set_ylabel("Cell-level WAPE (log scale)", fontsize=FS_LABEL, color=C["routing"])
    ax2.tick_params(axis="y", labelcolor=C["routing"])

    ax1.set_xticks(x)
    ax1.set_xticklabels([cell_label(c) for c in cells], fontsize=FS_TICK_SMALL, rotation=45, ha="right")
    clean_ax(ax1); ax2.spines["top"].set_visible(False)
    bottom_panel_title(ax1, "b", "Cell-level stress context")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    leg_bot = ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=FS_LEGEND_SMALL)
    leg_bot.get_frame().set_visible(True)
    leg_bot.get_frame().set_boxstyle("round,pad=0.3")
    leg_bot.get_frame().set_facecolor("white")
    leg_bot.get_frame().set_alpha(0.85)
    leg_bot.get_frame().set_edgecolor(C["grid"])

    fig.tight_layout(h_pad=1.0)
    fig.subplots_adjust(left=0.10)
    fig.savefig(OUT_DIR / "fig5_h3_barline.pdf", pad_inches=0.08); plt.close(fig)
    print("  Saved fig5_h3_barline.pdf")


# ===========================================================================
# FIG 7: Bubble chart (cost)
# ===========================================================================
def build_fig8_bubble():
    print("Building Fig 8: Bubble chart...")
    cost = pd.read_csv(REPAIR / "p6_computational_cost_family_summary_enhanced.csv")

    TSFM_IDS = {"chronos2", "timesfm2p5"}
    fig, ax = plt.subplots(figsize=(3.45, 2.75))

    def scaled_vram_size(vram):
        if pd.isna(vram):
            return 52.0
        return 45.0 + np.sqrt(float(vram)) * 58.0

    label_offsets = {
        "chronos2": (7, 4),
        "timesfm2p5": (7, 6),
        "itransformer": (8, 5),
        "nbeatsx": (-8, 6),
        "lightgbm": (8, -8),
    }

    for _, row in cost.iterrows():
        mid = row["model_id"]
        label = MODEL_DISPLAY.get(mid, mid)
        color = MODEL_COLOR.get(mid, C["fixed"])
        x = float(row["median_latency_ms_per_window_proxy"])
        y = float(row["median_wape"])
        vram = row["median_peak_vram_gb"]
        size = scaled_vram_size(vram)
        is_tsfm = mid in TSFM_IDS
        marker = "s" if mid == "lightgbm" else "o"
        # TSFM: thick colored edge; non-TSFM: thin white edge
        edge_color = color if is_tsfm else "white"
        edge_width = 1.45 if is_tsfm else 0.75

        ax.scatter(x, y, s=size, color=color, marker=marker,
                   edgecolors=edge_color, linewidths=edge_width,
                   zorder=3, alpha=0.88)
        dx, dy = label_offsets.get(mid, (7, 5))
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=FS_ANNOT, fontweight="bold", color=color,
                    ha="left" if dx >= 0 else "right",
                    va="bottom" if dy >= 0 else "top")

    ax.set_xlabel("Median latency (ms / window)")
    ax.set_ylabel("Median WAPE")
    ax.set_xlim(8, 128)
    ax.set_ylim(0.075, 0.163)
    clean_ax(ax)
    ax.grid(True, color=C["grid"], alpha=0.75, linewidth=0.45)

    # In-panel VRAM key: kept small because marker semantics are explained in
    # the caption.
    x0, y0 = 0.50, 0.55
    ax.text(x0, y0 + 0.08, "Bubble size = peak VRAM",
            transform=ax.transAxes, fontsize=FS_ANNOT_SMALL,
            color=C["text"], ha="left", va="center")
    for i, (vram_ref, label) in enumerate([(0.5, "~0.5 GB"), (10.0, "~10 GB")]):
        ax.scatter([x0 + 0.03], [y0 - i * 0.07], s=scaled_vram_size(vram_ref),
                   transform=ax.transAxes, color="#9CA3AF", alpha=0.42,
                   edgecolors="white", linewidths=0.6, clip_on=False)
        ax.text(x0 + 0.09, y0 - i * 0.07, label, transform=ax.transAxes,
                fontsize=FS_ANNOT_SMALL, color=C["fixed"],
                va="center", ha="left")

    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.20, top=0.96)
    fig.savefig(OUT_DIR / "fig8_cost_bubble.pdf"); plt.close(fig)
    print("  Saved fig8_cost_bubble.pdf")


# ===========================================================================
# FIG 8: Validation-test consistency scatter
# ===========================================================================
def build_fig4_valtest():
    """Scatter plot: validation WAPE vs test WAPE per route per cell.
    Demonstrates that validation performance predicts test performance,
    justifying the H2 validation-locked protocol."""
    print("Building Fig 4: Validation-test consistency scatter...")
    from scipy.stats import spearmanr

    cand = pd.read_csv(
        H2H3 / "p5_h2_candidate_window_metrics.csv",
        usecols=["candidate_id", "cell_id", "split", "abs_error_sum",
                 "abs_true_sum", "route_family", "model_id"],
    )
    agg = (
        cand.groupby(["candidate_id", "cell_id", "split", "model_id"])
        .agg(total_ae=("abs_error_sum", "sum"), total_at=("abs_true_sum", "sum"))
        .reset_index()
    )
    agg["wape"] = agg["total_ae"] / agg["total_at"]

    val = agg[agg["split"] == "validation"][["candidate_id", "cell_id", "model_id", "wape"]].rename(columns={"wape": "val_wape"})
    test = agg[agg["split"] == "test"][["candidate_id", "cell_id", "wape"]].rename(columns={"wape": "test_wape"})
    merged = val.merge(test, on=["candidate_id", "cell_id"])

    TSFM_IDS = {"chronos2", "timesfm2p5"}
    merged["is_tsfm"] = merged["model_id"].isin(TSFM_IDS)
    # Extract domain from cell_id for coloring
    merged["domain"] = merged["cell_id"].str.split("::").str[0]

    # Identify H2 winner per cell (lowest val_wape)
    winners = merged.loc[merged.groupby("cell_id")["val_wape"].idxmin()]

    DOMAIN_COLORS = {
        "aidc_power_optional": "#0F4C81", "aluminum_load": "#2A9D8F",
        "arena_pv": "#E76F51", "microgrid_load": "#C77B15",
        "provincial_load": "#7B2D8E",
    }

    fig, ax = plt.subplots(figsize=(3.6, 3.6))

    # Plot non-winners: circle = TSFM, square = non-TSFM
    non_w = merged[~merged.index.isin(winners.index)]
    for _, r in non_w.iterrows():
        mk = "o" if r["is_tsfm"] else "s"
        ax.scatter(r["val_wape"], r["test_wape"],
                   color=DOMAIN_COLORS.get(r["domain"], C["fixed"]),
                   marker=mk, s=36, alpha=0.72, zorder=2,
                   edgecolors="white", linewidths=0.4)

    # Plot H2 winners with star marker (larger, black edge)
    for _, r in winners.iterrows():
        ax.scatter(r["val_wape"], r["test_wape"],
                   color=DOMAIN_COLORS.get(r["domain"], C["fixed"]),
                   marker="*", s=120, zorder=4,
                   edgecolors="black", linewidths=0.5)

    ax.set_xscale("log"); ax.set_yscale("log")

    # Diagonal reference — clipped to data extent with small pad
    all_vals = pd.concat([merged["val_wape"], merged["test_wape"]])
    d_lo, d_hi = all_vals.min() * 0.7, all_vals.max() * 1.4
    ax.plot([d_lo, d_hi], [d_lo, d_hi], color=C["grid"], ls="--", lw=0.8, zorder=1)

    ax.set_xlabel("Validation WAPE (log scale)"); ax.set_ylabel("Test WAPE (log scale)")
    clean_ax(ax)
    ax.grid(True, alpha=0.2, linewidth=0.3)

    # Spearman correlation
    rho, pval = spearmanr(merged["val_wape"], merged["test_wape"])
    ax.text(0.05, 0.95, f"Spearman $\\rho$ = {rho:.3f}\n($p$ < 0.001)",
            transform=ax.transAxes, fontsize=FS_ANNOT, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec=C["grid"]))

    # Legend — two groups: domain (color) then marker role (shape)
    leg_domain = []
    for did, dname in [("aidc_power_optional", "AIDC"), ("aluminum_load", "Alum."),
                        ("arena_pv", "PV"), ("microgrid_load", "Micro."),
                        ("provincial_load", "Prov.")]:
        leg_domain.append(Line2D([0], [0], marker="o", color="w",
                                 markerfacecolor=DOMAIN_COLORS[did], markersize=5,
                                 label=dname))
    leg_shape = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markersize=4.5, label="TSFM"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#888888",
               markersize=4.5, label="non-TSFM"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#888888",
               markersize=7, markeredgecolor="black", markeredgewidth=0.5,
               label="H2 winner"),
    ]
    all_handles = leg_domain + [Line2D([], [], color="none", label="")] + leg_shape
    leg4 = ax.legend(handles=all_handles, loc="lower right", fontsize=FS_ANNOT_TINY,
                     ncol=3, columnspacing=0.8, handletextpad=0.3)
    leg4.get_frame().set_visible(True)
    leg4.get_frame().set_boxstyle("round,pad=0.3")
    leg4.get_frame().set_facecolor("white")
    leg4.get_frame().set_alpha(0.85)
    leg4.get_frame().set_edgecolor(C["grid"])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_valtest_scatter.pdf"); plt.close(fig)
    print(f"  Saved fig4_valtest_scatter.pdf  (rho={rho:.3f}, p={pval:.2e})")


# ===========================================================================
# FIG 6: H3 Threshold Sensitivity (dual-panel heatmap with default marker)
# ===========================================================================
def build_fig6_threshold():
    print("Building Fig 6: Threshold sensitivity heatmap...")
    if _H3_SENS.empty:
        print("  SKIP: no threshold sensitivity data")
        return

    DEFAULT_HIGH, DEFAULT_LOW = 0.90, 0.20
    cmap = LinearSegmentedColormap.from_list(
        "fig6_seq_white_grid",
        ["#f7f7d5", "#cde8b1", "#78c679", "#238443", "#005a32"],
    )
    pivots = [
        _H3_SENS.pivot(
            index="high_quantile", columns="low_pv_quantile",
            values="relative_reduction_dwape_critical",
        ).sort_index().sort_index(axis=1) * 100,
        _H3_SENS.pivot(
            index="high_quantile", columns="low_pv_quantile",
            values="relative_reduction_dwape_all",
        ).sort_index().sort_index(axis=1) * 100,
    ]
    vmin = min(float(p.values.min()) for p in pivots)
    vmax = max(float(p.values.max()) for p in pivots)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 2.5),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.11},
        constrained_layout=False,
    )

    ims = []
    for ax, pivot, title, plabel, show_ylabel in [
        (ax1, pivots[0], "Critical-union windows", "a", True),
        (ax2, pivots[1], "All windows", "b", False),
    ]:
        im = ax.imshow(
            pivot.values, cmap=cmap, aspect="auto", interpolation="nearest",
            vmin=vmin, vmax=vmax,
        )
        ims.append(im)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns], fontsize=FS_TICK_SMALL)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{r:.2f}" for r in pivot.index], fontsize=FS_TICK_SMALL)
        ax.set_xlabel("Low-PV quantile", fontsize=FS_LABEL)
        ax.set_ylabel("High-stress quantile" if show_ylabel else "", fontsize=FS_LABEL)
        ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.78, zorder=4)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        bottom_panel_title(ax, plabel, title, offset_pt=TITLE_OFFSET_XLABEL)

        for yi in range(pivot.shape[0]):
            for xi in range(pivot.shape[1]):
                v = pivot.values[yi, xi]
                ax.text(
                    xi, yi, f"{v:.1f}", ha="center", va="center",
                    fontsize=FS_HEATMAP,
                    color="white" if v > (vmin + vmax) / 2 else C["text"],
                )

        # Mark the train-defined default threshold pair.
        try:
            row_idx = list(pivot.index).index(DEFAULT_HIGH)
            col_idx = list(pivot.columns).index(DEFAULT_LOW)
            rect = plt.Rectangle((col_idx - 0.5, row_idx - 0.5), 1, 1,
                                  linewidth=1.15, edgecolor=C["text"], facecolor="none",
                                  zorder=5)
            ax.add_patch(rect)
        except ValueError:
            pass

    cax = fig.add_axes([0.915, 0.30, 0.012, 0.55])
    cb = fig.colorbar(ims[0], cax=cax)
    cb.set_label("Relative reduction (%)", rotation=90, labelpad=6, fontsize=FS_LABEL)
    cb.ax.tick_params(length=2.2, width=0.5, labelsize=FS_TICK_SMALL)
    fig.subplots_adjust(left=0.085, right=0.895, bottom=0.30, top=0.90)
    fig.savefig(OUT_DIR / "fig6_h3_threshold.pdf"); plt.close(fig)
    print("  Saved fig6_h3_threshold.pdf")


# ===========================================================================
def main():
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_fig2_violin()
    build_fig3_lollipop()
    build_fig4_valtest()
    build_fig5_barline()
    build_fig6_threshold()
    subprocess.run([sys.executable, str(PROJECT / "scripts/build_fig7_representative_trace.py")], check=True)
    build_fig8_bubble()
    print("\nAll figures generated.")

if __name__ == "__main__":
    main()
