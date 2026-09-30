"""Static PNG figures for the benchmark report (categorical colours in fixed slot order,
single-hue sequential ramp for weight magnitudes, recessive grid, one y-axis per chart)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import config as C  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODEL_ORDER = ["climatology", "persistence", "recent3", "anom_persistence", "static_blend", "gating_adaptive", "lgbm_season"]


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def rmse_bars(pm_season: pd.DataFrame, target: str, unit: str, path) -> None:
    d = pm_season[pm_season.target == target].pivot(index="season", columns="model", values="rmse").reindex(C.SEASONS)
    models = [m for m in MODEL_ORDER if m in d.columns]
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=150)
    x = np.arange(len(C.SEASONS))
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        ax.bar(x + (k - len(models) / 2 + 0.5) * w, d[m], w * 0.92, color=SERIES[k], label=m, zorder=2)
    ax.set_xticks(x, [s.replace("_", "-") for s in C.SEASONS], color=INK)
    ax.set_ylabel(f"RMSE ({unit})", color=INK2, fontsize=9)
    ax.set_title(f"{target}: test RMSE by season (leads 1-3 d pooled, lower is better)", fontsize=10, color=INK, loc="left")
    _style(ax)
    ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def weight_maps(w_cell: pd.DataFrame, target: str, path) -> None:
    d = w_cell[w_cell.target == target]
    fig, axes = plt.subplots(len(C.EXPERTS), len(C.SEASONS), figsize=(11, 10.5), dpi=130, sharex=True, sharey=True)
    im = None
    for i, e in enumerate(C.EXPERTS):
        for j, s in enumerate(C.SEASONS):
            ax = axes[i, j]
            g = d[d.season == s]
            im = ax.scatter(g.longitude, g.latitude, c=g[f"w_{e}"], cmap="Blues", vmin=0, vmax=1, s=4, marker="s", linewidths=0)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6, colors=INK2)
            for sp in ax.spines.values():
                sp.set_color(GRID)
            if i == 0:
                ax.set_title(s.replace("_", "-"), fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(e, fontsize=9, color=INK)
    cb = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.02)
    cb.set_label("mean gating weight (lead 1 d)", fontsize=8, color=INK2)
    fig.suptitle(f"Model weight maps - {target}: learned weight of each expert per grid cell and season", fontsize=11, color=INK, x=0.02, ha="left")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def reliability(rel: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, len(C.FLAGS), figsize=(11, 3.4), dpi=150)
    for ax, f in zip(axes, C.FLAGS):
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="perfect")
        for k, m in enumerate(["uncalibrated", "calibrated"]):
            g = rel[(rel.flag == f) & (rel.model == m) & (rel.n >= 30)]
            ax.plot(g.mean_pred, g.obs_freq, marker="o", ms=4, lw=2, color=SERIES[k], label=m)
        ax.set_title(f, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("forecast probability", fontsize=8, color=INK2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        _style(ax)
    axes[0].set_ylabel("observed frequency", fontsize=8, color=INK2)
    axes[-1].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def spread_skill(dis: pd.DataFrame, path) -> None:
    d = dis[dis.season == "ALL"]
    fig, axes = plt.subplots(1, len(C.TARGETS), figsize=(11, 3.0), dpi=150)
    for ax, (_, r) in zip(axes, d.iterrows()):
        vals = [r[f"Q{i}_rmse"] for i in range(1, 6)]
        ax.bar(range(1, 6), vals, 0.7, color=SERIES[0], zorder=2)
        ax.set_title(f"{r.target} (Spearman {r.spearman_spread_vs_abs_err:.2f})", fontsize=9, color=INK, loc="left")
        ax.set_xticks(range(1, 6), ["Q1\nlow", "Q2", "Q3", "Q4", "Q5\nhigh"])
        ax.set_xlabel("expert-disagreement quintile", fontsize=8, color=INK2)
        _style(ax)
    axes[0].set_ylabel("RMSE of adaptive blend", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
