"""Render story-led figures for the contrasting results from the frozen tables.

Reads only ``reports/analysis/tables/contrasting/*.csv`` (frozen, checksummed) and
writes PNG figures to ``reports/analysis/figures/contrasting/``. No recomputation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports/analysis/tables/contrasting"
FIGURES = ROOT / "reports/analysis/figures/contrasting"
FIGURES.mkdir(parents=True, exist_ok=True)

PREDICT = "#2b6cb0"
GENERATIVE = "#c05621"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 150})


def horizontal_split() -> None:
    d = pd.read_csv(TABLES / "horizontal_domain_contrast_full_corpus.csv")

    def share(dim: str, raw: str) -> pd.Series:
        s = d[(d.dimension_id == dim) & (d.raw_value == raw)]
        return s.set_index("domain_label")["share"] * 100

    predict = share("mechanism", "improves prediction")
    generative = share("technical_type", "generative AI")
    frame = pd.DataFrame({"predict": predict, "generative": generative}).dropna()
    frame = frame.sort_values("predict", ascending=True)

    y = np.arange(len(frame))
    height = 0.38
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(y + height / 2, frame["predict"], height, label="Improves prediction (of observed mechanisms)", color=PREDICT)
    ax.barh(y - height / 2, frame["generative"], height, label="Generative AI (of named types)", color=GENERATIVE)
    ax.axvline(42.9, color=PREDICT, ls=":", lw=1)
    ax.axvline(7.0, color=GENERATIVE, ls=":", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(frame.index)
    ax.set_xlabel("Share (%)")
    ax.set_title("Two theories of AI across domains: prediction versus generation")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.text(43.6, -0.9, "corpus 42.9%", color=PREDICT, fontsize=7)
    ax.text(7.6, -0.9, "corpus 7.0%", color=GENERATIVE, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "horizontal_prediction_vs_generative.png")
    plt.close(fig)
    print("wrote horizontal_prediction_vs_generative.png")


def _heatmap(pivot: pd.DataFrame, title: str, name: str, fmt: str = "{:.0f}") -> None:
    fig, ax = plt.subplots(figsize=(1.1 * len(pivot.columns) + 3, 0.6 * len(pivot.index) + 2))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0, vmax=np.nanmax(data))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v) and v > 0:
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        color="white" if v > np.nanmax(data) * 0.6 else "black", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row share (%)")
    fig.tight_layout()
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print("wrote", name)


def vertical_role_by_level() -> None:
    d = pd.read_csv(TABLES / "vertical_dimension_by_level.csv")
    d = d[d.row_dimension == "ai_role"].copy()
    d["pct"] = d["share_within_row"] * 100
    pivot = d.pivot_table(index="row_value", columns="column_value", values="pct", aggfunc="first")
    # order rows and columns by total papers for readability
    role_order = d.groupby("row_value")["papers"].sum().sort_values(ascending=False).index
    level_order = d.groupby("column_value")["papers"].sum().sort_values(ascending=False).index[:6]
    pivot = pivot.reindex(index=role_order, columns=level_order)
    _heatmap(pivot, "AI role by level of analysis (row %, entrepreneurship corpus)",
             "vertical_role_by_level.png")


def structuring_role_by_mechanism() -> None:
    d = pd.read_csv(TABLES / "structuring_pair_matrices.csv")
    d = d[d.pair == "ai_role__mechanism"].copy()
    d["pct"] = d["share_within_row"] * 100
    pivot = d.pivot_table(index="row_value", columns="column_value", values="pct", aggfunc="first")
    role_order = d.groupby("row_value")["papers"].sum().sort_values(ascending=False).index
    mech_order = d.groupby("column_value")["papers"].sum().sort_values(ascending=False).index[:7]
    pivot = pivot.reindex(index=role_order, columns=mech_order)
    _heatmap(pivot, "AI role by observed mechanism (row %, entrepreneurship corpus)",
             "structuring_role_by_mechanism.png")


if __name__ == "__main__":
    horizontal_split()
    vertical_role_by_level()
    structuring_role_by_mechanism()
