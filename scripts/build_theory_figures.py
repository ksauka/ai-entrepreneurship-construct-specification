"""Theory-elaboration matrices used in the manuscript results.

Reads the frozen primary dataset and the frozen domain assignments. Writes PNGs
to reports/analysis/figures/contrasting/. No new coding is introduced. The
vertical matrices use the registered ``level_of_analysis`` field directly and,
where a more compact display is required, aggregate its controlled categories
using the documented mapping below.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "data/processed/analysis/primary_analysis_dataset.csv"
DOMAINS = ROOT / "data/processed/analysis/theory_elaboration/domains/business_domain_assignments.csv"
FIG = ROOT / "reports/analysis/figures/contrasting"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

# Reader-facing aggregation of the frozen level_of_analysis categories. The
# direct registered-level matrix is retained separately.
LEVEL_MAP = {
    "individual entrepreneur": "Individual",
    "founding team": "Team",
    "venture": "Org. entity",
    "firm": "Org. entity",
    "platform": "Platform",
    "industry": "Market/ecosystem",
    "ecosystem": "Market/ecosystem",
    "national system": "Institutional/system",
    "institutional environment": "Institutional/system",
    "multi-level": "Multi-level",
}
LEVEL_ORDER = ["Individual", "Team", "Org. entity", "Platform",
               "Market/ecosystem", "Institutional/system", "Multi-level"]
ROLE_EXCLUDE = {"", "AI as unspecified label"}
MECH_EXCLUDE = {"", "mechanism missing"}
TYPE_EXCLUDE = {"", "unspecified AI"}


def heatmap(pivot, title, name, cbar="Row share (%)"):
    fig, ax = plt.subplots(figsize=(1.15 * len(pivot.columns) + 3.5, 0.62 * len(pivot.index) + 2))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0, vmax=np.nanmax(data))
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v) and v > 0:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if v > np.nanmax(data) * 0.6 else "black", fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar)
    fig.tight_layout(); fig.savefig(FIG / name, bbox_inches="tight"); plt.close(fig)
    print("wrote", name)


def main():
    d = pd.read_csv(PRIMARY, dtype=str, keep_default_na=False)
    q3 = d["in_query_3"].str.strip().str.lower().isin({"1", "true", "yes", "y", "x"})
    q4 = d["in_query_4"].str.strip().str.lower().isin({"1", "true", "yes", "y", "x"})
    ent = d[q3 | q4].copy()
    ent["level_bucket"] = ent["level_of_analysis"].str.strip().map(LEVEL_MAP)

    unmapped = set(ent["level_of_analysis"].str.strip().unique()) - set(LEVEL_MAP) - {"", "unspecified level"}
    if unmapped:
        print("WARNING unmapped level values:", unmapped)

    role = ent["ai_role_function"].str.strip()
    mech = ent["ai_mechanism_analysis"].str.strip()

    # Vertical: role by collapsed level (row % = where each role lives)
    v = ent[~role.isin(ROLE_EXCLUDE) & ent["level_bucket"].notna()]
    t = pd.crosstab(v["ai_role_function"].str.strip(), v["level_bucket"])
    t = t.reindex(columns=[c for c in LEVEL_ORDER if c in t.columns])
    t = t.loc[t.sum(axis=1).sort_values(ascending=False).index]
    heatmap(t.div(t.sum(axis=1), axis=0) * 100,
            "Vertical contrast: AI role by level (row %, entrepreneurship corpus)\n"
            "Registered levels grouped for a compact display",
            "vertical_role_by_collapsed_level.png")

    # Vertical: mechanism by collapsed level
    vm = ent[~mech.isin(MECH_EXCLUDE) & ent["level_bucket"].notna()]
    tm = pd.crosstab(vm["ai_mechanism_analysis"].str.strip(), vm["level_bucket"])
    tm = tm.reindex(columns=[c for c in LEVEL_ORDER if c in tm.columns])
    tm = tm.loc[tm.sum(axis=1).sort_values(ascending=False).index]
    heatmap(tm.div(tm.sum(axis=1), axis=0) * 100,
            "Vertical contrast: mechanism by level (row %, entrepreneurship corpus)\n"
            "Registered levels grouped for a compact display",
            "vertical_mechanism_by_collapsed_level.png")

    # Construct specification: technical type by theoretical role. Percentages
    # are conditional on each named technical type, so the rows answer whether
    # the same technical label is assigned different theoretical work.
    tr = ent[
        ~ent["ai_type_form"].str.strip().isin(TYPE_EXCLUDE)
        & ~ent["ai_role_function"].str.strip().isin(ROLE_EXCLUDE)
    ]
    ttr = pd.crosstab(
        tr["ai_type_form"].str.strip(), tr["ai_role_function"].str.strip()
    )
    ttr = ttr.loc[ttr.sum(axis=1).sort_values(ascending=False).index]
    ttr = ttr.reindex(columns=ttr.sum(axis=0).sort_values(ascending=False).index)
    heatmap(
        ttr.div(ttr.sum(axis=1), axis=0) * 100,
        "Construct specification: theoretical role within each named AI type "
        "(row %, entrepreneurship corpus)",
        "specification_type_by_role.png",
        cbar="Share within technical type (%)",
    )

    # Horizontal: AI role composition across business domains (col % within domain)
    dom = pd.read_csv(DOMAINS, dtype=str, keep_default_na=False)[["paper_id", "domain_label"]]
    merged = d.merge(dom, on="paper_id", how="inner")
    mr = merged[~merged["ai_role_function"].str.strip().isin(ROLE_EXCLUDE)]
    ct = pd.crosstab(mr["ai_role_function"].str.strip(), mr["domain_label"])
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
    ct = ct.reindex(columns=ct.sum(axis=0).sort_values(ascending=False).index)
    heatmap(ct.div(ct.sum(axis=0), axis=1) * 100,
            "Horizontal contrast: AI role composition by business domain (column %)",
            "horizontal_role_by_domain.png", cbar="Share within domain (%)")


if __name__ == "__main__":
    main()
