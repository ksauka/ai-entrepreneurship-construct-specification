"""Remaining blueprint exhibits: entrepreneurship-inclusive role-by-domain figure,
the construct-clarification framework diagram, and the core-vs-other and
recurring-configuration tables. Read-only on the frozen primary dataset.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "data/processed/analysis/primary_analysis_dataset.csv"
DOMAINS = ROOT / "data/processed/analysis/theory_elaboration/domains/business_domain_assignments.csv"
FIG = ROOT / "reports/analysis/figures/contrasting"
TAB = ROOT / "reports/analysis/tables/contrasting"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

ROLE_EXCLUDE = {"", "AI as unspecified label"}
TYPE_EXCLUDE = {"", "unspecified AI"}
MECH_EXCLUDE = {"", "mechanism missing"}


def _truthy(s):
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "x"})


def role_by_domain_with_ent(d):
    dom = pd.read_csv(DOMAINS, dtype=str, keep_default_na=False)[["paper_id", "domain_label"]]
    merged = d.merge(dom, on="paper_id", how="inner")
    frames = [merged[["paper_id", "ai_role_function", "domain_label"]]]
    q2, q3, q4 = _truthy(d["in_query_2"]), _truthy(d["in_query_3"]), _truthy(d["in_query_4"])
    for mask, label in [(q3, "Core entrepr."), (q4, "Other entrepr."),
                        (q3 | q4, "Combined entrepr."), (q2, "FT50")]:
        sub = d[mask][["paper_id", "ai_role_function"]].copy()
        sub["domain_label"] = label
        frames.append(sub)
    allrows = pd.concat(frames, ignore_index=True)
    allrows = allrows[~allrows["ai_role_function"].str.strip().isin(ROLE_EXCLUDE)]
    ct = pd.crosstab(allrows["ai_role_function"].str.strip(), allrows["domain_label"])
    ent_cols = ["Core entrepr.", "Other entrepr.", "Combined entrepr.", "FT50"]
    biz = [c for c in ct.columns if c not in ent_cols]
    biz = ct[biz].sum().sort_values(ascending=False).index.tolist()
    ct = ct.reindex(columns=biz + [c for c in ent_cols if c in ct.columns])
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
    pct = ct.div(ct.sum(axis=0), axis=1) * 100

    fig, ax = plt.subplots(figsize=(1.05 * len(pct.columns) + 3, 0.62 * len(pct.index) + 2))
    data = pct.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0, vmax=np.nanmax(data))
    ax.set_xticks(range(len(pct.columns))); ax.set_xticklabels(pct.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pct.index))); ax.set_yticklabels(pct.index)
    # separator before entrepreneurship columns
    sep = len(biz) - 0.5
    ax.axvline(sep, color="#c05621", lw=2)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if v > np.nanmax(data) * 0.6 else "black", fontsize=8)
    ax.set_title("Horizontal contrast: AI role composition by domain, entrepreneurship at right (column %)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Share within domain (%)")
    fig.tight_layout(); fig.savefig(FIG / "horizontal_role_by_domain_with_ent.png", bbox_inches="tight"); plt.close(fig)
    print("wrote horizontal_role_by_domain_with_ent.png")


def framework_figure():
    steps = [
        ("Construct specification", "What does each study mean by AI?"),
        ("Horizontal contrasting", "Does that meaning change across business domains?"),
        ("Vertical contrasting", "Does it change across levels of analysis?"),
        ("Structuring", "Which role-mechanism-level-scope combinations recur?"),
        ("Configurational identity", "The theoretical identity of AI is in the configuration, not the label"),
        ("Entrepreneurship insight", "As AI expands prediction and generation, the bottleneck moves to evaluation and commitment"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 10))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")
    y = 11
    colors = ["#2b6cb0", "#2b6cb0", "#2b6cb0", "#2b6cb0", "#276749", "#c05621"]
    for (title, sub), c in zip(steps, colors):
        box = FancyBboxPatch((1, y - 1.1), 8, 1.25, boxstyle="round,pad=0.08",
                             linewidth=1.5, edgecolor=c, facecolor="white")
        ax.add_patch(box)
        ax.text(5, y - 0.32, title, ha="center", va="center", fontsize=12, fontweight="bold", color=c)
        ax.text(5, y - 0.82, sub, ha="center", va="center", fontsize=8.5, wrap=True)
        if y > 3:
            ax.add_patch(FancyArrowPatch((5, y - 1.15), (5, y - 1.75),
                         arrowstyle="-|>", mutation_scale=18, color="#555"))
        y -= 1.85
    ax.text(5, 0.3, "Construct-clarification framework", ha="center", fontsize=11,
            style="italic", color="#333")
    fig.tight_layout(); fig.savefig(FIG / "framework_diagram.png", bbox_inches="tight"); plt.close(fig)
    print("wrote framework_diagram.png")


def observed_share(sub, col, exclude):
    v = sub[col].str.strip()
    v = v[~v.isin(exclude)]
    if len(v) == 0:
        return {}
    return (v.value_counts() / len(v) * 100).round(1).to_dict()


def core_vs_other_table(d):
    q3, q4 = _truthy(d["in_query_3"]), _truthy(d["in_query_4"])
    pops = {"Core": d[q3], "Other": d[q4], "Combined": d[q3 | q4]}
    dims = [("Study status", "ai_method_or_phenomenon", {"", "unclear"}),
            ("AI role", "ai_role_function", ROLE_EXCLUDE),
            ("Technical type", "ai_type_form", TYPE_EXCLUDE),
            ("Mechanism", "ai_mechanism_analysis", MECH_EXCLUDE),
            ("Level", "level_of_analysis", {"", "unspecified level"})]
    rows = []
    for dlabel, col, excl in dims:
        cats = set()
        dist = {p: observed_share(sub, col, excl) for p, sub in pops.items()}
        for p in dist:
            cats |= set(dist[p])
        for cat in sorted(cats, key=lambda c: -dist["Combined"].get(c, 0)):
            rows.append({"dimension": dlabel, "category": cat,
                         "core_pct": dist["Core"].get(cat, 0.0),
                         "other_pct": dist["Other"].get(cat, 0.0),
                         "combined_pct": dist["Combined"].get(cat, 0.0)})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "table3_core_vs_other.csv", index=False)
    print("wrote table3_core_vs_other.csv", df.shape)


def configurations_table(d):
    q3, q4 = _truthy(d["in_query_3"]), _truthy(d["in_query_4"])
    ent = d[q3 | q4].copy()
    role = ent["ai_role_function"].str.strip()
    mech = ent["ai_mechanism_analysis"].str.strip()
    obs = ent[~role.isin(ROLE_EXCLUDE) & ~mech.isin(MECH_EXCLUDE)].copy()
    obs["config"] = obs["ai_role_function"].str.strip() + " x " + obs["ai_mechanism_analysis"].str.strip()
    obs["is_core"] = _truthy(obs["in_query_3"])
    rows = []
    for cfg, grp in obs.groupby("config"):
        if len(grp) < 10:
            continue
        titles = grp["title"].head(2).tolist() if "title" in grp.columns else []
        rows.append({"configuration": cfg, "papers": len(grp),
                     "share_of_ent_observed": round(len(grp) / len(obs) * 100, 1),
                     "core_papers": int(grp["is_core"].sum()),
                     "other_papers": int((~grp["is_core"]).sum()),
                     "example_1": titles[0] if titles else "",
                     "example_2": titles[1] if len(titles) > 1 else ""})
    df = pd.DataFrame(rows).sort_values("papers", ascending=False)
    df.to_csv(TAB / "table4_configurations.csv", index=False)
    print("wrote table4_configurations.csv", df.shape)


def main():
    d = pd.read_csv(PRIMARY, dtype=str, keep_default_na=False)
    role_by_domain_with_ent(d)
    framework_figure()
    core_vs_other_table(d)
    configurations_table(d)


if __name__ == "__main__":
    main()
