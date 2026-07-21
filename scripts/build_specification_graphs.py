"""Render the construct-specification graphs per entrepreneurship population.

For each population (full corpus, top-tier/core entrepreneurship, other
entrepreneurship, combined) this renders the per-dimension observed-composition
panels that are the Stage 1 construct-specification portrait. It reuses the
platform's analyze_observed_composition so the graphs match the interactive
Observed Composition view exactly.

Read-only on the frozen primary dataset. Writes PNGs to
reports/analysis/figures/specification/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from aecsp.analytics.observed_composition import analyze_observed_composition

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "data/processed/analysis/primary_analysis_dataset.csv"
FIGURES = ROOT / "reports/analysis/figures/specification"
FIGURES.mkdir(parents=True, exist_ok=True)

BAR = "#2b6cb0"
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "x"})


def populations(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    q3 = _truthy(frame["in_query_3"]) if "in_query_3" in frame.columns else pd.Series(False, index=frame.index)
    q4 = _truthy(frame["in_query_4"]) if "in_query_4" in frame.columns else pd.Series(False, index=frame.index)
    return {
        "full_corpus": (frame, "Full corpus"),
        "core_entrepreneurship": (frame[q3], "Top-tier entrepreneurship (Query 3)"),
        "other_entrepreneurship": (frame[q4], "Other entrepreneurship (Query 4)"),
        "combined_entrepreneurship": (frame[q3 | q4], "Combined entrepreneurship"),
    }


def render_population(name: str, frame: pd.DataFrame, label: str) -> None:
    result = analyze_observed_composition(frame, study_status="all")
    panels = [p for p in result["panels"] if p["observed_n"] > 0]
    n = len(panels)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.5 * nrow))
    axes = axes.flatten()
    for ax, panel in zip(axes, panels):
        cats = panel["chart_categories"][:8][::-1]
        names = [c["value"] for c in cats]
        shares = [c["share"] * 100 for c in cats]
        ax.barh(range(len(names)), shares, color=BAR)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(
            f"{panel['label']}  (n={panel['observed_n']:,}, "
            f"{panel['observed_share']*100:.0f}% of papers)",
            fontsize=9,
        )
        ax.set_xlabel("Share of observed codes (%)", fontsize=8)
        for i, s in enumerate(shares):
            ax.text(s + 0.5, i, f"{s:.0f}", va="center", fontsize=7)
        ax.set_xlim(0, max(shares) * 1.18 if shares else 1)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(
        f"Construct specification: what {label} contains, by dimension "
        f"(observed view, N={len(frame):,})",
        fontsize=12, y=1.002,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out = FIGURES / f"specification_{name}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}  ({label}, N={len(frame):,})")


def main() -> None:
    frame = pd.read_csv(PRIMARY, dtype=str, keep_default_na=False)
    for name, (sub, label) in populations(frame).items():
        render_population(name, sub, label)


if __name__ == "__main__":
    main()
