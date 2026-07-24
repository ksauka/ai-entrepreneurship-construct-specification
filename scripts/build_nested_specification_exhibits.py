"""Build manuscript and supplementary exhibits for nested specification.

The Construct Specification platform can condition every distribution on an
exact value of any of the eight coding dimensions.  This script renders the
study-status slice requested for the manuscript and supplementary appendix
from the frozen tidy tables produced by ``freeze_contrasting_tables.py``.

No paper-level data are changed.  Percentages are conditional on the observed
denominator for the displayed outcome dimension unless a figure explicitly
reports observability against the full controlled subset.
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
SOURCE = TABLES / "study_status_conditioned_specification.csv"

STATUS_ORDER = ("phenomenon", "method", "both")
STATUS_LABELS = {
    "phenomenon": "Phenomenon",
    "method": "Method",
    "both": "Both",
}
DIMENSION_ORDER = (
    "ai_role",
    "technical_type",
    "mechanism",
    "level",
    "process_stage",
    "scope",
    "definition",
)
DIMENSION_LABELS = {
    "ai_role": "AI role",
    "technical_type": "Technical type",
    "mechanism": "Mechanism",
    "level": "Level",
    "process_stage": "Process stage",
    "scope": "Scope",
    "definition": "Definition",
}
CENTRAL_TOP_N = {
    "ai_role": 5,
    "technical_type": 5,
    "mechanism": 6,
    "level": 5,
}

plt.rcParams.update({"font.size": 9, "figure.dpi": 180})


def load_status_table() -> pd.DataFrame:
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE}. Run scripts/freeze_contrasting_tables.py first."
        )
    table = pd.read_csv(SOURCE, keep_default_na=False)
    table["share"] = pd.to_numeric(table["share"], errors="coerce").fillna(0.0)
    table["papers"] = pd.to_numeric(table["papers"], errors="coerce").fillna(0).astype(int)
    table["denominator"] = (
        pd.to_numeric(table["denominator"], errors="coerce").fillna(0).astype(int)
    )
    table["control_papers"] = (
        pd.to_numeric(table["control_papers"], errors="coerce").fillna(0).astype(int)
    )
    return table


def dimension_slice(
    table: pd.DataFrame,
    population: str,
    dimension: str,
) -> pd.DataFrame:
    return table.loc[
        table["population"].eq(population)
        & table["outcome_dimension"].eq(dimension)
        & table["distribution"].eq("observed")
        & table["control_raw_value"].isin(STATUS_ORDER)
    ].copy()


def category_order(frame: pd.DataFrame, limit: int | None = None) -> list[str]:
    totals = frame.groupby("category", sort=False)["papers"].sum().sort_values(ascending=False)
    categories = totals.index.astype(str).tolist()
    return categories if limit is None else categories[:limit]


def status_matrix(
    frame: pd.DataFrame,
    categories: list[str],
) -> tuple[np.ndarray, list[str]]:
    pivot = frame.pivot_table(
        index="category",
        columns="control_raw_value",
        values="share",
        aggfunc="first",
        fill_value=0.0,
    )
    pivot = pivot.reindex(index=categories, columns=STATUS_ORDER, fill_value=0.0)
    denominators = []
    for status in STATUS_ORDER:
        values = frame.loc[frame["control_raw_value"].eq(status), "denominator"]
        denominators.append(int(values.iloc[0]) if len(values) else 0)
    return pivot.to_numpy(dtype=float) * 100, [
        f"{STATUS_LABELS[status]}\nn={denominator:,}"
        for status, denominator in zip(STATUS_ORDER, denominators)
    ]


def annotate_heatmap(ax, values: np.ndarray, threshold: float | None = None) -> None:
    maximum = float(np.nanmax(values)) if values.size else 0.0
    threshold = maximum * 0.58 if threshold is None else threshold
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            ax.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if value >= threshold and threshold > 0 else "#1f2933",
            )


def build_central_profile(table: pd.DataFrame) -> None:
    rows: list[tuple[str, str, list[float]]] = []
    denominators: dict[str, list[str]] = {}
    for dimension, limit in CENTRAL_TOP_N.items():
        frame = dimension_slice(table, "combined", dimension)
        categories = category_order(frame, limit)
        values, column_labels = status_matrix(frame, categories)
        denominators[dimension] = column_labels
        for category, row in zip(categories, values):
            rows.append((dimension, category, row.tolist()))

    values = np.asarray([row[2] for row in rows], dtype=float)
    labels = [f"{DIMENSION_LABELS[dimension]}: {category}" for dimension, category, _ in rows]
    fig, ax = plt.subplots(figsize=(8.7, max(8.2, len(rows) * 0.36 + 2.0)))
    image = ax.imshow(values, cmap="Blues", aspect="auto", vmin=0, vmax=max(1, values.max()))
    ax.set_xticks(range(len(STATUS_ORDER)))
    # Denominators differ by dimension, so the central figure uses status labels
    # only; the corresponding denominators are reported in the table/caption.
    ax.set_xticklabels([STATUS_LABELS[status] for status in STATUS_ORDER])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    annotate_heatmap(ax, values)
    cursor = 0
    for dimension, limit in CENTRAL_TOP_N.items():
        cursor += min(limit, len(category_order(dimension_slice(table, "combined", dimension))))
        if cursor < len(rows):
            ax.axhline(cursor - 0.5, color="white", linewidth=2.5)
    ax.set_title(
        "Nested construct specification by study status\n"
        "Combined entrepreneurship, share within each observed dimension (%)",
        fontsize=11,
    )
    fig.colorbar(image, ax=ax, fraction=0.028, pad=0.025, label="Share of observed codes (%)")
    fig.tight_layout()
    path = FIGURES / "nested_status_central_dimensions.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.name}")


def build_all_dimension_panels(table: pd.DataFrame) -> None:
    # A 3 x 3 layout keeps the complete figure and its caption on one portrait
    # supplementary page while retaining enough height for long category lists.
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 15))
    axes = axes.flatten()
    for ax, dimension in zip(axes, DIMENSION_ORDER):
        frame = dimension_slice(table, "combined", dimension)
        categories = category_order(frame)
        values, column_labels = status_matrix(frame, categories)
        image = ax.imshow(
            values,
            cmap="Blues",
            aspect="auto",
            vmin=0,
            vmax=max(1, float(values.max()) if values.size else 1),
        )
        ax.set_xticks(range(len(STATUS_ORDER)))
        ax.set_xticklabels(column_labels, fontsize=8)
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories, fontsize=7.5)
        annotate_heatmap(ax, values)
        ax.set_title(DIMENSION_LABELS[dimension], fontweight="bold")
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    for ax in axes[len(DIMENSION_ORDER) :]:
        ax.axis("off")
    fig.suptitle(
        "Complete study-status-conditioned construct composition\n"
        "Combined entrepreneurship, observed view; each panel has its own disclosed denominator",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = FIGURES / "nested_status_all_dimensions.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.name}")


def build_observability_figure(table: pd.DataFrame) -> None:
    values = []
    for dimension in DIMENSION_ORDER:
        frame = dimension_slice(table, "combined", dimension)
        row = []
        for status in STATUS_ORDER:
            selected = frame.loc[frame["control_raw_value"].eq(status)]
            denominator = int(selected["denominator"].iloc[0]) if len(selected) else 0
            control_n = int(selected["control_papers"].iloc[0]) if len(selected) else 0
            row.append(denominator / control_n * 100 if control_n else 0.0)
        values.append(row)
    matrix = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(7.7, 5.8))
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(STATUS_ORDER)))
    ax.set_xticklabels([STATUS_LABELS[status] for status in STATUS_ORDER])
    ax.set_yticks(range(len(DIMENSION_ORDER)))
    ax.set_yticklabels([DIMENSION_LABELS[item] for item in DIMENSION_ORDER])
    annotate_heatmap(ax, matrix, threshold=62)
    ax.set_title(
        "Observability by study status\n"
        "Observed denominator as a share of each status subset (%)"
    )
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03, label="Papers with an observed code (%)")
    fig.tight_layout()
    path = FIGURES / "nested_status_observability.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.name}")


def build_core_additional_contrasts(table: pd.DataFrame) -> None:
    selected = table.loc[
        table["population"].isin(["core", "other"])
        & table["distribution"].eq("observed")
        & table["control_raw_value"].isin(STATUS_ORDER)
    ].copy()
    keys = [
        "control_raw_value",
        "outcome_dimension",
        "outcome_dimension_label",
        "category",
        "raw_value",
    ]
    share = selected.pivot_table(
        index=keys,
        columns="population",
        values="share",
        aggfunc="first",
        fill_value=0.0,
    ).reset_index()
    papers = selected.pivot_table(
        index=keys,
        columns="population",
        values="papers",
        aggfunc="first",
        fill_value=0,
    ).reset_index()
    denominators = selected.pivot_table(
        index=["control_raw_value", "outcome_dimension"],
        columns="population",
        values="denominator",
        aggfunc="first",
        fill_value=0,
    ).reset_index()
    result = share.rename(columns={"core": "core_share", "other": "additional_share"})
    result = result.merge(
        papers.rename(columns={"core": "core_papers", "other": "additional_papers"}),
        on=keys,
        how="left",
    ).merge(
        denominators.rename(
            columns={"core": "core_denominator", "other": "additional_denominator"}
        ),
        on=["control_raw_value", "outcome_dimension"],
        how="left",
    )
    result["core_minus_additional_pp"] = (
        (result["core_share"] - result["additional_share"]) * 100
    ).round(2)
    result.sort_values(
        ["control_raw_value", "outcome_dimension", "core_minus_additional_pp"],
        ascending=[True, True, False],
    ).to_csv(TABLES / "nested_status_core_additional_contrasts.csv", index=False)
    print("wrote nested_status_core_additional_contrasts.csv")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    table = load_status_table()
    build_central_profile(table)
    build_all_dimension_panels(table)
    build_observability_figure(table)
    build_core_additional_contrasts(table)


if __name__ == "__main__":
    main()
