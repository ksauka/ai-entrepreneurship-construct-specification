"""Build data-specific Stage 4 topic tables and figures.

Inputs: the checksummed primary analysis dataset, the final global topic table,
and the four native query-scope assignment tables. Outputs: a one-to-one
topic-enriched study table, scope-keyed tables, figures, and a run manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / "data/interim/runtime_cache/matplotlib")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aecsp.analytics.observed_composition import (  # noqa: E402
    OBSERVED_COMPOSITION_PANELS,
)
from aecsp.corpus.scopes import SCOPE_BY_ID  # noqa: E402

PRIMARY = PROJECT_ROOT / "data/processed/analysis/primary_analysis_dataset.csv"
PRIMARY_MANIFEST = PROJECT_ROOT / "data/processed/analysis/dataset_manifest.json"
TOPICS = PROJECT_ROOT / "data/processed/master_corpus_topics.csv"
TOPIC_RUN_MANIFEST = PROJECT_ROOT / "data/processed/topics/final_run_manifest.json"
ENRICHED = (
    PROJECT_ROOT
    / "data/processed/analysis/primary_analysis_dataset_with_topics.csv"
)
DATA_OUTPUT = PROJECT_ROOT / "data/processed/analysis/stage4"
TABLE_OUTPUT = PROJECT_ROOT / "reports/analysis/tables/stage4"
FIGURE_OUTPUT = PROJECT_ROOT / "reports/analysis/figures/stage4"
TOPIC_LABEL_REVIEW = DATA_OUTPUT / "topic_label_review.csv"

GLOBAL_TOPIC_COLUMNS = (
    "bertopic_topic",
    "bertopic_topic_label",
    "bertopic_topic_prob",
    "bertopic_was_outlier",
    "ai_terms",
    "ai_term_count",
    "ent_terms",
    "ent_term_count",
    "keybert_phrases",
)
# Backward-compatible name used by tests and external imports.
TOPIC_COLUMNS = GLOBAL_TOPIC_COLUMNS

SCOPE_CONFIG = {
    "full_corpus": {
        "display": SCOPE_BY_ID["full_corpus"].label,
        "model": "global",
        "flag": None,
        "topic_id": "bertopic_topic",
        "topic_label": "bertopic_topic_label",
        "topic_prob": "bertopic_topic_prob",
        "was_outlier": "bertopic_was_outlier",
        "expected_topics": 53,
        "assignment_path": None,
    },
    "query_1": {
        "display": SCOPE_BY_ID["query_1"].label,
        "model": "native",
        "flag": "in_query_1",
        "topic_id": "query_1_topic_id",
        "topic_label": "query_1_topic_label",
        "topic_prob": "query_1_topic_prob",
        "was_outlier": "query_1_was_outlier",
        "expected_topics": 50,
        "assignment_path": PROJECT_ROOT
        / "data/processed/topics/native/query_1/assignments.csv",
    },
    "query_2": {
        "display": SCOPE_BY_ID["query_2"].label,
        "model": "native",
        "flag": "in_query_2",
        "topic_id": "query_2_topic_id",
        "topic_label": "query_2_topic_label",
        "topic_prob": "query_2_topic_prob",
        "was_outlier": "query_2_was_outlier",
        "expected_topics": 13,
        "assignment_path": PROJECT_ROOT
        / "data/processed/topics/native/query_2/assignments.csv",
    },
    "query_3": {
        "display": SCOPE_BY_ID["query_3"].label,
        "model": "native",
        "flag": "in_query_3",
        "topic_id": "query_3_topic_id",
        "topic_label": "query_3_topic_label",
        "topic_prob": "query_3_topic_prob",
        "was_outlier": "query_3_was_outlier",
        "expected_topics": 6,
        "assignment_path": PROJECT_ROOT
        / "data/processed/topics/native/query_3/assignments.csv",
    },
    "query_4": {
        "display": SCOPE_BY_ID["query_4"].label,
        "model": "native",
        "flag": "in_query_4",
        "topic_id": "query_4_topic_id",
        "topic_label": "query_4_topic_label",
        "topic_prob": "query_4_topic_prob",
        "was_outlier": "query_4_was_outlier",
        "expected_topics": 8,
        "assignment_path": PROJECT_ROOT
        / "data/processed/topics/native/query_4/assignments.csv",
    },
}

NATIVE_SOURCE_COLUMNS = {
    "native_topic_id": "topic_id",
    "native_topic_label": "topic_label",
    "native_topic_prob": "topic_prob",
    "native_was_outlier": "was_outlier",
}

DIMENSIONS = (
    "ai_method_or_phenomenon",
    "ai_role_function",
    "ai_type_form",
    "ai_mechanism_analysis",
    "level_of_analysis",
    "entrepreneurial_process_stage",
    "scope_conditions",
    "definition_construct_clarity",
)

CONTRASTS = (
    ("study_status_x_type", "ai_method_or_phenomenon", "ai_type_form"),
    ("study_status_x_role", "ai_method_or_phenomenon", "ai_role_function"),
    ("type_x_role", "ai_type_form", "ai_role_function"),
    ("type_x_mechanism", "ai_type_form", "ai_mechanism_analysis"),
    ("level_x_mechanism", "level_of_analysis", "ai_mechanism_analysis"),
    (
        "process_stage_x_mechanism",
        "entrepreneurial_process_stage",
        "ai_mechanism_analysis",
    ),
)

ERA_ORDER = (
    "Before 2000",
    "2000-2015",
    "2016-2020",
    "2021-2023",
    "2024-2026 (as at 8 July 2026)",
    "2027 issue year (indexed by 8 July 2026)",
    "Unknown year",
)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_unique(frame: pd.DataFrame, label: str) -> None:
    """Require a non-empty, unique paper identifier."""

    if "paper_id" not in frame.columns:
        raise ValueError(f"{label} has no paper_id column")
    if frame["paper_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{label} contains blank paper IDs")
    duplicates = int(frame["paper_id"].duplicated().sum())
    if duplicates:
        raise ValueError(f"{label} contains {duplicates} duplicate paper IDs")


def _is_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(("1", "true", "yes"))


def _native_paths_from_config() -> dict[str, Path]:
    return {
        scope: config["assignment_path"]
        for scope, config in SCOPE_CONFIG.items()
        if config["assignment_path"] is not None
    }


def load_and_join(
    primary_path: Path = PRIMARY,
    topic_path: Path = TOPICS,
    manifest_path: Path = PRIMARY_MANIFEST,
    native_paths: dict[str, Path] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Verify and join the global and native scope assignments without row loss."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["primary_dataset"]
    actual_hash = sha256(primary_path)
    if actual_hash != expected["sha256"]:
        raise ValueError(
            "Primary dataset checksum does not match dataset_manifest.json: "
            f"expected {expected['sha256']}, observed {actual_hash}"
        )

    primary = pd.read_csv(primary_path, dtype=str, keep_default_na=False)
    topics = pd.read_csv(topic_path, dtype=str, keep_default_na=False)
    require_unique(primary, "primary dataset")
    require_unique(topics, "global topic dataset")
    if len(primary) != int(expected["rows"]):
        raise ValueError("Primary dataset row count does not match its manifest")
    if set(primary["paper_id"]) != set(topics["paper_id"]):
        raise ValueError("Primary and global topic datasets do not contain identical paper IDs")

    missing = [column for column in GLOBAL_TOPIC_COLUMNS if column not in topics.columns]
    if missing:
        raise ValueError(f"Global topic dataset is missing required columns: {missing}")
    overlap = [column for column in GLOBAL_TOPIC_COLUMNS if column in primary.columns]
    if overlap:
        raise ValueError(f"Primary dataset already contains topic columns: {overlap}")

    enriched = primary.merge(
        topics[["paper_id", *GLOBAL_TOPIC_COLUMNS]],
        on="paper_id",
        how="left",
        validate="one_to_one",
    )
    joined_columns = list(GLOBAL_TOPIC_COLUMNS)

    paths = _native_paths_from_config() if native_paths is None else native_paths
    for scope, assignment_path in paths.items():
        if scope not in SCOPE_CONFIG or scope == "full_corpus":
            raise ValueError(f"Unknown native topic scope: {scope}")
        config = SCOPE_CONFIG[scope]
        flag = str(config["flag"])
        if flag not in primary.columns:
            raise ValueError(f"Primary dataset is missing scope flag {flag}")
        native = pd.read_csv(assignment_path, dtype=str, keep_default_na=False)
        require_unique(native, f"{scope} native assignment dataset")
        missing_native = set(NATIVE_SOURCE_COLUMNS) - set(native.columns)
        if missing_native:
            raise ValueError(
                f"{scope} native assignments are missing columns: {sorted(missing_native)}"
            )
        scope_ids = set(primary.loc[_is_true(primary[flag]), "paper_id"])
        assignment_ids = set(native["paper_id"])
        outside = assignment_ids - scope_ids
        if outside:
            raise ValueError(
                f"{scope} native assignments include {len(outside)} papers outside {flag}"
            )

        rename = {
            source: str(config[target])
            for source, target in NATIVE_SOURCE_COLUMNS.items()
        }
        native = native[["paper_id", *NATIVE_SOURCE_COLUMNS]].rename(columns=rename)
        new_columns = list(rename.values())
        overlap = [column for column in new_columns if column in enriched.columns]
        if overlap:
            raise ValueError(f"Topic-enriched dataset already contains columns: {overlap}")
        enriched = enriched.merge(native, on="paper_id", how="left", validate="one_to_one")
        enriched[new_columns] = enriched[new_columns].fillna("")
        joined_columns.extend(new_columns)

    if len(enriched) != len(primary) or enriched["paper_id"].nunique() != len(primary):
        raise ValueError("Topic joins violated the one-to-one corpus invariant")
    return enriched, joined_columns


def clean_values(series: pd.Series) -> pd.Series:
    """Represent blank categorical values explicitly in exported tables."""

    values = series.astype(str).str.strip()
    return values.mask(values.eq(""), "(missing)")


def median_numeric(series: pd.Series) -> float:
    """Return a numeric median without warning for an all-missing group."""

    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if not values.empty else float("nan")


def add_publication_era(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the declared descriptive publication-era bands."""

    result = frame.copy()
    years = pd.to_numeric(result["Year"], errors="coerce")
    era = pd.Series("Unknown year", index=result.index, dtype="object")
    era.loc[years.le(1999)] = "Before 2000"
    era.loc[years.between(2000, 2015)] = "2000-2015"
    era.loc[years.between(2016, 2020)] = "2016-2020"
    era.loc[years.between(2021, 2023)] = "2021-2023"
    era.loc[years.between(2024, 2026)] = "2024-2026 (as at 8 July 2026)"
    era.loc[years.eq(2027)] = "2027 issue year (indexed by 8 July 2026)"
    result["publication_era"] = pd.Categorical(
        era, categories=ERA_ORDER, ordered=True
    )
    return result


def _automatic_label_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for config in SCOPE_CONFIG.values():
        label_column = str(config["topic_label"])
        if label_column in result.columns:
            result[f"{label_column}_automatic"] = result[label_column]
    return result


def apply_topic_label_review(
    frame: pd.DataFrame,
    review_path: Path = TOPIC_LABEL_REVIEW,
    scope_config: dict[str, dict[str, object]] = SCOPE_CONFIG,
) -> tuple[pd.DataFrame, str, int]:
    """Apply labels only after all scope-topic pairs have human approval."""

    result = _automatic_label_columns(frame)
    if not review_path.exists():
        return result, "not_prepared", 0

    review = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    required = {"scope", "topic_id", "approved_label", "review_status"}
    missing = required - set(review.columns)
    if missing:
        return result, "legacy_or_incomplete", 0
    if review.duplicated(["scope", "topic_id"]).any():
        raise ValueError("Topic label review contains duplicate scope-topic keys")

    expected_keys: set[tuple[str, int]] = set()
    for scope, config in scope_config.items():
        expected_keys.update(
            (scope, topic_id) for topic_id in range(int(config["expected_topics"]))
        )
    observed_keys = set(
        zip(
            review["scope"].astype(str),
            pd.to_numeric(review["topic_id"], errors="raise").astype(int),
        )
    )
    reviewed = int(review["review_status"].eq("approved").sum())
    complete = (
        observed_keys == expected_keys
        and reviewed == len(expected_keys)
        and review["approved_label"].astype(str).str.strip().ne("").all()
    )
    if not complete:
        return result, "pending", reviewed

    for scope, config in scope_config.items():
        mapping = (
            review.loc[review["scope"].eq(scope), ["topic_id", "approved_label"]]
            .assign(topic_id=lambda x: pd.to_numeric(x["topic_id"], errors="raise").astype(int))
            .set_index("topic_id")["approved_label"]
            .astype(str)
            .str.strip()
            .to_dict()
        )
        topic_column = str(config["topic_id"])
        label_column = str(config["topic_label"])
        topic_ids = pd.to_numeric(result[topic_column], errors="coerce")
        assigned = topic_ids.notna()
        result.loc[assigned, label_column] = topic_ids[assigned].astype(int).map(mapping)
        if result.loc[assigned, label_column].isna().any():
            raise ValueError(f"Approved labels do not cover every {scope} assignment")
    return result, "approved", reviewed


def scope_frame(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Return one scope using the topic model trained specifically for that scope."""

    if scope not in SCOPE_CONFIG:
        raise ValueError(f"Unknown topic scope: {scope}")
    config = SCOPE_CONFIG[scope]
    if config["flag"] is None:
        result = frame.copy()
    else:
        result = frame[_is_true(frame[str(config["flag"])])].copy()
    result["analysis_scope"] = scope
    result["analysis_topic_model"] = str(config["model"])
    result["analysis_topic_id"] = result[str(config["topic_id"])]
    result["analysis_topic_label"] = result[str(config["topic_label"])]
    result["analysis_topic_prob"] = result[str(config["topic_prob"])]
    result["analysis_was_outlier"] = result[str(config["was_outlier"])]
    return result


def topic_prevalence(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize one scope's topic prevalence with explicit denominators."""

    scope = frame["analysis_scope"].iat[0]
    model = frame["analysis_topic_model"].iat[0]
    labels = clean_values(frame["analysis_topic_label"])
    ids = clean_values(frame["analysis_topic_id"])
    work = frame.assign(topic_id=ids, topic_label=labels)
    grouped = (
        work.groupby(["topic_id", "topic_label"], observed=True, dropna=False)
        .agg(
            papers=("paper_id", "size"),
            median_probability=("analysis_topic_prob", median_numeric),
            reassigned_papers=(
                "analysis_was_outlier",
                lambda x: x.astype(str).str.lower().eq("true").sum(),
            ),
        )
        .reset_index()
    )
    assigned_n = int(labels.ne("(missing)").sum())
    grouped.insert(0, "scope", scope)
    grouped.insert(1, "topic_model", model)
    grouped["scope_papers"] = len(frame)
    grouped["assigned_papers"] = assigned_n
    grouped["share_of_scope"] = grouped["papers"] / len(frame)
    grouped["share_of_assigned"] = np.where(
        grouped["topic_label"].eq("(missing)"),
        np.nan,
        grouped["papers"] / assigned_n if assigned_n else np.nan,
    )
    return grouped.sort_values(
        ["topic_label", "papers"], ascending=[True, False], kind="stable"
    )


def grouped_topic_distribution(
    frame: pd.DataFrame,
    group_column: str,
    group_label: str,
) -> pd.DataFrame:
    """Return data-specific topic prevalence within a declared subgroup."""

    scope = frame["analysis_scope"].iat[0]
    model = frame["analysis_topic_model"].iat[0]
    work = frame.copy()
    work["topic_id"] = clean_values(work["analysis_topic_id"])
    work["topic_label"] = clean_values(work["analysis_topic_label"])
    work[group_column] = clean_values(work[group_column])
    denominators = work.groupby(group_column, observed=True).size().rename("group_papers")
    assigned = (
        work.groupby([group_column, "topic_id", "topic_label"], observed=True)
        .size()
        .rename("papers")
        .reset_index()
    )
    assigned = assigned.merge(denominators, on=group_column, validate="many_to_one")
    assigned["share_of_group"] = assigned["papers"] / assigned["group_papers"]
    assigned = assigned.rename(columns={group_column: group_label})
    assigned.insert(0, "scope", scope)
    assigned.insert(1, "topic_model", model)
    return assigned.sort_values(
        [group_label, "papers"], ascending=[True, False], kind="stable"
    )


def topic_dimension_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Produce one scope's topic-by-specification distributions."""

    scope = frame["analysis_scope"].iat[0]
    model = frame["analysis_topic_model"].iat[0]
    assigned = frame[
        clean_values(frame["analysis_topic_label"]).ne("(missing)")
    ].copy()
    assigned["topic_id"] = clean_values(assigned["analysis_topic_id"])
    assigned["topic_label"] = clean_values(assigned["analysis_topic_label"])
    topic_sizes = assigned.groupby(["topic_id", "topic_label"]).size().rename("topic_papers")
    outputs = []
    for dimension in DIMENSIONS:
        work = assigned.assign(category=clean_values(assigned[dimension]))
        table = (
            work.groupby(["topic_id", "topic_label", "category"], observed=True)
            .size()
            .rename("papers")
            .reset_index()
            .merge(topic_sizes.reset_index(), on=["topic_id", "topic_label"])
        )
        table.insert(0, "scope", scope)
        table.insert(1, "topic_model", model)
        table.insert(2, "dimension", dimension)
        table["share_within_topic"] = table["papers"] / table["topic_papers"]
        outputs.append(table)
    return pd.concat(outputs, ignore_index=True)


def construct_contrasts(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build locked full-corpus construct cross-tabs and a paper evidence index."""

    summaries = []
    evidence = []
    evidence_columns = ["paper_id", "Title", "Year", "Source title", "query_sources"]
    for name, row_column, column_column in CONTRASTS:
        work = frame.copy()
        work["row_value"] = clean_values(work[row_column])
        work["column_value"] = clean_values(work[column_column])
        row_totals = work.groupby("row_value").size().rename("row_papers")
        table = (
            work.groupby(["row_value", "column_value"])
            .size()
            .rename("papers")
            .reset_index()
            .merge(row_totals.reset_index(), on="row_value", validate="many_to_one")
        )
        table.insert(0, "scope", "full_corpus")
        table.insert(1, "contrast", name)
        table.insert(2, "row_dimension", row_column)
        table.insert(3, "column_dimension", column_column)
        table["share_within_row"] = table["papers"] / table["row_papers"]
        summaries.append(table)

        detail = work[evidence_columns].copy()
        detail.insert(0, "scope", "full_corpus")
        detail.insert(1, "contrast", name)
        detail.insert(2, "row_dimension", row_column)
        detail.insert(3, "row_value", work["row_value"])
        detail.insert(4, "column_dimension", column_column)
        detail.insert(5, "column_value", work["column_value"])
        evidence.append(detail)
    return pd.concat(summaries, ignore_index=True), pd.concat(evidence, ignore_index=True)


def topic_paper_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Return paper-level provenance for one data-specific topic model."""

    columns = [
        "analysis_scope",
        "analysis_topic_model",
        "paper_id",
        "Title",
        "Year",
        "Source title",
        "query_sources",
        "analysis_topic_id",
        "analysis_topic_label",
        "analysis_topic_prob",
        "analysis_was_outlier",
        *DIMENSIONS,
    ]
    return frame[columns].rename(
        columns={
            "analysis_scope": "scope",
            "analysis_topic_model": "topic_model",
            "analysis_topic_id": "topic_id",
            "analysis_topic_label": "topic_label",
            "analysis_topic_prob": "topic_probability",
            "analysis_was_outlier": "was_outlier_before_reassignment",
        }
    )


def _topic_number(topic_id: object) -> str:
    """Return a concise one-based display number for a fitted topic ID."""

    number = pd.to_numeric(pd.Series([topic_id]), errors="coerce").iat[0]
    return str(int(number) + 1) if pd.notna(number) else str(topic_id)


def _topic_axis_labels(topic_ids: list[object]) -> list[str]:
    """Return short, stable labels for topic-chart axes."""

    return [f"Topic {_topic_number(topic_id)}" for topic_id in topic_ids]


def _prevalence_axis_labels(plotted: pd.DataFrame) -> list[str]:
    """Retain editable topic interpretations on the prevalence chart."""

    return [
        f"T{row.topic_id}: {row.topic_label.replace('_', ' ')}"
        for row in plotted.itertuples()
    ]


def _prevalence_plot_rows(prevalence: pd.DataFrame) -> pd.DataFrame:
    """Return every assigned topic ordered for the horizontal bar chart."""

    return prevalence[prevalence["topic_label"].ne("(missing)")].sort_values(
        "papers"
    )


def _top_topic_records(frame: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Select largest topics by stable ID, independent of editable labels."""

    work = pd.DataFrame(
        {
            "topic_id": pd.to_numeric(frame["analysis_topic_id"], errors="coerce"),
            "topic_label": clean_values(frame["analysis_topic_label"]),
        }
    )
    work = work[work["topic_id"].ge(0) & work["topic_label"].ne("(missing)")]
    return (
        work.groupby(["topic_id", "topic_label"], observed=True)
        .size()
        .rename("papers")
        .reset_index()
        .sort_values(["papers", "topic_id"], ascending=[False, True])
        .head(n)
        .reset_index(drop=True)
    )


def _scope_title(frame: pd.DataFrame) -> str:
    scope = frame["analysis_scope"].iat[0]
    config = SCOPE_CONFIG[scope]
    model = "global model" if config["model"] == "global" else "native model"
    return f"{config['display']} ({model})"


def plot_topic_prevalence(prevalence: pd.DataFrame, output: Path, title: str) -> None:
    """Plot every assigned topic in one scope."""

    plotted = _prevalence_plot_rows(prevalence)
    labels = _prevalence_axis_labels(plotted)
    fig, ax = plt.subplots(figsize=(12, max(6, len(plotted) * 0.42)))
    bars = ax.barh(labels, plotted["papers"], color="#2878B5")
    ax.bar_label(bars, labels=[f"{value:,}" for value in plotted["papers"]], padding=3)
    ax.set_xlabel("Papers assigned to topic")
    ax.set_title(f"Topic prevalence: {title}")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_group_heatmap(
    frame: pd.DataFrame,
    group_column: str,
    group_order: list[str],
    output: Path,
    title: str,
) -> None:
    """Plot the share of each subgroup occupied by one scope's largest topics."""

    selected_topics = _top_topic_records(frame)
    topic_ids = selected_topics["topic_id"].tolist()
    work = frame.copy()
    work["topic_id"] = pd.to_numeric(work["analysis_topic_id"], errors="coerce")
    work[group_column] = clean_values(work[group_column])
    denominators = work.groupby(group_column, observed=True).size()
    counts = pd.crosstab(work[group_column], work["topic_id"])
    matrix = counts.reindex(index=group_order, columns=topic_ids, fill_value=0)
    matrix = matrix.div(denominators.reindex(group_order), axis=0).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(14, max(5, len(group_order) * 0.9)))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(topic_ids)))
    ax.set_xticklabels(
        _topic_axis_labels(topic_ids), rotation=45, ha="right", fontsize=9
    )
    ax.set_yticks(range(len(group_order)))
    ax.set_yticklabels(group_order)
    ax.set_title(title)
    maximum = matrix.to_numpy().max() if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iat[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if maximum and value > maximum * 0.55 else "black",
            )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label("Share of subgroup papers (%)")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_topic_observability(frame: pd.DataFrame, output: Path, title: str) -> None:
    """Plot dimension observability within one scope's largest topics."""

    selected_topics = _top_topic_records(frame)
    topic_ids = selected_topics["topic_id"].tolist()
    work = frame.copy()
    work["topic_id"] = pd.to_numeric(work["analysis_topic_id"], errors="coerce")
    matrix = []
    labels = []
    for panel in OBSERVED_COMPOSITION_PANELS:
        column = panel["column"]
        if column not in work.columns and panel.get("fallback_column") in work.columns:
            column = panel["fallback_column"]
        if column not in work.columns:
            continue
        values = clean_values(work[column])
        observed = values.ne("(missing)") & ~values.isin(panel["excluded"])
        rates = []
        for topic_id in topic_ids:
            topic_mask = work["topic_id"].eq(topic_id)
            rates.append(
                float(observed[topic_mask].mean() * 100) if topic_mask.any() else 0.0
            )
        matrix.append(rates)
        labels.append(panel["label"])

    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(14, 7))
    image = ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(range(len(topic_ids)))
    ax.set_xticklabels(
        _topic_axis_labels(topic_ids), rotation=45, ha="right", fontsize=9
    )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value >= 60 else "black",
            )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label("Papers with an observed code (%)")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_scope_outputs(enriched: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, object]]]:
    tables: dict[str, list[pd.DataFrame]] = {
        "scope_topic_prevalence.csv": [],
        "scope_topic_by_era.csv": [],
        "scope_topic_by_journal.csv": [],
        "scope_topic_dimension_distribution.csv": [],
        "scope_topic_paper_index.csv": [],
    }
    coverage: dict[str, dict[str, object]] = {}
    for scope, config in SCOPE_CONFIG.items():
        scoped = scope_frame(enriched, scope)
        prevalence = topic_prevalence(scoped)
        tables["scope_topic_prevalence.csv"].append(prevalence)
        tables["scope_topic_by_era.csv"].append(
            grouped_topic_distribution(scoped, "publication_era", "publication_era")
        )
        tables["scope_topic_by_journal.csv"].append(
            grouped_topic_distribution(scoped, "Source title", "journal")
        )
        tables["scope_topic_dimension_distribution.csv"].append(
            topic_dimension_distribution(scoped)
        )
        tables["scope_topic_paper_index.csv"].append(topic_paper_index(scoped))

        labels = clean_values(scoped["analysis_topic_label"])
        assigned = labels.ne("(missing)")
        coverage[scope] = {
            "topic_model": config["model"],
            "scope_papers": len(scoped),
            "assigned_topics": int(assigned.sum()),
            "unassigned_or_text_ineligible": int((~assigned).sum()),
            "observed_topic_count": int(labels[assigned].nunique()),
            "expected_topic_count": int(config["expected_topics"]),
            "topic_id_column": config["topic_id"],
            "topic_label_column": config["topic_label"],
        }

        scope_dir = FIGURE_OUTPUT / scope
        scope_dir.mkdir(parents=True, exist_ok=True)
        title = _scope_title(scoped)
        plot_topic_prevalence(prevalence, scope_dir / "topic_prevalence.png", title)
        era_groups = [
            value
            for value in ERA_ORDER
            if value in set(scoped["publication_era"].astype(str))
        ]
        plot_group_heatmap(
            scoped,
            "publication_era",
            era_groups,
            scope_dir / "topic_by_era_heatmap.png",
            f"Largest topics by publication era: {title}",
        )
        status_groups = [
            value
            for value in ("phenomenon", "method", "both", "unclear", "(missing)")
            if value in set(clean_values(scoped["ai_method_or_phenomenon"]))
        ]
        plot_group_heatmap(
            scoped,
            "ai_method_or_phenomenon",
            status_groups,
            scope_dir / "topic_by_study_status_heatmap.png",
            f"Largest topics by AI positioning: {title}",
        )
        plot_topic_observability(
            scoped,
            scope_dir / "topic_observability_heatmap.png",
            f"Construct observability within largest topics: {title}",
        )

    return (
        {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()},
        coverage,
    )


def main() -> None:
    """Run the complete, non-mutating, data-specific Stage 4 export."""

    for directory in (DATA_OUTPUT, TABLE_OUTPUT, FIGURE_OUTPUT, ENRICHED.parent):
        directory.mkdir(parents=True, exist_ok=True)

    enriched, topic_columns = load_and_join()
    enriched = add_publication_era(enriched)
    enriched, label_status, reviewed_topics = apply_topic_label_review(enriched)
    enriched.to_csv(ENRICHED, index=False, encoding="utf-8-sig")

    tables, coverage = _build_scope_outputs(enriched)
    topic_run_manifest = json.loads(TOPIC_RUN_MANIFEST.read_text(encoding="utf-8"))
    coverage["full_corpus"]["model_eligible_records"] = int(
        topic_run_manifest["models"]["full_corpus"]["eligible_papers"]
    )
    for scope, path in _native_paths_from_config().items():
        coverage[scope]["model_eligible_records"] = len(
            pd.read_csv(path, usecols=["paper_id"])
        )
    contrasts, contrast_evidence = construct_contrasts(enriched)
    tables["construct_contrasts.csv"] = contrasts
    tables["construct_contrast_papers.csv"] = contrast_evidence
    for name, table in tables.items():
        table.to_csv(TABLE_OUTPUT / name, index=False, encoding="utf-8-sig")

    output_paths = [ENRICHED, *[TABLE_OUTPUT / name for name in tables]]
    native_inputs = {
        str(path.relative_to(PROJECT_ROOT)): sha256(path)
        for path in _native_paths_from_config().values()
    }
    required_topics = sum(
        int(config["expected_topics"]) for config in SCOPE_CONFIG.values()
    )
    figure_paths = sorted(FIGURE_OUTPUT.glob("*/*.png"))
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "status": (
            "completed" if label_status == "approved" else "completed_with_pending_topic_labels"
        ),
        "analytical_status": (
            "descriptive; scope-specific topics; automatic labels remain provisional"
            if label_status != "approved"
            else "descriptive; scope-specific topics; topic labels human-approved"
        ),
        "scope_contract": {
            scope: {
                "topic_model": config["model"],
                "expected_topics": config["expected_topics"],
                "membership_flag": config["flag"],
            }
            for scope, config in SCOPE_CONFIG.items()
        },
        "topic_label_review": {
            "status": label_status,
            "approved_topics": reviewed_topics,
            "required_topics": required_topics,
            "unique_key": ["scope", "topic_id"],
            "path": str(TOPIC_LABEL_REVIEW.relative_to(PROJECT_ROOT)),
            "sha256": (
                sha256(TOPIC_LABEL_REVIEW) if TOPIC_LABEL_REVIEW.exists() else ""
            ),
        },
        "inputs": {
            str(PRIMARY.relative_to(PROJECT_ROOT)): sha256(PRIMARY),
            str(TOPICS.relative_to(PROJECT_ROOT)): sha256(TOPICS),
            str(PRIMARY_MANIFEST.relative_to(PROJECT_ROOT)): sha256(PRIMARY_MANIFEST),
            str(TOPIC_RUN_MANIFEST.relative_to(PROJECT_ROOT)): sha256(TOPIC_RUN_MANIFEST),
            **native_inputs,
        },
        "topic_columns_joined": topic_columns,
        "coverage": {
            "rows": len(enriched),
            "unique_paper_ids": int(enriched["paper_id"].nunique()),
            "by_scope": coverage,
        },
        "outputs": {
            str(path.relative_to(PROJECT_ROOT)): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in output_paths
        },
        "figures": [str(path.relative_to(PROJECT_ROOT)) for path in figure_paths],
    }
    manifest_path = DATA_OUTPUT / "stage4_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"Topic-enriched dataset: {len(enriched):,} rows x "
        f"{len(enriched.columns):,} columns -> {ENRICHED}"
    )
    print(
        f"Topic label review: {label_status} "
        f"({reviewed_topics}/{required_topics} approved)"
    )
    for scope, values in coverage.items():
        print(
            f"{scope}: {values['assigned_topics']:,}/{values['scope_papers']:,} "
            f"assigned with the {values['topic_model']} model; "
            f"{values['observed_topic_count']} topics"
        )
    print(f"Tables -> {TABLE_OUTPUT}")
    print(f"Figures ({len(figure_paths)}) -> {FIGURE_OUTPUT}")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
