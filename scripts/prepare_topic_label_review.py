"""Prepare and validate scope-specific human review of final topic labels.

Inputs: selected optimization candidates, representative papers, and the
topic-enriched dataset. Output: one review row per scope-topic pair. Existing
decisions are preserved by the composite key ``scope, topic_id``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENRICHED = (
    PROJECT_ROOT
    / "data/processed/analysis/primary_analysis_dataset_with_topics.csv"
)
OUTPUT = PROJECT_ROOT / "data/processed/analysis/stage4/topic_label_review.csv"

SCOPE_REVIEW_CONFIG = {
    "full_corpus": {
        "min_topic_size": 50,
        "topic_column": "bertopic_topic",
        "expected_topics": 53,
    },
    "query_1": {
        "min_topic_size": 50,
        "topic_column": "query_1_topic_id",
        "expected_topics": 50,
    },
    "query_2": {
        "min_topic_size": 8,
        "topic_column": "query_2_topic_id",
        "expected_topics": 13,
    },
    "query_3": {
        "min_topic_size": 18,
        "topic_column": "query_3_topic_id",
        "expected_topics": 6,
    },
    "query_4": {
        "min_topic_size": 20,
        "topic_column": "query_4_topic_id",
        "expected_topics": 8,
    },
}

DECISION_COLUMNS = (
    "approved_label",
    "review_status",
    "reviewer_notes",
    "last_updated_at",
    "last_reviewer",
)
ALLOWED_STATUSES = ("pending", "approved", "revise")


def candidate_paths(scope: str, min_topic_size: int) -> tuple[Path, Path]:
    directory = (
        PROJECT_ROOT
        / f"data/processed/topics/optimization/{scope}/candidates"
        / f"min_topic_size_{min_topic_size}"
    )
    return directory / "topics.csv", directory / "representative_papers.csv"


def build_scope_review(
    scope: str,
    candidate_topics: pd.DataFrame,
    representatives: pd.DataFrame,
    enriched: pd.DataFrame,
    topic_column: str,
) -> pd.DataFrame:
    """Build topic evidence rows for one independently fitted topic model."""

    topics = candidate_topics.copy()
    topics.columns = [str(column).lstrip("\ufeff") for column in topics.columns]
    topics["topic_id"] = pd.to_numeric(topics["topic_id"], errors="raise").astype(int)
    topics = topics.sort_values("topic_id").rename(
        columns={
            "paper_count": "fitted_papers",
            "topic_label": "candidate_term_label",
        }
    )
    topics["automatic_label"] = (
        topics["candidate_term_label"].astype(str).str.replace(" | ", " / ", regex=False)
    )

    assigned = enriched[enriched[topic_column].astype(str).str.strip().ne("")].copy()
    assigned["topic_id"] = pd.to_numeric(
        assigned[topic_column], errors="raise"
    ).astype(int)
    final_counts = assigned.groupby("topic_id").size().rename("final_assigned_papers")
    topics = topics.merge(
        final_counts.reset_index(), on="topic_id", how="left", validate="one_to_one"
    )
    topics["final_assigned_papers"] = (
        topics["final_assigned_papers"].fillna(0).astype(int)
    )

    reps = representatives.copy()
    reps.columns = [str(column).lstrip("\ufeff") for column in reps.columns]
    reps["topic_id"] = pd.to_numeric(reps["topic_id"], errors="raise").astype(int)
    rep_rows = []
    for topic_id, group in reps.groupby("topic_id"):
        row: dict[str, object] = {"topic_id": int(topic_id)}
        for representative in group.sort_values("representative_rank").itertuples():
            rank = int(representative.representative_rank)
            row[f"representative_{rank}_paper_id"] = representative.paper_id
            row[f"representative_{rank}_title"] = representative.title
            row[f"representative_{rank}_centroid_similarity"] = (
                representative.similarity_to_topic_centroid
            )
        rep_rows.append(row)
    topics = topics.merge(
        pd.DataFrame(rep_rows), on="topic_id", how="left", validate="one_to_one"
    )

    evidence_columns = [
        "topic_id",
        "automatic_label",
        "top_terms",
        "fitted_papers",
        "final_assigned_papers",
        "representative_1_paper_id",
        "representative_1_title",
        "representative_1_centroid_similarity",
        "representative_2_paper_id",
        "representative_2_title",
        "representative_2_centroid_similarity",
        "representative_3_paper_id",
        "representative_3_title",
        "representative_3_centroid_similarity",
    ]
    review = topics[evidence_columns].copy()
    review.insert(0, "scope", scope)
    review["approved_label"] = ""
    review["review_status"] = "pending"
    review["reviewer_notes"] = ""
    review["last_updated_at"] = ""
    review["last_reviewer"] = ""
    return review


def build_review(
    scope_inputs: dict[str, tuple[pd.DataFrame, pd.DataFrame, str]],
    enriched: pd.DataFrame,
    previous: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build all scope-topic rows while preserving composite-keyed decisions."""

    outputs = [
        build_scope_review(scope, topics, reps, enriched, topic_column)
        for scope, (topics, reps, topic_column) in scope_inputs.items()
    ]
    review = pd.concat(outputs, ignore_index=True)

    if previous is not None and not previous.empty:
        if "scope" not in previous.columns:
            nonblank_labels = previous.get("approved_label", pd.Series(dtype=str)).astype(
                str
            ).str.strip().ne("").any()
            approved = previous.get("review_status", pd.Series(dtype=str)).astype(
                str
            ).eq("approved").any()
            if nonblank_labels or approved:
                raise ValueError(
                    "Legacy global-only review contains decisions and cannot be "
                    "automatically converted to scope-specific keys"
                )
        else:
            if previous.duplicated(["scope", "topic_id"]).any():
                raise ValueError("Existing review contains duplicate scope-topic keys")
            for column in DECISION_COLUMNS:
                if column not in previous.columns:
                    previous[column] = ""
            decisions = previous[["scope", "topic_id", *DECISION_COLUMNS]].copy()
            decisions["topic_id"] = pd.to_numeric(
                decisions["topic_id"], errors="raise"
            ).astype(int)
            review = review.drop(columns=list(DECISION_COLUMNS)).merge(
                decisions,
                on=["scope", "topic_id"],
                how="left",
                validate="one_to_one",
            )
            review["approved_label"] = review["approved_label"].fillna("")
            review["review_status"] = review["review_status"].fillna("pending")
            review["reviewer_notes"] = review["reviewer_notes"].fillna("")

    expected = {
        scope: len(topics)
        for scope, (topics, _representatives, _topic_column) in scope_inputs.items()
    }
    validate_review(review, expected_by_scope=expected)
    scope_order = {scope: index for index, scope in enumerate(SCOPE_REVIEW_CONFIG)}
    return (
        review.assign(_scope_order=review["scope"].map(scope_order))
        .sort_values(["_scope_order", "topic_id"])
        .drop(columns="_scope_order")
        .reset_index(drop=True)
    )


def validate_review(
    review: pd.DataFrame,
    expected_by_scope: dict[str, int] | None = None,
) -> None:
    """Validate scope-topic coverage and decision-state invariants."""

    required = {"scope", "topic_id", *DECISION_COLUMNS}
    missing = required - set(review.columns)
    if missing:
        raise ValueError(f"Review sheet is missing columns: {sorted(missing)}")
    if review.duplicated(["scope", "topic_id"]).any():
        raise ValueError("Review sheet contains duplicate scope-topic keys")
    expected = expected_by_scope or {
        scope: int(config["expected_topics"])
        for scope, config in SCOPE_REVIEW_CONFIG.items()
    }
    observed_scopes = set(review["scope"].astype(str))
    if observed_scopes != set(expected):
        raise ValueError(
            f"Expected scopes {sorted(expected)}, found {sorted(observed_scopes)}"
        )
    for scope, count in expected.items():
        scoped = review[review["scope"].eq(scope)]
        if len(scoped) != count:
            raise ValueError(f"Expected {count} {scope} topics, found {len(scoped)}")
        topic_ids = set(pd.to_numeric(scoped["topic_id"], errors="raise").astype(int))
        if topic_ids != set(range(count)):
            raise ValueError(f"{scope} topic IDs are not the complete range 0-{count - 1}")
    invalid = sorted(
        set(review["review_status"].astype(str)) - set(ALLOWED_STATUSES)
    )
    if invalid:
        raise ValueError(f"Review sheet contains invalid statuses: {invalid}")
    approved_without_label = review["review_status"].eq("approved") & review[
        "approved_label"
    ].astype(str).str.strip().eq("")
    if approved_without_label.any():
        keys = review.loc[approved_without_label, ["scope", "topic_id"]].to_dict(
            "records"
        )
        raise ValueError(f"Approved topics have blank labels: {keys}")


def _load_scope_inputs() -> dict[str, tuple[pd.DataFrame, pd.DataFrame, str]]:
    inputs = {}
    for scope, config in SCOPE_REVIEW_CONFIG.items():
        topics_path, reps_path = candidate_paths(scope, int(config["min_topic_size"]))
        inputs[scope] = (
            pd.read_csv(topics_path, dtype=str, keep_default_na=False),
            pd.read_csv(reps_path, dtype=str, keep_default_na=False),
            str(config["topic_column"]),
        )
    return inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the existing review sheet without rewriting it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        review = pd.read_csv(OUTPUT, dtype=str, keep_default_na=False)
        validate_review(review)
        counts = review.groupby(["scope", "review_status"]).size().to_dict()
        print(f"Topic label review PASS: {len(review)} scope-topic pairs; statuses={counts}")
        return

    enriched = pd.read_csv(ENRICHED, dtype=str, keep_default_na=False)
    previous = (
        pd.read_csv(OUTPUT, dtype=str, keep_default_na=False) if OUTPUT.exists() else None
    )
    if previous is not None and "scope" not in previous.columns:
        archive = OUTPUT.with_name("topic_label_review_global_53_superseded.csv")
        if not archive.exists():
            shutil.copy2(OUTPUT, archive)
            print(f"Archived superseded global-only review -> {archive}")
    review = build_review(_load_scope_inputs(), enriched, previous)
    validate_review(review)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"Topic label review: {len(review)} scope-topic pairs -> {OUTPUT}")
    print(review.groupby("scope").size().to_string())
    print(
        "Complete approved_label, set review_status to approved, and add "
        "reviewer_notes as needed."
    )


if __name__ == "__main__":
    main()
