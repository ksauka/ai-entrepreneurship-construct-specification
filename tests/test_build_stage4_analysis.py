import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_stage4_analysis import (
    SCOPE_CONFIG,
    TOPIC_COLUMNS,
    _prevalence_axis_labels,
    _prevalence_plot_rows,
    _top_topic_records,
    _topic_axis_labels,
    add_publication_era,
    apply_topic_label_review,
    construct_contrasts,
    load_and_join,
    scope_frame,
    topic_dimension_distribution,
    topic_prevalence,
)
from aecsp.corpus.scopes import SCOPE_BY_ID


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _primary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "paper_id": ["p1", "p2", "p3"],
            "Title": ["One", "Two", "Three"],
            "Year": [2014, 2022, 2025],
            "Source title": ["J1", "J1", "J2"],
            "query_sources": ["query_1", "query_1;query_2", "query_4"],
            "in_query_1": [1, 1, 0],
            "in_query_2": [0, 1, 0],
            "in_query_3": [0, 0, 0],
            "in_query_4": [0, 0, 1],
            "ai_method_or_phenomenon": ["phenomenon", "method", "both"],
            "ai_role_function": ["AI as tool", "AI as method", "AI as capability"],
            "ai_type_form": ["machine learning", "analytics", "unspecified AI"],
            "ai_mechanism_analysis": ["prediction", "measurement", "mechanism missing"],
            "level_of_analysis": ["firm", "individual", "firm"],
            "entrepreneurial_process_stage": ["opportunity discovery", "static input", "process unspecified"],
            "scope_conditions": ["sector-specific", "country-specific", "scope missing"],
            "definition_construct_clarity": ["definition fits claim", "no definition", "no definition"],
        }
    )


def _topics() -> pd.DataFrame:
    frame = pd.DataFrame({"paper_id": ["p1", "p2", "p3"]})
    values = {
        "bertopic_topic": [0, 1, ""],
        "bertopic_topic_label": ["Global prediction", "Global measurement", ""],
        "bertopic_topic_prob": [0.8, 0.6, ""],
        "bertopic_was_outlier": [False, True, ""],
        "ai_terms": ["machine learning", "analytics", ""],
        "ai_term_count": [1, 1, 0],
        "ent_terms": ["venture", "firm", ""],
        "ent_term_count": [1, 1, 0],
        "keybert_phrases": ["machine learning", "analytics", ""],
    }
    for column in TOPIC_COLUMNS:
        frame[column] = values[column]
    return frame


def _native_query_1() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "paper_id": ["p1", "p2"],
            "native_topic_id": [1, 0],
            "native_topic_label": ["Native ventures", "Native finance"],
            "native_topic_prob": [0.91, 0.84],
            "native_was_outlier": [False, False],
        }
    )


def _write_inputs(tmp_path: Path):
    primary_path = tmp_path / "primary.csv"
    topic_path = tmp_path / "topics.csv"
    native_path = tmp_path / "query_1.csv"
    manifest_path = tmp_path / "manifest.json"
    _primary().to_csv(primary_path, index=False)
    _topics().to_csv(topic_path, index=False)
    _native_query_1().to_csv(native_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {"primary_dataset": {"rows": 3, "sha256": _sha256(primary_path)}}
        )
    )
    return primary_path, topic_path, native_path, manifest_path


def test_join_preserves_corpus_and_adds_native_assignments(tmp_path):
    primary_path, topic_path, native_path, manifest_path = _write_inputs(tmp_path)

    joined, columns = load_and_join(
        primary_path,
        topic_path,
        manifest_path,
        native_paths={"query_1": native_path},
    )

    assert len(joined) == 3
    assert joined.paper_id.nunique() == 3
    assert set(TOPIC_COLUMNS).issubset(columns)
    assert "query_1_topic_id" in columns
    assert joined.loc[joined.paper_id == "p1", "query_1_topic_label"].iat[0] == "Native ventures"
    assert joined.loc[joined.paper_id == "p3", "query_1_topic_label"].iat[0] == ""

    manifest = json.loads(manifest_path.read_text())
    manifest["primary_dataset"]["sha256"] = "wrong"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        load_and_join(
            primary_path,
            topic_path,
            manifest_path,
            native_paths={"query_1": native_path},
        )


def test_native_assignment_must_belong_to_declared_scope(tmp_path):
    primary_path, topic_path, native_path, manifest_path = _write_inputs(tmp_path)
    native = _native_query_1()
    native.loc[0, "paper_id"] = "p3"
    native.to_csv(native_path, index=False)
    with pytest.raises(ValueError, match="outside in_query_1"):
        load_and_join(
            primary_path,
            topic_path,
            manifest_path,
            native_paths={"query_1": native_path},
        )


def test_scope_tables_use_native_query_topics_not_global_topics(tmp_path):
    primary_path, topic_path, native_path, manifest_path = _write_inputs(tmp_path)
    joined, _ = load_and_join(
        primary_path,
        topic_path,
        manifest_path,
        native_paths={"query_1": native_path},
    )
    frame = add_publication_era(joined)

    query = scope_frame(frame, "query_1")
    assert len(query) == 2
    assert query.analysis_topic_model.unique().tolist() == ["native"]
    assert set(query.analysis_topic_label) == {"Native ventures", "Native finance"}
    assert not set(query.analysis_topic_label) & {"Global prediction", "Global measurement"}

    prevalence = topic_prevalence(query)
    assert prevalence.scope.unique().tolist() == ["query_1"]
    assert prevalence.scope_papers.unique().tolist() == [2]

    dimensions = topic_dimension_distribution(query)
    assert dimensions.scope.unique().tolist() == ["query_1"]
    assert set(dimensions.dimension) == {
        "ai_method_or_phenomenon",
        "ai_role_function",
        "ai_type_form",
        "ai_mechanism_analysis",
        "level_of_analysis",
        "entrepreneurial_process_stage",
        "scope_conditions",
        "definition_construct_clarity",
    }


def test_topic_figures_use_canonical_scope_names_and_short_stable_axes():
    assert {
        scope: config["display"] for scope, config in SCOPE_CONFIG.items()
    } == {
        scope: SCOPE_BY_ID[scope].label for scope in SCOPE_CONFIG
    }

    frame = pd.DataFrame(
        {
            "analysis_topic_id": [0, 0, 1, 1, 1, 2],
            "analysis_topic_label": [
                "same editable label",
                "same editable label",
                "same editable label",
                "same editable label",
                "same editable label",
                "another label",
            ],
        }
    )
    selected = _top_topic_records(frame)
    assert selected["topic_id"].tolist() == [1, 0, 2]
    assert selected["papers"].tolist() == [3, 2, 1]
    assert _topic_axis_labels(selected["topic_id"].tolist()) == [
        "Topic 2",
        "Topic 1",
        "Topic 3",
    ]
    assert _prevalence_axis_labels(
        pd.DataFrame(
            {
                "topic_id": [0, 1],
                "topic_label": ["Humanized capability", "AI_governance"],
            }
        )
    ) == ["T0: Humanized capability", "T1: AI governance"]


def test_publication_era_distinguishes_indexed_2027_issues_from_unknown_years():
    frame = add_publication_era(
        pd.DataFrame({"Year": ["2026", "2027", "", "not a year"]})
    )
    assert frame["publication_era"].astype(str).tolist() == [
        "2024-2026 (as at 8 July 2026)",
        "2027 issue year (indexed by 8 July 2026)",
        "Unknown year",
        "Unknown year",
    ]


def test_topic_prevalence_plot_includes_every_assigned_topic():
    prevalence = pd.DataFrame(
        {
            "topic_id": list(range(25)) + ["(missing)"],
            "topic_label": [f"Topic label {index}" for index in range(25)]
            + ["(missing)"],
            "papers": list(range(1, 26)) + [3],
        }
    )
    plotted = _prevalence_plot_rows(prevalence)
    assert len(plotted) == 25
    assert set(plotted["topic_id"]) == set(range(25))


def test_construct_contrasts_are_full_corpus_and_traceable():
    summary, evidence = construct_contrasts(_primary())
    assert summary.contrast.nunique() == 6
    assert summary.scope.unique().tolist() == ["full_corpus"]
    assert len(evidence) == len(_primary()) * 6
    assert evidence.paper_id.notna().all()


def test_join_rejects_global_topic_set_mismatch(tmp_path):
    primary_path, topic_path, native_path, manifest_path = _write_inputs(tmp_path)
    _topics().iloc[:2].to_csv(topic_path, index=False)
    with pytest.raises(ValueError, match="identical paper IDs"):
        load_and_join(
            primary_path,
            topic_path,
            manifest_path,
            native_paths={"query_1": native_path},
        )


def test_topic_labels_apply_only_after_complete_scope_approval(tmp_path):
    frame = _primary().merge(_topics(), on="paper_id")
    frame = frame.merge(
        _native_query_1().rename(
            columns={
                "native_topic_id": "query_1_topic_id",
                "native_topic_label": "query_1_topic_label",
                "native_topic_prob": "query_1_topic_prob",
                "native_was_outlier": "query_1_was_outlier",
            }
        ),
        on="paper_id",
        how="left",
    ).fillna("")
    review_path = tmp_path / "labels.csv"
    config = {
        "full_corpus": {
            "topic_id": "bertopic_topic",
            "topic_label": "bertopic_topic_label",
            "expected_topics": 2,
        },
        "query_1": {
            "topic_id": "query_1_topic_id",
            "topic_label": "query_1_topic_label",
            "expected_topics": 2,
        },
    }
    pending = pd.DataFrame(
        {
            "scope": ["full_corpus", "full_corpus", "query_1", "query_1"],
            "topic_id": [0, 1, 0, 1],
            "approved_label": ["G0", "G1", "Q0", "Q1"],
            "review_status": ["approved", "approved", "approved", "pending"],
        }
    )
    pending.to_csv(review_path, index=False)

    unchanged, status, reviewed = apply_topic_label_review(
        frame, review_path, scope_config=config
    )
    assert status == "pending"
    assert reviewed == 3
    assert unchanged.bertopic_topic_label.tolist() == frame.bertopic_topic_label.tolist()
    assert "query_1_topic_label_automatic" in unchanged

    pending["review_status"] = "approved"
    pending.to_csv(review_path, index=False)
    approved, status, reviewed = apply_topic_label_review(
        frame, review_path, scope_config=config
    )
    assert status == "approved"
    assert reviewed == 4
    assert approved.loc[approved.paper_id == "p1", "bertopic_topic_label"].iat[0] == "G0"
    assert approved.loc[approved.paper_id == "p1", "query_1_topic_label"].iat[0] == "Q1"
