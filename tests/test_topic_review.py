import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest
from fastapi import Request

from aecsp.api.auth import ADMIN_ROLE
from aecsp.topics.review import (
    EXPECTED_BY_SCOPE,
    FIGURE_NAMES,
    TopicReviewStore,
    file_sha256,
)
from aecsp.api import main


def _review_frame(status: str = "pending") -> pd.DataFrame:
    rows = []
    for scope, count in EXPECTED_BY_SCOPE.items():
        for topic_id in range(count):
            row = {
                "scope": scope,
                "topic_id": topic_id,
                "automatic_label": f"Automatic {scope} {topic_id}",
                "top_terms": "artificial intelligence; entrepreneurship; innovation",
                "fitted_papers": 20,
                "final_assigned_papers": 24,
                "approved_label": (
                    f"Approved {scope} {topic_id}" if status == "approved" else ""
                ),
                "review_status": status,
                "reviewer_notes": "",
                "last_updated_at": "",
                "last_reviewer": "",
            }
            for rank in (1, 2, 3):
                row[f"representative_{rank}_paper_id"] = f"{scope}-{topic_id}-{rank}"
                row[f"representative_{rank}_title"] = f"Paper {rank}"
                row[f"representative_{rank}_centroid_similarity"] = "0.8"
            rows.append(row)
    return pd.DataFrame(rows)


def _store(tmp_path: Path, status: str = "pending") -> TopicReviewStore:
    store = TopicReviewStore(tmp_path)
    store.review_path.parent.mkdir(parents=True)
    _review_frame(status).to_csv(
        store.review_path, index=False, encoding="utf-8-sig"
    )
    return store


def test_summary_and_records_are_scope_specific(tmp_path):
    store = _store(tmp_path)
    summary = store.summary()

    assert summary["total_topics"] == 130
    assert summary["approved"] == 0
    assert summary["pending"] == 130
    assert summary["complete"] is False
    assert len(store.records("query_3")) == 6
    assert len(store.records("query_4", query="Automatic query_4 7")) == 1


def test_update_is_audited_and_requires_label_for_approval(tmp_path):
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="non-empty label"):
        store.update(
            "query_2",
            3,
            approved_label="",
            review_status="approved",
            reviewer_notes="",
            reviewer="Researcher A",
        )

    result = store.update(
        "query_2",
        3,
        approved_label="AI-supported opportunity evaluation",
        review_status="approved",
        reviewer_notes="Terms and all three representatives inspected.",
        reviewer="Researcher A",
    )
    assert result["approved_label"] == "AI-supported opportunity evaluation"
    assert result["last_reviewer"] == "Researcher A"
    assert store.summary()["approved"] == 1

    audit = [json.loads(line) for line in store.audit_path.read_text().splitlines()]
    assert len(audit) == 1
    assert audit[0]["scope"] == "query_2"
    assert audit[0]["topic_id"] == 3
    assert audit[0]["before"]["review_status"] == "pending"
    assert audit[0]["after"]["review_status"] == "approved"


def test_outputs_current_requires_matching_complete_review_hash(tmp_path):
    store = _store(tmp_path, status="approved")
    store.manifest_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-07-15T12:00:00+02:00",
                "topic_label_review": {
                    "status": "approved",
                    "sha256": file_sha256(store.review_path),
                },
            }
        )
    )
    assert store.summary()["outputs_current"] is True

    store.update(
        "full_corpus",
        0,
        approved_label="A changed interpretation",
        review_status="approved",
        reviewer_notes="Revised after closer reading.",
        reviewer="Researcher B",
    )
    summary = store.summary()
    assert summary["complete"] is True
    assert summary["outputs_current"] is False


def test_figure_path_is_whitelisted(tmp_path):
    store = _store(tmp_path)
    figure = store.figure_root / "query_4" / FIGURE_NAMES[0]
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"png")
    assert store.figure_path("query_4", FIGURE_NAMES[0]) == figure
    with pytest.raises(ValueError, match="Unknown topic figure"):
        store.figure_path("query_4", "../../secret")


def test_preview_figure_cache_changes_with_review_checksum(tmp_path, monkeypatch):
    store = _store(tmp_path)

    def render(scope, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in FIGURE_NAMES:
            (output_dir / name).write_bytes(
                f"{scope}:{file_sha256(store.review_path)}:{name}".encode()
            )

    monkeypatch.setattr(store, "_render_preview_figures", render)
    first = store.preview_figure_path("query_2", FIGURE_NAMES[0])
    first_content = first.read_bytes()
    store.update(
        "query_2",
        0,
        approved_label="A new draft label",
        review_status="revise",
        reviewer_notes="Testing cache invalidation",
        reviewer="Researcher F",
    )
    second = store.preview_figure_path("query_2", FIGURE_NAMES[0])
    assert first != second
    assert first_content != second.read_bytes()


def test_graph_preview_uses_stable_identity_and_latest_saved_label(tmp_path):
    store = _store(tmp_path)
    before = store.graph_preview("query_3")
    topic = next(node for node in before["nodes"] if node["nodeType"] == "Topic")
    assert topic["id"] == "Topic::query_3:0"
    assert topic["caption"] == "Automatic query_3 0"

    store.update(
        "query_3",
        0,
        approved_label="Humanized opportunity recognition",
        review_status="approved",
        reviewer_notes="Inspected terms and representative papers.",
        reviewer="Researcher D",
    )
    after = store.graph_preview("query_3")
    updated = next(node for node in after["nodes"] if node["id"] == topic["id"])
    assert updated["caption"] == "Humanized opportunity recognition"
    assert updated["properties"]["automatic_label"] == "Automatic query_3 0"


def test_fitted_papers_exclude_reassigned_outliers_and_respect_limit(
    tmp_path, monkeypatch
):
    store = _store(tmp_path)
    enriched = pd.DataFrame(
        [
            {
                "paper_id": "p1",
                "Title": "High probability fitted paper",
                "Authors": "Author A",
                "Year": "2025",
                "Source title": "Journal A",
                "Cited by": "5",
                "DOI": "10.1/p1",
                "Link": "https://www.scopus.com/record/display.uri?eid=p1",
                "query_4_topic_id": "0",
                "query_4_topic_prob": "0.95",
                "query_4_was_outlier": "False",
            },
            {
                "paper_id": "p2",
                "Title": "Reassigned outlier",
                "Authors": "Author B",
                "Year": "2024",
                "Source title": "Journal B",
                "Cited by": "20",
                "DOI": "10.1/p2",
                "Link": "",
                "query_4_topic_id": "0",
                "query_4_topic_prob": "0.99",
                "query_4_was_outlier": "True",
            },
            {
                "paper_id": "p3",
                "Title": "Lower probability fitted paper",
                "Authors": "Author C",
                "Year": "2023",
                "Source title": "Journal C",
                "Cited by": "2",
                "DOI": "",
                "Link": "",
                "query_4_topic_id": "0",
                "query_4_topic_prob": "0.75",
                "query_4_was_outlier": "False",
            },
        ]
    )
    enriched.to_csv(
        tmp_path
        / "data/processed/analysis/primary_analysis_dataset_with_topics.csv",
        index=False,
    )

    result = store.fitted_papers("query_4", 0, limit=1)
    assert result["total"] == 2
    assert result["returned"] == 1
    assert result["papers"][0]["paper_id"] == "p1"
    assert result["papers"][0]["DOI"] == "10.1/p1"
    assert result["papers"][0]["Link"].startswith("https://www.scopus.com/")
    assert result["papers"][0]["topic_probability"] == 0.95

    monkeypatch.setitem(main.state, "topic_review", store)
    api_result = main.topic_review_fitted_papers("query_4", 0, 100)
    assert [paper["paper_id"] for paper in api_result["papers"]] == ["p1", "p3"]


def test_topic_download_contains_draft_labels_and_checksum_manifest(
    tmp_path, monkeypatch
):
    store = _store(tmp_path)
    store.update(
        "query_4",
        0,
        approved_label="Humanized AI venture topic",
        review_status="approved",
        reviewer_notes="Reviewed",
        reviewer="Researcher E",
    )
    table_dir = tmp_path / "tables"
    table_dir.mkdir()
    pd.DataFrame(
        {
            "scope": ["query_4"],
            "topic_id": [0],
            "topic_label": ["Automatic query_4 0"],
            "papers": [24],
        }
    ).to_csv(table_dir / "scope_topic_prevalence.csv", index=False)
    figure = store.figure_root / "query_4" / "topic_prevalence.png"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"png")

    def render(scope, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in FIGURE_NAMES:
            (output_dir / name).write_bytes(f"{scope}:{name}".encode())

    monkeypatch.setattr(store, "_render_preview_figures", render)

    monkeypatch.setitem(main.state, "topic_review", store)
    monkeypatch.setattr(main, "TOPIC_TABLE_DIR", table_dir)
    monkeypatch.setattr(main, "TOPIC_ENRICHED_DATASET", tmp_path / "missing.csv")
    monkeypatch.setattr(main, "GRAPH_EXPORT_DIR", tmp_path / "graph")

    response = main.topic_review_download("release", scope="query_4")
    assert response.media_type == "application/zip"
    assert response.headers["x-etv-release-state"] == "draft"
    with ZipFile(BytesIO(response.body)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["scope"] == "query_4"
        assert manifest["release_state"] == "draft"
        assert manifest["files"]
        prevalence = pd.read_csv(
            archive.open("topics/query_4/scope_topic_prevalence.csv"),
            dtype=str,
        )
        assert prevalence["topic_label"].iat[0] == "Humanized AI venture topic"
        assert prevalence["automatic_label"].iat[0] == "Automatic query_4 0"
        assert "figures/query_4/topic_prevalence.png" in archive.namelist()
        assert "graph/query_4/draft_topic_nodes.csv" in archive.namelist()


def test_api_write_requires_authentication_and_updates_store(tmp_path, monkeypatch):
    store = _store(tmp_path)
    monkeypatch.setitem(main.state, "topic_review", store)
    request = main.TopicReviewUpdateRequest(
        approved_label="AI adoption",
        review_status="approved",
        reviewer_notes="Reviewed",
        reviewer="Researcher C",
    )
    http_request = Request(
        {
            "type": "http",
            "method": "PATCH",
            "path": "/api/topic-review/query_4/0",
            "headers": [],
        }
    )
    http_request.state.dashboard_access_role = ADMIN_ROLE

    monkeypatch.delenv("ETV_DASHBOARD_REQUIRE_AUTH", raising=False)
    with pytest.raises(main.HTTPException) as blocked:
        main.update_topic_review("query_4", 0, request, http_request)
    assert blocked.value.status_code == 403

    monkeypatch.setenv("ETV_DASHBOARD_REQUIRE_AUTH", "true")
    response = main.update_topic_review("query_4", 0, request, http_request)
    assert response["topic"]["approved_label"] == "AI adoption"
    assert response["summary"]["approved"] == 1
