"""Contract tests for the Stage 3 GraphService (CSV mode, no Neo4j)."""

from pathlib import Path

import pandas as pd
import pytest

from aecsp.api.graph_service import GraphService


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    papers = pd.DataFrame(
        [
            {"paper_id": "P1", "Title": "AI tool for founders", "Source title": "JBV", "Year": "2024",
             "in_query_1": "1", "in_query_2": "0", "in_query_3": "1", "in_query_4": "0",
             "query_sources": "query_1;query_3", "bertopic_topic_label": "0_ai_startup",
             "ai_role_function": "AI as tool", "ai_type_form": "predictive AI"},
            {"paper_id": "P2", "Title": "AI actor in ventures", "Source title": "JBV", "Year": "2025",
             "in_query_1": "1", "in_query_2": "0", "in_query_3": "0", "in_query_4": "0",
             "query_sources": "query_1", "bertopic_topic_label": "0_ai_startup",
             "ai_role_function": "AI as actor/agent", "ai_type_form": "predictive AI"},
        ]
    )
    papers.to_csv(tmp_path / "master_corpus_topics.csv", index=False)
    return tmp_path


def test_scopes_report_overlapping_counts(processed_dir):
    svc = GraphService(processed_dir=processed_dir)
    by_id = {s["id"]: s["papers"] for s in svc.scopes()}
    assert by_id["full_corpus"] == 2
    assert by_id["query_1"] == 2
    assert by_id["query_3"] == 1


def test_overview_reports_convergence(processed_dir):
    svc = GraphService(processed_dir=processed_dir)
    ov = svc.scope_overview("full_corpus")
    assert ov["paper_count"] == 2
    assert ov["has_specifications"] is True
    # Both papers share AI type (converges) but differ on role (diverges).
    assert ov["dimension_convergence"]["ai_type_form"] == 1.0
    assert ov["dimension_convergence"]["ai_role_function"] == 0.0


def test_evidence_returns_the_paper_list(processed_dir):
    svc = GraphService(processed_dir=processed_dir)
    ev = svc.evidence("full_corpus", "ai_role_function", "AI as tool")
    assert [p["paper_id"] for p in ev] == ["P1"]
    assert ev[0]["Title"] == "AI tool for founders"


def test_contrast_flags_same_type_different_role(processed_dir):
    svc = GraphService(processed_dir=processed_dir)
    contrast = svc.contrast("full_corpus", "ai_type_form", "ai_role_function")
    predictive = next(c for c in contrast if c["shared_value"] == "predictive AI")
    assert predictive["contrast_value_count"] == 2


def test_paper_returns_profile_with_neighbours(processed_dir):
    svc = GraphService(processed_dir=processed_dir)
    paper = svc.paper("P1")
    assert paper["ai_role_function"] == "AI as tool"
    # P2 shares AI type but differs on role -> contrasting, not convergent.
    contrasting_ids = {p["paper_id"] for p in paper["contrasting_papers"]}
    assert "P2" in contrasting_ids
