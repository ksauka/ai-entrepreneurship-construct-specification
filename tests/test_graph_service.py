"""Contract tests for the Stage 3 GraphService (CSV mode, no Neo4j)."""

from pathlib import Path

import pandas as pd
import pytest

from aecsp.api.graph_service import GraphService


class _SecurityResult:
    def __init__(self, roles):
        self.roles = roles

    def single(self):
        return {"user": "etv_app", "roles": self.roles}


class _SecuritySession:
    def __init__(self, roles):
        self.roles = roles

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, query, *args, **kwargs):
        assert "SHOW CURRENT USER" in query
        return _SecurityResult(self.roles)


class _SecurityDriver:
    def __init__(self, roles):
        self.roles = roles

    def session(self, **kwargs):
        return _SecuritySession(self.roles)


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
    assert ov["mean_concentration_score"] == ov["overall_specification_clarity_score"]
    role_profile = ov["dimension_profiles"]["ai_role_function"]
    assert role_profile["category_count"] == 2
    assert role_profile["dominant_share"] == 0.5


def test_dimension_profile_returns_inspectable_categories(processed_dir):
    svc = GraphService(processed_dir=processed_dir)
    profile = svc.dimension_profile("full_corpus", "ai_role_function")
    assert profile["scope"] == "full_corpus"
    assert {row["value"] for row in profile["categories"]} == {
        "AI as tool",
        "AI as actor/agent",
    }


def test_evidence_returns_the_paper_list(processed_dir):
    svc = GraphService(processed_dir=processed_dir)
    ev = svc.evidence("full_corpus", "ai_role_function", "AI as tool")
    assert [p["paper_id"] for p in ev] == ["P1"]
    assert ev[0]["Title"] == "AI tool for founders"


def test_evidence_limit_is_respected(processed_dir):
    svc = GraphService(processed_dir=processed_dir)
    ev = svc.evidence("full_corpus", "ai_type_form", "predictive AI", limit=1)
    assert len(ev) == 1


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


def test_graph_seed_falls_back_without_backend_colours_or_groups(processed_dir):
    svc = GraphService(processed_dir=processed_dir)
    graph = svc.graph_seed("full_corpus", limit=1)
    assert graph["available"] is False
    assert graph["backend"] == "csv"
    assert graph["nodes"]
    assert all("nodeType" in node for node in graph["nodes"])
    assert all("group" not in node and "color" not in node for node in graph["nodes"])


def test_graph_seed_aggregates_repeated_relationship_ids(tmp_path):
    papers = pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "Title": "First collaboration",
                "Authors": "Huang M.-H.; Rust R.T.",
            },
            {
                "paper_id": "P2",
                "Title": "Second collaboration",
                "Authors": "Huang M.-H.; Rust R.T.",
            },
        ]
    )
    papers.to_csv(tmp_path / "master_corpus_topics.csv", index=False)

    graph = GraphService(processed_dir=tmp_path).graph_seed(
        "full_corpus",
        limit=2,
        node_types={"Publication", "Author"},
    )
    coauthor_edges = [
        edge for edge in graph["edges"] if edge["type"] == "CO_AUTHORED_WITH"
    ]

    assert len(coauthor_edges) == 1
    assert coauthor_edges[0]["properties"]["weight"] == 2
    assert len({edge["id"] for edge in graph["edges"]}) == len(graph["edges"])


def test_saved_topic_labels_update_scope_specific_graph_nodes(tmp_path):
    analysis = tmp_path / "analysis"
    review_dir = analysis / "stage4"
    review_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "Title": "Scope-specific topic paper",
                "in_query_1": "1",
                "bertopic_topic": "4",
                "bertopic_topic_label": "automatic full label",
                "query_1_topic_id": "2",
                "query_1_topic_label": "automatic query label",
            }
        ]
    ).to_csv(analysis / "primary_analysis_dataset_with_topics.csv", index=False)
    pd.DataFrame(
        [
            {
                "scope": "full_corpus",
                "topic_id": "4",
                "automatic_label": "automatic full label",
                "approved_label": "Human full label",
                "review_status": "approved",
            },
            {
                "scope": "query_1",
                "topic_id": "2",
                "automatic_label": "automatic query label",
                "approved_label": "Human query label",
                "review_status": "revise",
            },
        ]
    ).to_csv(review_dir / "topic_label_review.csv", index=False)

    service = GraphService(processed_dir=tmp_path)
    graph = service.graph_seed("query_1", limit=1, node_types={"Topic"})
    topic_nodes = [node for node in graph["nodes"] if node["nodeType"] == "Topic"]

    assert len(topic_nodes) == 1
    assert topic_nodes[0]["properties"]["uid"] == "query_1:2"
    assert topic_nodes[0]["properties"]["display_label"] == "Human query label"
    assert topic_nodes[0]["properties"]["automatic_label"] == "automatic query label"
    assert topic_nodes[0]["properties"]["review_status"] == "revise"
    assert topic_nodes[0]["caption"] == "Human query label"


def test_graph_focus_degrades_clearly_without_neo4j(processed_dir):
    svc = GraphService(processed_dir=processed_dir)
    result = svc.graph_neighborhood("full_corpus", "Publication::P1")
    assert result["available"] is False
    assert "requires Neo4j" in result["message"]


def test_graph_rejects_reachable_principal_without_pure_reader_role(processed_dir):
    admin_service = GraphService(
        processed_dir=processed_dir,
        neo4j_driver=_SecurityDriver(["PUBLIC", "admin"]),
    )
    assert admin_service.neo4j_available() is False
    assert admin_service.graph_status()["neo4j_reachable"] is True

    reader_service = GraphService(
        processed_dir=processed_dir,
        neo4j_driver=_SecurityDriver(["PUBLIC", "reader"]),
    )
    assert reader_service.neo4j_available() is True
    assert reader_service.graph_status()["read_only_verified"] is True
