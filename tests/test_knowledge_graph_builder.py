from etv_v2.knowledge_graph import build_publication_graph


def test_build_publication_graph_creates_query_topic_and_specification_nodes() -> None:
    graph = build_publication_graph(
        [
            {
                "paper_id": "P1",
                "title": "AI in new venture evaluation",
                "authors": "Ada Lovelace; Grace Hopper",
                "source_title": "Journal of Business Venturing",
                "year": 2026,
                "in_query_1": 1,
                "keywords": "machine learning; opportunity evaluation",
                "vos_cluster": "cluster_2",
                "ai_role_function": "AI as tool",
                "ai_type_form": "predictive AI",
                "ai_mechanism": "reduces uncertainty",
                "level_of_analysis": "venture",
                "entrepreneurial_process_stage": "opportunity evaluation",
                "scope_conditions": "early-stage ventures",
                "definition_construct_clarity": "partial definition",
                "specification_problem": "scope conditions missing",
            }
        ]
    )

    assert graph.node_count("Publication") == 1
    assert graph.node_count("Author") == 2
    assert graph.node_count("SearchQuery") == 1
    assert graph.node_count("Topic") == 2
    assert graph.node_count("SpecificationProfile") == 1
    assert graph.node_count("SpecificationProblem") == 1

    assert graph.relationship_count("WROTE") == 2
    assert graph.relationship_count("CAPTURED_BY") == 1
    assert graph.relationship_count("HAS_TOPIC") == 2
    assert graph.relationship_count("HAS_SPECIFICATION") == 1
    assert graph.relationship_count("SPECIFIES_ROLE") == 1
    assert graph.relationship_count("HAS_SPECIFICATION_PROBLEM") == 1


def test_graph_draft_exports_node_and_relationship_rows() -> None:
    graph = build_publication_graph(
        [
            {
                "paper_id": "P2",
                "title": "Generative AI and founders",
                "query_sources": "query_2",
                "ai_role_function": "AI as actor/agent",
            }
        ]
    )

    node_rows = graph.to_node_rows()
    relationship_rows = graph.to_relationship_rows()

    assert any(row["label"] == "Publication" and row["value"] == "P2" for row in node_rows)
    assert any(row["relationship_type"] == "CAPTURED_BY" for row in relationship_rows)
    assert any(row["relationship_type"] == "SPECIFIES_ROLE" for row in relationship_rows)
