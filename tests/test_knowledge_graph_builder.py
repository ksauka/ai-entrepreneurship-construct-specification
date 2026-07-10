from aecsp.knowledge_graph import build_publication_graph


def test_enriched_graph_creates_all_node_and_relationship_types() -> None:
    graph = build_publication_graph(
        [
            {
                "paper_id": "P1",
                "Title": "AI in new venture evaluation",
                "Authors": "Ada Lovelace; Grace Hopper",
                "Authors with affiliations": (
                    "Ada Lovelace, Analytical Institute, London, UK; "
                    "Grace Hopper, Yale University, New Haven, USA"
                ),
                "Source title": "Journal of Business Venturing",
                "Year": "2026",
                "DOI": "10.1000/aaa",
                "in_query_1": 1,
                "Author Keywords": "Machine Learning; Opportunity Evaluation",
                "bertopic_topic": "AI adoption in ventures",
                "ai_role_function": "AI as tool",
                "ai_type_form": "predictive AI",
                "ai_mechanism": "reduces uncertainty",
                "level_of_analysis": "venture",
                "entrepreneurial_process_stage": "opportunity evaluation",
                "scope_conditions": "early-stage ventures",
                "definition_construct_clarity": "partial definition",
                "specification_problem": "scope conditions missing",
            },
            {
                "paper_id": "P2",
                "Title": "Generative AI and founders",
                "Authors": "John Doe",
                "DOI": "10.2000/bbb",
                "in_query_1": 1,
                "References": "Lovelace A., (2026) https://doi.org/10.1000/aaa; Other X., (2019) 10.9999/zzz",
            },
        ]
    )

    # nodes
    assert graph.node_count("Publication") == 2
    assert graph.node_count("Author") == 3          # Lovelace, Hopper, Doe
    assert graph.node_count("Institution") == 2     # Analytical Institute, Yale University
    assert graph.node_count("Keyword") == 2         # machine learning, opportunity evaluation
    assert graph.node_count("Topic") == 1           # BERTopic label only, not keywords
    assert graph.node_count("Reference") == 1       # external DOI 10.9999/zzz
    assert graph.node_count("SpecificationProfile") == 1

    # relationships
    assert graph.relationship_count("WROTE") == 3
    assert graph.relationship_count("CO_AUTHORED_WITH") == 1   # Lovelace-Hopper pair
    assert graph.relationship_count("AFFILIATED_WITH") == 2
    assert graph.relationship_count("HAS_KEYWORD") == 2
    assert graph.relationship_count("HAS_TOPIC") == 1
    assert graph.relationship_count("CITES") == 1             # P2 -> P1 (internal DOI match)
    assert graph.relationship_count("REFERENCES") == 1        # P2 -> external reference
    assert graph.relationship_count("SPECIFIES_ROLE") == 1


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
