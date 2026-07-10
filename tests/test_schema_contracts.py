from aecsp.corpus.query_provenance import QUERY_ONE_HOT_COLUMNS, SEARCH_QUERIES
from aecsp.knowledge_graph.schema import NODE_SPECS, RELATIONSHIP_SPECS
from aecsp.specification.schema import SPECIFICATION_DIMENSIONS


def test_four_search_queries_are_defined() -> None:
    assert len(SEARCH_QUERIES) == 4
    assert QUERY_ONE_HOT_COLUMNS == (
        "in_query_1",
        "in_query_2",
        "in_query_3",
        "in_query_4",
    )


def test_seven_specification_dimensions_are_defined() -> None:
    assert len(SPECIFICATION_DIMENSIONS) == 7


def test_kg_contains_specification_backbone() -> None:
    node_labels = {node.label for node in NODE_SPECS}
    relationship_types = {relationship.relationship for relationship in RELATIONSHIP_SPECS}

    assert "SpecificationProfile" in node_labels
    assert "SpecificationProblem" in node_labels
    assert "HAS_SPECIFICATION" in relationship_types
    assert "HAS_SPECIFICATION_PROBLEM" in relationship_types
