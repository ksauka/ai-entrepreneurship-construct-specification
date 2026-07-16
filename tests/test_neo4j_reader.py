"""Security and identifier contracts for Neo4j graph exploration."""

import pytest

from aecsp.knowledge_graph.neo4j_reader import (
    GraphQueryError,
    graph_node_id,
    parse_graph_node_id,
    validate_read_only_cypher,
)


def test_graph_node_identifier_round_trips_reserved_characters() -> None:
    node_id = graph_node_id("Publication", "eid:2-s2.0-123::suffix/value")
    assert parse_graph_node_id(node_id) == (
        "Publication",
        "id",
        "eid:2-s2.0-123::suffix/value",
    )


def test_read_only_cypher_accepts_parameterized_match() -> None:
    query = "MATCH (p:Publication {id: $paper_id}) RETURN p LIMIT 1"
    assert validate_read_only_cypher(query) == query
    assert validate_read_only_cypher("RETURN 'CREATE is text' AS example")


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n) DELETE n RETURN n",
        "MATCH (n) SET n.changed = true RETURN n",
        "MERGE (n:Publication {id: 'P1'}) RETURN n",
        "CALL db.labels()",
        "SHOW USERS",
        "MATCH (n) RETURN n; MATCH (m) RETURN m",
        "MATCH (n) // hidden write\nRETURN n",
        "MATCH (n) /* hidden write */ RETURN n",
    ],
)
def test_read_only_cypher_rejects_writes_admin_and_multiple_statements(
    query: str,
) -> None:
    with pytest.raises(GraphQueryError):
        validate_read_only_cypher(query)


def test_node_identifier_rejects_labels_outside_locked_contract() -> None:
    with pytest.raises(GraphQueryError, match="Unknown node label"):
        graph_node_id("VOSCluster", "1")
