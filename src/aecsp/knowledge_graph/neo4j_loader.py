"""Load an in-memory graph draft into Neo4j.

Inputs: a Neo4j driver, GraphDraft, and load options.
Outputs: idempotently merged graph records and ingestion counts.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from aecsp.knowledge_graph.records import GraphDraft
from aecsp.knowledge_graph.schema import NODE_SPECS
from aecsp.progress import ProgressReporter


def connect(uri: str, user: str, password: str):
    """Return a Neo4j driver (raises a clear error if the driver is missing)."""

    try:
        from neo4j import GraphDatabase
    except ImportError as error:  # pragma: no cover - environment guard
        raise RuntimeError(
            "neo4j driver not installed. Run: pip install neo4j"
        ) from error
    return GraphDatabase.driver(uri, auth=(user, password))


def create_constraints(session) -> None:
    """One uniqueness constraint per node label on its key property."""

    for spec in NODE_SPECS:
        session.run(
            f"CREATE CONSTRAINT {_constraint_name(spec.label)} IF NOT EXISTS "
            f"FOR (n:`{spec.label}`) REQUIRE n.`{spec.key}` IS UNIQUE"
        )


def load_graph(
    driver,
    graph: GraphDraft,
    wipe: bool = False,
    batch_size: int = 1000,
    *,
    show_progress: bool = False,
) -> dict:
    """Ingest all nodes and relationships; return counts."""

    with driver.session() as session:
        if wipe:
            session.run("MATCH (n) DETACH DELETE n")
        create_constraints(session)
        node_progress = (
            ProgressReporter("Neo4j nodes", graph.node_count(), every=batch_size)
            if show_progress else None
        )
        nodes = _load_nodes(session, graph, batch_size, node_progress)
        relationship_progress = (
            ProgressReporter(
                "Neo4j relationships", graph.relationship_count(), every=batch_size
            )
            if show_progress else None
        )
        rels = _load_relationships(session, graph, batch_size, relationship_progress)
    return {"nodes": nodes, "relationships": rels}


def _load_nodes(session, graph: GraphDraft, batch_size: int, progress=None) -> int:
    # Group by (label, key) so each MERGE statement is homogeneous.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in graph.to_node_rows():
        label, key = row["label"], row["key"]
        props = {k: v for k, v in row.items() if k not in {"label", "key", "value"}}
        grouped[(label, key)].append({"value": row["value"], "props": props})

    total = 0
    for (label, key), rows in grouped.items():
        query = (
            f"UNWIND $rows AS row "
            f"MERGE (n:`{label}` {{`{key}`: row.value}}) "
            f"SET n += row.props"
        )
        for chunk in _chunks(rows, batch_size):
            session.run(query, rows=chunk)
            total += len(chunk)
            if progress is not None:
                progress.update(total, detail=label, force=True)
    return total


def _load_relationships(session, graph: GraphDraft, batch_size: int, progress=None) -> int:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in graph.to_relationship_rows():
        signature = (
            row["start_label"],
            row["start_key"],
            row["relationship_type"],
            row["end_label"],
            row["end_key"],
        )
        props = {
            k: v
            for k, v in row.items()
            if k
            not in {
                "start_label",
                "start_key",
                "start_value",
                "relationship_type",
                "end_label",
                "end_key",
                "end_value",
            }
        }
        grouped[signature].append(
            {"start": row["start_value"], "end": row["end_value"], "props": props}
        )

    total = 0
    for (s_label, s_key, rel_type, e_label, e_key), rows in grouped.items():
        query = (
            f"UNWIND $rows AS row "
            f"MATCH (a:`{s_label}` {{`{s_key}`: row.start}}) "
            f"MATCH (b:`{e_label}` {{`{e_key}`: row.end}}) "
            f"MERGE (a)-[r:`{rel_type}`]->(b) "
            f"SET r += row.props"
        )
        for chunk in _chunks(rows, batch_size):
            session.run(query, rows=chunk)
            total += len(chunk)
            if progress is not None:
                progress.update(total, detail=rel_type, force=True)
    return total


def _constraint_name(label: str) -> str:
    return "uniq_" + "".join(ch if ch.isalnum() else "_" for ch in label).lower()


def _chunks(rows: list, size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]
