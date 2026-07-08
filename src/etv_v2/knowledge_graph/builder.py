"""Build the ETV_V2 graph draft from paper-level records."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from etv_v2.corpus.query_provenance import QUERY_SOURCE_COLUMN, SEARCH_QUERIES
from etv_v2.knowledge_graph.records import GraphDraft, NodeRef
from etv_v2.specification.schema import (
    SPECIFICATION_DIMENSIONS,
    SPECIFICATION_PROBLEM_COLUMN,
)

PAPER_ID_COLUMNS = ("paper_id", "publication_id", "id", "eid", "doi", "title")
TITLE_COLUMNS = ("title", "Title", "document_title")
ABSTRACT_COLUMNS = ("abstract", "Abstract", "description")
AUTHOR_COLUMNS = ("authors", "Authors", "author_names")
JOURNAL_COLUMNS = ("journal", "source_title", "Source title", "source", "publication_name")
YEAR_COLUMNS = ("year", "Year", "publication_year")
TOPIC_COLUMNS = ("topics", "topic", "bertopic_topic", "keywords", "keyphrases")
VOS_CLUSTER_COLUMNS = ("vos_cluster", "cluster", "vosviewer_cluster")


class GraphBuildError(ValueError):
    """Raised when a record cannot be converted into the graph contract."""


def build_publication_graph(rows: Iterable[Mapping[str, Any]]) -> GraphDraft:
    """Build a graph draft from paper records.

    The input can be dictionaries from CSV rows, pandas ``DataFrame.to_dict``
    records, or any mapping with equivalent fields. Missing optional fields are
    skipped, but each record must have at least one stable publication id field.
    """

    graph = GraphDraft()

    for index, row in enumerate(rows, start=1):
        publication_ref = _add_publication(graph, row, index)
        _add_authors(graph, publication_ref, row)
        _add_journal(graph, publication_ref, row)
        _add_year(graph, publication_ref, row)
        _add_queries(graph, publication_ref, row)
        _add_topics(graph, publication_ref, row)
        _add_vos_cluster(graph, publication_ref, row)
        _add_specification(graph, publication_ref, row)

    return graph


def _add_publication(graph: GraphDraft, row: Mapping[str, Any], index: int) -> NodeRef:
    publication_id = _first_value(row, PAPER_ID_COLUMNS)
    if publication_id is None:
        raise GraphBuildError(f"Record {index} has no usable publication id")

    return graph.add_node(
        "Publication",
        "id",
        publication_id,
        title=_first_value(row, TITLE_COLUMNS),
        abstract=_first_value(row, ABSTRACT_COLUMNS),
        doi=_clean(row.get("doi") or row.get("DOI")),
        source_record_index=index,
    )


def _add_authors(graph: GraphDraft, publication_ref: NodeRef, row: Mapping[str, Any]) -> None:
    for author in _split_multi_value(_first_value(row, AUTHOR_COLUMNS)):
        author_ref = graph.add_node("Author", "name", author)
        graph.add_relationship(author_ref, "WROTE", publication_ref)


def _add_journal(graph: GraphDraft, publication_ref: NodeRef, row: Mapping[str, Any]) -> None:
    journal = _first_value(row, JOURNAL_COLUMNS)
    if journal is None:
        return
    journal_ref = graph.add_node("Journal", "name", journal)
    graph.add_relationship(publication_ref, "PUBLISHED_IN", journal_ref)


def _add_year(graph: GraphDraft, publication_ref: NodeRef, row: Mapping[str, Any]) -> None:
    year = _first_value(row, YEAR_COLUMNS)
    if year is None:
        return
    year_ref = graph.add_node("Year", "value", year)
    graph.add_relationship(publication_ref, "PUBLISHED_IN_YEAR", year_ref)


def _add_queries(graph: GraphDraft, publication_ref: NodeRef, row: Mapping[str, Any]) -> None:
    explicit_sources = {source.lower().replace(" ", "_") for source in _split_multi_value(row.get(QUERY_SOURCE_COLUMN))}

    for query in SEARCH_QUERIES:
        one_hot_value = row.get(query.one_hot_column)
        captured = _is_truthy(one_hot_value) or query.id in explicit_sources or query.label.lower().replace(" ", "_") in explicit_sources
        if not captured:
            continue

        query_ref = graph.add_node(
            "SearchQuery",
            "id",
            query.id,
            display_label=query.label,
            description=query.description,
            one_hot_column=query.one_hot_column,
        )
        graph.add_relationship(publication_ref, "CAPTURED_BY", query_ref)


def _add_topics(graph: GraphDraft, publication_ref: NodeRef, row: Mapping[str, Any]) -> None:
    for column in TOPIC_COLUMNS:
        for topic in _split_multi_value(row.get(column)):
            topic_ref = graph.add_node("Topic", "label", topic)
            graph.add_relationship(
                publication_ref,
                "HAS_TOPIC",
                topic_ref,
                extraction_method=column,
            )


def _add_vos_cluster(graph: GraphDraft, publication_ref: NodeRef, row: Mapping[str, Any]) -> None:
    cluster = _first_value(row, VOS_CLUSTER_COLUMNS)
    if cluster is None:
        return
    cluster_ref = graph.add_node("VOSCluster", "id", cluster)
    graph.add_relationship(publication_ref, "IN_VOS_CLUSTER", cluster_ref)


def _add_specification(graph: GraphDraft, publication_ref: NodeRef, row: Mapping[str, Any]) -> None:
    values = {
        dimension.column: _clean(row.get(dimension.column))
        for dimension in SPECIFICATION_DIMENSIONS
    }
    problem_values = _split_multi_value(row.get(SPECIFICATION_PROBLEM_COLUMN))

    if not any(values.values()) and not problem_values:
        return

    profile_ref = graph.add_node(
        "SpecificationProfile",
        "id",
        f"{publication_ref.value}:specification",
        publication_id=publication_ref.value,
    )
    graph.add_relationship(publication_ref, "HAS_SPECIFICATION", profile_ref)

    for dimension in SPECIFICATION_DIMENSIONS:
        value = values[dimension.column]
        if value is None:
            continue
        dimension_ref = graph.add_node(dimension.graph_node_label, "name", value)
        graph.add_relationship(profile_ref, dimension.relationship_type, dimension_ref)

    for problem in problem_values:
        problem_ref = graph.add_node("SpecificationProblem", "name", problem)
        graph.add_relationship(profile_ref, "HAS_SPECIFICATION_PROBLEM", problem_ref)


def _first_value(row: Mapping[str, Any], columns: Iterable[str]) -> str | None:
    for column in columns:
        value = _clean(row.get(column))
        if value is not None:
            return value
    return None


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _split_multi_value(value: Any) -> list[str]:
    text = _clean(value)
    if text is None:
        return []
    values = [part.strip() for part in re.split(r";|\||,", text)]
    return [value for value in values if value]


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _clean(value)
    if text is None:
        return False
    return text.lower() in {"1", "true", "yes", "y", "x"}
