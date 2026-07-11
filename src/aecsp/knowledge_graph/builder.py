"""Build an in-memory knowledge graph from paper-level records.

Inputs: corpus records with bibliographic, query, topic, and specification data.
Outputs: a GraphDraft containing normalized nodes and relationships.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from aecsp.corpus.query_provenance import QUERY_SOURCE_COLUMN, SEARCH_QUERIES
from aecsp.knowledge_graph.records import GraphDraft, NodeRef
from aecsp.progress import ProgressReporter
from aecsp.specification.schema import (
    SPECIFICATION_DIMENSIONS,
    SPECIFICATION_PROBLEM_COLUMN,
)

PAPER_ID_COLUMNS = ("paper_id", "publication_id", "id", "eid", "doi", "title")
TITLE_COLUMNS = ("title", "Title", "document_title")
ABSTRACT_COLUMNS = ("abstract", "Abstract", "description")
AUTHOR_COLUMNS = ("authors", "Authors", "author_names")
JOURNAL_COLUMNS = ("journal", "source_title", "Source title", "source", "publication_name")
YEAR_COLUMNS = ("year", "Year", "publication_year")
AFFILIATION_COLUMNS = ("Authors with affiliations", "authors_with_affiliations")
REFERENCE_COLUMNS = ("References", "references")

# BERTopic assignments become Topic nodes; keywords are handled separately.
TOPIC_COLUMNS = ("bertopic_topic", "topic", "topics")

# (column, source tag) for keyword-bearing fields.
KEYWORD_SOURCES = (
    ("Author Keywords", "author"),
    ("author_keywords", "author"),
    ("Index Keywords", "index"),
    ("index_keywords", "index"),
    ("keyphrases", "keybert"),
)

INSTITUTION_HINTS = (
    "universit", "institut", "college", "school", "academ", "hospital",
    "laborator", "centre", "center", "polytechnic", "foundation",
)

_DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", re.IGNORECASE)

# Papers with more authors than this skip co-authorship pairing (avoids the
# quadratic blow-up on large consortium author lists).
MAX_COAUTHORS = 25


class GraphBuildError(ValueError):
    """Raised when a record cannot be converted into the graph contract."""


def build_publication_graph(
    rows: Iterable[Mapping[str, Any]], *, show_progress: bool = False
) -> GraphDraft:
    """Build a graph draft from paper records.

    Rows are materialised so a first pass can index DOIs (for internal CITES
    resolution) before the main build pass.
    """

    rows = list(rows)
    doi_index = _build_doi_index(rows)
    graph = GraphDraft()
    progress = (
        ProgressReporter("Graph publications", len(rows), every=250)
        if show_progress
        else None
    )

    for index, row in enumerate(rows, start=1):
        publication_ref = _add_publication(graph, row, index)
        author_refs = _add_authors(graph, publication_ref, row)
        _add_coauthorship(graph, author_refs)
        _add_affiliations(graph, author_refs, row)
        _add_journal(graph, publication_ref, row)
        _add_year(graph, publication_ref, row)
        _add_queries(graph, publication_ref, row)
        _add_topics(graph, publication_ref, row)
        _add_keywords(graph, publication_ref, row)
        _add_references(graph, publication_ref, row, doi_index)
        _add_specification(graph, publication_ref, row)
        if progress is not None:
            progress.update(index, detail=str(publication_ref.value))

    return graph


def _build_doi_index(rows: list[Mapping[str, Any]]) -> dict[str, str]:
    index: dict[str, str] = {}
    for row in rows:
        pub_id = _first_value(row, PAPER_ID_COLUMNS)
        doi = _normalize_doi(row.get("DOI") or row.get("doi"))
        if pub_id and doi:
            index.setdefault(doi, pub_id)
    return index


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
        year=_first_value(row, YEAR_COLUMNS),
        source_record_index=index,
    )


def _add_authors(
    graph: GraphDraft, publication_ref: NodeRef, row: Mapping[str, Any]
) -> dict[str, NodeRef]:
    author_refs: dict[str, NodeRef] = {}
    for author in _split_list(_first_value(row, AUTHOR_COLUMNS)):
        author_ref = graph.add_node("Author", "name", author)
        graph.add_relationship(author_ref, "WROTE", publication_ref)
        author_refs[author] = author_ref
    return author_refs


def _add_coauthorship(graph: GraphDraft, author_refs: dict[str, NodeRef]) -> None:
    names = sorted(author_refs)
    if len(names) < 2 or len(names) > MAX_COAUTHORS:
        return
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            graph.add_relationship(
                author_refs[names[i]], "CO_AUTHORED_WITH", author_refs[names[j]]
            )


def _add_affiliations(
    graph: GraphDraft, author_refs: dict[str, NodeRef], row: Mapping[str, Any]
) -> None:
    raw = _first_value(row, AFFILIATION_COLUMNS)
    if raw is None:
        return
    for entry in raw.split(";"):
        parts = [p.strip() for p in entry.split(",") if p.strip()]
        if len(parts) < 2:
            continue
        name = parts[0]
        institution = _extract_institution(parts[1:])
        if institution is None:
            continue
        author_ref = author_refs.get(name) or graph.add_node("Author", "name", name)
        institution_ref = graph.add_node("Institution", "name", institution)
        graph.add_relationship(author_ref, "AFFILIATED_WITH", institution_ref)


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
    explicit_sources = {
        source.lower().replace(" ", "_")
        for source in _split_list(row.get(QUERY_SOURCE_COLUMN))
    }
    for query in SEARCH_QUERIES:
        captured = (
            _is_truthy(row.get(query.one_hot_column))
            or query.id in explicit_sources
        )
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
        for topic in _split_list(row.get(column)):
            topic_ref = graph.add_node("Topic", "label", topic)
            graph.add_relationship(
                publication_ref, "HAS_TOPIC", topic_ref, model="global"
            )


def _add_keywords(graph: GraphDraft, publication_ref: NodeRef, row: Mapping[str, Any]) -> None:
    seen: set[tuple[str, str]] = set()
    for column, source in KEYWORD_SOURCES:
        for term in _split_list(row.get(column)):
            normalized = _normalize_keyword(term)
            if not normalized or (normalized, source) in seen:
                continue
            seen.add((normalized, source))
            keyword_ref = graph.add_node("Keyword", "term", normalized, text=term)
            graph.add_relationship(
                publication_ref, "HAS_KEYWORD", keyword_ref, source=source
            )


def _add_references(
    graph: GraphDraft,
    publication_ref: NodeRef,
    row: Mapping[str, Any],
    doi_index: dict[str, str],
) -> None:
    raw = _first_value(row, REFERENCE_COLUMNS)
    if raw is None:
        return
    seen: set[str] = set()
    for match in _DOI_PATTERN.findall(raw):
        doi = _normalize_doi(match)
        if not doi or doi in seen:
            continue
        seen.add(doi)
        target_id = doi_index.get(doi)
        if target_id and target_id != publication_ref.value:
            target_ref = graph.add_node("Publication", "id", target_id)
            graph.add_relationship(publication_ref, "CITES", target_ref)
        elif not target_id:
            reference_ref = graph.add_node("Reference", "doi", doi)
            graph.add_relationship(publication_ref, "REFERENCES", reference_ref)


def _add_specification(graph: GraphDraft, publication_ref: NodeRef, row: Mapping[str, Any]) -> None:
    values = {
        dimension.column: _clean(row.get(dimension.column))
        for dimension in SPECIFICATION_DIMENSIONS
    }
    problem_values = _split_list(row.get(SPECIFICATION_PROBLEM_COLUMN))

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


def _extract_institution(segments: list[str]) -> str | None:
    for segment in segments:
        if any(hint in segment.lower() for hint in INSTITUTION_HINTS):
            return segment
    return segments[0] if segments else None


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


def _split_list(value: Any) -> list[str]:
    """Split a Scopus multi-value cell on semicolons (authors, keywords)."""

    text = _clean(value)
    if text is None:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def _normalize_keyword(term: str) -> str:
    text = str(term).strip().lower()
    return re.sub(r"\s+", " ", text)


def _normalize_doi(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"\s+", "", text)
    return text.rstrip(".,;)")


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _clean(value)
    if text is None:
        return False
    return text.lower() in {"1", "true", "yes", "y", "x"}
