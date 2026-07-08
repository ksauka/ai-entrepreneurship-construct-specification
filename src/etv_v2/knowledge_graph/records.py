"""In-memory graph records used before Neo4j export/ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, order=True)
class NodeRef:
    """Stable reference to one graph node."""

    label: str
    key: str
    value: str


@dataclass(frozen=True)
class GraphNode:
    """Node payload with a stable identity and properties."""

    ref: NodeRef
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphRelationship:
    """Relationship payload connecting two stable node references."""

    start: NodeRef
    relationship_type: str
    end: NodeRef
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphDraft:
    """Small graph container for CSV export, tests, and later Neo4j ingestion."""

    _nodes: dict[NodeRef, GraphNode] = field(default_factory=dict)
    _relationships: list[GraphRelationship] = field(default_factory=list)

    @property
    def nodes(self) -> tuple[GraphNode, ...]:
        return tuple(self._nodes.values())

    @property
    def relationships(self) -> tuple[GraphRelationship, ...]:
        return tuple(self._relationships)

    def add_node(
        self,
        label: str,
        key: str,
        value: Any,
        **properties: Any,
    ) -> NodeRef:
        """Add or update a node and return its stable reference."""

        clean_value = _clean_identifier(value)
        ref = NodeRef(label=label, key=key, value=clean_value)
        existing = self._nodes.get(ref)
        node_properties = {key: val for key, val in properties.items() if val not in (None, "")}

        if existing is None:
            self._nodes[ref] = GraphNode(ref=ref, properties=node_properties)
            return ref

        merged_properties = {**existing.properties, **node_properties}
        self._nodes[ref] = GraphNode(ref=ref, properties=merged_properties)
        return ref

    def add_relationship(
        self,
        start: NodeRef,
        relationship_type: str,
        end: NodeRef,
        **properties: Any,
    ) -> GraphRelationship:
        """Add a relationship between two already referenced nodes."""

        relationship_properties = {
            key: val for key, val in properties.items() if val not in (None, "")
        }
        relationship = GraphRelationship(
            start=start,
            relationship_type=relationship_type,
            end=end,
            properties=relationship_properties,
        )
        self._relationships.append(relationship)
        return relationship

    def node_count(self, label: str | None = None) -> int:
        if label is None:
            return len(self._nodes)
        return sum(1 for node in self._nodes.values() if node.ref.label == label)

    def relationship_count(self, relationship_type: str | None = None) -> int:
        if relationship_type is None:
            return len(self._relationships)
        return sum(
            1
            for relationship in self._relationships
            if relationship.relationship_type == relationship_type
        )

    def to_node_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for node in self.nodes:
            rows.append(
                {
                    "label": node.ref.label,
                    "key": node.ref.key,
                    "value": node.ref.value,
                    **node.properties,
                }
            )
        return rows

    def to_relationship_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for relationship in self.relationships:
            rows.append(
                {
                    "start_label": relationship.start.label,
                    "start_key": relationship.start.key,
                    "start_value": relationship.start.value,
                    "relationship_type": relationship.relationship_type,
                    "end_label": relationship.end.label,
                    "end_key": relationship.end.key,
                    "end_value": relationship.end.value,
                    **relationship.properties,
                }
            )
        return rows


def _clean_identifier(value: Any) -> str:
    clean_value = str(value).strip()
    if not clean_value:
        raise ValueError("Graph node identifiers cannot be empty")
    return clean_value
