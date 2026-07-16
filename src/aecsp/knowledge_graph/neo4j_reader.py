"""Read the project graph from Neo4j without mutating database state.

Inputs: a Neo4j driver, a locked analytical scope, and exploration filters.
Outputs: small, serializable subgraphs and search/query results for the API.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, unquote

from aecsp.corpus.scopes import SCOPE_BY_ID
from aecsp.knowledge_graph.schema import NODE_SPECS, RELATIONSHIP_SPECS


NODE_KEYS = {spec.label: spec.key for spec in NODE_SPECS}
NODE_LABELS = frozenset(NODE_KEYS)
RELATIONSHIP_TYPES = frozenset(spec.relationship for spec in RELATIONSHIP_SPECS)
SPECIFICATION_LABELS = frozenset(
    {
        "AIRole",
        "AIType",
        "Mechanism",
        "LevelOfAnalysis",
        "ProcessStage",
        "ScopeCondition",
        "DefinitionClarity",
        "SpecificationProblem",
    }
)

_WRITE_TOKENS = frozenset(
    {
        "ALTER",
        "CALL",
        "CREATE",
        "DATABASE",
        "DELETE",
        "DENY",
        "DETACH",
        "DROP",
        "FOREACH",
        "GRANT",
        "INDEX",
        "LOAD",
        "MERGE",
        "REMOVE",
        "RENAME",
        "REVOKE",
        "ROLE",
        "SET",
        "START",
        "STOP",
        "TERMINATE",
        "TRANSACTION",
        "USE",
        "USER",
    }
)
_READ_PREFIXES = frozenset({"MATCH", "OPTIONAL", "RETURN", "UNWIND", "WITH"})


class GraphQueryError(ValueError):
    """Raised when a graph request violates the read-only query contract."""


def graph_node_id(label: str, value: object) -> str:
    """Return a URL-safe, stable identifier derived from the schema key."""

    if label not in NODE_KEYS:
        raise GraphQueryError(f"Unknown node label: {label}")
    return f"{quote(label, safe='')}::{quote(str(value), safe='')}"


def parse_graph_node_id(node_id: str) -> tuple[str, str, str]:
    """Return ``(label, key, value)`` for one application node identifier."""

    if "::" not in node_id:
        raise GraphQueryError("Invalid graph node identifier")
    encoded_label, encoded_value = node_id.split("::", 1)
    label = unquote(encoded_label)
    value = unquote(encoded_value)
    if label not in NODE_KEYS or not value:
        raise GraphQueryError("Invalid graph node identifier")
    return label, NODE_KEYS[label], value


def validate_read_only_cypher(query: str) -> str:
    """Validate the deliberately narrow raw-Cypher read surface.

    Raw Cypher is optional and is accepted only when the database principal is
    independently verified as a Neo4j ``reader``. This lexical gate is a second
    boundary: it permits query clauses, rejects administration/procedure/write
    clauses, comments, multiple statements, and write keywords outside strings.
    """

    text = str(query or "").strip()
    if not text:
        raise GraphQueryError("Cypher query cannot be empty")
    if len(text) > 10_000:
        raise GraphQueryError("Cypher query exceeds the 10,000-character limit")
    if ";" in text:
        raise GraphQueryError("Multiple Cypher statements are not allowed")
    if "//" in text or "/*" in text or "*/" in text:
        raise GraphQueryError("Cypher comments are not allowed")

    scrubbed = re.sub(r"'(?:\\.|[^'\\])*'", "''", text)
    scrubbed = re.sub(r'"(?:\\.|[^"\\])*"', '""', scrubbed)
    tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", scrubbed.upper())
    if not tokens or tokens[0] not in _READ_PREFIXES:
        raise GraphQueryError(
            "Raw Cypher must begin with MATCH, OPTIONAL MATCH, WITH, UNWIND, or RETURN"
        )
    blocked = sorted(set(tokens) & _WRITE_TOKENS)
    if blocked:
        raise GraphQueryError(
            "Raw Cypher contains a prohibited clause: " + ", ".join(blocked)
        )
    return text


class Neo4jGraphReader:
    """Execute bounded exploration queries through a Neo4j read principal."""

    def __init__(self, driver, database: str = "neo4j") -> None:
        self.driver = driver
        self.database = database
        self._security = self._inspect_security()

    @property
    def security(self) -> dict[str, Any]:
        return dict(self._security)

    @property
    def raw_cypher_enabled(self) -> bool:
        return bool(self._security.get("read_only_verified"))

    def _inspect_security(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "connected": True,
            "principal": None,
            "roles": [],
            "read_only_verified": False,
            "message": "Neo4j connected, but the database role could not be verified.",
        }
        try:
            with self.driver.session(database="system") as session:
                record = session.run(
                    "SHOW CURRENT USER YIELD user, roles RETURN user, roles"
                ).single()
            if record is None:
                return status
            principal = record.get("user")
            roles = sorted(str(role) for role in (record.get("roles") or []))
            forbidden = {"admin", "architect", "publisher", "editor"}
            verified = "reader" in roles and not forbidden.intersection(roles)
            status.update(
                {
                    "principal": principal,
                    "roles": roles,
                    "read_only_verified": verified,
                    "message": (
                        "Neo4j reader role verified."
                        if verified
                        else "The Neo4j principal is not a verified reader role. Raw Cypher is disabled."
                    ),
                }
            )
        except Exception as error:
            status["message"] = (
                "Neo4j is connected, but role verification failed: "
                f"{type(error).__name__}. Raw Cypher is disabled."
            )
        return status

    def counts(self) -> dict[str, Any]:
        """Return database counts without assuming optional node types exist."""

        query = (
            "MATCH (n) UNWIND labels(n) AS label "
            "WITH collect({label: label}) AS labels "
            "MATCH ()-[r]->() "
            "RETURN labels, collect({type: type(r)}) AS relationships"
        )
        with self.driver.session(database=self.database) as session:
            record = session.run(query).single()
        label_counts = Counter(item["label"] for item in (record["labels"] if record else []))
        relationship_counts = Counter(
            item["type"] for item in (record["relationships"] if record else [])
        )
        return {
            "nodes": sum(label_counts.values()),
            "relationships": sum(relationship_counts.values()),
            "node_labels": dict(sorted(label_counts.items())),
            "relationship_types": dict(sorted(relationship_counts.items())),
        }

    def seed(
        self,
        scope_id: str,
        *,
        limit: int = 30,
        node_types: set[str] | None = None,
        relationship_types: set[str] | None = None,
        specification_label: str | None = None,
        specification_value: str | None = None,
    ) -> dict[str, Any]:
        """Return a small, high-degree publication-centred starting graph."""

        node_types = self._labels(node_types)
        relationship_types = self._relationships(relationship_types)
        scope_clause = self._publication_scope_clause("p", scope_id)
        specification_clause = ""
        parameters: dict[str, Any] = {
            "limit": min(max(int(limit), 1), 100),
            "node_types": sorted(node_types),
            "relationship_types": sorted(relationship_types),
            "edge_limit": 1_500,
        }
        if specification_label or specification_value:
            if specification_label not in SPECIFICATION_LABELS or not specification_value:
                raise GraphQueryError(
                    "A valid specification label and code value must be supplied together"
                )
            specification_clause = (
                "AND EXISTS { MATCH (p)-[:HAS_SPECIFICATION]->"
                "(:SpecificationProfile)-[]->(spec:`"
                + specification_label
                + "`) WHERE spec.name = $specification_value } "
            )
            parameters["specification_value"] = specification_value

        query = (
            "MATCH (p:Publication) "
            f"WHERE {scope_clause} {specification_clause}"
            "OPTIONAL MATCH (p)-[degree_rel]-() "
            "WITH p, count(degree_rel) AS graph_degree "
            "ORDER BY graph_degree DESC, coalesce(toInteger(p.citations), 0) DESC "
            "LIMIT $limit "
            "OPTIONAL MATCH (p)-[r]-(n) "
            "WHERE (size($relationship_types) = 0 OR type(r) IN $relationship_types) "
            "AND (size($node_types) = 0 OR any(label IN labels(n) WHERE label IN $node_types)) "
            "RETURN p, r, n LIMIT $edge_limit"
        )
        return self._subgraph(query, parameters, scope_id, "seed")

    def neighborhood(
        self,
        scope_id: str,
        node_id: str,
        *,
        relationship_types: set[str] | None = None,
    ) -> dict[str, Any]:
        """Return only one node and its directly connected neighbors."""

        label, key, value = parse_graph_node_id(node_id)
        relationship_types = self._relationships(relationship_types)
        scope_clause = self._node_scope_clause("center", scope_id)
        neighbor_scope = self._neighbor_scope_clause("neighbor", scope_id)
        query = (
            f"MATCH (center:`{label}` {{`{key}`: $value}}) "
            f"WHERE {scope_clause} "
            "OPTIONAL MATCH (center)-[r]-(neighbor) "
            "WHERE (size($relationship_types) = 0 OR type(r) IN $relationship_types) "
            f"AND ({neighbor_scope}) "
            "RETURN center, r, neighbor LIMIT 1500"
        )
        return self._subgraph(
            query,
            {"value": value, "relationship_types": sorted(relationship_types)},
            scope_id,
            "neighborhood",
            focus=node_id,
        )

    def expand(
        self,
        scope_id: str,
        node_id: str,
        *,
        relationship_types: set[str] | None = None,
    ) -> dict[str, Any]:
        """Return the next one-hop neighborhood for client-side merging."""

        result = self.neighborhood(
            scope_id, node_id, relationship_types=relationship_types
        )
        result["action"] = "expand"
        return result

    def search(
        self,
        scope_id: str,
        text: str,
        *,
        node_types: set[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search node identifiers and display properties within one scope."""

        query_text = str(text or "").strip().lower()
        if not query_text:
            return []
        node_types = self._labels(node_types)
        scope_clause = self._node_scope_clause("n", scope_id)
        query = (
            "MATCH (n) "
            "WHERE (size($node_types) = 0 OR any(label IN labels(n) WHERE label IN $node_types)) "
            "AND any(property IN keys(n) WHERE "
            "toLower(toString(n[property])) CONTAINS $text) "
            f"AND ({scope_clause}) "
            "OPTIONAL MATCH (n)-[r]-() "
            "RETURN n, count(r) AS degree ORDER BY degree DESC LIMIT $limit"
        )
        with self.driver.session(database=self.database) as session:
            records = session.run(
                query,
                text=query_text,
                node_types=sorted(node_types),
                limit=min(max(int(limit), 1), 50),
            )
            return [
                self._node_payload(record["n"], degree=int(record["degree"] or 0))
                for record in records
            ]

    def raw_query(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
        *,
        limit: int = 500,
    ) -> dict[str, Any]:
        """Execute bounded raw Cypher only for a verified Neo4j reader role."""

        if not self.raw_cypher_enabled:
            raise GraphQueryError(
                "Raw Cypher is disabled until the app principal is verified as a Neo4j reader role"
            )
        safe_query = validate_read_only_cypher(query)
        safe_parameters = dict(parameters or {})
        with self.driver.session(database=self.database, default_access_mode="READ") as session:
            result = session.run(safe_query, safe_parameters)
            records = []
            nodes: dict[str, dict[str, Any]] = {}
            edges: dict[str, dict[str, Any]] = {}
            for index, record in enumerate(result):
                if index >= min(max(int(limit), 1), 500):
                    break
                plain: dict[str, Any] = {}
                for key in record.keys():
                    value = record[key]
                    self._extract_graph_values(value, nodes, edges)
                    plain[key] = self._plain_value(value)
                records.append(plain)
        graph = self._finish_graph(nodes, edges, "full_corpus", "cypher")
        graph["rows"] = records
        graph["columns"] = list(records[0]) if records else []
        return graph

    def _subgraph(
        self,
        query: str,
        parameters: Mapping[str, Any],
        scope_id: str,
        action: str,
        *,
        focus: str | None = None,
    ) -> dict[str, Any]:
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[str, dict[str, Any]] = {}
        with self.driver.session(database=self.database, default_access_mode="READ") as session:
            for record in session.run(query, dict(parameters)):
                for value in record.values():
                    self._extract_graph_values(value, nodes, edges)
        result = self._finish_graph(nodes, edges, scope_id, action)
        result["focus"] = focus
        return result

    def _extract_graph_values(
        self,
        value: Any,
        nodes: dict[str, dict[str, Any]],
        edges: dict[str, dict[str, Any]],
    ) -> None:
        if value is None:
            return
        if self._is_node(value):
            payload = self._node_payload(value)
            nodes[payload["id"]] = payload
            return
        if self._is_relationship(value):
            payload = self._relationship_payload(value)
            edges[payload["id"]] = payload
            self._extract_graph_values(value.start_node, nodes, edges)
            self._extract_graph_values(value.end_node, nodes, edges)
            return
        if hasattr(value, "nodes") and hasattr(value, "relationships"):
            for node in value.nodes:
                self._extract_graph_values(node, nodes, edges)
            for relationship in value.relationships:
                self._extract_graph_values(relationship, nodes, edges)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                self._extract_graph_values(item, nodes, edges)

    def _node_payload(self, node: Any, *, degree: int | None = None) -> dict[str, Any]:
        labels = sorted(str(label) for label in node.labels)
        label = next((item for item in labels if item in NODE_KEYS), labels[0])
        key = NODE_KEYS.get(label)
        properties = {str(k): self._plain_value(v) for k, v in dict(node).items()}
        value = properties.get(key) if key else None
        if value in (None, ""):
            value = getattr(node, "element_id", None) or getattr(node, "id", "unknown")
        caption = self._caption(label, properties, value)
        return {
            "id": graph_node_id(label, value),
            "nodeType": label,
            "caption": caption,
            "degree": int(degree or 0),
            "properties": properties,
        }

    def _relationship_payload(self, relationship: Any) -> dict[str, Any]:
        start = self._node_payload(relationship.start_node)
        end = self._node_payload(relationship.end_node)
        rel_id = getattr(relationship, "element_id", None) or (
            f"{start['id']}::{relationship.type}::{end['id']}"
        )
        return {
            "id": str(rel_id),
            "from": start["id"],
            "to": end["id"],
            "type": str(relationship.type),
            "properties": {
                str(k): self._plain_value(v) for k, v in dict(relationship).items()
            },
        }

    def _finish_graph(
        self,
        nodes: dict[str, dict[str, Any]],
        edges: dict[str, dict[str, Any]],
        scope_id: str,
        action: str,
    ) -> dict[str, Any]:
        degrees = Counter()
        for edge in edges.values():
            degrees[edge["from"]] += 1
            degrees[edge["to"]] += 1
        for node_id, node in nodes.items():
            node["degree"] = max(int(node.get("degree") or 0), degrees[node_id])
        counts = Counter(node["nodeType"] for node in nodes.values())
        return {
            "available": True,
            "backend": "neo4j",
            "scope": scope_id,
            "action": action,
            "nodes": list(nodes.values()),
            "edges": list(edges.values()),
            "counts": dict(sorted(counts.items())),
            "security": self.security,
        }

    def _publication_scope_clause(self, variable: str, scope_id: str) -> str:
        scope = SCOPE_BY_ID.get(scope_id)
        if scope is None:
            raise GraphQueryError(f"Unknown graph scope: {scope_id}")
        if scope.filter_column is None:
            return "true"
        return f"coalesce(toInteger({variable}.`{scope.filter_column}`), 0) = 1"

    def _node_scope_clause(self, variable: str, scope_id: str) -> str:
        publication_clause = self._publication_scope_clause(variable, scope_id)
        linked_clause = self._publication_scope_clause("scope_publication", scope_id)
        return (
            f"(('Publication' IN labels({variable}) AND {publication_clause}) OR "
            f"(NOT ('Publication' IN labels({variable})) AND EXISTS {{ "
            f"MATCH ({variable})-[*1..2]-(scope_publication:Publication) "
            f"WHERE {linked_clause} }}))"
        )

    def _neighbor_scope_clause(self, neighbor: str, scope_id: str) -> str:
        return self._node_scope_clause(neighbor, scope_id)

    @staticmethod
    def _labels(values: set[str] | None) -> set[str]:
        labels = set(values or ())
        unknown = labels - NODE_LABELS
        if unknown:
            raise GraphQueryError("Unknown node type: " + ", ".join(sorted(unknown)))
        return labels

    @staticmethod
    def _relationships(values: set[str] | None) -> set[str]:
        relationships = set(values or ())
        unknown = relationships - RELATIONSHIP_TYPES
        if unknown:
            raise GraphQueryError(
                "Unknown relationship type: " + ", ".join(sorted(unknown))
            )
        return relationships

    @staticmethod
    def _caption(label: str, properties: Mapping[str, Any], value: Any) -> str:
        if label == "Publication":
            return str(properties.get("title") or value)
        return str(
            properties.get("name")
            or properties.get("label")
            or properties.get("term")
            or properties.get("value")
            or properties.get("id")
            or properties.get("doi")
            or value
        )

    @staticmethod
    def _is_node(value: Any) -> bool:
        return hasattr(value, "labels") and hasattr(value, "items")

    @staticmethod
    def _is_relationship(value: Any) -> bool:
        return (
            hasattr(value, "type")
            and hasattr(value, "start_node")
            and hasattr(value, "end_node")
        )

    @classmethod
    def _plain_value(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {str(k): cls._plain_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._plain_value(item) for item in value]
        if cls._is_node(value):
            return cls._node_payload_static(value)
        if cls._is_relationship(value):
            return {
                "type": str(value.type),
                "properties": {
                    str(k): cls._plain_value(v) for k, v in dict(value).items()
                },
            }
        return str(value)

    @classmethod
    def _node_payload_static(cls, node: Any) -> dict[str, Any]:
        return {
            "labels": sorted(str(label) for label in node.labels),
            "properties": {
                str(k): cls._plain_value(v) for k, v in dict(node).items()
            },
        }
