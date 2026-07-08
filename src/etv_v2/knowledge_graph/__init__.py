"""Knowledge graph schema and construction tools for ETV_V2."""

from etv_v2.knowledge_graph.builder import GraphBuildError, build_publication_graph
from etv_v2.knowledge_graph.records import (
    GraphDraft,
    GraphNode,
    GraphRelationship,
    NodeRef,
)
from etv_v2.knowledge_graph.schema import (
    CONTRAST_EDGE,
    CONVERGENCE_EDGE,
    NODE_SPECS,
    RELATIONSHIP_SPECS,
    NodeSpec,
    RelationshipSpec,
)

__all__ = [
    "CONTRAST_EDGE",
    "CONVERGENCE_EDGE",
    "NODE_SPECS",
    "RELATIONSHIP_SPECS",
    "GraphBuildError",
    "GraphDraft",
    "GraphNode",
    "GraphRelationship",
    "NodeRef",
    "NodeSpec",
    "RelationshipSpec",
    "build_publication_graph",
]
