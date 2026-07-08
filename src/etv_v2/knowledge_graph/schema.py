"""Target knowledge graph schema for ETV_V2.

This module defines the contract for the new graph. It intentionally keeps the
old workable Publication -> Topic path while adding query provenance,
bibliometric context, VOS membership, and AI specification structure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeSpec:
    label: str
    key: str
    description: str


@dataclass(frozen=True)
class RelationshipSpec:
    source: str
    relationship: str
    target: str
    description: str


NODE_SPECS: tuple[NodeSpec, ...] = (
    NodeSpec("Publication", "id", "Deduplicated bibliometric record."),
    NodeSpec("Author", "name", "Paper author."),
    NodeSpec("Journal", "name", "Source title or journal."),
    NodeSpec("Year", "value", "Publication year."),
    NodeSpec("SearchQuery", "id", "Scopus query provenance node."),
    NodeSpec("Topic", "label", "BERTopic topic, KeyBERT phrase, or extracted keyword."),
    NodeSpec("VOSCluster", "id", "Cluster imported from VOSviewer exports."),
    NodeSpec("SpecificationProfile", "id", "Paper-level construct specification profile."),
    NodeSpec("AIRole", "name", "Role or function assigned to AI."),
    NodeSpec("AIType", "name", "Form or type of AI named in the paper."),
    NodeSpec("Mechanism", "name", "Claimed causal or theoretical mechanism."),
    NodeSpec("LevelOfAnalysis", "name", "Analytical level of the AI claim."),
    NodeSpec("ProcessStage", "name", "Entrepreneurship process stage."),
    NodeSpec("ScopeCondition", "name", "Boundary condition for the AI claim."),
    NodeSpec("DefinitionClarity", "name", "Degree of construct-definition clarity."),
    NodeSpec("SpecificationProblem", "name", "Diagnosed construct specification issue."),
)


RELATIONSHIP_SPECS: tuple[RelationshipSpec, ...] = (
    RelationshipSpec("Author", "WROTE", "Publication", "Authorship."),
    RelationshipSpec("Publication", "PUBLISHED_IN", "Journal", "Journal/source title."),
    RelationshipSpec("Publication", "PUBLISHED_IN_YEAR", "Year", "Publication year."),
    RelationshipSpec("Publication", "CAPTURED_BY", "SearchQuery", "Search query membership."),
    RelationshipSpec("Publication", "HAS_TOPIC", "Topic", "Topic, keyword, or keyphrase assignment."),
    RelationshipSpec("Publication", "IN_VOS_CLUSTER", "VOSCluster", "VOSviewer cluster membership."),
    RelationshipSpec(
        "Publication",
        "HAS_SPECIFICATION",
        "SpecificationProfile",
        "Paper-level construct coding profile.",
    ),
    RelationshipSpec("SpecificationProfile", "SPECIFIES_ROLE", "AIRole", "AI role/function."),
    RelationshipSpec("SpecificationProfile", "SPECIFIES_TYPE", "AIType", "AI type/form."),
    RelationshipSpec(
        "SpecificationProfile",
        "SPECIFIES_MECHANISM",
        "Mechanism",
        "Mechanism connected to AI.",
    ),
    RelationshipSpec(
        "SpecificationProfile",
        "SPECIFIES_LEVEL",
        "LevelOfAnalysis",
        "Level of analysis.",
    ),
    RelationshipSpec(
        "SpecificationProfile",
        "SPECIFIES_PROCESS",
        "ProcessStage",
        "Entrepreneurship process stage.",
    ),
    RelationshipSpec(
        "SpecificationProfile",
        "SPECIFIES_SCOPE",
        "ScopeCondition",
        "Scope condition or boundary.",
    ),
    RelationshipSpec(
        "SpecificationProfile",
        "HAS_DEFINITION_CLARITY",
        "DefinitionClarity",
        "Construct definition clarity.",
    ),
    RelationshipSpec(
        "SpecificationProfile",
        "HAS_SPECIFICATION_PROBLEM",
        "SpecificationProblem",
        "Diagnosed specification issue.",
    ),
)


CONVERGENCE_EDGE = "CONVERGES_WITH"
CONTRAST_EDGE = "CONTRASTS_WITH"
