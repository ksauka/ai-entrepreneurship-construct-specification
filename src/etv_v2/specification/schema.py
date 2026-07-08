"""Schema for paper-level AI construct specification coding (Stage 2A.5).

Column names follow the project brief exactly so the coded dataset, the graph
builder, and the manuscript tables all speak the same language.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpecificationDimension:
    """One controlled construct-specification dimension."""

    id: str
    label: str
    column: str
    graph_node_label: str
    relationship_type: str
    question: str
    allowed_values: tuple[str, ...] = field(default=())


SPECIFICATION_DIMENSIONS: tuple[SpecificationDimension, ...] = (
    SpecificationDimension(
        id="ai_role",
        label="AI Role / Function",
        column="ai_role_function",
        graph_node_label="AIRole",
        relationship_type="SPECIFIES_ROLE",
        question="What role does AI play in the argument?",
        allowed_values=(
            "AI as tool",
            "AI as firm capability",
            "AI as actor/agent",
            "AI as context",
            "AI as research method",
            "AI as infrastructure",
            "AI as unspecified label",
        ),
    ),
    SpecificationDimension(
        id="ai_type",
        label="AI Type / Form",
        column="ai_type_form",
        graph_node_label="AIType",
        relationship_type="SPECIFIES_TYPE",
        question="What kind of AI is being discussed?",
        allowed_values=(
            "predictive AI",
            "generative AI",
            "machine learning",
            "deep learning",
            "natural language processing",
            "computer vision",
            "recommender systems",
            "large language models",
            "automation",
            "analytics",
            "general AI",
            "unspecified AI",
        ),
    ),
    SpecificationDimension(
        id="mechanism",
        label="Mechanism",
        column="ai_mechanism",
        graph_node_label="Mechanism",
        relationship_type="SPECIFIES_MECHANISM",
        question="What exactly is AI doing that changes entrepreneurial outcomes?",
        allowed_values=(
            "reduces uncertainty",
            "expands search",
            "alters judgment",
            "automates decisions",
            "reshapes experimentation",
            "supports learning",
            "changes access to resources",
            "transforms stakeholder interaction",
            "improves prediction",
            "enables personalisation",
            "mechanism missing",
        ),
    ),
    SpecificationDimension(
        id="level_of_analysis",
        label="Level of Analysis",
        column="level_of_analysis",
        graph_node_label="LevelOfAnalysis",
        relationship_type="SPECIFIES_LEVEL",
        question="Where does the AI-related claim operate?",
        allowed_values=(
            "individual entrepreneur",
            "founding team",
            "venture",
            "firm",
            "platform",
            "ecosystem",
            "industry",
            "national system",
            "institutional environment",
            "multi-level",
            "unspecified level",
        ),
    ),
    SpecificationDimension(
        id="process_stage",
        label="Process / Sequence",
        column="entrepreneurial_process_stage",
        graph_node_label="ProcessStage",
        relationship_type="SPECIFIES_PROCESS",
        question="When and how does AI matter in the entrepreneurial process?",
        allowed_values=(
            "opportunity recognition",
            "ideation",
            "opportunity evaluation",
            "venture creation",
            "resource acquisition",
            "experimentation",
            "innovation",
            "market entry",
            "scaling",
            "survival",
            "exit",
            "static input",
            "process unspecified",
        ),
    ),
    SpecificationDimension(
        id="scope_condition",
        label="Scope Conditions",
        column="scope_conditions",
        graph_node_label="ScopeCondition",
        relationship_type="SPECIFIES_SCOPE",
        question="Under what conditions does the AI-related claim hold?",
        allowed_values=(
            "early-stage ventures",
            "established firms",
            "SMEs",
            "high-tech startups",
            "digital platforms",
            "ecosystems",
            "sector-specific",
            "country-specific",
            "AI-form-specific",
            "generalised without scope",
            "scope missing",
        ),
    ),
    SpecificationDimension(
        id="definition_clarity",
        label="Definition / Construct Clarity",
        column="definition_construct_clarity",
        graph_node_label="DefinitionClarity",
        relationship_type="HAS_DEFINITION_CLARITY",
        question="Does the paper define AI, and does the definition fit the claim?",
        allowed_values=(
            "explicit definition, fits claim",
            "explicit definition, does not fit claim",
            "partial definition",
            "definition by example only",
            "no definition",
        ),
    ),
)


SPECIFICATION_PROBLEM_COLUMN = "specification_problem"

SPECIFICATION_PROBLEM_VALUES: tuple[str, ...] = (
    "role ambiguity",
    "AI type underspecified",
    "mechanism missing",
    "scope conditions missing",
    "level mismatch",
    "process not specified",
    "AI definition absent",
    "AI treated as loose label",
    "construct contrast",
    "construct fragmentation",
)

# Additional paper-level indicator columns from the brief that support the
# seven dimensions but are not graph dimension nodes themselves.
AUXILIARY_COLUMNS: tuple[str, ...] = (
    "ai_method_or_phenomenon",
    "process_sequence_specified",
    "ai_definition_present",
    "ai_distinction_present",
    "construct_clarity_score",
)

# Constructs the paper should distinguish AI from when checking definition
# clarity (brief section 6.7).
AI_DISTINCTION_TARGETS: tuple[str, ...] = (
    "algorithms",
    "analytics",
    "automation",
    "digitalisation",
    "information systems",
    "decision support",
    "data-driven tools",
    "general technology",
)

SPECIFICATION_COLUMNS: tuple[str, ...] = (
    tuple(dimension.column for dimension in SPECIFICATION_DIMENSIONS)
    + AUXILIARY_COLUMNS
    + (SPECIFICATION_PROBLEM_COLUMN,)
)
