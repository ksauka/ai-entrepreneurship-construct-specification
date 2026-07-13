"""Define controlled dimensions for paper-level AI specification coding.

Inputs: none. Outputs: dimension metadata, allowed values, and column contracts.
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
    # What this dimension diagnoses in the AI-entrepreneurship literature
    # (the third column of the brief's dimension table); feeds the coder
    # prompt so the LLM codes with the diagnosis in mind.
    diagnosis: str = ""


SPECIFICATION_DIMENSIONS: tuple[SpecificationDimension, ...] = (
    SpecificationDimension(
        id="ai_role",
        label="AI Role / Function",
        column="ai_role_function",
        graph_node_label="AIRole",
        relationship_type="SPECIFIES_ROLE",
        question="What role does AI play in the argument?",
        diagnosis=(
            "ROLE AMBIGUITY: is AI a tool entrepreneurs use, a capability firms "
            "develop, an actor/agent that makes or shapes decisions, a context "
            "that changes entrepreneurial conditions, or a research method used "
            "by the scholars themselves?"
        ),
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
        diagnosis=(
            "AI IS NOT ALL THE SAME (Obschonka et al.): distinguish the form "
            "(predictive vs generative, ML, NLP, computer vision, LLMs) and "
            "whether AI is the PHENOMENON being studied or the METHOD the "
            "authors use."
        ),
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
        diagnosis=(
            "BLACK-BOX CLAIMS (Wiklund; Maula et al.): the task is not to list "
            "effects but to identify the causal logic. A paper may say 'AI "
            "improves entrepreneurship' without explaining the mechanism; that "
            "is mechanism missing."
        ),
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
        diagnosis=(
            "LEVEL CLARITY (Fisher & Aguinis treat cross-level comparison as a "
            "theory-elaboration tactic): individual entrepreneur, team, "
            "venture/firm, platform, ecosystem, industry, national or "
            "institutional level."
        ),
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
        diagnosis=(
            "STATIC-INPUT TREATMENT (Burnell et al.; Shepherd & Suddaby on "
            "sequence in theory building): is AI treated as a timeless input, "
            "or located in the entrepreneurial process?"
        ),
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
        diagnosis=(
            "MISSING SCOPE (Suddaby: constructs need contextual conditions "
            "covering space, time, and values): early-stage ventures, "
            "established firms, platforms, ecosystems, solo entrepreneurs, all "
            "entrepreneurship? Generative AI only, or AI generally?"
        ),
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
        diagnosis=(
            "LOOSE-LABEL USAGE (Suddaby): does the paper define AI, distinguish "
            "it from algorithms, analytics, automation, digitalisation, or "
            "decision support, and does the definition fit the theoretical "
            "argument? Judging definition FIT usually needs the full paper: "
            "flag it in needs_full_text."
        ),
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

# Whether AI is the object of inquiry, an analytical method used by the
# researchers, or both. This is distinct from the technical AI form recorded
# in ``ai_type_form``.
AI_STUDY_STATUS_COLUMN = "ai_method_or_phenomenon"
AI_STUDY_STATUS_VALUES: tuple[str, ...] = (
    "method",
    "phenomenon",
    "both",
    "unclear",
)

AI_STUDY_STATUS_FIELD = SpecificationDimension(
    id="ai_study_status",
    label="AI Study Status",
    column=AI_STUDY_STATUS_COLUMN,
    graph_node_label="AIStudyStatus",
    relationship_type="CLASSIFIES_AI_AS",
    question=(
        "Is AI the phenomenon being studied, a research method used by the "
        "authors, both, or unclear from the available evidence?"
    ),
    allowed_values=AI_STUDY_STATUS_VALUES,
)

# The model protocol remains the seven evidence-bearing dimensions above.
# Curation additionally reviews AI study status, which is a required analytic
# classification but does not have its own model-generated evidence fields.
CURATABLE_SPECIFICATION_FIELDS: tuple[SpecificationDimension, ...] = (
    *SPECIFICATION_DIMENSIONS,
    AI_STUDY_STATUS_FIELD,
)

# Additional paper-level indicator columns from the brief that support the
# seven dimensions but are not graph dimension nodes themselves.
AUXILIARY_COLUMNS: tuple[str, ...] = (
    AI_STUDY_STATUS_COLUMN,
    "process_sequence_specified",
    "ai_definition_present",
    "ai_distinction_present",
    "construct_clarity_score",
)

# How the coder must label the epistemic status of each dimension code.
EVIDENCE_TYPES: tuple[str, ...] = ("stated", "inferred", "absent")

# Per-dimension sub-columns emitted alongside each dimension code so the
# curation step can accept/queue dimensions independently.
DIMENSION_SUBFIELDS: tuple[str, ...] = ("evidence", "evidence_type", "confidence")

# Fields the coder may flag as requiring the full paper. Definition fit and
# theory use usually cannot be judged from an abstract; the curator can
# defer these corpus-wide (human batch review or a later full-text pass).
FULL_TEXT_SENSITIVE: tuple[str, ...] = (
    "definition_construct_clarity",
    "theories_mentioned",
)

# Additional fields that preserve epistemic and theoretical coding detail.
# theories_mentioned mirrors the workbook's "Ent Theories Discussed"
# annotation; ai_mechanism_logic captures the causal logic in the paper's
# own terms (the black-box diagnosis is auditable only with this).
ENRICHMENT_COLUMNS: tuple[str, ...] = (
    "ai_mechanism_logic",
    "theories_mentioned",
    "needs_full_text",
    "adversarial_review",
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
