"""Contract tests for Stage 2A.5 curation logic (no UI, no files)."""

from aecsp.specification.curation import (
    AUTO_ACCEPTED,
    DEFERRED,
    HUMAN_ACCEPTED,
    HUMAN_OVERRIDDEN,
    LLM_UNREVIEWED,
    build_review_queue,
    curated_frame,
    curation_status,
    empty_overrides,
    is_auto_accepted,
    record_decision,
)
from aecsp.specification.schema import SPECIFICATION_DIMENSIONS

N_DIMENSIONS = len(SPECIFICATION_DIMENSIONS)


def make_record(paper_id: str = "eid:1", **overrides) -> dict:
    """One coded paper: every dimension stated at 0.9 unless overridden."""

    record = {"paper_id": paper_id}
    for dimension in SPECIFICATION_DIMENSIONS:
        record[dimension.column] = dimension.allowed_values[0]
        record[f"{dimension.column}_evidence"] = "quoted evidence"
        record[f"{dimension.column}_evidence_type"] = "stated"
        record[f"{dimension.column}_confidence"] = 0.9
    record.update(overrides)
    return record


def test_auto_accept_requires_stated_and_threshold():
    record = make_record(
        ai_mechanism_evidence_type="inferred",   # high confidence but inferred
        ai_type_form_confidence=0.5,             # stated but low confidence
        scope_conditions_confidence=None,        # missing confidence
    )
    assert is_auto_accepted(record, "ai_role_function")
    assert not is_auto_accepted(record, "ai_mechanism")
    assert not is_auto_accepted(record, "ai_type_form")
    assert not is_auto_accepted(record, "scope_conditions")


def test_queue_holds_only_unaccepted_dimensions_least_confident_first():
    weak = make_record(
        "eid:weak",
        ai_mechanism_evidence_type="inferred",
        ai_mechanism_confidence=0.7,
        ai_type_form_evidence_type="inferred",
        ai_type_form_confidence=0.3,
    )
    strong = make_record("eid:strong")

    queue = build_review_queue([weak, strong])

    assert {(item["paper_id"], item["column"]) for item in queue} == {
        ("eid:weak", "ai_type_form"),
        ("eid:weak", "ai_mechanism"),
    }
    # Grouped by dimension in schema order; within a dimension, ascending confidence.
    assert [item["column"] for item in queue] == ["ai_type_form", "ai_mechanism"]


def test_queue_respects_decisions_and_deferrals():
    record = make_record(
        ai_mechanism_evidence_type="inferred",
        definition_construct_clarity_evidence_type="absent",
        definition_construct_clarity_confidence=0.0,
    )
    overrides = empty_overrides()
    overrides["dimension_deferrals"].append("definition_construct_clarity")
    record_decision(overrides, "eid:1", "ai_mechanism", "accept")

    assert build_review_queue([record], overrides=overrides) == []


def test_curation_status_precedence():
    record = make_record(
        ai_mechanism_evidence_type="inferred",
        definition_construct_clarity_evidence_type="absent",
        definition_construct_clarity_confidence=0.0,
    )
    overrides = empty_overrides()
    overrides["dimension_deferrals"].append("definition_construct_clarity")
    record_decision(overrides, "eid:1", "ai_mechanism", "override", code="alters judgment")
    record_decision(overrides, "eid:1", "ai_role_function", "accept")

    assert curation_status(record, "ai_mechanism", overrides) == (
        "alters judgment", HUMAN_OVERRIDDEN
    )
    assert curation_status(record, "ai_role_function", overrides)[1] == HUMAN_ACCEPTED
    assert curation_status(record, "definition_construct_clarity", overrides)[1] == DEFERRED
    assert curation_status(record, "ai_type_form", overrides)[1] == AUTO_ACCEPTED


def test_curated_frame_applies_overrides_and_reports_status():
    record = make_record(ai_mechanism_evidence_type="inferred")
    overrides = empty_overrides()
    record_decision(overrides, "eid:1", "ai_mechanism", "override", code="expands search")

    frame = curated_frame([record], overrides)

    row = frame.iloc[0]
    assert row["ai_mechanism"] == "expands search"
    assert row["ai_mechanism_curation"] == HUMAN_OVERRIDDEN
    assert row["ai_role_function_curation"] == AUTO_ACCEPTED
    # Original LLM evidence columns survive untouched for the audit trail.
    assert row["ai_mechanism_evidence"] == "quoted evidence"
    assert len([c for c in frame.columns if c.endswith("_curation")]) == N_DIMENSIONS


def test_unreviewed_below_threshold_is_marked():
    record = make_record(ai_type_form_confidence=0.4)
    frame = curated_frame([record], empty_overrides())
    assert frame.iloc[0]["ai_type_form_curation"] == LLM_UNREVIEWED


def test_study_status_is_preserved_and_human_curatable():
    record = make_record(
        ai_method_or_phenomenon="phenomenon",
        ai_role_function_evidence="AI adoption changes venture decisions.",
    )

    queue = build_review_queue([record])
    status_item = next(
        item for item in queue if item["column"] == "ai_method_or_phenomenon"
    )
    assert status_item["code"] == "phenomenon"
    assert status_item["allowed_values"] == ["method", "phenomenon", "both", "unclear"]

    overrides = empty_overrides()
    record_decision(
        overrides,
        "eid:1",
        "ai_method_or_phenomenon",
        "override",
        code="both",
    )
    row = curated_frame([record], overrides).iloc[0]
    assert row["ai_method_or_phenomenon"] == "both"
    assert row["ai_method_or_phenomenon_curation"] == HUMAN_OVERRIDDEN
