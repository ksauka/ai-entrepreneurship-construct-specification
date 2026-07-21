"""Tests for blind, resumable, multi-human annotation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from aecsp.human_annotation import HUMAN_DIMENSIONS, HumanAnnotationStore
from aecsp.specification.llm_coder import cache_key


def _project(tmp_path: Path) -> HumanAnnotationStore:
    sample_dir = tmp_path / "data/interim/theory_elaboration"
    sample_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "Title": "First paper",
                "Abstract": "AI supports learning in a venture.",
                "Author Keywords": "artificial intelligence; learning",
                "Source title": "Journal A",
                "Year": "2024",
                "workbook_code": "must remain blinded",
            },
            {
                "paper_id": "P2",
                "Title": "Second paper",
                "Abstract": "Machine learning predicts demand.",
                "Author Keywords": "machine learning; prediction",
                "Source title": "Journal B",
                "Year": "2025",
                "workbook_code": "must remain blinded",
            },
            {
                "paper_id": "P3",
                "Title": "Third paper",
                "Abstract": "AI is discussed without a mechanism.",
                "Author Keywords": "artificial intelligence",
                "Source title": "Journal C",
                "Year": "2026",
                "workbook_code": "must remain blinded",
            },
        ]
    ).to_csv(
        sample_dir / "theory_elaboration_probability_overlap_23.csv",
        index=False,
    )

    specification_dir = tmp_path / "data/processed/specification"
    specification_dir.mkdir(parents=True)
    for model, status in (
        ("model-one", "phenomenon"),
        ("model-two", "method"),
    ):
        rows = []
        for paper_id in ("P1", "P2", "P3"):
            rows.append(
                {
                    "paper_id": paper_id,
                    "coding_model": model,
                    "ai_method_or_phenomenon": status,
                    "ai_type_form": "machine learning",
                    "ai_role_function": "AI as tool",
                    "ai_mechanism": "supports learning",
                    "ai_mechanism_logic": "AI supports learning.",
                    "level_of_analysis": "venture",
                    "scope_conditions": "sector-specific",
                    "entrepreneurial_process_stage": "innovation",
                    "definition_construct_clarity": "partial definition",
                }
            )
        pd.DataFrame(rows).to_csv(
            specification_dir
            / f"paper_specifications_{model}_spec-v3.csv",
            index=False,
        )

    llama_cache = tmp_path / "data/interim/spec_cache/spec-v3/llama3.2"
    llama_cache.mkdir(parents=True)
    for paper_id in ("P1", "P2"):
        payload = {
            "paper_id": paper_id,
            "coding_model": "llama3.2",
            "coding_protocol": "spec-v3",
            "ai_method_or_phenomenon": "phenomenon",
            "ai_type_form": "machine learning",
            "ai_role_function": "AI as tool",
            "ai_mechanism": "supports learning",
            "ai_mechanism_logic": "AI supports learning.",
            "level_of_analysis": "venture",
            "scope_conditions": "sector-specific",
            "entrepreneurial_process_stage": "innovation",
            "definition_construct_clarity": "partial definition",
        }
        (llama_cache / cache_key(paper_id)).write_text(
            json.dumps(payload), encoding="utf-8"
        )
    return HumanAnnotationStore(tmp_path)


def _annotation(*, status: str = "phenomenon") -> dict:
    values = {
        "ai_method_or_phenomenon": status,
        "ai_type_form": "machine learning",
        "ai_role_function": "AI as tool",
        "ai_mechanism": "supports learning",
        "level_of_analysis": "venture",
        "scope_conditions": "sector-specific",
        "entrepreneurial_process_stage": "innovation",
        "definition_construct_clarity": "partial definition",
    }
    return {
        "dimensions": {
            contract["column"]: {
                "code": values[contract["column"]],
                "evidence": f"Evidence for {contract['label']}",
                "evidence_type": "stated",
                "confidence": 0.9,
            }
            for contract in HUMAN_DIMENSIONS
        },
        "ai_mechanism_logic": "AI supports learning.",
        "needs_full_text": [],
        "annotator_notes": "",
    }


def test_annotators_have_separate_records_and_one_common_blinded_order(
    tmp_path: Path,
):
    store = _project(tmp_path)
    instrument = store.instrument()
    assert instrument["model_protocol_id"] == "spec-v3"
    assert instrument["model_protocol_label"] == "System prompt V3"
    assert "CODING DISCIPLINE:" in instrument["full_model_prompt"]
    assert "Mechanism requires causal logic" in instrument["full_model_prompt"]
    assert len(instrument["model_protocol_fingerprint"]) == 64

    first = store.paper("human_a")
    second = store.paper("human_b")
    assert [item["paper_id"] for item in first["navigation"]] == [
        item["paper_id"] for item in second["navigation"]
    ]
    assert set(first["paper"]) == {
        "paper_id",
        "Title",
        "Abstract",
        "Author Keywords",
        "Source title",
        "Year",
    }
    assert "workbook_code" not in first["paper"]

    store.save("human_a", "P1", _annotation(), submit=True)
    store.save("human_b", "P1", _annotation(status="method"), submit=True)
    progress = {
        row["annotator_id"]: row
        for row in store.progress()["annotators"]
    }
    assert progress["human_a"]["completed_papers"] == 1
    assert progress["human_b"]["completed_papers"] == 1
    assert store.paper("human_a", "P1")["annotation"]["dimensions"][
        "ai_method_or_phenomenon"
    ]["code"] == "phenomenon"
    assert store.paper("human_b", "P1")["annotation"]["dimensions"][
        "ai_method_or_phenomenon"
    ]["code"] == "method"


def test_reliability_uses_exact_common_ids_not_the_smallest_record_count(
    tmp_path: Path,
):
    store = _project(tmp_path)
    for paper_id in ("P1", "P2"):
        store.save("human_a", paper_id, _annotation(), submit=True)
    for paper_id in ("P2", "P3"):
        store.save("human_b", paper_id, _annotation(), submit=True)

    result = store.reliability(
        annotator_ids={"human_a", "human_b"},
        model_ids={"model-one", "model-two"},
    )
    assert result["balanced_common_papers"] == 1
    assert len(result["raters"]) == 4
    assert len(result["pairs"]) == 6
    assert all(pair["intersection_papers"] == 1 for pair in result["pairs"])
    human_pair = next(
        pair
        for pair in result["pairs"]
        if pair["left_model"].startswith("human::")
        and pair["right_model"].startswith("human::")
    )
    assert len(human_pair["dimensions"]) == 8
    assert sum(
        item["classification"] == "Core"
        for item in human_pair["dimensions"]
    ) == 6
    assert all(
        item["comparable_papers"] == 1
        for item in human_pair["dimensions"]
    )


def test_partial_llama_cache_is_available_but_not_selected_by_default(
    tmp_path: Path,
):
    store = _project(tmp_path)
    llama = next(
        item
        for item in store.rater_catalog()
        if item["id"] == "model::llama3.2"
    )
    assert llama["label"] == "Llama 3.2"
    assert llama["available_papers"] == 2
    assert llama["target_papers"] == 3
    assert llama["default_selected"] is False

    store.save("human_a", "P1", _annotation(), submit=True)
    result = store.reliability(
        annotator_ids={"human_a"},
        model_ids={"llama3.2"},
    )
    assert result["balanced_common_papers"] == 1
    assert {item["id"] for item in result["raters"]} == {
        "human::human_a",
        "model::llama3.2",
    }


def test_every_save_is_audited_and_completion_requires_evidence(
    tmp_path: Path,
):
    store = _project(tmp_path)
    payload = _annotation()
    payload["dimensions"]["ai_type_form"]["evidence"] = ""
    with pytest.raises(ValueError, match="All eight dimensions"):
        store.save("human_a", "P1", payload, submit=True)

    store.save("human_a", "P1", payload, submit=False)
    payload["dimensions"]["ai_type_form"]["evidence"] = "Machine learning"
    store.save("human_a", "P1", payload, submit=True)
    with sqlite3.connect(store.database_path) as connection:
        revisions = connection.execute(
            """
            SELECT payload_json, is_complete
            FROM annotation_audit
            WHERE annotator_id = ? AND paper_id = ?
            ORDER BY revision_id
            """,
            ("human_a", "P1"),
        ).fetchall()
    assert len(revisions) == 2
    assert [row[1] for row in revisions] == [0, 1]
    assert json.loads(revisions[-1][0])["dimensions"]["ai_type_form"][
        "evidence"
    ] == "Machine learning"


def test_absent_evidence_type_allows_an_empty_evidence_field(tmp_path: Path):
    store = _project(tmp_path)
    payload = _annotation()
    entry = payload["dimensions"]["definition_construct_clarity"]
    entry["code"] = "no definition"
    entry["evidence"] = ""
    entry["evidence_type"] = "absent"

    saved = store.save("human_a", "P1", payload, submit=True)
    assert saved["is_complete"] is True


def test_export_preserves_annotator_identity_and_completion_state(
    tmp_path: Path,
):
    store = _project(tmp_path)
    store.save("human_a", "P1", _annotation(), submit=True)
    store.save("human_b", "P2", _annotation(status="method"), submit=True)

    exported = store.export()
    assert set(exported["annotator_id"]) == {"human_a", "human_b"}
    assert set(exported["paper_id"]) == {"P1", "P2"}
    assert len(exported) == 16
    assert exported["is_complete"].all()
