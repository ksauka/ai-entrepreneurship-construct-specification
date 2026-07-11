"""Methodological contract tests for registered paths, corrections, and QC."""

import json

import pandas as pd

from aecsp.analytics.reliability import pairwise_dimension_agreement
from aecsp.corpus.scope_overlap import pairwise_overlap
from aecsp.specification.analysis_columns import enrich_for_analysis
from aecsp.specification.paths import specification_csv_path
from aecsp.specification.validators import validate_profile


def test_registered_specification_path(tmp_path):
    register = tmp_path / "register.json"
    register.write_text(json.dumps({"primary_model": "gpt:x", "protocol_id": "spec-v3"}))
    assert specification_csv_path(tmp_path, register_path=register).name == "paper_specifications_gpt_x_spec-v3.csv"


def test_mechanism_correction_preserves_raw_value():
    result = enrich_for_analysis(pd.DataFrame([{"ai_mechanism": "supports learning", "ai_mechanism_logic": "", "specification_problem": ""}])).iloc[0]
    assert result.ai_mechanism_raw == "supports learning"
    assert result.ai_mechanism_analysis == "mechanism missing"
    assert result.mechanism_corrected == 1


def test_validator_flags_rule7_without_mutation():
    record = {"ai_mechanism": "supports learning", "ai_mechanism_logic": ""}
    result = validate_profile(record, {"title": "AI learning", "abstract": "", "keywords": ""})
    assert result["validation_passed"] is False
    assert record == {"ai_mechanism": "supports learning", "ai_mechanism_logic": ""}


def test_pairwise_reliability_uses_exact_intersection():
    left = pd.DataFrame({"paper_id": ["1", "2"], "code": ["a", "b"]})
    right = pd.DataFrame({"paper_id": ["2", "3"], "code": ["b", "a"]})
    result = pairwise_dimension_agreement(left, right, "code")
    assert result["comparable"] == 1
    assert result["percent_agreement"] == 1.0
    assert result["paper_ids"] == ["2"]


def test_scope_overlap_reports_repeated_papers():
    master = pd.DataFrame({"paper_id": ["1", "2", "3"], "in_query_1": [1, 1, 0], "in_query_2": [0, 1, 1]})
    result = pairwise_overlap(master, "query_1", "query_2")
    assert result["n_intersection"] == 1
    assert result["jaccard"] == 1 / 3
