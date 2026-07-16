"""Tests for longitudinal, source-aware keyword prevalence."""

import pandas as pd

from aecsp.analytics.keyword_trends import (
    SEARCH_CUTOFF_DATE,
    analyze_keyword_evolution,
    keyword_evidence_mask,
    load_keyword_aliases,
    normalize_keyword,
    search_keyword_series,
)


def _papers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "Year": "2019",
                "Author Keywords": "AI; Machine-Learning",
                "Index Keywords": "Artificial Intelligence; Forecasting",
            },
            {
                "paper_id": "P2",
                "Year": "2021",
                "Author Keywords": "Machine Learning; Predictive Analytics",
                "Index Keywords": "Forecasting",
            },
            {
                "paper_id": "P3",
                "Year": "2022",
                "Author Keywords": "GenAI; LLMs",
                "Index Keywords": "Generative Artificial Intelligence",
            },
            {
                "paper_id": "P4",
                "Year": "2027",
                "Author Keywords": "AI",
                "Index Keywords": "",
            },
        ]
    )


def test_normalize_keyword_uses_conservative_aliases():
    aliases = {
        "ai": "artificial intelligence",
        "genai": "generative ai",
        "llms": "large language models",
    }
    assert normalize_keyword("AI", aliases) == "artificial intelligence"
    assert normalize_keyword("Machine-Learning", aliases) == "machine learning"
    assert normalize_keyword("LLMs", aliases) == "large language models"


def test_controlled_vocabulary_merges_safe_variants_without_stemming_ambiguities():
    aliases = load_keyword_aliases()
    assert normalize_keyword("AI", aliases) == "artificial intelligence"
    assert normalize_keyword("Decision Support System", aliases) == "decision support systems"
    assert normalize_keyword("Decision Support Systems", aliases) == "decision support systems"
    assert normalize_keyword("DSSs", aliases) == "decision support systems"
    assert normalize_keyword("Decision Support Systems (DSSs)", aliases) == "decision support systems"
    assert normalize_keyword("Neural Network", aliases) == "neural networks"
    assert normalize_keyword("Neural Networks", aliases) == "neural networks"
    assert normalize_keyword("machine learnings", aliases) == "machine learning"
    assert normalize_keyword("organisational behaviour", aliases) == "organizational behavior"
    assert normalize_keyword("optimisation models", aliases) == "optimization models"

    # These are related strings but not proven equivalents, so blind stemming
    # must not collapse them.
    assert normalize_keyword("economic", aliases) == "economic"
    assert normalize_keyword("economics", aliases) == "economics"
    assert normalize_keyword("analytic", aliases) == "analytic"
    assert normalize_keyword("analytics", aliases) == "analytics"


def test_keyword_evolution_deduplicates_aliases_within_each_paper():
    papers = pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "Year": "2024",
                "Author Keywords": "AI; Artificial Intelligence; Decision Support System; Decision Support Systems",
                "Index Keywords": "",
            },
            {
                "paper_id": "P2",
                "Year": "2024",
                "Author Keywords": "DSS",
                "Index Keywords": "",
            },
        ]
    )
    result = analyze_keyword_evolution(papers, "author", minimum_mover_papers=1)
    recent = next(period for period in result["periods"] if period["id"] == "2024_2026")
    by_keyword = {item["keyword"]: item for item in recent["top_keywords"]}
    assert by_keyword["artificial intelligence"]["papers"] == 1
    assert by_keyword["decision support systems"]["papers"] == 2
    assert "decision support system" not in by_keyword


def test_evolution_uses_keyword_bearing_denominator_and_excludes_forthcoming():
    result = analyze_keyword_evolution(_papers(), "author", minimum_mover_papers=1)
    periods = {period["id"]: period for period in result["periods"]}
    recent = periods["2021_2023"]
    assert recent["papers"] == 2
    assert recent["keyword_papers"] == 2
    assert result["excluded"]["after_2026"] == 1
    generative = next(item for item in recent["top_keywords"] if item["keyword"] == "generative ai")
    assert generative["papers"] == 1
    assert generative["prevalence"] == 0.5
    assert SEARCH_CUTOFF_DATE.isoformat() == "2026-07-08"
    assert result["search_cutoff"]["label"] == "8 July 2026"
    assert result["periods"][-1]["label"] == "2024–2026 (as at 8 July 2026)"
    assert result["all_time"]["label"] == "All time (to 8 July 2026)"
    assert result["annual_years"] == list(range(2019, 2027))


def test_combined_source_deduplicates_same_canonical_term_per_paper():
    result = analyze_keyword_evolution(_papers(), "combined", minimum_mover_papers=1)
    period = next(item for item in result["periods"] if item["id"] == "2016_2020")
    artificial_intelligence = next(
        item for item in period["top_keywords"] if item["keyword"] == "artificial intelligence"
    )
    assert artificial_intelligence["papers"] == 1


def test_keyword_evidence_mask_applies_period_and_alias():
    mask = keyword_evidence_mask(_papers(), "author", "generative ai", "2021_2023")
    assert _papers().loc[mask, "paper_id"].tolist() == ["P3"]

    annual_mask = keyword_evidence_mask(
        _papers(), "author", "generative ai", None, year=2022
    )
    assert _papers().loc[annual_mask, "paper_id"].tolist() == ["P3"]

    all_time_mask = keyword_evidence_mask(
        _papers(), "author", "artificial intelligence", "all_time"
    )
    assert _papers().loc[all_time_mask, "paper_id"].tolist() == ["P1"]


def test_dynamic_series_include_period_leaders_and_search_trajectories():
    result = analyze_keyword_evolution(
        _papers(), "author", series_top_n=1, minimum_mover_papers=1
    )
    series_names = {series["keyword"] for series in result["series"]}
    assert "generative ai" in series_names
    assert len(result["default_overall_keywords"]) == 1
    generative_series = next(
        series for series in result["series"] if series["keyword"] == "generative ai"
    )
    annual_2022 = next(
        value for value in generative_series["annual_values"] if value["year"] == 2022
    )
    assert annual_2022["prevalence"] == 1.0
    assert annual_2022["papers"] == 1
    assert annual_2022["denominator"] == 1

    matches = search_keyword_series(_papers(), "author", "generative", limit=5)
    assert [match["keyword"] for match in matches] == ["generative ai"]
    recent = next(
        value for value in matches[0]["values"] if value["period"] == "2021_2023"
    )
    assert recent["prevalence"] == 0.5
    searched_2022 = next(
        value for value in matches[0]["annual_values"] if value["year"] == 2022
    )
    assert searched_2022["prevalence"] == 1.0
    assert searched_2022["denominator"] == 1
