"""Contract tests for the pandas ingestion pipeline (Stages 0-1.6)."""

from pathlib import Path

import pandas as pd
import pytest

from aecsp.corpus.ingest import IngestError, merge_query_frames, query_view
from aecsp.corpus.relevance import load_relevance_config, score_relevance
from aecsp.corpus.source_titles import (
    load_source_title_universe,
    normalize_source_title,
    validate_source_titles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


@pytest.fixture
def query_frames() -> dict[str, pd.DataFrame]:
    q1 = pd.DataFrame(
        [
            {
                "EID": "2-s2.0-1",
                "DOI": "10.1/a",
                "Title": "AI and entrepreneurship",
                "Year": "2024",
                "Source title": "Journal of Business Venturing",
                "Abstract": "Machine learning helps startup founders.",
                "Author Keywords": "Artificial Intelligence; New Venture",
            },
            {
                "EID": "2-s2.0-2",
                "DOI": "",
                "Title": "Deep learning in finance",
                "Year": "2023",
                "Source title": "Journal of Finance",
                "Abstract": "Deep learning predicts returns.",
                "Author Keywords": "Deep Learning",
            },
        ]
    )
    q3 = pd.DataFrame(
        [
            {
                "EID": "2-s2.0-1",
                "DOI": "10.1/a",
                "Title": "AI and entrepreneurship",
                "Year": "2024",
                "Source title": "Journal of Business Venturing",
                "Abstract": "Machine learning helps startup founders.",
                "Author Keywords": "Artificial Intelligence; New Venture",
            },
            {
                "EID": "2-s2.0-3",
                "DOI": "10.1/c",
                "Title": "Generative AI and venture creation",
                "Year": "2025",
                "Source title": "Entrepreneurship Theory and Practice",
                "Abstract": "Generative AI reshapes venture creation by entrepreneurs.",
                "Author Keywords": "Generative AI; Entrepreneurship",
            },
        ]
    )
    return {"query_1": q1, "query_3": q3}


def test_merge_preserves_query_provenance(query_frames):
    master = merge_query_frames(query_frames)

    assert len(master) == 3
    shared = master[master["EID"] == "2-s2.0-1"].iloc[0]
    assert shared["in_query_1"] == 1
    assert shared["in_query_3"] == 1
    assert shared["query_count"] == 2
    assert shared["query_sources"] == "query_1;query_3"
    assert master["paper_id"].str.startswith("eid:").all()


def test_query_views_are_overlapping_not_exclusive(query_frames):
    master = merge_query_frames(query_frames)

    q1_view = query_view(master, "query_1")
    q3_view = query_view(master, "query_3")
    assert len(q1_view) == 2
    assert len(q3_view) == 2
    assert "2-s2.0-1" in set(q1_view["EID"]) & set(q3_view["EID"])


def test_merge_rejects_unknown_query(query_frames):
    with pytest.raises(IngestError):
        merge_query_frames({"query_9": query_frames["query_1"]})


def test_source_title_universe_matches_july_2026_queries():
    universe = load_source_title_universe(CONFIG_DIR / "search_queries_july2026_q1_q4.yaml")

    assert len(universe["Search Query 1"]) == 695
    assert len(universe["Search Query 2"]) == 50
    assert normalize_source_title("Journal of Business Venturing") in universe["combined"]


def test_validate_source_titles_flags_out_of_universe(query_frames):
    master = merge_query_frames(query_frames)
    universe = {normalize_source_title("Journal of Business Venturing")}

    validated = validate_source_titles(master, universe)

    by_eid = validated.set_index("EID")["source_title_valid"]
    assert by_eid["2-s2.0-1"] == 1
    assert by_eid["2-s2.0-2"] == 0


def test_relevance_requires_both_domains(query_frames):
    master = merge_query_frames(query_frames)
    config = load_relevance_config(CONFIG_DIR / "ai_keyword_config.yaml")

    scored = score_relevance(master, config)

    by_eid = scored.set_index("EID")["ai_ent_relevant"]
    assert by_eid["2-s2.0-1"] == 1  # AI + entrepreneurship terms
    assert by_eid["2-s2.0-2"] == 0  # AI terms only
    assert by_eid["2-s2.0-3"] == 1
