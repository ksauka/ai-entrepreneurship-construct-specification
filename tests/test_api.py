"""Tests for the performance analysis, report, and HTTP endpoints."""

from pathlib import Path

import pandas as pd
import pytest

from aecsp.api.graph_service import GraphService
from aecsp.api.report import build_scope_report


@pytest.fixture
def service(tmp_path: Path) -> GraphService:
    papers = pd.DataFrame(
        [
            {"paper_id": "P1", "Title": "AI and new ventures", "Authors": "Obschonka M.; Audretsch D.B.",
             "Source title": "Journal of Business Venturing", "Year": "2020", "Cited by": "10",
             "DOI": "10.1/a", "Link": "https://scopus.com/p1",
             "in_query_1": "1", "in_query_2": "0", "in_query_3": "1", "in_query_4": "0"},
            {"paper_id": "P2", "Title": "Predictive analytics for founders", "Authors": "Smith J.",
             "Source title": "Journal of Business Venturing", "Year": "2021", "Cited by": "5",
             "DOI": "10.1/b", "Link": "",
             "in_query_1": "1", "in_query_2": "0", "in_query_3": "0", "in_query_4": "0"},
            {"paper_id": "P3", "Title": "Generative AI and venture teams", "Authors": "Doe A.; Roe B.; Lee C.",
             "Source title": "Entrepreneurship Theory and Practice", "Year": "2021", "Cited by": "0",
             "DOI": "", "Link": "",
             "in_query_1": "0", "in_query_2": "0", "in_query_3": "0", "in_query_4": "1"},
        ]
    )
    papers.to_csv(tmp_path / "master_corpus.csv", index=False)
    return GraphService(processed_dir=tmp_path)


def test_performance_summary(service):
    perf = service.performance("full_corpus")
    s = perf["summary"]
    assert s["papers"] == 3
    assert s["total_citations"] == 15
    assert s["mean_citations"] == 5.0
    assert s["year_min"] == 2020 and s["year_max"] == 2021
    assert round(s["cited_share"], 2) == 0.67


def test_performance_rankings(service):
    perf = service.performance("full_corpus")
    assert perf["annual_production"][0]["year"] == 2020
    assert perf["most_cited"][0]["paper_id"] == "P1"
    assert perf["most_cited"][0]["citations"] == 10
    top_journal = perf["top_journals"][0]
    assert top_journal["Source title"] == "Journal of Business Venturing"
    assert top_journal["papers"] == 2 and top_journal["citations"] == 15


def test_performance_is_scope_aware(service):
    q4 = service.performance("query_4")
    assert q4["summary"]["papers"] == 1
    assert q4["most_cited"][0]["paper_id"] == "P3"


def test_report_includes_performance_with_citation_and_link(service):
    report = build_scope_report(service, "full_corpus")
    assert "Performance analysis" in report
    assert "Most cited papers" in report
    assert "https://doi.org/10.1/a" in report          # DOI link for P1
    assert "Obschonka and Audretsch (2020)" in report  # in-text citation for P1 (two authors)


def test_endpoint_handlers_serve_performance_and_report(service):
    from aecsp.api import main

    main.state["service"] = service
    health = main.health()
    assert health["papers_loaded"] == 3

    perf = main.performance("full_corpus")
    assert perf["summary"]["total_citations"] == 15

    report = main.scope_report("full_corpus")
    assert "Performance analysis" in report
