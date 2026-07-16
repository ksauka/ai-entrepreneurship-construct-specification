"""Tests for the performance analysis, report, and HTTP endpoints."""

import asyncio
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from aecsp.api.graph_service import GraphService
from aecsp.api.report import build_composition_report, build_scope_report


@pytest.fixture
def service(tmp_path: Path) -> GraphService:
    papers = pd.DataFrame(
        [
            {"paper_id": "P1", "Title": "AI and new ventures", "Authors": "Obschonka M.; Audretsch D.B.",
             "Source title": "Journal of Business Venturing", "Year": "2020", "Cited by": "10",
             "Author Keywords": "AI; Machine-Learning", "Index Keywords": "Artificial Intelligence; Forecasting",
             "ai_type_form": "machine learning", "ai_role_function": "AI as tool",
             "ai_method_or_phenomenon": "phenomenon", "ai_mechanism_analysis": "improves prediction",
             "DOI": "10.1/a", "Link": "https://scopus.com/p1",
             "in_query_1": "1", "in_query_2": "0", "in_query_3": "1", "in_query_4": "0"},
            {"paper_id": "P2", "Title": "Predictive analytics for founders", "Authors": "Smith J.",
             "Source title": "Journal of Business Venturing", "Year": "2021", "Cited by": "5",
             "Author Keywords": "Machine Learning; Predictive Analytics", "Index Keywords": "Forecasting",
             "ai_type_form": "machine learning", "ai_role_function": "AI as research method",
             "ai_method_or_phenomenon": "method", "ai_mechanism_analysis": "mechanism missing",
             "DOI": "10.1/b", "Link": "",
             "in_query_1": "1", "in_query_2": "0", "in_query_3": "0", "in_query_4": "0"},
            {"paper_id": "P3", "Title": "Generative AI and venture teams", "Authors": "Doe A.; Roe B.; Lee C.",
             "Source title": "Entrepreneurship Theory and Practice", "Year": "2021", "Cited by": "0",
             "Author Keywords": "Generative AI; LLMs", "Index Keywords": "Generative Artificial Intelligence",
             "ai_type_form": "generative AI", "ai_role_function": "AI as actor/agent",
             "ai_method_or_phenomenon": "both", "ai_mechanism_analysis": "supports learning",
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


def test_service_prefers_frozen_primary_analysis_dataset(tmp_path):
    pd.DataFrame(
        [{"paper_id": "legacy", "Title": "Legacy corpus", "Year": "2020"}]
    ).to_csv(tmp_path / "master_corpus.csv", index=False)
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    pd.DataFrame(
        [{"paper_id": "primary", "Title": "Frozen primary", "Year": "2021"}]
    ).to_csv(analysis / "primary_analysis_dataset.csv", index=False)
    loaded = GraphService(processed_dir=tmp_path).papers
    assert loaded["paper_id"].tolist() == ["primary"]


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


def test_scope_export_is_scope_and_filter_aware(service):
    q4 = service.export_scope("query_4")
    assert q4["paper_id"].tolist() == ["P3"]

    filtered = service.export_scope(
        "full_corpus", {"ai_type_form": "machine learning"}
    )
    assert filtered["paper_id"].tolist() == ["P1", "P2"]

    with pytest.raises(ValueError, match="Unknown export filter column"):
        service.export_scope("full_corpus", {"not_a_column": "value"})


def test_keyword_evolution_and_evidence_are_scope_aware(service):
    result = service.keyword_evolution("full_corpus", source="author")
    assert result["source_label"] == "Author keywords"
    assert result["keyword_papers"] == 3
    period = next(item for item in result["periods"] if item["id"] == "2021_2023")
    assert period["papers"] == 2
    evidence = service.keyword_evidence(
        "query_4", "author", "generative ai", "2021_2023"
    )
    assert [item["paper_id"] for item in evidence] == ["P3"]

    annual_evidence = service.keyword_evidence(
        "query_4", "author", "generative ai", year=2021
    )
    assert [item["paper_id"] for item in annual_evidence] == ["P3"]


def test_contrast_evidence_returns_roles_and_supporting_papers(service):
    detail = service.contrast_evidence(
        "full_corpus",
        shared_column="ai_type_form",
        shared_value="machine learning",
        contrast_column="ai_role_function",
    )
    assert detail["total_papers"] == 2
    assert {item["value"] for item in detail["values"]} == {
        "AI as tool",
        "AI as research method",
    }
    assert {paper["paper_id"] for paper in detail["papers"]} == {"P1", "P2"}


def test_keyword_search_returns_period_series(service):
    matches = service.keyword_search(
        "full_corpus", source="author", query="generative", limit=5
    )
    assert matches[0]["keyword"] == "generative ai"
    assert len(matches[0]["values"]) == 5


def test_observed_composition_is_filtered_and_inspectable(service):
    composition = service.observed_composition("full_corpus", study_status="method")
    assert composition["filtered_papers"] == 1
    type_panel = next(
        panel for panel in composition["panels"] if panel["id"] == "technical_type"
    )
    assert type_panel["categories"][0]["value"] == "machine learning"

    papers = service.observed_composition_evidence(
        "full_corpus",
        study_status="method",
        column="ai_type_form",
        value="machine learning",
    )
    assert [paper["paper_id"] for paper in papers] == ["P2"]

    limited = service.observed_composition_evidence(
        "full_corpus",
        study_status="all",
        column="ai_type_form",
        value="machine learning",
        limit=1,
    )
    assert len(limited) == 1


def test_observed_composition_uses_selected_model_and_its_coverage(tmp_path):
    papers = pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "Title": "First paper",
                "ai_method_or_phenomenon": "phenomenon",
                "ai_type_form": "machine learning",
                "ai_mechanism": "improves prediction",
                "ai_mechanism_analysis": "improves prediction",
            },
            {
                "paper_id": "P2",
                "Title": "Second paper",
                "ai_method_or_phenomenon": "method",
                "ai_type_form": "generative AI",
                "ai_mechanism": "supports learning",
                "ai_mechanism_analysis": "supports learning",
            },
            {
                "paper_id": "P3",
                "Title": "Third paper",
                "ai_method_or_phenomenon": "both",
                "ai_type_form": "automation",
                "ai_mechanism": "automates decisions",
                "ai_mechanism_analysis": "automates decisions",
            },
        ]
    )
    papers.to_csv(tmp_path / "master_corpus.csv", index=False)
    specification_dir = tmp_path / "specification"
    specification_dir.mkdir()
    pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "coding_model": "gpt-4.1-nano-2025-04-14",
                "ai_method_or_phenomenon": "method",
                "ai_type_form": "analytics",
                "ai_mechanism": "supports learning",
                "ai_mechanism_logic": "",
            },
            {
                "paper_id": "P2",
                "coding_model": "gpt-4.1-nano-2025-04-14",
                "ai_method_or_phenomenon": "method",
                "ai_type_form": "analytics",
                "ai_mechanism": "improves prediction",
                "ai_mechanism_logic": "AI predicts the outcome.",
            },
        ]
    ).to_csv(
        specification_dir
        / "paper_specifications_gpt-4.1-nano-2025-04-14_spec-v3.csv",
        index=False,
    )

    model_service = GraphService(processed_dir=tmp_path)
    models = model_service.composition_models()
    nano = next(item for item in models if item["id"].startswith("gpt-4.1-nano"))
    assert nano["coded_papers"] == 2
    assert nano["missing_papers"] == 1
    assert nano["coverage_share"] == 0.666667

    composition = model_service.observed_composition(
        "full_corpus",
        model="gpt-4.1-nano-2025-04-14",
    )
    assert composition["model_label"] == "GPT-4.1 Nano"
    assert composition["scope_papers"] == 2
    assert composition["corpus_scope_papers"] == 3
    assert composition["model_missing_papers"] == 1
    assert composition["model_coverage_share"] == 0.666667
    type_panel = next(
        panel for panel in composition["panels"] if panel["id"] == "technical_type"
    )
    assert type_panel["categories"] == [
        {"value": "analytics", "count": 2, "share": 1.0}
    ]
    mechanism_panel = next(
        panel for panel in composition["panels"] if panel["id"] == "mechanism"
    )
    assert mechanism_panel["observed_n"] == 1
    assert mechanism_panel["categories"][0]["value"] == "improves prediction"

    evidence = model_service.observed_composition_evidence(
        "full_corpus",
        study_status="all",
        column="ai_type_form",
        value="analytics",
        model="gpt-4.1-nano-2025-04-14",
    )
    assert {paper["paper_id"] for paper in evidence} == {"P1", "P2"}

    irr = model_service.composition_irr(
        "full_corpus",
        "gpt-5.4-mini-2026-03-17",
        "gpt-4.1-nano-2025-04-14",
    )
    assert irr["intersection_papers"] == 2
    assert len(irr["dimensions"]) == 8
    status_irr = next(
        item for item in irr["dimensions"] if item["column"] == "ai_method_or_phenomenon"
    )
    assert status_irr["comparable_papers"] == 2
    assert status_irr["agreements"] == 1
    assert status_irr["percent_agreement"] == 0.5
    assert status_irr["classification"] == "Core"
    process_irr = next(
        item for item in irr["dimensions"] if item["column"] == "entrepreneurial_process_stage"
    )
    assert process_irr["classification"] == "Exploratory"

    irr_matrix = model_service.composition_irr_matrix("full_corpus")
    assert [item["label"] for item in irr_matrix["models"]] == [
        "GPT-5.4 Mini",
        "GPT-4.1 Nano",
    ]
    assert irr_matrix["dimension_count"] == 8
    assert irr_matrix["core_dimension_count"] == 6
    assert len(irr_matrix["pairs"]) == 1
    assert irr_matrix["pairs"][0]["intersection_papers"] == 2
    assert irr_matrix["pairs"][0]["mean_percent_agreement"] is not None

    exported = model_service.composition_export(
        "full_corpus",
        "gpt-4.1-nano-2025-04-14",
        "method",
    )
    assert exported["paper_id"].tolist() == ["P1", "P2"]
    assert exported["ai_type_form"].tolist() == ["analytics", "analytics"]

    report = build_composition_report(
        model_service,
        "full_corpus",
        "gpt-4.1-nano-2025-04-14",
        "method",
    )
    assert "Observed construct composition" in report
    assert "GPT-4.1 Nano" in report
    assert "Study-status filter:</strong> Method" in report
    assert "Distribution:</strong> Compare full and observed" in report
    assert "Model inter-rater reliability" in report
    assert "Krippendorff alpha" in report
    assert "Full papers" in report
    assert "Observed share" in report
    assert "Exploratory" in report

    observed_report = build_composition_report(
        model_service,
        "full_corpus",
        "gpt-4.1-nano-2025-04-14",
        "method",
        "observed",
    )
    assert "Distribution:</strong> Observed only" in observed_report
    assert "Share of observed" in observed_report
    assert "Share of full" not in observed_report

    from aecsp.api import main

    main.state["service"] = model_service
    response = main.observed_composition_download(
        "full_corpus",
        "release",
        model="gpt-4.1-nano-2025-04-14",
        study_status="method",
        distribution="observed",
    )
    assert response.media_type == "application/zip"
    with ZipFile(BytesIO(response.body)) as archive:
        names = set(archive.namelist())
        assert "composition/observed_composition_summary.csv" in names
        assert "composition/observed_composition_papers.csv" in names
        assert "irr/pairwise_irr_matrix.csv" in names
        assert "irr/pairwise_irr_by_dimension.csv" in names
        assert any(name.startswith("irr/common_paper_ratings/") for name in names)
        assert "report/observed_composition_report.html" in names
        composition_summary = pd.read_csv(
            archive.open("composition/observed_composition_summary.csv")
        )
        assert {"distribution", "denominator", "papers", "share"}.issubset(
            composition_summary.columns
        )
        assert composition_summary["distribution"].unique().tolist() == ["observed"]
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["model"] == "gpt-4.1-nano-2025-04-14"
        assert manifest["irr_pair_count"] == 1
        assert manifest["irr_dimension_count"] == 8
        assert manifest["irr_core_dimension_count"] == 6
        assert manifest["study_status_filter"] == "method"
        assert manifest["distribution_view"] == "observed"
        assert manifest["filtered_papers"] == 2
        assert manifest["irr_pairs"][0]["common_papers"] == 2
        assert manifest["raw_model_records_changed"] is False


def test_report_includes_performance_with_citation_and_link(service):
    report = build_scope_report(service, "full_corpus")
    assert "Performance analysis" in report
    assert "Most cited papers" in report
    assert "https://scopus.com/p1" in report  # Scopus is primary when available.
    assert "https://doi.org/10.1/b" in report  # DOI remains the fallback.
    assert "Obschonka and Audretsch (2020)" in report  # in-text citation for P1 (two authors)
    assert "Specification-code distributions" not in report
    assert "Journal-level specification-code diversity" not in report


def test_endpoint_handlers_serve_performance_and_report(service):
    from aecsp.api import main

    main.state["service"] = service
    health = main.health()
    assert health["papers_loaded"] == 3

    perf = main.performance("full_corpus")
    assert perf["summary"]["total_citations"] == 15

    keywords = main.keyword_evolution(
        "full_corpus", source="author", series_top_n=10, period_top_n=20
    )
    assert keywords["keyword_papers"] == 3

    contrast_detail = main.contrast_evidence(
        "full_corpus",
        shared="ai_type_form",
        value="machine learning",
        differ="ai_role_function",
        limit=100,
    )
    assert contrast_detail["total_papers"] == 2

    profile = main.dimension_profile(
        "full_corpus", column="ai_role_function"
    )
    assert profile["coded_papers"] == 3

    composition = main.observed_composition(
        "full_corpus", study_status="phenomenon", model=None
    )
    assert composition["filtered_papers"] == 1

    composition_evidence = main.observed_composition_evidence(
        "full_corpus",
        study_status="all",
        column="ai_type_form",
        value="machine learning",
        limit=50000,
        model=None,
    )
    assert {paper["paper_id"] for paper in composition_evidence} == {"P1", "P2"}

    report = main.scope_report("full_corpus")
    assert "Performance analysis" in report

    download = main.scope_download("query_4", filters="")
    assert download.media_type == "application/zip"
    assert download.headers["x-etv-scope"] == "query_4"
    assert download.headers["x-etv-paper-count"] == "1"
    with ZipFile(BytesIO(download.body)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["scope_id"] == "query_4"
        assert manifest["scope_label"] == "Additional entrepreneurship journals"
        assert manifest["paper_count"] == 1
        csv_name = manifest["data_file"]
        exported = pd.read_csv(archive.open(csv_name), dtype=str)
        assert exported["paper_id"].tolist() == ["P3"]


def test_dashboard_entry_pages_are_current_and_not_cached():
    from aecsp.api import main

    for handler in (
        main.index,
        main.graph_page,
        main.assistant_page,
        main.composition_page,
        main.topic_review_page,
    ):
        response = handler()
        assert response.headers["cache-control"] == "no-store, max-age=0"

    redirect = main.legacy_graph_page()
    assert redirect.status_code == 307
    assert redirect.headers["location"] == "/knowledge-graph"
    assert redirect.headers["cache-control"] == "no-store, max-age=0"

    for filename in (
        "index.html",
        "knowledge_graph.html",
        "assistant.html",
        "observed_composition.html",
        "topic_review.html",
    ):
        html = (main.STATIC_DIR / filename).read_text(encoding="utf-8")
        assert 'href="/composition"' in html
        assert 'href="/topic-review"' in html
        assert 'href="/knowledge-graph"' in html
        assert 'src="/static/citation.js?v=scopus-first-20260716"' in html
        assert "Observed Composition" in html

    topic_review_html = (main.STATIC_DIR / "topic_review.html").read_text(
        encoding="utf-8"
    )
    assert 'id="reviewerName"' in topic_review_html
    assert 'id="topicPrevalenceChart"' in topic_review_html
    assert "function renderTopicPrevalence" in topic_review_html
    assert "papers assigned across" in topic_review_html
    assert "unassignedPapers.toLocaleString()" in topic_review_html
    assert "Topic label humanization" in topic_review_html
    assert "Review each topic's keywords and supporting papers" in topic_review_html
    assert "showEvidence(ranked[elements[0].index].topic_id)" in topic_review_html
    assert "prevalenceTopics.find(item => item.topic_id === topicId)" in topic_review_html
    assert "ranked.map(topic => `T${topic.topic_id}: ${topic.display_label}`)" in topic_review_html
    assert "topic_prevalence.png" not in topic_review_html
    assert "Topics by publication era" not in topic_review_html
    assert "Topics by AI study status" not in topic_review_html
    assert "Construct observability by topic" not in topic_review_html
    assert "Download topic prevalence" in topic_review_html
    assert "/api/topic-review/fitted-papers" in topic_review_html
    assert 'id="fittedPaperLimit"' in topic_review_html
    assert "Centroid-nearest representative papers" in topic_review_html
    assert "Fitted papers" in topic_review_html
    assert "function paperEvidenceLink" in topic_review_html
    assert 'src="/static/citation.js?v=scopus-first-20260716"' in topic_review_html
    assert "const url = paperHref(paper)" in topic_review_html
    assert "paperEvidenceLink(item.paper, item.title" in topic_review_html
    assert "paperEvidenceLink(paper, paper.Title" in topic_review_html
    assert '<a href="/api/paper/' not in topic_review_html
    assert 'api("/api/scopes")' in topic_review_html
    assert 'id="reviewScope"></select>' in topic_review_html
    assert "Dataset review progress" not in topic_review_html
    assert 'id="approvedCount"' not in topic_review_html
    assert 'id="pendingCount"' not in topic_review_html
    assert 'id="reviseCount"' not in topic_review_html
    assert 'id="outputState"' not in topic_review_html
    assert 'id="scopeProgress"' in topic_review_html
    assert "Search topics" not in topic_review_html
    assert 'id="topicSearch"' not in topic_review_html
    assert "searchTimer" not in topic_review_html
    assert 'id="finalizeTopics"' in topic_review_html
    assert 'id="finalizeTopics" class="btn" type="button" hidden' in topic_review_html
    assert 'id="finalizeHelp"' not in topic_review_html
    assert "/api/topic-review/finalize" in topic_review_html
    assert 'id="topicGraphPreview"' not in topic_review_html
    assert 'id="topicGraphFullscreen"' not in topic_review_html
    assert "Topic graph label preview" not in topic_review_html
    assert "Downloads are marked as drafts" not in topic_review_html
    assert "provisional automatic labels" not in topic_review_html
    assert "vis-network.min.js" not in topic_review_html
    assert 'data-topic-download="topics"' in topic_review_html
    assert 'data-topic-download="figures"' in topic_review_html
    assert 'data-topic-download="graph"' in topic_review_html
    assert 'data-topic-download="release"' in topic_review_html
    assert "/api/topic-review/download/" in topic_review_html
    assert "topic-figure-card-wide" not in topic_review_html
    assert '<a href="${source}"' not in topic_review_html
    assert "cursor: zoom-in" not in (main.STATIC_DIR / "esd.css").read_text(
        encoding="utf-8"
    )

    graph_html = (main.STATIC_DIR / "knowledge_graph.html").read_text(
        encoding="utf-8"
    )
    graph_js = (main.STATIC_DIR / "knowledge_graph.js").read_text(encoding="utf-8")
    assert 'src="/static/knowledge_graph.js"' in graph_html
    assert 'id="nodeTypeFilters"' in graph_html
    assert 'id="relationshipTypeFilters"' in graph_html
    assert 'id="cypherQuery"' in graph_html
    assert 'id="graphFullscreen"' in graph_html
    assert 'id="togglePhysics"' in graph_html
    assert 'id="exitGraphFullscreen"' in graph_html
    assert "const LABEL_COLOURS" in graph_js
    assert 'network.on("click"' in graph_js
    assert 'network.on("doubleClick"' in graph_js
    assert "focusNode" in graph_js
    assert "focusVisibleNode" in graph_js
    assert "expandNode" in graph_js
    assert "uniqueById" in graph_js
    assert "freezeLayout" in graph_js
    assert "requestFullscreen" in graph_js
    assert "loadConnectedPreview" in graph_js
    assert "Nearest convergent papers" in graph_js
    assert "group:" not in graph_js

    composition_html = (main.STATIC_DIR / "observed_composition.html").read_text(
        encoding="utf-8"
    )
    assert 'id="codingModel"' in composition_html
    assert 'id="distributionView"' in composition_html
    assert "Compare full and observed" in composition_html
    assert "Full only" in composition_html
    assert "Observed only" in composition_html
    assert 'api("/api/specification/models")' in composition_html
    assert "model: codingModel" in composition_html
    assert "distribution: distributionView" in composition_html
    assert "data.model_coverage_share" in composition_html
    assert "data.model_label" in composition_html
    assert "Research artifacts" in composition_html
    assert "Generate filtered report" in composition_html
    assert 'data-composition-download="composition"' in composition_html
    assert 'data-composition-download="irr"' in composition_html
    assert 'data-composition-download="release"' in composition_html
    assert "Model inter-rater reliability" in composition_html
    assert composition_html.index('id="compositionGrid"') < composition_html.index(
        "Model inter-rater reliability"
    )
    assert 'id="irrAgreementMatrix"' in composition_html
    assert 'id="irrAlphaMatrix"' in composition_html
    assert 'id="irrLeftModel"' not in composition_html
    assert 'id="irrRightModel"' not in composition_html
    assert "/composition/irr/matrix" in composition_html
    assert "Full = all successfully coded papers for the selected model" not in composition_html
    assert "Full distribution (n =" in composition_html
    assert "Observed distribution (n =" in composition_html
    assert 'distributionView === "full"' in composition_html
    assert 'distributionView === "observed"' in composition_html
    assert "eight dimensions; six core" in composition_html
    assert "/composition/report?" in composition_html
    assert "/composition/download/" in composition_html
    assert 'id="compositionEvidenceLimit"' in composition_html
    assert "50000 ? \"All\"" in composition_html

    dashboard_html = (main.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert 'text: "Publication year"' in dashboard_html
    assert "keywordData.all_time" in dashboard_html
    assert 'id="keywordFormula"' in dashboard_html
    assert "The denominator is not every paper published that year" not in dashboard_html
    assert 'data-action="roles"' in dashboard_html
    assert 'data-action="papers"' in dashboard_html
    assert "View roles" in dashboard_html
    assert "View papers" in dashboard_html
    assert "Distinct roles: inspect" in dashboard_html
    for pattern in ("*.html", "*.js"):
        for static_page in main.STATIC_DIR.glob(pattern):
            platform_copy = static_page.read_text(encoding="utf-8")
            assert "—" not in platform_copy
            assert "&mdash;" not in platform_copy.lower()
    assert "showContrastRoles" in dashboard_html
    assert "showContrastPapers" in dashboard_html
    assert "showRolePapers" in dashboard_html
    assert "Click any chart point to inspect its supporting papers" in dashboard_html
    assert "period incomplete" not in dashboard_html
    assert "Generate scope report" in dashboard_html
    assert "Download scope data" in dashboard_html
    assert "downloadScopeData" in dashboard_html
    assert "papers selected" in dashboard_html
    assert 'id="cards"' not in dashboard_html
    assert "Clear study status" not in dashboard_html
    assert "Named technical type" not in dashboard_html
    assert "Observable mechanism" not in dashboard_html
    assert "${s.papers.toLocaleString()}" in dashboard_html
    assert "Total citations" in dashboard_html
    assert "Mean citations per paper" in dashboard_html
    assert "Papers cited at least once" in dashboard_html
    assert "Publication years" in dashboard_html

    citation_js = (main.STATIC_DIR / "citation.js").read_text(encoding="utf-8")
    assert "Best available link for a paper: Scopus first, then DOI" in citation_js
    assert citation_js.index('const link = (p.Link || "").trim()') < citation_js.index(
        'const doi = (p.DOI || "").trim()'
    )

    assistant_html = (main.STATIC_DIR / "assistant.html").read_text(encoding="utf-8")
    assert "assistant-contrast" in assistant_html
    assert "Documents per view" in assistant_html
    assert "View roles" in assistant_html
    assert "View papers" in assistant_html
    assert "showContrastRoles" in assistant_html
    assert "showRolePapers" in assistant_html


def test_static_assets_are_never_cached(monkeypatch):
    from aecsp.api import main

    async def base_response(_self, _path, _scope):
        return main.Response("asset")

    monkeypatch.setattr(main.StaticFiles, "get_response", base_response)
    static_files = main.NoCacheStaticFiles(directory=main.STATIC_DIR)
    response = asyncio.run(
        static_files.get_response("citation.js", {"method": "GET"})
    )
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert response.headers["pragma"] == "no-cache"
