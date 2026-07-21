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
             "Abstract": "Machine learning improves prediction for new ventures.",
             "Author Keywords": "AI; Machine-Learning", "Index Keywords": "Artificial Intelligence; Forecasting",
             "ai_type_form": "machine learning", "ai_role_function": "AI as tool",
             "ai_type_form_evidence": "Machine learning",
             "ai_type_form_evidence_type": "stated", "ai_type_form_confidence": "0.98",
             "ai_role_function_evidence": "used by new ventures",
             "ai_role_function_evidence_type": "inferred", "ai_role_function_confidence": "0.75",
             "ai_method_or_phenomenon": "phenomenon", "ai_mechanism_analysis": "improves prediction",
             "level_of_analysis": "venture", "entrepreneurial_process_stage": "venture creation",
             "scope_conditions": "sector-specific", "definition_construct_clarity": "partial definition",
             "DOI": "10.1/a", "Link": "https://scopus.com/p1",
             "in_query_1": "1", "in_query_2": "0", "in_query_3": "1", "in_query_4": "0"},
            {"paper_id": "P2", "Title": "Predictive analytics for founders", "Authors": "Smith J.",
             "Source title": "Journal of Business Venturing", "Year": "2021", "Cited by": "5",
             "Abstract": "The authors use machine learning to analyse founders.",
             "Author Keywords": "Machine Learning; Predictive Analytics", "Index Keywords": "Forecasting",
             "ai_type_form": "machine learning", "ai_role_function": "AI as research method",
             "ai_method_or_phenomenon": "method", "ai_mechanism_analysis": "mechanism missing",
             "level_of_analysis": "individual entrepreneur", "entrepreneurial_process_stage": "process unspecified",
             "scope_conditions": "scope missing", "definition_construct_clarity": "no definition",
             "DOI": "10.1/b", "Link": "",
             "in_query_1": "1", "in_query_2": "0", "in_query_3": "0", "in_query_4": "0"},
            {"paper_id": "P3", "Title": "Generative AI and venture teams", "Authors": "Doe A.; Roe B.; Lee C.",
             "Source title": "Entrepreneurship Theory and Practice", "Year": "2021", "Cited by": "0",
             "Abstract": "Generative AI supports learning in venture teams.",
             "Author Keywords": "Generative AI; LLMs", "Index Keywords": "Generative Artificial Intelligence",
             "ai_type_form": "generative AI", "ai_role_function": "AI as actor/agent",
             "ai_method_or_phenomenon": "both", "ai_mechanism_analysis": "supports learning",
             "level_of_analysis": "founding team", "entrepreneurial_process_stage": "opportunity recognition",
             "scope_conditions": "high-tech startups", "definition_construct_clarity": "explicit definition, fits claim",
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
    assert perf["annual_production"][0]["cumulative_papers"] == 1
    assert perf["annual_production"][1]["papers"] == 2
    assert perf["annual_production"][1]["cumulative_papers"] == 3
    assert perf["trend_reconciliation"]["matches_scope_papers"] is True
    assert perf["trend_reconciliation"]["final_cumulative_papers"] == 3
    assert perf["search_cutoff"]["date"] == "2026-07-08"
    period = next(
        item
        for item in perf["publication_growth"]
        if item["start_year"] == 2020 and item["end_year"] == 2023
    )
    assert period["start_cumulative_papers"] == 1
    assert period["end_cumulative_papers"] == 3
    assert period["added_papers"] == 2
    assert period["percent_growth"] == 2.0
    assert perf["most_cited"][0]["paper_id"] == "P1"
    assert perf["most_cited"][0]["citations"] == 10
    top_journal = perf["top_journals"][0]
    assert top_journal["Source title"] == "Journal of Business Venturing"
    assert top_journal["papers"] == 2 and top_journal["citations"] == 15

    annual_papers = service.performance_papers(
        "full_corpus", 2021, mode="annual", limit=1
    )
    assert annual_papers["total_papers"] == 2
    assert annual_papers["returned_papers"] == 1
    assert annual_papers["papers"][0]["paper_id"] in {"P2", "P3"}
    cumulative_papers = service.performance_papers(
        "full_corpus", 2021, mode="cumulative", limit=100
    )
    assert cumulative_papers["total_papers"] == 3
    assert {item["paper_id"] for item in cumulative_papers["papers"]} == {
        "P1",
        "P2",
        "P3",
    }


def test_performance_is_scope_aware(service):
    q4 = service.performance("query_4")
    assert q4["summary"]["papers"] == 1
    assert q4["annual_production"][-1]["cumulative_papers"] == 1
    assert q4["most_cited"][0]["paper_id"] == "P3"

    comparison = service.publication_growth_comparison()
    labels = {item["label"] for item in comparison["views"]}
    assert "Full corpus (all papers)" in labels
    assert "Additional entrepreneurship journals" in labels
    assert comparison["scopes_overlap"] is True
    assert comparison["search_cutoff"]["label"] == "8 July 2026"


def test_performance_trend_preserves_later_dated_records_and_reconciles_total(
    tmp_path,
):
    pd.DataFrame(
        [
            {"paper_id": "P1", "Year": "2025", "Cited by": "0"},
            {"paper_id": "P2", "Year": "2026", "Cited by": "0"},
            {"paper_id": "P3", "Year": "2027", "Cited by": "0"},
        ]
    ).to_csv(tmp_path / "master_corpus.csv", index=False)
    result = GraphService(processed_dir=tmp_path).performance("full_corpus")
    assert result["summary"]["papers"] == 3
    assert result["summary"]["year_max"] == 2027
    assert [item["year"] for item in result["annual_production"]] == [
        2025,
        2026,
        2027,
    ]
    assert (
        result["annual_production"][-1][
            "publication_year_after_retrieval_year"
        ]
        is True
    )
    assert result["annual_production"][-1]["cumulative_papers"] == 3
    assert (
        result["trend_reconciliation"][
            "records_dated_after_retrieval_year"
        ]
        == 1
    )
    assert result["trend_reconciliation"]["matches_scope_papers"] is True


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
    assert papers[0]["Abstract"].startswith("The authors use")
    assert papers[0]["Author Keywords"] == "Machine Learning; Predictive Analytics"
    inspection = papers[0]["_inspection"]
    assert inspection["evidence_boundary"] == "Title, abstract, and author keywords"
    selected = [item for item in inspection["dimensions"] if item["selected"]]
    assert len(selected) == 1
    assert selected[0]["column"] == "ai_type_form"

    limited = service.observed_composition_evidence(
        "full_corpus",
        study_status="all",
        column="ai_type_form",
        value="machine learning",
        limit=1,
    )
    assert len(limited) == 1


def test_construct_contrasting_is_corpus_bounded_and_traceable(service):
    model = service.composition_models()[0]["id"]

    metadata = service.theory_contrasting_metadata(model)
    populations = {item["id"]: item["papers"] for item in metadata["populations"]}
    assert populations == {"core": 1, "other": 1, "combined": 2}
    controls = {item["id"]: item for item in metadata["horizontal_controls"]}
    assert set(controls) == {
        "study_status",
        "ai_role",
        "technical_type",
        "mechanism",
        "level",
        "process_stage",
        "scope",
        "definition",
    }
    assert controls["study_status"]["values"] == ["both", "method", "phenomenon"]
    assert controls["technical_type"]["values"] == [
        "generative AI",
        "machine learning",
    ]
    population_labels = {item["id"]: item["label"] for item in metadata["populations"]}
    assert population_labels["other"] == "Additional entrepreneurship"
    core_domain = next(
        item for item in metadata["domains"] if item["id"] == "core_entrepreneurship"
    )
    assert core_domain["assignment_type"] == "Registered journal population"
    assert core_domain["source_title_count"] == 1
    assert core_domain["source_titles"][0]["title"] == "Journal of Business Venturing"
    assert "do not retrieve or add papers" in metadata["domain_methodology"]["construction"]

    construct = service.theory_construct_specification(
        model, "combined", "all", "all"
    )
    assert construct["filtered_papers"] == 2

    horizontal = service.theory_horizontal_contrast(
        model, "ai_role", "observed", "all", "all"
    )
    group_ids = {item["id"] for item in horizontal["groups"]}
    assert {
        "core_entrepreneurship",
        "other_entrepreneurship",
        "combined_entrepreneurship",
    }.issubset(group_ids)
    assert sum(
        item["papers"]
        for item in next(
            group
            for group in horizontal["groups"]
            if group["id"] == "combined_entrepreneurship"
        )["categories"]
    ) == 2

    controlled_horizontal = service.theory_horizontal_contrast(
        model,
        "ai_role",
        "observed",
        "all",
        "all",
        "study_status",
        "phenomenon",
    )
    assert controlled_horizontal["control"] == {
        "dimension_id": "study_status",
        "dimension_label": "Study status",
        "column": "ai_method_or_phenomenon",
        "value": "phenomenon",
    }
    assert controlled_horizontal["baseline"]["full_n"] == 1

    controlled_type = service.theory_horizontal_contrast(
        model,
        "ai_role",
        "observed",
        "all",
        "all",
        "technical_type",
        "machine learning",
    )
    assert controlled_type["baseline"]["full_n"] == 2

    vertical = service.theory_vertical_contrast(
        model, "combined", "ai_role", "observed", "all", "all"
    )
    assert vertical["analyzed_n"] == 2
    assert {cell["papers"] for cell in vertical["cells"]} == {0, 1}

    reversed_vertical = service.theory_vertical_contrast(
        model,
        "combined",
        "level",
        "observed",
        "all",
        "all",
        "technical_type",
    )
    assert reversed_vertical["row_dimension"] == "level"
    assert reversed_vertical["column_dimension"] == "technical_type"

    with pytest.raises(
        ValueError, match="requires Level of analysis on one axis"
    ):
        service.theory_vertical_contrast(
            model,
            "combined",
            "ai_role",
            "observed",
            "all",
            "all",
            "mechanism",
        )

    structuring = service.theory_structuring(
        model,
        "combined",
        "ai_role__mechanism",
        "observed",
        "all",
        "all",
        1,
    )
    assert structuring["matrix"]["analyzed_n"] == 2
    assert structuring["configurations"]["analyzed_n"] == 2
    assert len(structuring["configurations"]["configurations"]) == 2

    evidence = service.theory_contrasting_evidence(
        model,
        domain_id="combined_entrepreneurship",
        filters={"ai_role_function": "AI as tool"},
        limit=100,
    )
    assert evidence["total_papers"] == 1
    assert evidence["papers"][0]["paper_id"] == "P1"
    paper = evidence["papers"][0]
    selected = [item for item in paper["_inspection"]["dimensions"] if item["selected"]]
    assert selected[0]["label"] == "AI Role / Function"
    assert selected[0]["evidence"] == "used by new ventures"


def test_ft50_horizontal_replication_uses_ft50_baseline_without_tautology(service):
    service.papers.loc[service.papers["paper_id"].eq("P1"), "in_query_2"] = "1"
    service._composition_frames.clear()
    model = service.composition_models()[0]["id"]

    result = service.theory_horizontal_contrast(
        model,
        "ai_role",
        distribution_view="observed",
        journal_scope="ft50",
        study_status="all",
    )

    assert result["baseline_label"] == "FT50 corpus"
    assert result["baseline"]["full_n"] == 1
    assert result["baseline"]["denominator"] == 1
    assert "also in the FT50 corpus" in result["comparison_definition"]
    assert "ft50" not in {group["id"] for group in result["groups"]}
    other = next(
        group
        for group in result["groups"]
        if group["id"] == "other_entrepreneurship"
    )
    assert other["label"] == "Additional entrepreneurship"
    assert other["eligible"] is False
    assert other["full_n"] == 0
    assert other["denominator"] == 0


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
    assert "Construct specification" in report
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
    papers = main.performance_papers(
        "full_corpus", year=2021, mode="annual", limit=100
    )
    assert papers["total_papers"] == 2
    growth = main.publication_growth_comparison()
    assert growth["search_cutoff"]["date"] == "2026-07-08"
    assert len(growth["views"]) == 5

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


def test_construct_contrasting_endpoints_and_release_are_reproducible(service):
    from aecsp.api import main

    main.state["service"] = service
    model = service.composition_models()[0]["id"]

    metadata = main.theory_contrasting_metadata(model)
    assert metadata["model"] == model

    horizontal = main.theory_horizontal_contrast(
        model=model,
        dimension="ai_role",
        distribution="observed",
        journal_scope="all",
        study_status="all",
        control_dimension=None,
        control_value=None,
    )
    assert horizontal["dimension_id"] == "ai_role"

    entrepreneurship = service.theory_entrepreneurship_comparison(
        model, "ai_role", "observed", "all", 1
    )
    assert [group["id"] for group in entrepreneurship["groups"]] == [
        "core",
        "other",
        "combined",
    ]
    assert entrepreneurship["groups"][2]["comparison_role"] == "Union benchmark"
    assert entrepreneurship["configurations"]
    assert len(entrepreneurship["configurations"][0]["population_values"]) == 3

    report = main.theory_contrasting_report(
        tactic="horizontal",
        model=model,
        population="combined",
        journal_scope="all",
        study_status="all",
        distribution="observed",
        dimension="ai_role",
        row_dimension="ai_role",
        column_dimension="level",
        pair="ai_role__mechanism",
        min_support=1,
        control_dimension=None,
        control_value=None,
    )
    assert "Construct contrasting: horizontal" in report
    assert "Evidence Boundary" in report

    current = main.theory_contrasting_download(
        bundle="current",
        tactic="horizontal",
        model=model,
        population="combined",
        journal_scope="all",
        study_status="all",
        distribution="observed",
        dimension="ai_role",
        row_dimension="ai_role",
        column_dimension="level",
        pair="ai_role__mechanism",
        min_support=1,
        control_dimension=None,
        control_value=None,
    )
    assert current.media_type.startswith("text/csv")
    current_rows = pd.read_csv(BytesIO(current.body))
    assert {"domain", "category", "papers", "share"}.issubset(
        current_rows.columns
    )

    entrepreneurship_current = main.theory_contrasting_download(
        bundle="current",
        tactic="entrepreneurship",
        model=model,
        population="combined",
        journal_scope="all",
        study_status="all",
        distribution="observed",
        dimension="ai_role",
        row_dimension="ai_role",
        column_dimension="level",
        pair="ai_role__mechanism",
        min_support=1,
        control_dimension=None,
        control_value=None,
    )
    entrepreneurship_rows = pd.read_csv(BytesIO(entrepreneurship_current.body))
    assert set(entrepreneurship_rows["record_type"]) == {
        "specification_distribution",
        "recurring_configuration",
    }

    release = main.theory_contrasting_download(
        bundle="release",
        tactic="construct",
        model=model,
        population="combined",
        journal_scope="all",
        study_status="all",
        distribution="observed",
        dimension="ai_role",
        row_dimension="ai_role",
        column_dimension="level",
        pair="ai_role__mechanism",
        min_support=1,
        control_dimension=None,
        control_value=None,
    )
    assert release.media_type == "application/zip"
    with ZipFile(BytesIO(release.body)) as archive:
        names = set(archive.namelist())
        assert "construct_specification/entrepreneurship_specification.csv" in names
        assert "horizontal/all/ai_role.csv" in names
        assert "within_entrepreneurship/specification/ai_role.csv" in names
        assert "within_entrepreneurship/recurring_configurations.csv" in names
        assert "vertical/ai_role_by_level.csv" in names
        assert "structuring/matrices/ai_role__mechanism.csv" in names
        assert "structuring/recurring_configurations.csv" in names
        assert "evidence/filtered_entrepreneurship_papers.csv" in names
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["corpus_papers"] == 3
        assert manifest["raw_model_records_changed"] is False
        assert manifest["journal_scope_by_tactic"] == {
            "construct_specification": "all",
            "horizontal_contrasting": "all",
            "vertical_contrasting": "all",
            "structuring": "all",
            "within_entrepreneurship": "core, additional, and combined journal sets",
        }
        assert all(item["sha256"] for item in manifest["files"])


def test_dashboard_entry_pages_are_current_and_not_cached():
    from aecsp.api import main

    for handler in (
        main.index,
        main.graph_page,
        main.assistant_page,
        main.composition_page,
        main.contrasting_page,
        main.topic_review_page,
        main.human_annotation_page,
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
        "construct_contrasting.html",
        "topic_review.html",
    ):
        html = (main.STATIC_DIR / filename).read_text(encoding="utf-8")
        assert 'href="/composition"' in html
        assert 'href="/topic-review"' in html
        assert 'href="/knowledge-graph"' in html
        assert 'href="/contrasting"' in html
        assert 'src="/static/citation.js?v=scopus-first-20260716"' in html
        assert "Construct Specification" in html

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

    composition_html = (
        main.STATIC_DIR / "observed_composition.html"
    ).read_text(encoding="utf-8")
    assert "<h1>Construct specification</h1>" in composition_html
    assert "Dataset scope" in composition_html
    assert "Compare full and observed" in composition_html
    assert "filtered papers · calculated live" in composition_html
    assert "Research artifacts" in composition_html
    assert 'href="/human-annotation"' in composition_html
    assert "Human-anchored model validation" in composition_html
    assert "/api/human-annotation/reliability" in composition_html
    assert 'id="humanRaterControls"' in composition_html
    assert "Balanced common papers" in composition_html
    assert "/static/paper_inspection.js" in composition_html
    assert "paperInspectionCard(paper, [column])" in composition_html

    contrasting_html = (
        main.STATIC_DIR / "construct_contrasting.html"
    ).read_text(encoding="utf-8")
    assert "Construct contrasting" in contrasting_html
    assert "Horizontal contrasting" in contrasting_html
    assert "Vertical contrasting" in contrasting_html
    assert 'data-tactic="entrepreneurship"' not in contrasting_html
    assert "entrepreneurship boundary comparison nested within Horizontal contrasting" in contrasting_html
    assert 'id="filterDimension"' in contrasting_html
    assert 'id="filterValue"' in contrasting_html
    assert 'id="verticalColumn"' in contrasting_html
    assert "vertical_column" in contrasting_html
    assert "Level of analysis remains on one matrix axis" in contrasting_html
    assert "Structuring" in contrasting_html
    assert 'id="distributionView"' in contrasting_html
    assert "/api/contrasting/metadata" in contrasting_html
    assert 'data-tactic="construct"' not in contrasting_html
    assert 'const VALID_TACTICS = ["horizontal", "vertical", "structuring"]' in contrasting_html
    assert "/api/contrasting/construct" not in contrasting_html
    assert "/api/contrasting/horizontal" in contrasting_html
    assert "/api/contrasting/vertical" in contrasting_html
    assert "/api/contrasting/structuring" in contrasting_html
    assert "/api/contrasting/evidence" in contrasting_html
    assert "/api/contrasting/report" in contrasting_html
    assert "/api/contrasting/download/" in contrasting_html
    assert 'id="artifactMenu"' in contrasting_html
    assert '<summary class="btn btn-outline">Research artifacts</summary>' in contrasting_html
    assert '<h2>Research artifacts' not in contrasting_html
    assert 'class="card contrasting-summary"' not in contrasting_html
    assert 'class="card contrasting-tabs"' not in contrasting_html
    assert 'class="contrasting-context-strip"' in contrasting_html
    assert 'class="card contrasting-workspace-controls"' in contrasting_html
    assert "/static/paper_inspection.js" in contrasting_html
    assert "paperInspectionCard(paper, Object.keys(evidenceFilters))" in contrasting_html
    assert "What is contrasted" in contrasting_html
    assert "percentage points" in contrasting_html
    assert "Supporting papers" in contrasting_html
    assert "descriptive compositional contrast" in contrasting_html
    assert "not a causal direction or temporal sequence" in contrasting_html
    assert 'id="populationControl"' not in contrasting_html
    assert 'id="tacticPopulation"' in contrasting_html
    assert "function populationOptions()" in contrasting_html
    assert "selected comparison corpus defines both the baseline" in contrasting_html
    assert 'id="journalScopeControl"' not in contrasting_html
    assert 'id="horizontalJournalScope"' in contrasting_html
    assert "All journals (full-corpus baseline)" in contrasting_html
    assert "FT50 papers only (FT50 baseline)" in contrasting_html
    assert "Percentage-point difference from selected baseline" in contrasting_html
    assert "FT50-only replication" in contrasting_html
    assert "Groups with no eligible papers remain visible" in contrasting_html
    assert "heatmap-unavailable" in contrasting_html
    assert 'id="analysisScopeInfoButton"' in contrasting_html
    assert 'id="domainInfoButton"' in contrasting_html
    assert 'id="selectedAnalysisScope"' in contrasting_html
    assert "function showAnalysisScopeInfo()" in contrasting_html
    assert 'const FISHER_AGUINIS_URL = "https://doi.org/10.1177/1094428116689707"' in contrasting_html
    assert 'fisherAguinisLink("Fisher and Aguinis, 2017, pp. 444-445")' in contrasting_html
    assert "Using Theory Elaboration to Make Theoretical Advancements" in contrasting_html
    assert "function showDomainInfo()" in contrasting_html
    assert "What the domains mean" in contrasting_html
    assert "Entrepreneurship analysis population" in contrasting_html
    assert 'function effectiveJournalScope()' in contrasting_html
    assert 'tactic === "horizontal" ? journalScope : "all"' in contrasting_html
    assert 'id="analysisNLabel"' in contrasting_html
    assert "Matrix-comparable papers" in contrasting_html
    assert "Five-field-complete papers" in contrasting_html
    assert contrasting_html.rstrip().endswith("</html>")
    assert contrasting_html.count(
        'el("tacticPopulation").onchange = event => {'
    ) == 2

    paper_inspection_js = (
        main.STATIC_DIR / "paper_inspection.js"
    ).read_text(encoding="utf-8")
    assert "Why this paper is included" in paper_inspection_js
    assert "Source evidence" in paper_inspection_js
    assert "Complete construct profile" in paper_inspection_js
    assert "Author keywords" in paper_inspection_js
    assert "Bibliographic and analytical metadata" in paper_inspection_js

    human_annotation_html = (
        main.STATIC_DIR / "human_annotation.html"
    ).read_text(encoding="utf-8")
    assert "Human annotation" in human_annotation_html
    assert 'id="annotatorId"' in human_annotation_html
    assert "/api/human-annotation/instrument" in human_annotation_html
    assert "/api/human-annotation/save" in human_annotation_html
    assert 'id="evidenceBoundary"' in human_annotation_html
    assert "fixed blinded paper order" in human_annotation_html
    assert 'id="instructions"' in human_annotation_html
    assert 'id="completionRequirements"' in human_annotation_html
    assert "View all eight dimensions and permitted codes" in human_annotation_html
    assert "View the full frozen system prompt" in human_annotation_html
    assert 'id="fullModelPrompt"' in human_annotation_html
    assert "instrument.full_model_prompt" in human_annotation_html
    assert "Fingerprint:" not in human_annotation_html
    assert 'id="protocolFingerprint"' not in human_annotation_html
    assert "Choose your own unique annotator ID" in human_annotation_html
    assert 'id="annotatorCollision"' in human_annotation_html
    assert 'id="confirmResume"' in human_annotation_html
    assert "Annotator ID already in use: choose another ID" in human_annotation_html

    index_html = (main.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "Annual and cumulative publication output" in index_html
    assert "Cumulative papers" in index_html
    assert "Citations accumulated by publication cohort" not in index_html
    assert "Cumulative publication growth across dataset views" not in index_html
    assert 'id="publicationGrowthTable"' not in index_html
    assert "/performance/papers" in index_html
    assert "showPublicationPapers(year)" in index_html
    assert 'id="publicationPaperLimit"' in index_html
    assert "Show all ${data.total_papers.toLocaleString()}" in index_html
    assert "The final cumulative point matches all" in index_html
    assert "publication year recorded in Scopus" in index_html
    assert "present in the Scopus export downloaded on" in index_html
    assert "`${a.year} (retrieved ${p.search_cutoff.label})`" in index_html
    assert "`${a.year} (Scopus year)`" not in index_html
    assert "minBarLength: 4" in index_html
    assert "order: 2, yAxisID: \"y\"" in index_html
    assert "order: 1, yAxisID: \"y1\"" in index_html
    assert "row.year >= p.search_cutoff.year ? 5 : 1.5" in index_html
    assert "autoSkip: false" in index_html
    assert 'class="chart-wrap publication-chart"' in index_html
    assert index_html.count('class="performance-table-scroll') == 3
    assert index_html.count("<thead></thead><tbody></tbody>") == 3
    assert "early access" not in index_html.lower()
    assert "post-cutoff" not in index_html.lower()
    assert "2026 is incomplete" not in index_html
    assert '<a href="${source}"' not in topic_review_html
    css = (main.STATIC_DIR / "esd.css").read_text(encoding="utf-8")
    assert "cursor: zoom-in" not in css
    assert ".chart-wrap.publication-chart { height: 520px; }" in css
    assert ".performance-table-scroll {" in css
    assert "position: sticky;" in css

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
    assert 'label: "Full"' in composition_html
    assert 'label: "Observed"' in composition_html
    assert "categoryValueLabels" not in composition_html
    assert "toLocaleString()} papers" in composition_html
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
