"""Generate printable specification reports for dataset scopes.

Inputs: scope analytics and their supporting paper records.
Outputs: self-contained HTML reports with traceable evidence.
"""

from __future__ import annotations

import html
import re
from datetime import date

from aecsp.corpus.scopes import SCOPE_BY_ID
PLATFORM_NAME = "AI-Entrepreneurship Construct Specification Platform"


def _citation(row: dict) -> str:
    """In-text citation such as 'Obschonka et al. (2019)'."""

    authors = [a.strip() for a in str(row.get("Authors", "")).split(";") if a.strip()]
    year = str(row.get("Year", "")).strip()

    def surname(name: str) -> str:
        return (name.split(",")[0] or name).strip().split(" ")[0]

    if not authors:
        who = "Anon."
    elif len(authors) == 1:
        who = surname(authors[0])
    elif len(authors) == 2:
        who = f"{surname(authors[0])} and {surname(authors[1])}"
    else:
        who = f"{surname(authors[0])} et al."
    return f"{who} ({year})" if year else who


def _paper_link(row: dict) -> str | None:
    link = str(row.get("Link", "")).strip()
    if link:
        return link
    doi = str(row.get("DOI", "")).strip()
    if doi:
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
        return "https://doi.org/" + doi
    return None


def build_scope_report(service, scope_id: str) -> str:
    """Return a full HTML document reporting on one dataset scope."""

    scope_label = service.scope_label(scope_id)
    overview = service.scope_overview(scope_id)
    n = overview["paper_count"]

    sections = [_intro(scope_label, n)]
    sections.append(_performance_section(service, scope_id))

    if overview["has_specifications"]:
        sections.append(_contrast_section(service, scope_id))
    else:
        sections.append(
            "<h2>Specification coding</h2><p>Specification codes have not been "
            "generated for this corpus yet. Once Stage 2A.5 has run, this report "
            "will describe the dominant AI role, type, mechanism, level, process "
            "stage, scope conditions, and definition clarity for the dataset, "
            "together with the construct contrasts found within it.</p>"
        )

    sections.append(_methods_note())
    body = "\n".join(sections)
    return _document(scope_label, body)


def build_composition_report(
    service,
    scope_id: str,
    model: str,
    study_status: str,
    distribution: str = "compare",
    filter_dimension: str | None = None,
    filter_value: str | None = None,
) -> str:
    """Return a filter-aware construct-specification and IRR report."""

    distribution_labels = {
        "compare": "Compare full and observed",
        "full": "Full only",
        "observed": "Observed only",
    }
    if distribution not in distribution_labels:
        raise ValueError(f"Unknown composition distribution: {distribution}")

    scope = SCOPE_BY_ID.get(scope_id)
    scope_label = scope.label if scope else scope_id
    composition = service.observed_composition(
        scope_id,
        study_status=study_status,
        model=model,
        filter_dimension=filter_dimension,
        filter_value=filter_value,
    )
    irr = service.composition_irr_matrix(scope_id)
    status_label = "All papers" if study_status == "all" else study_status.capitalize()
    control = composition.get("control")
    control_label = (
        f"{control['dimension_label']} = {control['value']}"
        if control
        else "No dimension filter"
    )
    coverage = composition["model_coverage_share"] * 100
    body = (
        "<h2>Construct specification</h2>"
        f"<p><strong>Dataset:</strong> {html.escape(scope_label)}<br />"
        f"<strong>Coding model:</strong> {html.escape(composition['model_label'])}<br />"
        f"<strong>Study-status filter:</strong> {html.escape(status_label)}<br />"
        f"<strong>Dimension filter:</strong> {html.escape(control_label)}<br />"
        f"<strong>Distribution:</strong> {html.escape(distribution_labels[distribution])}<br />"
        f"<strong>Model coverage:</strong> {composition['model_scope_papers']:,} of "
        f"{composition['corpus_scope_papers']:,} papers ({coverage:.2f}%)<br />"
        f"<strong>Papers after filter:</strong> {composition['filtered_papers']:,}</p>"
    )
    if distribution == "compare":
        body += (
            "<p>Each dimension reports two distributions. Full shares use every "
            "successfully coded paper for the selected model after the dataset-scope "
            "and study-status filters. Observed shares use the subset remaining after "
            "that dimension's declared missing or unspecified categories are excluded.</p>"
        )
    for panel in composition["panels"]:
        observed_percent = panel["observed_share"] * 100
        if distribution == "full":
            rows = "".join(
                f"<tr><td>{html.escape(str(category['value']))}</td>"
                f"<td>{category['full_count']:,}</td>"
                f"<td>{category['full_share'] * 100:.2f}%</td></tr>"
                for category in panel["comparison_categories"]
            )
            body += (
                f"<h3>{html.escape(panel['label'])}</h3>"
                f"<p>Full n = {panel['full_n']:,}.</p>"
                "<table><tr><th>Category</th><th>Papers</th><th>Share of full</th></tr>"
                f"{rows}</table>"
            )
        elif distribution == "observed":
            rows = "".join(
                f"<tr><td>{html.escape(str(category['value']))}</td>"
                f"<td>{category['observed_count']:,}</td>"
                f"<td>{category['observed_share'] * 100:.2f}%</td></tr>"
                for category in panel["comparison_categories"]
                if category["is_observed"]
            )
            body += (
                f"<h3>{html.escape(panel['label'])}</h3>"
                f"<p>Observed n = {panel['observed_n']:,} "
                f"({observed_percent:.2f}% of full).</p>"
                "<table><tr><th>Observed category</th><th>Papers</th>"
                "<th>Share of observed</th></tr>"
                f"{rows}</table>"
            )
        else:
            rows = "".join(
                f"<tr><td>{html.escape(str(category['value']))}</td>"
                f"<td>{category['full_count']:,}</td>"
                f"<td>{category['full_share'] * 100:.2f}%</td>"
                f"<td>{category['observed_count']:,}</td>"
                f"<td>{_format_metric(category['observed_share'], percent=True) if category['is_observed'] else 'Excluded'}</td></tr>"
                for category in panel["comparison_categories"]
            )
            body += (
                f"<h3>{html.escape(panel['label'])}</h3>"
                f"<p>Full n = {panel['full_n']:,}; observed n = {panel['observed_n']:,} "
                f"({observed_percent:.2f}% observable).</p>"
                "<table><tr><th>Category</th><th>Full papers</th><th>Full share</th>"
                "<th>Observed papers</th><th>Observed share</th></tr>"
                f"{rows}</table>"
            )

    matrix_head = "".join(
        f"<th>{html.escape(item['label'])}</th>" for item in irr["models"]
    )

    def matrix_value(left: str, right: str, metric: str, percent: bool) -> str:
        if left == right:
            return "100.00%" if percent else "1.00"
        pair = next(
            (
                item
                for item in irr["pairs"]
                if {item["left_model"], item["right_model"]} == {left, right}
            ),
            None,
        )
        value = pair[metric] if pair else None
        if not percent and "alpha" in metric:
            return _format_alpha(value)
        return _format_metric(value, percent=percent)

    def matrix_rows(metric: str, percent: bool) -> str:
        return "".join(
            f"<tr><th>{html.escape(left['label'])}</th>"
            + "".join(
                f"<td>{matrix_value(left['id'], right['id'], metric, percent)}</td>"
                for right in irr["models"]
            )
            + "</tr>"
            for left in irr["models"]
        )

    irr_rows = "".join(
        f"<tr><td>{html.escape(pair['left_label'])} / "
        f"{html.escape(pair['right_label'])}</td>"
        f"<td>{pair['intersection_papers']:,}</td>"
        f"<td>{html.escape(str(row['label']))}</td>"
        f"<td>{html.escape(str(row['classification']))} dimension</td>"
        f"<td>{row['comparable_papers']:,}</td>"
        f"<td>{_format_metric(row['percent_agreement'], percent=True)}</td>"
        f"<td>{_format_alpha(row['krippendorff_alpha'])}</td>"
        f"<td>{_format_metric(row['observability_percent_agreement'], percent=True)}</td>"
        f"<td>{_format_alpha(row['observability_krippendorff_alpha'])}</td>"
        f"<td>{row['jointly_observed_papers']:,}</td>"
        f"<td>{_format_metric(row['observed_category_percent_agreement'], percent=True)}</td>"
        f"<td>{_format_alpha(row['observed_category_krippendorff_alpha'])}</td></tr>"
        for pair in irr["pairs"]
        for row in pair["dimensions"]
    )
    body += (
        "<h2>Model inter-rater reliability</h2>"
        f"<p>All {len(irr['pairs']):,} available model pairs are compared on one "
        f"balanced intersection of {irr['balanced_common_papers']:,} papers shared by every "
        f"displayed model within the {html.escape(str(irr.get('reference_label') or 'reference-model'))} "
        f"successful-paper cohort (n = {irr['reference_cohort_papers']:,}). IRR uses the selected dataset scope but is not restricted "
        "by the study-status filter because study status is itself a rated dimension.</p>"
        "<h3>Mean exact agreement across six dimensions</h3>"
        f"<table><tr><th>Model</th>{matrix_head}</tr>"
        f"{matrix_rows('mean_percent_agreement', True)}</table>"
        "<h3>Mean pairwise nominal Krippendorff’s α across six dimensions</h3>"
        f"<table><tr><th>Model</th>{matrix_head}</tr>"
        f"{matrix_rows('mean_krippendorff_alpha', False)}</table>"
        "<p>The matrix means are orientation summaries across the six core dimensions: "
        "study status, technical AI type, AI role, mechanism, level of analysis, and scope conditions. "
        "The table reports all eight displayed dimensions. Process stage and definition clarity are "
        "marked exploratory for different empirical reasons. Process-stage models disagree primarily about "
        "whether a stage is observable; definition clarity remains weak both at detecting a definitional "
        "signal and at classifying its form. They remain available as dimension-level results but are not "
        "included in the heatmap averages. Definition clarity records only the signal observable in the "
        "title, abstract, or author keywords and is not a verdict on full-paper quality.</p>"
        "<p><strong>Agreement layers:</strong> All-category agreement retains missing or unspecified "
        "values as categories. Evidence-presence agreement asks whether both models find substantive "
        "evidence. Category agreement where both found evidence then compares their substantive "
        "category choices within that common-evidence subset. Every α coefficient below is pairwise "
        "nominal Krippendorff’s α.</p>"
        "<table><tr><th>Model pair</th><th>Balanced papers</th><th>Dimension</th><th>Analytical status</th>"
        "<th>All balanced papers</th><th>All-category exact agreement</th>"
        "<th>All-category pairwise nominal Krippendorff’s α</th>"
        "<th>Evidence-presence exact agreement</th>"
        "<th>Evidence-presence pairwise nominal Krippendorff’s α</th>"
        "<th>Both found evidence</th><th>Category exact agreement where both found evidence</th>"
        "<th>Category pairwise nominal Krippendorff’s α where both found evidence</th></tr>"
        f"{irr_rows}</table>"
        "<h2>Interpretation boundary</h2>"
        "<p>Model agreement measures consistency, not accuracy. Exact agreement "
        "must be interpreted together with nominal Krippendorff α because "
        "dominant categories can produce high raw agreement. Human coding remains "
        "the accuracy anchor.</p>"
    )
    return _document(f"{scope_label}: construct specification", body)


def build_theory_contrasting_report(
    title: str,
    context: dict[str, object],
    rows: list[dict[str, object]],
) -> str:
    """Return a printable report for one filtered construct-contrasting view."""

    context_rows = "".join(
        f"<tr><th>{html.escape(str(key).replace('_', ' ').title())}</th>"
        f"<td>{html.escape(str(value))}</td></tr>"
        for key, value in context.items()
    )
    if rows:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        head = "".join(
            f"<th>{html.escape(str(column).replace('_', ' ').title())}</th>"
            for column in columns
        )
        body_rows = "".join(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(row.get(column, '')))}</td>"
                for column in columns
            )
            + "</tr>"
            for row in rows
        )
        result_table = f"<table><thead><tr>{head}</tr></thead><tbody>{body_rows}</tbody></table>"
    else:
        result_table = "<p>No rows are available under the selected filters.</p>"
    content = (
        f"<h2>{html.escape(title)}</h2>"
        "<h3>Analysis context</h3>"
        f"<table>{context_rows}</table>"
        "<h3>Results</h3>"
        f"{result_table}"
        "<h2>Interpretation boundary</h2>"
        "<p>Results describe title, abstract, and author-keyword evidence. "
        "Domain memberships may overlap and must not be summed. Structuring "
        "outputs describe recurring co-occurrences rather than causal sequences.</p>"
    )
    return _document(title, content)


def _format_metric(value: float | None, percent: bool = False) -> str:
    if value is None:
        return "Not estimable"
    return f"{value * 100:.2f}%" if percent else f"{value:.2f}"


def _format_alpha(value: float | None) -> str:
    """Format a bounded coefficient without a leading zero (APA style)."""

    if value is None:
        return "Not estimable"
    text = f"{value:.2f}"
    if text.startswith("-0"):
        return f"-{text[2:]}"
    return text[1:] if text.startswith("0") else text


def _intro(scope_label: str, n: int) -> str:
    return (
        f"<h2>Overview</h2>"
        f"<p>This report examines how artificial intelligence is specified across "
        f"the {html.escape(scope_label)} dataset, which contains "
        f"{n:,} papers.</p>"
    )


def _performance_section(service, scope_id: str) -> str:
    perf = service.performance(scope_id)
    s = perf["summary"]
    span = (
        f"{s['year_min']} to {s['year_max']}"
        if s.get("year_min")
        else "an unrecorded period"
    )
    lead = (
        "<h2>Performance analysis</h2>"
        f"<p>The dataset spans {span} and has attracted {s['total_citations']:,} "
        f"citations in total, an average of {s['mean_citations']:.2f} per paper. "
        f"About {s['cited_share'] * 100:.2f} percent of the papers have been "
        f"cited at least once.</p>"
    )

    journals = perf["top_journals"][:10]
    journal_rows = "".join(
        f"<tr><td>{html.escape(str(j['Source title']))}</td>"
        f"<td>{j['papers']}</td><td>{j['citations']:,}</td></tr>"
        for j in journals
    )
    journal_table = (
        "<h3>Most productive journals</h3>"
        "<table><tr><th>Journal</th><th>Papers</th><th>Citations</th></tr>"
        + journal_rows
        + "</table>"
        if journal_rows
        else ""
    )

    cited_rows = ""
    for m in perf["most_cited"][:10]:
        link = _paper_link(m)
        title = html.escape(m.get("Title", "") or "(untitled)")
        title_html = (
            f'<a href="{html.escape(link)}">{title}</a>' if link else title
        )
        cited_rows += (
            f"<tr><td>{html.escape(_citation(m))}</td>"
            f"<td>{title_html}</td><td>{m['citations']:,}</td></tr>"
        )
    cited_table = (
        "<h3>Most cited papers</h3>"
        "<table><tr><th>Citation</th><th>Title</th><th>Citations</th></tr>"
        + cited_rows
        + "</table>"
        if cited_rows
        else ""
    )

    return lead + journal_table + cited_table


def _contrast_section(service, scope_id: str) -> str:
    rows = service.contrast(scope_id, "ai_type_form", "ai_role_function")
    if not rows:
        return ""
    items = "".join(
        f"<tr><td>{html.escape(str(r['shared_value']))}</td>"
        f"<td>{r['contrast_value_count']}</td><td>{r['paper_count']}</td></tr>"
        for r in rows[:15]
    )
    return (
        "<h2>Construct contrast</h2>"
        "<p>The papers below share the same AI type but assign it different roles. "
        "The distinct-role count identifies types whose role assignments warrant "
        "closer inspection; it does not by itself establish theoretical incompatibility.</p>"
        "<table><tr><th>AI type</th><th>Distinct roles</th><th>Papers</th></tr>"
        + items
        + "</table>"
    )


def _methods_note() -> str:
    return (
        "<h2>Notes on method</h2>"
        "<p>Papers were drawn from Scopus searches, merged with their query "
        "provenance preserved, validated against the source title list, and "
        "filtered for relevance to AI and entrepreneurship or business research. "
        "Specification codes follow a seven dimension framework covering role, "
        "type, mechanism, level of analysis, process stage, scope conditions, and "
        "definition clarity. Every figure can be traced to the underlying papers "
        "in the platform.</p>"
    )


def _document(scope_label: str, body: str) -> str:
    generated = date.today().isoformat()
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<title>{html.escape(PLATFORM_NAME)}: {html.escape(scope_label)}</title>
<style>
  body {{ font-family: Georgia, 'Times New Roman', serif; color: #222; max-width: 820px;
         margin: 40px auto; padding: 0 24px; line-height: 1.6; }}
  h1 {{ font-size: 1.6em; color: #2c3e50; }}
  h2 {{ font-size: 1.15em; color: #2c3e50; margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  h3 {{ font-size: 1.0em; color: #2c3e50; margin-top: 18px; }}
  a {{ color: #2c3e50; }}
  .meta {{ color: #777; font-size: 0.9em; margin-bottom: 24px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.92em; }}
  th, td {{ text-align: left; padding: 7px 9px; border-bottom: 1px solid #e5e5e5; }}
  th {{ color: #555; }}
  .actions {{ margin: 20px 0; }}
  button {{ font-size: 0.95em; padding: 8px 16px; border: 1px solid #2c3e50; background: #2c3e50;
           color: #fff; border-radius: 6px; cursor: pointer; }}
  @media print {{ .actions {{ display: none; }} body {{ margin: 0; }} }}
</style></head><body>
<h1>{html.escape(PLATFORM_NAME)}</h1>
<div class="meta">Dataset: {html.escape(scope_label)}. Generated {generated}.</div>
<div class="actions"><button onclick="window.print()">Print or save as PDF</button></div>
{body}
</body></html>"""
