"""Scope report generation for the platform.

Produces a self-contained, printable HTML report describing how artificial
intelligence is specified across one dataset scope. Written in plain prose so a
researcher can read it, save it as PDF from the browser, or paste sections into
a manuscript. No decorative symbols are used.
"""

from __future__ import annotations

import html
import re
from datetime import date

from aecsp.corpus.scopes import SCOPE_BY_ID
from aecsp.specification.schema import SPECIFICATION_DIMENSIONS

DIM_LABELS = {d.column: d.label for d in SPECIFICATION_DIMENSIONS}

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
    doi = str(row.get("DOI", "")).strip()
    if doi:
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
        return "https://doi.org/" + doi
    link = str(row.get("Link", "")).strip()
    return link or None


def build_scope_report(service, scope_id: str) -> str:
    """Return a full HTML document reporting on one dataset scope."""

    scope = SCOPE_BY_ID.get(scope_id)
    scope_label = scope.label if scope else scope_id
    overview = service.scope_overview(scope_id)
    n = overview["paper_count"]

    sections = [_intro(scope_label, n, overview)]
    sections.append(_performance_section(service, scope_id))

    if overview["has_specifications"]:
        sections.append(_dimensions_section(overview))
        sections.append(_contrast_section(service, scope_id))
        sections.append(_journal_section(service, scope_id))
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


def _intro(scope_label: str, n: int, overview: dict) -> str:
    clarity = overview["overall_specification_clarity_score"]
    fragmentation = overview["fragmentation_score"]
    lead = (
        f"<h2>Overview</h2>"
        f"<p>This report examines how artificial intelligence is specified across "
        f"the {html.escape(scope_label)} dataset, which contains "
        f"{n:,} papers.</p>"
    )
    if overview["has_specifications"]:
        lead += (
            f"<p>Across the seven specification dimensions, the papers reach an "
            f"average specification clarity of {clarity:.2f} on a zero to one "
            f"scale, where one means the papers agree closely on how AI is "
            f"specified and zero means they diverge. The corresponding "
            f"fragmentation score is {fragmentation:.2f}. Higher fragmentation "
            f"points to a literature that uses the label AI in competing ways.</p>"
        )
    return lead


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
        f"About {round(s['cited_share'] * 100)} percent of the papers have been "
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


def _dimensions_section(overview: dict) -> str:
    rows = []
    for column, label in DIM_LABELS.items():
        score = overview["dimension_convergence"].get(column, 0.0)
        dominant = overview["dominant_values"].get(column) or "not specified"
        reading = "converges" if score >= 0.6 else "is divided" if score >= 0.3 else "is fragmented"
        rows.append(
            f"<tr><td>{html.escape(label)}</td>"
            f"<td>{html.escape(str(dominant))}</td>"
            f"<td>{score:.2f}</td><td>{reading}</td></tr>"
        )
    return (
        "<h2>Specification dimensions</h2>"
        "<p>For each dimension, the table shows the most common value in this "
        "dataset and how strongly the papers agree on it.</p>"
        "<table><tr><th>Dimension</th><th>Most common value</th>"
        "<th>Convergence</th><th>Reading</th></tr>"
        + "".join(rows)
        + "</table>"
    )


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
        "A high number of distinct roles for one AI type indicates a construct "
        "contrast, where the same technology is theorised in incompatible ways.</p>"
        "<table><tr><th>AI type</th><th>Distinct roles</th><th>Papers</th></tr>"
        + items
        + "</table>"
    )


def _journal_section(service, scope_id: str) -> str:
    rows = service.group_convergence_table(scope_id, "Source title")
    if not rows:
        return ""
    items = "".join(
        f"<tr><td>{html.escape(str(r['Source title']))}</td>"
        f"<td>{r['paper_count']}</td>"
        f"<td>{r['fragmentation_score']:.2f}</td></tr>"
        for r in rows[:15]
    )
    return (
        "<h2>Journals with the most fragmented specification</h2>"
        "<p>These journals show the widest internal disagreement on how AI is "
        "specified, which can indicate a venue where the construct is still "
        "unsettled.</p>"
        "<table><tr><th>Journal</th><th>Papers</th><th>Fragmentation</th></tr>"
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
        "definition clarity. Convergence is measured from the spread of codes "
        "within a group, so every figure in this report can be traced back to the "
        "underlying papers in the platform.</p>"
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
