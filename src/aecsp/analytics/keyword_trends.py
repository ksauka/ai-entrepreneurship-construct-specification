"""Calculate source-aware keyword prevalence across publication periods.

Inputs: paper-level years and semicolon-delimited author or index keywords.
Outputs: normalized period trends, prevalence changes, and evidence masks.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ALIAS_PATH = PROJECT_ROOT / "configs" / "keyword_aliases.json"
DEFAULT_SEARCH_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "search_queries_july2026_q1_q4.yaml"
)


def load_search_cutoff(path: Path = DEFAULT_SEARCH_CONFIG_PATH) -> date:
    """Return the documented source-search cut-off date."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    value = payload.get("date_updated")
    if not value:
        raise ValueError(f"Search cut-off date is missing from {path}")
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


SEARCH_CUTOFF_DATE = load_search_cutoff()
SEARCH_CUTOFF_LABEL = f"{SEARCH_CUTOFF_DATE.day} {SEARCH_CUTOFF_DATE:%B %Y}"
SEARCH_CUTOFF_YEAR = SEARCH_CUTOFF_DATE.year

KEYWORD_SOURCES = {
    "author": ("Author Keywords",),
    "index": ("Index Keywords",),
    "combined": ("Author Keywords", "Index Keywords"),
}

KEYWORD_SOURCE_LABELS = {
    "author": "Author keywords",
    "index": "Scopus index keywords",
    "combined": "Author and Scopus index keywords",
}

# Only heads whose singular/plural forms represent the same bibliometric
# concept are included. This deliberately excludes ambiguous pairs such as
# analytic/analytics, economic/economics, dynamic/dynamics and human/humans.
_COUNT_NOUN_HEADS = (
    ("agent", "agents"),
    ("algorithm", "algorithms"),
    ("application", "applications"),
    ("architecture", "architectures"),
    ("capability", "capabilities"),
    ("category", "categories"),
    ("chatbot", "chatbots"),
    ("city", "cities"),
    ("cluster", "clusters"),
    ("country", "countries"),
    ("dataset", "datasets"),
    ("destination", "destinations"),
    ("ecosystem", "ecosystems"),
    ("emission", "emissions"),
    ("enterprise", "enterprises"),
    ("forest", "forests"),
    ("framework", "frameworks"),
    ("goal", "goals"),
    ("graph", "graphs"),
    ("industry", "industries"),
    ("interface", "interfaces"),
    ("machine", "machines"),
    ("market", "markets"),
    ("mechanism", "mechanisms"),
    ("method", "methods"),
    ("model", "models"),
    ("network", "networks"),
    ("ontology", "ontologies"),
    ("platform", "platforms"),
    ("review", "reviews"),
    ("robot", "robots"),
    ("set", "sets"),
    ("startup", "startups"),
    ("structure", "structures"),
    ("study", "studies"),
    ("survey", "surveys"),
    ("system", "systems"),
    ("technique", "techniques"),
    ("technology", "technologies"),
    ("tool", "tools"),
    ("tree", "trees"),
    ("twin", "twins"),
)
_KEYWORD_HEAD_ALIASES = {
    variant: plural
    for singular, plural in _COUNT_NOUN_HEADS
    for variant in (singular, plural)
}

# Explicit British/American spelling equivalents. These substitutions operate
# on whole tokens only; they do not use edit distance or guess at misspellings.
_ORTHOGRAPHIC_TOKEN_ALIASES = {
    "behaviour": "behavior",
    "behaviours": "behaviors",
    "centre": "center",
    "centres": "centers",
    "colour": "color",
    "colours": "colors",
    "conceptualisation": "conceptualization",
    "conceptualisations": "conceptualizations",
    "digitisation": "digitization",
    "digitisations": "digitizations",
    "globalisation": "globalization",
    "globalisations": "globalizations",
    "internationalisation": "internationalization",
    "internationalisations": "internationalizations",
    "labour": "labor",
    "modelling": "modeling",
    "optimisation": "optimization",
    "optimisations": "optimizations",
    "organisation": "organization",
    "organisations": "organizations",
    "organisational": "organizational",
    "personalisation": "personalization",
    "personalised": "personalized",
    "specialisation": "specialization",
    "specialisations": "specializations",
    "standardisation": "standardization",
    "standardisations": "standardizations",
    "utilisation": "utilization",
    "visualisation": "visualization",
    "visualisations": "visualizations",
}

KEYWORD_PERIODS = (
    {"id": "pre_2000", "label": "Before 2000", "start": None, "end": 1999, "incomplete": False},
    {"id": "2000_2015", "label": "2000–2015", "start": 2000, "end": 2015, "incomplete": False},
    {"id": "2016_2020", "label": "2016–2020", "start": 2016, "end": 2020, "incomplete": False},
    {"id": "2021_2023", "label": "2021–2023", "start": 2021, "end": 2023, "incomplete": False},
    {
        "id": "2024_2026",
        "label": f"2024–2026 (as at {SEARCH_CUTOFF_LABEL})",
        "start": 2024,
        "end": SEARCH_CUTOFF_YEAR,
        "incomplete": False,
    },
)


def _base_normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"(?<=\w)[‐‑‒–—-](?=\w)", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_keyword_aliases(path: Path = DEFAULT_ALIAS_PATH) -> dict[str, str]:
    """Load exact aliases without changing any source keyword column."""

    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        _base_normalize(source): _base_normalize(target)
        for source, target in payload.get("aliases", {}).items()
    }


def normalize_keyword(value: object, aliases: dict[str, str] | None = None) -> str:
    """Return a controlled, auditable canonical keyword label."""

    normalized = _base_normalize(value)
    normalized = (aliases or {}).get(normalized, normalized)
    words = [
        _ORTHOGRAPHIC_TOKEN_ALIASES.get(word, word)
        for word in normalized.split()
    ]
    if words and words[-1] in _KEYWORD_HEAD_ALIASES:
        words[-1] = _KEYWORD_HEAD_ALIASES[words[-1]]
    return " ".join(words)


def paper_keywords(
    row: pd.Series,
    source: str,
    aliases: dict[str, str] | None = None,
) -> set[str]:
    """Return unique canonical terms for one paper and keyword source."""

    if source not in KEYWORD_SOURCES:
        raise ValueError(f"Unknown keyword source: {source}")
    values: set[str] = set()
    for column in KEYWORD_SOURCES[source]:
        for term in str(row.get(column, "") or "").split(";"):
            normalized = normalize_keyword(term, aliases)
            if normalized:
                values.add(normalized)
    return values


def period_mask(years: pd.Series, period_id: str) -> pd.Series:
    """Select one locked publication period from a numeric year series."""

    if period_id == "all_time":
        return years.notna() & (years <= SEARCH_CUTOFF_YEAR)
    period = next((item for item in KEYWORD_PERIODS if item["id"] == period_id), None)
    if period is None:
        raise ValueError(f"Unknown keyword period: {period_id}")
    mask = years.notna()
    if period["start"] is not None:
        mask &= years >= period["start"]
    if period["end"] is not None:
        mask &= years <= period["end"]
    return mask


def _keyword_record(
    keyword: str,
    current_count: int,
    denominator: int,
    previous_count: int | None,
    previous_denominator: int | None,
) -> dict:
    prevalence = current_count / denominator if denominator else 0.0
    previous = (
        previous_count / previous_denominator
        if previous_count is not None and previous_denominator
        else None
    )
    return {
        "keyword": keyword,
        "papers": int(current_count),
        "prevalence": round(prevalence, 6),
        "previous_prevalence": round(previous, 6) if previous is not None else None,
        "change_pp": round((prevalence - previous) * 100, 3) if previous is not None else None,
    }


def _annual_keyword_counts(
    years: pd.Series,
    keyword_sets: pd.Series,
) -> tuple[list[int], dict[int, Counter], dict[int, int]]:
    """Count canonical keywords and keyword-bearing papers by publication year."""

    valid = years.notna() & (years <= SEARCH_CUTOFF_YEAR)
    if not valid.any():
        return [], {}, {}
    start_year = int(years[valid].min())
    annual_years = list(range(start_year, SEARCH_CUTOFF_YEAR + 1))
    annual_counts: dict[int, Counter] = {}
    annual_denominators: dict[int, int] = {}
    for year in annual_years:
        selected = keyword_sets[years == year]
        eligible = selected[selected.map(bool)]
        annual_counts[year] = Counter(term for terms in eligible for term in terms)
        annual_denominators[year] = len(eligible)
    return annual_years, annual_counts, annual_denominators


def _annual_values(
    keyword: str,
    annual_years: list[int],
    annual_counts: dict[int, Counter],
    annual_denominators: dict[int, int],
) -> list[dict]:
    """Return one keyword's annual paper count and prevalence trajectory."""

    values = []
    for year in annual_years:
        count = annual_counts[year].get(keyword, 0)
        denominator = annual_denominators[year]
        values.append(
            {
                "year": year,
                "papers": int(count),
                "denominator": int(denominator),
                "prevalence": round(count / denominator, 6) if denominator else 0.0,
            }
        )
    return values


def analyze_keyword_evolution(
    frame: pd.DataFrame,
    source: str = "author",
    *,
    series_top_n: int = 10,
    period_top_n: int = 20,
    mover_top_n: int = 10,
    minimum_mover_papers: int = 10,
    aliases: dict[str, str] | None = None,
) -> dict:
    """Summarize longitudinal keyword prevalence for one corpus scope.

    Prevalence uses papers with at least one keyword from the selected source as
    the period denominator. A term is counted at most once per paper.
    """

    if source not in KEYWORD_SOURCES:
        raise ValueError(f"Unknown keyword source: {source}")
    aliases = load_keyword_aliases() if aliases is None else aliases
    years = pd.to_numeric(frame.get("Year", pd.Series(index=frame.index)), errors="coerce")
    keyword_sets = frame.apply(paper_keywords, axis=1, source=source, aliases=aliases)
    annual_years, annual_counts, annual_denominators = _annual_keyword_counts(
        years, keyword_sets
    )

    period_counts: list[Counter] = []
    denominators: list[int] = []
    period_rows: list[dict] = []
    overall = Counter()

    for period in KEYWORD_PERIODS:
        mask = period_mask(years, period["id"])
        selected = keyword_sets[mask]
        eligible = selected[selected.map(bool)]
        counts = Counter(term for terms in eligible for term in terms)
        period_counts.append(counts)
        denominator = len(eligible)
        denominators.append(denominator)
        overall.update(counts)

        previous_counts = period_counts[-2] if len(period_counts) > 1 else None
        previous_denominator = denominators[-2] if len(denominators) > 1 else None
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        top_keywords = [
            _keyword_record(
                keyword,
                count,
                denominator,
                previous_counts.get(keyword, 0) if previous_counts is not None else None,
                previous_denominator,
            )
            for keyword, count in ranked[:period_top_n]
        ]

        emerging: list[dict] = []
        declining: list[dict] = []
        if previous_counts is not None:
            candidates = set(counts) | set(previous_counts)
            records = [
                _keyword_record(
                    keyword,
                    counts.get(keyword, 0),
                    denominator,
                    previous_counts.get(keyword, 0),
                    previous_denominator,
                )
                for keyword in candidates
                if max(counts.get(keyword, 0), previous_counts.get(keyword, 0))
                >= minimum_mover_papers
            ]
            emerging = sorted(
                (record for record in records if (record["change_pp"] or 0) > 0),
                key=lambda record: (-record["change_pp"], -record["papers"], record["keyword"]),
            )[:mover_top_n]
            declining = sorted(
                (record for record in records if (record["change_pp"] or 0) < 0),
                key=lambda record: (record["change_pp"], -record["papers"], record["keyword"]),
            )[:mover_top_n]

        period_rows.append(
            {
                **period,
                "papers": int(mask.sum()),
                "keyword_papers": denominator,
                "coverage": round(denominator / int(mask.sum()), 6) if mask.any() else 0.0,
                "top_keywords": top_keywords,
                "emerging": emerging,
                "declining": declining,
            }
        )

    overall_keywords = [keyword for keyword, _ in overall.most_common(series_top_n)]
    dynamic_keywords = {
        item["keyword"]
        for period in period_rows
        for group in ("top_keywords", "emerging", "declining")
        for item in period[group]
    }
    series_keywords = [
        *overall_keywords,
        *sorted(dynamic_keywords - set(overall_keywords)),
    ]
    series = []
    for keyword in series_keywords:
        values = []
        for period, counts, denominator in zip(KEYWORD_PERIODS, period_counts, denominators):
            values.append(
                {
                    "period": period["id"],
                    "papers": int(counts.get(keyword, 0)),
                    "prevalence": round(counts.get(keyword, 0) / denominator, 6)
                    if denominator
                    else 0.0,
                }
            )
        series.append(
            {
                "keyword": keyword,
                "total_paper_periods": int(overall[keyword]),
                "values": values,
                "annual_values": _annual_values(
                    keyword,
                    annual_years,
                    annual_counts,
                    annual_denominators,
                ),
            }
        )

    valid_year = years.notna() & (years <= SEARCH_CUTOFF_YEAR)
    valid_keyword_sets = keyword_sets[valid_year]
    all_time_denominator = int(valid_keyword_sets.map(bool).sum())
    all_time_ranked = sorted(overall.items(), key=lambda item: (-item[1], item[0]))
    all_time = {
        "id": "all_time",
        "label": f"All time (to {SEARCH_CUTOFF_LABEL})",
        "start": annual_years[0] if annual_years else None,
        "end": SEARCH_CUTOFF_YEAR,
        "papers": int(valid_year.sum()),
        "keyword_papers": all_time_denominator,
        "coverage": round(all_time_denominator / int(valid_year.sum()), 6)
        if valid_year.any()
        else 0.0,
        "top_keywords": [
            _keyword_record(keyword, count, all_time_denominator, None, None)
            for keyword, count in all_time_ranked[:period_top_n]
        ],
        "emerging": [],
        "declining": [],
    }

    covered = keyword_sets.map(bool)
    return {
        "source": source,
        "source_label": KEYWORD_SOURCE_LABELS[source],
        "denominator_definition": (
            "For each year or period, papers with at least one keyword from the "
            "selected source form the denominator; each canonical keyword is counted "
            "once per paper."
        ),
        "papers": len(frame),
        "keyword_papers": int(covered.sum()),
        "coverage": round(float(covered.mean()), 6) if len(frame) else 0.0,
        "search_cutoff": {
            "date": SEARCH_CUTOFF_DATE.isoformat(),
            "label": SEARCH_CUTOFF_LABEL,
            "year": SEARCH_CUTOFF_YEAR,
        },
        "annual_years": annual_years,
        "all_time": all_time,
        "periods": period_rows,
        "series": series,
        "default_overall_keywords": overall_keywords,
        "excluded": {
            "invalid_year": int(years.isna().sum()),
            "after_2026": int((years > SEARCH_CUTOFF_YEAR).sum()),
        },
    }


def search_keyword_series(
    frame: pd.DataFrame,
    source: str,
    query: str,
    *,
    limit: int = 20,
    aliases: dict[str, str] | None = None,
) -> list[dict]:
    """Find canonical keywords and return their full period trajectories."""

    if source not in KEYWORD_SOURCES:
        raise ValueError(f"Unknown keyword source: {source}")
    aliases = load_keyword_aliases() if aliases is None else aliases
    normalized_query = normalize_keyword(query, aliases)
    years = pd.to_numeric(frame.get("Year", pd.Series(index=frame.index)), errors="coerce")
    keyword_sets = frame.apply(paper_keywords, axis=1, source=source, aliases=aliases)
    annual_years, annual_counts, annual_denominators = _annual_keyword_counts(
        years, keyword_sets
    )

    period_counts: list[Counter] = []
    denominators: list[int] = []
    overall = Counter()
    for period in KEYWORD_PERIODS:
        selected = keyword_sets[period_mask(years, period["id"])]
        eligible = selected[selected.map(bool)]
        counts = Counter(term for terms in eligible for term in terms)
        period_counts.append(counts)
        denominators.append(len(eligible))
        overall.update(counts)

    candidates = [
        (keyword, count)
        for keyword, count in overall.items()
        if not normalized_query or normalized_query in keyword
    ]
    candidates.sort(
        key=lambda item: (
            0 if item[0] == normalized_query else 1,
            0 if item[0].startswith(normalized_query) else 1,
            -item[1],
            item[0],
        )
    )
    results = []
    for keyword, total in candidates[:limit]:
        values = []
        for period, counts, denominator in zip(KEYWORD_PERIODS, period_counts, denominators):
            count = counts.get(keyword, 0)
            values.append(
                {
                    "period": period["id"],
                    "papers": int(count),
                    "prevalence": round(count / denominator, 6) if denominator else 0.0,
                }
            )
        results.append(
            {
                "keyword": keyword,
                "total_paper_periods": int(total),
                "values": values,
                "annual_values": _annual_values(
                    keyword,
                    annual_years,
                    annual_counts,
                    annual_denominators,
                ),
            }
        )
    return results


def keyword_year_summary(
    frame: pd.DataFrame,
    source: str,
    year: int,
    *,
    limit: int = 20,
    aliases: dict[str, str] | None = None,
) -> dict:
    """Rank canonical keywords within one recorded publication year."""

    if source not in KEYWORD_SOURCES:
        raise ValueError(f"Unknown keyword source: {source}")
    if not 1 <= int(limit) <= 100:
        raise ValueError("Keyword-year limit must be between 1 and 100")
    aliases = load_keyword_aliases() if aliases is None else aliases
    years = pd.to_numeric(
        frame.get("Year", pd.Series(index=frame.index)), errors="coerce"
    )
    selected = frame.loc[years.eq(year)]
    keyword_sets = selected.apply(
        paper_keywords, axis=1, source=source, aliases=aliases
    )
    eligible = keyword_sets[keyword_sets.map(bool)]
    counts = Counter(term for terms in eligible for term in terms)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    denominator = len(eligible)
    return {
        "year": int(year),
        "source": source,
        "source_label": KEYWORD_SOURCE_LABELS[source],
        "papers": len(selected),
        "keyword_papers": denominator,
        "coverage": round(denominator / len(selected), 6) if len(selected) else 0.0,
        "denominator_definition": (
            "Papers published in the selected year with at least one keyword "
            "from the selected source form the denominator; each canonical "
            "keyword is counted once per paper."
        ),
        "top_keywords": [
            {
                "keyword": keyword,
                "papers": int(count),
                "prevalence": round(count / denominator, 6) if denominator else 0.0,
            }
            for keyword, count in ranked[:limit]
        ],
    }


def keyword_evidence_mask(
    frame: pd.DataFrame,
    source: str,
    keyword: str,
    period_id: str | None,
    aliases: dict[str, str] | None = None,
    *,
    year: int | None = None,
) -> pd.Series:
    """Return papers containing a canonical keyword in one period or year."""

    aliases = load_keyword_aliases() if aliases is None else aliases
    target = normalize_keyword(keyword, aliases)
    years = pd.to_numeric(frame.get("Year", pd.Series(index=frame.index)), errors="coerce")
    keywords = frame.apply(paper_keywords, axis=1, source=source, aliases=aliases)
    if year is not None:
        time_mask = years == year
    elif period_id is not None:
        time_mask = period_mask(years, period_id)
    else:
        raise ValueError("A keyword evidence period or year is required")
    return time_mask & keywords.map(lambda terms: target in terms)
