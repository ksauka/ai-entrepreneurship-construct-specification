"""Calculate traceable cumulative publication-stock growth.

The growth denominator is the cumulative number of papers through the start
year, not the number published during the start year.  Keeping this logic in a
shared module prevents the manuscript and interactive platform from silently
using different endpoint definitions.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


DEFAULT_GROWTH_PERIODS: tuple[tuple[int, int], ...] = (
    (1976, 2026),
    (1976, 2000),
    (2000, 2026),
    (2000, 2010),
    (2010, 2020),
    (2020, 2023),
    (2023, 2026),
)


def publication_years(frame: pd.DataFrame, column: str = "Year") -> pd.Series:
    """Return numeric publication years aligned to ``frame``."""

    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def cumulative_papers_through(
    frame: pd.DataFrame,
    year: int,
    column: str = "Year",
) -> int:
    """Count papers with a valid recorded publication year through ``year``."""

    years = publication_years(frame, column)
    return int(((years > 0) & (years <= year)).sum())


def growth_record(
    frame: pd.DataFrame,
    start_year: int,
    end_year: int,
    column: str = "Year",
) -> dict:
    """Describe cumulative-stock growth between two inclusive endpoints.

    ``percent_growth`` is stored as a proportion for consistency with the API:
    ``1.0`` means 100 percent growth.
    """

    if end_year <= start_year:
        raise ValueError("Growth end year must be later than start year")
    start_count = cumulative_papers_through(frame, start_year, column)
    end_count = cumulative_papers_through(frame, end_year, column)
    added = end_count - start_count
    return {
        "start_year": int(start_year),
        "end_year": int(end_year),
        "start_cumulative_papers": start_count,
        "end_cumulative_papers": end_count,
        "added_papers": added,
        "percent_growth": (
            round(added / start_count, 6) if start_count > 0 else None
        ),
    }


def growth_records(
    frame: pd.DataFrame,
    periods: Iterable[tuple[int, int]] = DEFAULT_GROWTH_PERIODS,
    column: str = "Year",
) -> list[dict]:
    """Calculate each registered cumulative-stock growth window."""

    return [
        growth_record(frame, start_year, end_year, column)
        for start_year, end_year in periods
    ]


def cumulative_trace(
    frame: pd.DataFrame,
    start_year: int = 2000,
    end_year: int = 2026,
    column: str = "Year",
) -> list[dict]:
    """Return annual counts and the running corpus stock for a chart window.

    The cumulative value at the first displayed year includes papers published
    before that year.  This is required for the endpoint to match
    :func:`growth_record` exactly.
    """

    if end_year < start_year:
        raise ValueError("Trace end year must not precede start year")
    years = publication_years(frame, column)
    return [
        {
            "year": year,
            "papers": int((years == year).sum()),
            "cumulative_papers": int(((years > 0) & (years <= year)).sum()),
        }
        for year in range(start_year, end_year + 1)
    ]
