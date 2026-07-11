"""Derive locked analysis fields without mutating raw model records."""

from __future__ import annotations

import pandas as pd


def enrich_for_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    raw = result.get("ai_mechanism", pd.Series("", index=result.index)).fillna("").astype(str)
    logic = result.get("ai_mechanism_logic", pd.Series("", index=result.index)).fillna("").astype(str)
    substantive = raw.str.strip().ne("") & raw.str.strip().ne("mechanism missing")
    corrected = substantive & logic.str.strip().eq("")
    result["ai_mechanism_raw"] = raw
    result["ai_mechanism_analysis"] = raw.mask(corrected, "mechanism missing")
    result["mechanism_corrected"] = corrected.astype(int)
    problems = result.get("specification_problem", pd.Series("", index=result.index)).fillna("").astype(str)
    result["mechanism_black_box"] = (
        result["ai_mechanism_analysis"].eq("mechanism missing")
        | problems.str.split(";").map(lambda values: "mechanism missing" in values)
    ).astype(int)
    result["specification_problem_count"] = problems.map(
        lambda value: len([part for part in value.split(";") if part.strip()])
    )
    return result
