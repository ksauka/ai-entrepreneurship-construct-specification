"""Score lexical relevance to AI and entrepreneurship or business.

Inputs: corpus text, query provenance, and configured keyword groups.
Outputs: broad corpus and strict analytical relevance flags and match details.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

AI_MATCH_COUNT = "ai_match_count"
AI_MATCHED_TERMS = "ai_matched_terms"
ENT_MATCH_COUNT = "ent_match_count"
ENT_MATCHED_TERMS = "ent_matched_terms"
BUSINESS_MATCH_COUNT = "business_match_count"
BUSINESS_MATCHED_TERMS = "business_matched_terms"
ENT_VENUE_COLUMN = "in_entrepreneurship_journal"
STRICT_RELEVANT_COLUMN = "ai_ent_relevant"
RELEVANT_COLUMN = "corpus_relevant"

TEXT_COLUMNS = ("Title", "Abstract", "Author Keywords", "Index Keywords")

# Query views whose journals are entrepreneurship outlets by construction;
# membership satisfies the entrepreneurship domain without lexical matches.
ENTREPRENEURSHIP_VENUE_COLUMNS = ("in_query_3", "in_query_4")


def load_relevance_config(config_path: Path) -> dict[str, list[str]]:
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return {
        "ai_terms": list(config.get("ai_seed_terms_expanded", [])),
        "ent_terms": list(config.get("entrepreneurship_seed_terms", [])),
        "weak_ent_terms": list(config.get("weak_entrepreneurship_seed_terms", [])),
        "business_terms": list(config.get("business_seed_terms", [])),
    }


def build_search_text(frame: pd.DataFrame) -> pd.Series:
    available = [col for col in TEXT_COLUMNS if col in frame.columns]
    if not available:
        return pd.Series([""] * len(frame), index=frame.index)
    return (
        frame[available]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def score_relevance(frame: pd.DataFrame, config: dict[str, list[str]]) -> pd.DataFrame:
    """Add AI/entrepreneurship/business match columns and both relevance flags."""

    frame = frame.copy()
    text = build_search_text(frame)

    ai_matches = _match_matrix(text, config["ai_terms"])
    ent_matches = _match_matrix(text, config["ent_terms"])
    business_matches = _match_matrix(text, config.get("business_terms", []))
    weak_terms = {term for term in config["weak_ent_terms"]}

    frame[AI_MATCH_COUNT] = ai_matches.sum(axis=1)
    frame[AI_MATCHED_TERMS] = _matched_labels(ai_matches)
    frame[ENT_MATCH_COUNT] = ent_matches.sum(axis=1)
    frame[ENT_MATCHED_TERMS] = _matched_labels(ent_matches)
    frame[BUSINESS_MATCH_COUNT] = business_matches.sum(axis=1)
    frame[BUSINESS_MATCHED_TERMS] = _matched_labels(business_matches)

    frame[ENT_VENUE_COLUMN] = _entrepreneurship_venue(frame)

    strong_columns = [col for col in ent_matches.columns if col not in weak_terms]
    strong_count = ent_matches[strong_columns].sum(axis=1) if strong_columns else 0

    ai_ok = frame[AI_MATCH_COUNT] >= 1
    ent_ok = (strong_count >= 1) | (frame[ENT_MATCH_COUNT] >= 2)
    venue_ok = frame[ENT_VENUE_COLUMN] == 1
    business_ok = frame[BUSINESS_MATCH_COUNT] >= 1

    frame[STRICT_RELEVANT_COLUMN] = (ai_ok & (ent_ok | venue_ok)).astype(int)
    frame[RELEVANT_COLUMN] = (ai_ok & (ent_ok | venue_ok | business_ok)).astype(int)
    return frame


def _entrepreneurship_venue(frame: pd.DataFrame) -> pd.Series:
    venue = pd.Series([0] * len(frame), index=frame.index)
    for column in ENTREPRENEURSHIP_VENUE_COLUMNS:
        if column in frame.columns:
            member = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
            venue = venue | (member == 1)
    return venue.astype(int)


def _match_matrix(text: pd.Series, terms: Iterable[str]) -> pd.DataFrame:
    """Boolean DataFrame: one column per seed term, True where the text matches."""

    columns: dict[str, pd.Series] = {}
    for term in terms:
        label = str(term).strip()
        if not label or label in columns:
            continue
        if "*" in label:
            pattern = re.escape(label).replace(r"\*", r"\w*")
            regex = rf"\b{pattern}\b"
        else:
            regex = rf"(?<!\w){re.escape(label)}(?!\w)"
        columns[label] = text.str.contains(regex, case=False, regex=True)
    return pd.DataFrame(columns, index=text.index)


def _matched_labels(matches: pd.DataFrame) -> pd.Series:
    if matches.empty:
        return pd.Series([""] * len(matches), index=matches.index)
    labels = matches.columns.to_numpy()
    return matches.apply(lambda row: ";".join(labels[row.to_numpy()]), axis=1)
