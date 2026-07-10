"""
extraction.py

Extract topics, document assignments, and domain-specific terms (AI, entrepreneurship)
from trained BERTopic models. Generates SKOS construction contract artifacts.

KEY FEATURES:
- Data-driven topic labels from top keywords
- Outlier handling with probability-based reassignment
- SKOS contract generation (3 tables: topics, topic_terms, doc_topics)
- Paper-level keyword extraction for Stage 2B integration
- Topic-level keyword extraction for domain analysis

USAGE:
    from theory_elaboration.topic_modeling import (
        extract_topics,
        assign_document_topics,
        export_bertopic_artifacts,
        extract_ai_terms_by_topic,
        extract_ai_terms_by_paper
    )

    # Extract topic summaries
    topics_df = extract_topics(model)

    # Assign documents to topics
    doc_topics_df = assign_document_topics(
        model, documents,
        min_reassign_prob=0.05,
        reassign_outliers=True
    )

    # Generate SKOS contract (3 tables)
    export_bertopic_artifacts(
        model, doc_topics_df, doc_index_df, out_dir
    )
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from aecsp.progress import ProgressReporter

if TYPE_CHECKING:
    from bertopic import BERTopic

logger = logging.getLogger(__name__)


_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_SIMPLE_PHRASE_PATTERN = re.compile(r"^[^\W_]+(?:[ _][^\W_]+)*$", re.UNICODE)


@dataclass(frozen=True)
class _PhraseIndex:
    by_width: dict[int, dict[tuple[str, ...], set[str]]]
    regex_fallback: dict[str, re.Pattern]


def _build_phrase_index(
    phrases: set[str], *, uppercase_ai_only: bool = False
) -> _PhraseIndex:
    """Compile reusable token sequences for a domain vocabulary."""

    by_width: dict[int, dict[tuple[str, ...], set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    regex_fallback: dict[str, re.Pattern] = {}
    for phrase in phrases:
        if uppercase_ai_only and phrase == "ai":
            continue
        if not _SIMPLE_PHRASE_PATTERN.fullmatch(str(phrase)):
            spaced = phrase.replace("_", " ")
            regex_fallback[phrase] = re.compile(
                rf"\b{re.escape(phrase)}\b|\b{re.escape(spaced)}\b", re.IGNORECASE
            )
            continue
        tokens = tuple(
            token.lower()
            for token in _TOKEN_PATTERN.findall(str(phrase).replace("_", " "))
        )
        if tokens:
            by_width[len(tokens)][tokens].add(phrase)
    return _PhraseIndex(dict(by_width), regex_fallback)


def _phrase_counts(
    text: object,
    phrases: set[str],
    *,
    uppercase_ai_only: bool = False,
    phrase_index: _PhraseIndex | None = None,
) -> Counter:
    """Count domain phrases in one document without a regex scan per phrase.

    Hybrid phrase detection emits one-to-four-word phrases.  Indexing those
    phrases by token length makes extraction proportional to document length
    instead of ``documents * phrases`` while retaining overlapping matches
    such as both ``machine learning`` and ``learning``.
    """

    original = str(text or "")
    token_matches = list(_TOKEN_PATTERN.finditer(original))
    lowered_tokens = [match.group().lower() for match in token_matches]
    if phrase_index is None:
        phrase_index = _build_phrase_index(
            phrases, uppercase_ai_only=uppercase_ai_only
        )

    counts: Counter = Counter()
    for width, aliases in phrase_index.by_width.items():
        if width > len(lowered_tokens):
            continue
        for start in range(len(lowered_tokens) - width + 1):
            matched = aliases.get(tuple(lowered_tokens[start : start + width]))
            if matched:
                surface = original[
                    token_matches[start].start() : token_matches[start + width - 1].end()
                ].lower()
                for phrase in matched:
                    if surface in {phrase.lower(), phrase.replace("_", " ").lower()}:
                        counts[phrase] += 1

    for phrase, pattern in phrase_index.regex_fallback.items():
        count = len(pattern.findall(original))
        if count:
            counts[phrase] = count

    if uppercase_ai_only and "ai" in phrases:
        ai_count = len(re.findall(r"\bAI\b", original))
        if ai_count:
            counts["ai"] = ai_count
    return counts


def extract_topics(model: BERTopic) -> pd.DataFrame:
    """
    Extract topic information from BERTopic model with data-driven labels.

    Generates topic labels from top 2 keywords (cleaned and formatted).

    Args:
        model: Trained BERTopic model

    Returns:
        DataFrame with columns:
        - topic_id: Topic ID
        - topic_size: Number of documents
        - topic_label: Data-driven label from top keywords
        - top_words_20: Top 20 words (comma-separated)
        - word_scores_20: c-TF-IDF scores for top 20 words
        - top_words_full: All words for topic
    """
    topic_info = model.get_topic_info()

    # BERTopic returns: Topic (ID), Count (size), Name, Representation (top words)
    # Reformat to match our expected structure
    rows = []
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:  # Skip outlier topic
            continue

        # Get detailed topic words and scores
        topic_words = model.get_topic(topic_id)
        if topic_words:
            words = [word for word, score in topic_words[:20]]
            scores = [score for word, score in topic_words[:20]]
            all_words = [word for word, score in topic_words]
        else:
            words, scores, all_words = [], [], []

        # Generate data-driven topic label from top 2 keywords
        # Remove underscores, title case, join with underscore
        label_parts = []
        for w in words[:2]:  # Top 2 keywords
            cleaned = w.replace('_', ' ').title().replace(' ', '_')
            # Remove special characters
            cleaned = ''.join(c for c in cleaned if c.isalnum() or c == '_')
            label_parts.append(cleaned)
        topic_label = '_'.join(label_parts) if label_parts else f"Topic_{topic_id}"

        rows.append({
            "topic_id": int(topic_id),
            "topic_size": int(row['Count']),
            "topic_label": topic_label,
            "top_words_20": ", ".join(words),
            "word_scores_20": ", ".join(f"{s:.3f}" for s in scores),
            "top_words_full": ", ".join(all_words),
        })

    return pd.DataFrame(rows)


def assign_document_topics(
    model: BERTopic,
    documents: List[str],
    doc_ids: Optional[List] = None,
    min_reassign_prob: float = 0.05,
    reassign_outliers: bool = True
) -> pd.DataFrame:
    """
    Map each document to its assigned topic with probability.
    Handles BERTopic probability arrays correctly and optionally reassigns outliers.

    Args:
        model: Trained BERTopic model
        documents: List of documents
        doc_ids: Optional document IDs (defaults to indices)
        min_reassign_prob: Minimum probability threshold for outlier reassignment
        reassign_outliers: Whether to reassign outliers to best non-outlier topic

    Returns:
        DataFrame with columns:
        - doc_local_index: Document index
        - topic_id: Assigned topic ID (None for unassigned outliers)
        - topic_prob: Topic probability
        - was_outlier: Whether document was originally an outlier
    """
    topics, probs = model.transform(documents)
    n_docs = len(documents)

    if doc_ids is None:
        doc_ids = np.arange(n_docs)

    # Get mapping of probability columns to topic IDs (excludes -1)
    topic_info = model.get_topic_info()
    kept_topics = topic_info[topic_info["Topic"] != -1]["Topic"].to_numpy()

    rows = []
    for i in range(n_docs):
        t = int(topics[i])

        if probs is None:
            # No probabilities available
            rows.append({
                "doc_local_index": i,
                "topic_id": t if t != -1 else None,
                "topic_prob": None,
                "was_outlier": False
            })
            continue

        # probs is shape (n_docs, n_kept_topics)
        dist = probs[i]  # Probability distribution over kept topics

        if t == -1:
            # Outlier handling
            if reassign_outliers and dist.size > 0:
                best_col = int(np.argmax(dist))
                best_topic = int(kept_topics[best_col])
                best_prob = float(dist[best_col])

                if best_prob >= min_reassign_prob:
                    # Reassign to best topic
                    rows.append({
                        "doc_local_index": i,
                        "topic_id": best_topic,
                        "topic_prob": best_prob,
                        "was_outlier": True
                    })
                else:
                    # Keep as outlier
                    rows.append({
                        "doc_local_index": i,
                        "topic_id": None,
                        "topic_prob": 0.0,
                        "was_outlier": True
                    })
            else:
                rows.append({
                    "doc_local_index": i,
                    "topic_id": None,
                    "topic_prob": 0.0,
                    "was_outlier": True
                })
        else:
            # Regular topic assignment
            if t in kept_topics:
                # Find probability for assigned topic
                prob = float(dist[np.where(kept_topics == t)[0][0]])
            else:
                # Fallback to max probability
                prob = float(np.max(dist))

            rows.append({
                "doc_local_index": i,
                "topic_id": t,
                "topic_prob": prob,
                "was_outlier": False
            })

    return pd.DataFrame(rows)


def export_bertopic_artifacts(
    model: BERTopic,
    doc_topics_df: pd.DataFrame,
    doc_index_df: pd.DataFrame,
    out_dir: Path
) -> Tuple[Path, Path, Path]:
    """
    Export BERTopic artifacts as SKOS construction contract (3 tables).

    CONTRACT TABLES FOR SKOS:
    A. topics.csv → Application collections
       - topic_id, topic_label, topic_size

    B. topic_terms.csv → Technique↔Application links with usage
       - topic_id, term, weight, usageRate, source
       - weight = c-TF-IDF score
       - usageRate = normalized weight (sums to 1 within topic)
       - source = "bertopic"

    C. doc_topics.csv → prevalence (optional but useful)
       - doc_id, topic_id, weight
       - weight = BERTopic probability

    Args:
        model: Trained BERTopic model
        doc_topics_df: Document topic assignments with probabilities
        doc_index_df: Document index with EID/DOI mapping
        out_dir: Output directory

    Returns:
        Tuple of (topics_path, topic_terms_path, doc_topics_path)
    """
    logger.info("\nExporting BERTopic artifacts for SKOS...")

    # 1. Topics table
    topic_info = model.get_topic_info().copy()
    topic_info["topic_label"] = topic_info["Name"].fillna("").astype(str)

    topics_df = topic_info[["Topic", "topic_label", "Count"]].rename(
        columns={"Topic": "topic_id", "Count": "topic_size"}
    )

    topics_path = out_dir / "topics.csv"
    topics_df.to_csv(topics_path, index=False)
    logger.info(f"   Saved topics: {topics_path.name}")

    # 2. Topic-term weights (c-TF-IDF)
    kept_topics = topic_info[topic_info["Topic"] != -1]["Topic"].to_numpy()
    ctfidf = model.c_tf_idf_
    vocab = np.array(model.vectorizer_model.get_feature_names_out())

    rows = []
    for row_i, tid in enumerate(kept_topics):
        # Extract weights for this topic (handle sparse matrix properly)
        row_data = ctfidf[row_i, :]
        if hasattr(row_data, 'toarray'):
            weights = row_data.toarray().ravel()
        elif hasattr(row_data, 'A1'):
            weights = row_data.A1
        else:
            weights = np.array(row_data).ravel()

        # Filter to positive weights
        mask = weights > 0
        terms = vocab[mask]
        vals = weights[mask]

        # Normalize within topic to get usageRate (sum to 1)
        norm_vals = vals / (vals.sum() if vals.sum() > 0 else 1.0)

        for term, w, nw in zip(terms, vals, norm_vals):
            rows.append({
                "topic_id": int(tid),
                "term": term,
                "weight": float(w),
                "usageRate": float(nw),
                "source": "bertopic"
            })

    topic_terms_df = pd.DataFrame(rows)
    topic_terms_path = out_dir / "topic_terms.csv"
    topic_terms_df.to_csv(topic_terms_path, index=False)
    logger.info(f"   Saved topic terms: {topic_terms_path.name} ({len(topic_terms_df):,} rows)")

    # 3. Document-topic assignments (for prevalence)
    doc_topics_merged = doc_topics_df.merge(doc_index_df[['doc_local_index', 'EID', 'DOI']], on='doc_local_index', how='left')

    # Create doc_id: prefer EID > DOI > doc_local_index
    doc_topics_merged['doc_id'] = doc_topics_merged.apply(
        lambda row: row['EID'] if pd.notna(row['EID']) and row['EID']
                    else row['DOI'] if pd.notna(row['DOI']) and row['DOI']
                    else f"doc_{row['doc_local_index']}",
        axis=1
    )

    doc_topics_export = doc_topics_merged[['doc_id', 'topic_id', 'topic_prob']].rename(
        columns={'topic_prob': 'weight'}
    )
    # Remove outliers (topic_id = None)
    doc_topics_export = doc_topics_export[doc_topics_export['topic_id'].notna()]

    doc_topics_path = out_dir / "doc_topics.csv"
    doc_topics_export.to_csv(doc_topics_path, index=False)
    logger.info(f"   Saved doc topics: {doc_topics_path.name} ({len(doc_topics_export):,} rows)")

    return topics_path, topic_terms_path, doc_topics_path


def extract_ai_terms_by_topic(
    topics_df: pd.DataFrame,
    doc_topics_df: pd.DataFrame,
    doc_index_df: pd.DataFrame,
    df_work: pd.DataFrame,
    keyphrase_stats: Dict
) -> pd.DataFrame:
    """
    Extract which AI terms appear in documents of each topic.
    Shows BOTH: application domains (topic words) AND AI techniques used (AI terms).

    Uses word boundary detection with regex to prevent false positives:
    - Standalone "AI" is case-sensitive (capital letters only)
    - All other terms case-insensitive with word boundaries
    - Prevents substring matches (e.g., "ai" in "obtain")

    Args:
        topics_df: Topic information DataFrame
        doc_topics_df: Document topic assignments
        doc_index_df: Document index with original indices
        df_work: Working DataFrame with unified text
        keyphrase_stats: Keyphrase detection statistics

    Returns:
        DataFrame with columns:
        - topic_id, topic_size, application_domain, domain_keywords
        - ai_terms_count, top_ai_terms, ai_term_doc_counts, ai_term_percentages
    """
    logger.info("Extracting AI terms by topic...")

    # Get all AI phrases from keyphrase detection
    ai_phrases = set(keyphrase_stats['semantic_phrases']['ai'])
    ai_phrases_both = set(keyphrase_stats['semantic_phrases']['both'])
    all_ai = ai_phrases.union(ai_phrases_both)
    phrase_index = _build_phrase_index(all_ai, uppercase_ai_only=True)

    logger.info(f"  Searching for {len(all_ai)} AI phrases across topics...")

    # Merge to get original indices (skip outliers)
    merged = doc_topics_df[doc_topics_df['topic_id'].notna()].merge(doc_index_df, on='doc_local_index')

    # Build topic → documents mapping
    topic_docs = defaultdict(list)
    for _, row in merged.iterrows():
        topic_id = row['topic_id']
        orig_idx = row['original_index']
        topic_docs[topic_id].append(orig_idx)

    # For each topic, find which AI phrases appear in its documents
    topic_ai_terms = {}
    ai_topic_progress = ProgressReporter(
        "AI terms by topic", sum(len(indices) for indices in topic_docs.values()), every=250
    )
    ai_topic_done = 0

    for topic_id in sorted(topic_docs.keys()):
        doc_indices = topic_docs[topic_id]
        term_counts = Counter()

        for orig_idx in doc_indices:
            if orig_idx >= len(df_work):
                continue

            row = df_work.iloc[orig_idx]
            # Use the unified text field we already built
            full_text_original = row.get("__Unified_Text__", "")

            for phrase in _phrase_counts(
                full_text_original,
                all_ai,
                uppercase_ai_only=True,
                phrase_index=phrase_index,
            ):
                term_counts[phrase] += 1
            ai_topic_done += 1
            ai_topic_progress.update(ai_topic_done, detail=f"topic={topic_id}")

        topic_ai_terms[topic_id] = term_counts

    # Build output rows
    rows = []
    for topic_id in sorted(topic_ai_terms.keys()):
        topic_size = len(topic_docs[topic_id])
        topic_row = topics_df[topics_df['topic_id'] == topic_id]

        if topic_row.empty:
            continue

        topic_row = topic_row.iloc[0]
        topic_words = topic_row['top_words_20']

        ai_terms = topic_ai_terms[topic_id]

        # Top 20 AI terms for this topic
        top_ai = ai_terms.most_common(20)
        ai_terms_str = ', '.join([term for term, count in top_ai])
        ai_counts_str = ', '.join([str(count) for term, count in top_ai])
        ai_pcts_str = ', '.join([f'{(count/topic_size)*100:.1f}%' for term, count in top_ai])

        rows.append({
            'topic_id': topic_id,
            'topic_size': topic_size,
            'application_domain': topic_words.split(', ')[0] if topic_words else 'unknown',
            'domain_keywords': topic_words,
            'ai_terms_count': len(ai_terms),
            'top_ai_terms': ai_terms_str,
            'ai_term_doc_counts': ai_counts_str,
            'ai_term_percentages': ai_pcts_str
        })

    df_ai_by_topic = pd.DataFrame(rows)

    logger.info(f"   Analyzed {len(df_ai_by_topic)} topics")
    if not df_ai_by_topic.empty:
        logger.info(f"  → Average AI terms per topic: {df_ai_by_topic['ai_terms_count'].mean():.1f}")
        logger.info(f"  → Range: {df_ai_by_topic['ai_terms_count'].min()} - {df_ai_by_topic['ai_terms_count'].max()} unique AI terms")

    return df_ai_by_topic


def extract_entrepreneurship_terms_by_topic(
    topics_df: pd.DataFrame,
    doc_topics_df: pd.DataFrame,
    doc_index_df: pd.DataFrame,
    df_work: pd.DataFrame,
    keyphrase_stats: Dict
) -> pd.DataFrame:
    """
    Extract which entrepreneurship terms appear in documents of each topic.
    Parallel to extract_ai_terms_by_topic for domain validation.

    Args:
        topics_df: Topic information DataFrame
        doc_topics_df: Document topic assignments
        doc_index_df: Document index with original indices
        df_work: Working DataFrame with unified text
        keyphrase_stats: Keyphrase detection statistics

    Returns:
        DataFrame with columns:
        - topic_id, ent_terms_count, top_ent_terms, ent_term_doc_counts, ent_term_percentages
    """
    logger.info("Extracting entrepreneurship terms by topic...")

    # Get all entrepreneurship phrases from keyphrase detection
    ent_phrases = set(keyphrase_stats['semantic_phrases']['entrepreneurship'])
    ent_phrases_both = set(keyphrase_stats['semantic_phrases']['both'])
    all_ent = ent_phrases.union(ent_phrases_both)
    phrase_index = _build_phrase_index(all_ent)

    logger.info(f"  Searching for {len(all_ent)} entrepreneurship phrases across topics...")

    # Merge to get original indices (skip outliers)
    merged = doc_topics_df[doc_topics_df['topic_id'].notna()].merge(doc_index_df, on='doc_local_index')

    # Build topic → documents mapping
    topic_docs = defaultdict(list)
    for _, row in merged.iterrows():
        topic_id = row['topic_id']
        orig_idx = row['original_index']
        topic_docs[topic_id].append(orig_idx)

    # For each topic, find which entrepreneurship phrases appear
    topic_ent_terms = {}
    ent_topic_progress = ProgressReporter(
        "Entrepreneurship terms by topic",
        sum(len(indices) for indices in topic_docs.values()),
        every=250,
    )
    ent_topic_done = 0

    for topic_id in sorted(topic_docs.keys()):
        doc_indices = topic_docs[topic_id]
        term_counts = Counter()

        for orig_idx in doc_indices:
            if orig_idx >= len(df_work):
                continue

            row = df_work.iloc[orig_idx]
            full_text = row.get("__Unified_Text__", "")
            for phrase in _phrase_counts(
                full_text, all_ent, phrase_index=phrase_index
            ):
                term_counts[phrase] += 1
            ent_topic_done += 1
            ent_topic_progress.update(ent_topic_done, detail=f"topic={topic_id}")

        topic_ent_terms[topic_id] = term_counts

    # Build output rows
    rows = []
    for topic_id in sorted(topic_ent_terms.keys()):
        topic_size = len(topic_docs[topic_id])
        ent_terms = topic_ent_terms[topic_id]

        # Top 20 entrepreneurship terms for this topic
        top_ent = ent_terms.most_common(20)
        ent_terms_str = ', '.join([term for term, count in top_ent])
        ent_counts_str = ', '.join([str(count) for term, count in top_ent])
        ent_pcts_str = ', '.join([f'{(count/topic_size)*100:.1f}%' for term, count in top_ent])

        rows.append({
            'topic_id': topic_id,
            'ent_terms_count': len(ent_terms),
            'top_ent_terms': ent_terms_str,
            'ent_term_doc_counts': ent_counts_str,
            'ent_term_percentages': ent_pcts_str
        })

    df_ent_by_topic = pd.DataFrame(rows)

    logger.info(f"   Analyzed {len(df_ent_by_topic)} topics")
    if not df_ent_by_topic.empty:
        logger.info(f"  → Average entrepreneurship terms per topic: {df_ent_by_topic['ent_terms_count'].mean():.1f}")
        logger.info(f"  → Range: {df_ent_by_topic['ent_terms_count'].min()} - {df_ent_by_topic['ent_terms_count'].max()} unique terms")

    return df_ent_by_topic


def extract_ai_terms_by_paper(
    doc_index_df: pd.DataFrame,
    df_work: pd.DataFrame,
    keyphrase_stats: Dict
) -> pd.DataFrame:
    """
    Extract AI terms for each paper (paper-level).

    NEW FEATURE for SKOS integration:
    - Returns paper-level AI keyword counts and lists
    - Enables downstream filtering in Stage 2B without re-extraction
    - Searches full unified text (not just title+abstract)

    Uses word boundary detection with regex to prevent false positives:
    - Standalone "AI" is case-sensitive (capital letters only)
    - All other terms case-insensitive with word boundaries

    Args:
        doc_index_df: Document index with EID/DOI mapping
        df_work: Working DataFrame with unified text
        keyphrase_stats: Keyphrase detection statistics

    Returns:
        DataFrame with columns:
        - EID, doc_local_index, original_index, Title
        - ai_keywords (dict), ai_keyword_count
    """
    logger.info("\nExtracting AI terms by paper...")

    # Get all AI phrases from keyphrase detection
    ai_phrases = set(keyphrase_stats['semantic_phrases']['ai'])
    ai_phrases_both = set(keyphrase_stats['semantic_phrases']['both'])
    all_ai = ai_phrases.union(ai_phrases_both)
    phrase_index = _build_phrase_index(all_ai, uppercase_ai_only=True)

    logger.info(f"  Searching for {len(all_ai)} AI phrases across {len(doc_index_df)} papers...")

    rows = []
    ai_paper_progress = ProgressReporter(
        "AI terms by paper", len(doc_index_df), every=250
    )
    for paper_number, (_, doc_row) in enumerate(doc_index_df.iterrows(), start=1):
        orig_idx = doc_row['original_index']
        doc_local_idx = doc_row['doc_local_index']

        if orig_idx >= len(df_work):
            ai_paper_progress.update(paper_number, detail="skipped invalid index")
            continue

        paper = df_work.iloc[orig_idx]
        full_text_original = paper.get("__Unified_Text__", "")
        title = doc_row.get("Title", paper.get("Title", "Unknown"))
        eid = doc_row.get("EID", "Unknown")  # Get EID from doc_index_df, not df_work

        ai_found = _phrase_counts(
            full_text_original,
            all_ai,
            uppercase_ai_only=True,
            phrase_index=phrase_index,
        )

        rows.append({
            'EID': eid,
            'doc_local_index': doc_local_idx,
            'original_index': orig_idx,
            'Title': title,
            'ai_keywords': dict(ai_found),
            'ai_keyword_count': len(ai_found)
        })
        ai_paper_progress.update(paper_number, detail=str(eid))

    df_ai_by_paper = pd.DataFrame(rows)

    logger.info(f"   Analyzed {len(df_ai_by_paper)} papers")
    if not df_ai_by_paper.empty:
        logger.info(f"  → Average AI terms per paper: {df_ai_by_paper['ai_keyword_count'].mean():.2f}")
        logger.info(f"  → Papers with 0 AI terms: {(df_ai_by_paper['ai_keyword_count'] == 0).sum()} ({(df_ai_by_paper['ai_keyword_count'] == 0).mean()*100:.1f}%)")
        logger.info(f"  → Papers with ≥2 AI terms: {(df_ai_by_paper['ai_keyword_count'] >= 2).sum()} ({(df_ai_by_paper['ai_keyword_count'] >= 2).mean()*100:.1f}%)")

    return df_ai_by_paper


def extract_entrepreneurship_terms_by_paper(
    doc_index_df: pd.DataFrame,
    df_work: pd.DataFrame,
    keyphrase_stats: Dict
) -> pd.DataFrame:
    """
    Extract entrepreneurship terms for each paper (paper-level).

    NEW FEATURE for SKOS integration:
    - Returns paper-level entrepreneurship keyword counts and lists
    - Enables downstream filtering in Stage 2B without re-extraction
    - Searches full unified text (not just title+abstract)

    Args:
        doc_index_df: Document index with EID/DOI mapping
        df_work: Working DataFrame with unified text
        keyphrase_stats: Keyphrase detection statistics

    Returns:
        DataFrame with columns:
        - EID, doc_local_index, original_index, Title
        - ent_keywords (dict), ent_keyword_count
    """
    logger.info("\nExtracting entrepreneurship terms by paper...")

    # Get all entrepreneurship phrases from keyphrase detection
    ent_phrases = set(keyphrase_stats['semantic_phrases']['entrepreneurship'])
    ent_phrases_both = set(keyphrase_stats['semantic_phrases']['both'])
    all_ent = ent_phrases.union(ent_phrases_both)
    phrase_index = _build_phrase_index(all_ent)

    logger.info(f"  Searching for {len(all_ent)} entrepreneurship phrases across {len(doc_index_df)} papers...")

    rows = []
    ent_paper_progress = ProgressReporter(
        "Entrepreneurship terms by paper", len(doc_index_df), every=250
    )
    for paper_number, (_, doc_row) in enumerate(doc_index_df.iterrows(), start=1):
        orig_idx = doc_row['original_index']
        doc_local_idx = doc_row['doc_local_index']

        if orig_idx >= len(df_work):
            ent_paper_progress.update(paper_number, detail="skipped invalid index")
            continue

        paper = df_work.iloc[orig_idx]
        full_text_original = paper.get("__Unified_Text__", "")
        title = doc_row.get("Title", paper.get("Title", "Unknown"))
        eid = doc_row.get("EID", "Unknown")  # Get EID from doc_index_df, not df_work

        ent_found = _phrase_counts(
            full_text_original, all_ent, phrase_index=phrase_index
        )

        rows.append({
            'EID': eid,
            'doc_local_index': doc_local_idx,
            'original_index': orig_idx,
            'Title': title,
            'ent_keywords': dict(ent_found),
            'ent_keyword_count': len(ent_found)
        })
        ent_paper_progress.update(paper_number, detail=str(eid))

    df_ent_by_paper = pd.DataFrame(rows)

    logger.info(f"   Analyzed {len(df_ent_by_paper)} papers")
    if not df_ent_by_paper.empty:
        logger.info(f"  → Average ent terms per paper: {df_ent_by_paper['ent_keyword_count'].mean():.2f}")
        logger.info(f"  → Papers with 0 ent terms: {(df_ent_by_paper['ent_keyword_count'] == 0).sum()} ({(df_ent_by_paper['ent_keyword_count'] == 0).mean()*100:.1f}%)")
        logger.info(f"  → Papers with ≥2 ent terms: {(df_ent_by_paper['ent_keyword_count'] >= 2).sum()} ({(df_ent_by_paper['ent_keyword_count'] >= 2).mean()*100:.1f}%)")

    return df_ent_by_paper
