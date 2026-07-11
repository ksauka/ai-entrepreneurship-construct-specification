"""Filter and summarize BERTopic results for downstream analysis.

Inputs: topic assignments, document indexes, and paper-level domain terms.
Outputs: assigned-paper datasets, filtered datasets, and topic statistics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def filter_assigned_papers(
    doc_topics_df: pd.DataFrame,
    doc_index_df: pd.DataFrame,
    ai_terms_df: pd.DataFrame,
    ent_terms_df: pd.DataFrame,
    out_dir: Path,
    min_ai_terms: int = 2,
    min_ent_terms: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter BERTopic outputs to papers with valid topic assignments.

    Excludes outliers (topic == -1 or None) and optionally filters to papers
    with sufficient AI and entrepreneurship keyword coverage for Stage 2B.

    Args:
        doc_topics_df: Document topic assignments from assign_document_topics()
        doc_index_df: Document index with EID/DOI mapping
        ai_terms_df: AI terms by paper from extract_ai_terms_by_paper()
        ent_terms_df: Entrepreneurship terms by paper from extract_entrepreneurship_terms_by_paper()
        out_dir: Output directory for filtered datasets
        min_ai_terms: Minimum AI terms for Stage 2B readiness (default: 2)
        min_ent_terms: Minimum entrepreneurship terms for Stage 2B readiness (default: 2)

    Returns:
        Tuple of (assigned_papers, stage2b_ready_papers)
        - assigned_papers: All papers with valid topic assignments
        - stage2b_ready_papers: Papers meeting Stage 2B criteria (≥2 AI + ≥2 ent terms)
    """
    logger.info("\n" + "=" * 80)
    logger.info("POST-PROCESSING: Filter Assigned Papers")
    logger.info("=" * 80)

    # Create output directory
    filtered_dir = out_dir / "filtered"
    filtered_dir.mkdir(exist_ok=True, parents=True)

    # Original counts
    total_papers = len(doc_topics_df)
    outliers = len(doc_topics_df[doc_topics_df['topic_id'].isna()])
    assigned = total_papers - outliers

    logger.info(f"\nORIGINAL DATA:")
    logger.info(f"  Total papers: {total_papers:,}")
    logger.info(f"  → Assigned to topics: {assigned:,} ({assigned/total_papers*100:.1f}%)")
    logger.info(f"  → Outliers (noise): {outliers:,} ({outliers/total_papers*100:.1f}%)")

    # Filter: Only papers with valid topic assignments
    assigned_papers = doc_topics_df[doc_topics_df['topic_id'].notna()].copy()

    logger.info(f"\nFILTERING: Papers with Valid Topic Assignments")

    # Merge with document index for metadata
    assigned_with_meta = assigned_papers.merge(doc_index_df, on='doc_local_index', how='left')

    # Merge with AI terms
    assigned_with_ai = assigned_with_meta.merge(
        ai_terms_df[['doc_local_index', 'ai_keywords', 'ai_keyword_count']],
        on='doc_local_index',
        how='left'
    )

    # Merge with entrepreneurship terms
    assigned_full = assigned_with_ai.merge(
        ent_terms_df[['doc_local_index', 'ent_keywords', 'ent_keyword_count']],
        on='doc_local_index',
        how='left'
    )

    # Fill NaN keyword counts with 0
    assigned_full['ai_keyword_count'] = assigned_full['ai_keyword_count'].fillna(0).astype(int)
    assigned_full['ent_keyword_count'] = assigned_full['ent_keyword_count'].fillna(0).astype(int)

    # Statistics
    logger.info(f"  Filtered papers: {len(assigned_full):,}")
    logger.info(f"\nKeyword coverage:")
    logger.info(f"  → Papers with ≥1 AI term: {(assigned_full['ai_keyword_count'] >= 1).sum():,}")
    logger.info(f"  → Papers with ≥{min_ai_terms} AI terms: {(assigned_full['ai_keyword_count'] >= min_ai_terms).sum():,}")
    logger.info(f"  → Papers with ≥1 ent term: {(assigned_full['ent_keyword_count'] >= 1).sum():,}")
    logger.info(f"  → Papers with ≥{min_ent_terms} ent terms: {(assigned_full['ent_keyword_count'] >= min_ent_terms).sum():,}")

    # Double-filter: ≥min_ai_terms AI terms AND ≥min_ent_terms ent terms (Stage 2B criterion)
    stage2b_ready = assigned_full[
        (assigned_full['ai_keyword_count'] >= min_ai_terms) &
        (assigned_full['ent_keyword_count'] >= min_ent_terms)
    ].copy()

    logger.info(f"\nSTAGE 2B READY (≥{min_ai_terms} AI + ≥{min_ent_terms} ent terms):")
    logger.info(f"  → Papers: {len(stage2b_ready):,} ({len(stage2b_ready)/len(assigned_full)*100:.1f}% of assigned)")
    logger.info(f"  → Topics covered: {stage2b_ready['topic_id'].nunique()}")

    # Save outputs
    logger.info(f"\n{'='*80}")
    logger.info(f"SAVING FILTERED DATASETS")
    logger.info(f"{'='*80}")

    # 1. All assigned papers with full metadata
    out_path = filtered_dir / "papers_with_topics.csv"
    assigned_full.to_csv(out_path, index=False)
    logger.info(f" Saved: {out_path.name} ({len(assigned_full):,} papers)")

    # 2. Stage 2B ready papers (≥min_ai_terms AI + ≥min_ent_terms ent terms)
    out_path = filtered_dir / "papers_stage2b_ready.csv"
    stage2b_ready.to_csv(out_path, index=False)
    logger.info(f" Saved: {out_path.name} ({len(stage2b_ready):,} papers)")

    # 3. Topic distribution
    topic_dist = assigned_full.groupby('topic_id').agg({
        'doc_local_index': 'count',
        'ai_keyword_count': 'mean',
        'ent_keyword_count': 'mean'
    }).rename(columns={
        'doc_local_index': 'paper_count',
        'ai_keyword_count': 'avg_ai_terms',
        'ent_keyword_count': 'avg_ent_terms'
    }).reset_index()

    out_path = filtered_dir / "topic_distribution.csv"
    topic_dist.to_csv(out_path, index=False)
    logger.info(f" Saved: {out_path.name} ({len(topic_dist)} topics)")

    # 4. Summary statistics
    summary = {
        'total_papers_input': int(total_papers),
        'outliers_excluded': int(outliers),
        'papers_with_topics': int(len(assigned_full)),
        'papers_stage2b_ready': int(len(stage2b_ready)),
        'topics_discovered': int(assigned_full['topic_id'].nunique()),
        'avg_ai_terms_per_paper': float(assigned_full['ai_keyword_count'].mean()),
        'avg_ent_terms_per_paper': float(assigned_full['ent_keyword_count'].mean()),
        'coverage_rate': float(len(assigned_full) / total_papers),
        'stage2b_readiness_rate': float(len(stage2b_ready) / len(assigned_full)),
        'filtering_criteria': {
            'min_ai_terms': min_ai_terms,
            'min_ent_terms': min_ent_terms
        }
    }

    out_path = filtered_dir / "filtering_summary.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f" Saved: {out_path.name}")

    logger.info(f"\n{'='*80}")
    logger.info(f"Output directory: {filtered_dir}")
    logger.info(f"{'='*80}\n")

    return assigned_full, stage2b_ready


def load_and_filter_from_disk(
    run_dir: Path,
    output_dir: Path = None,
    min_ai_terms: int = 2,
    min_ent_terms: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load BERTopic outputs from disk and apply filtering.

    Convenience function for post-hoc filtering of saved BERTopic runs.

    Args:
        run_dir: Path to stage2a_modeling directory containing BERTopic outputs
        output_dir: Optional output directory (defaults to run_dir/filtered)
        min_ai_terms: Minimum AI terms for Stage 2B readiness (default: 2)
        min_ent_terms: Minimum entrepreneurship terms for Stage 2B readiness (default: 2)

    Returns:
        Tuple of (assigned_papers, stage2b_ready_papers)
    """
    logger.info(f"Loading data from: {run_dir}")

    # Load core files
    doc_topics = pd.read_csv(run_dir / "document_topics.csv")
    doc_index = pd.read_csv(run_dir / "documents_index.csv")
    ai_terms = pd.read_csv(run_dir / "ai_terms_by_paper.csv")
    ent_terms = pd.read_csv(run_dir / "ent_terms_by_paper.csv")

    # Parse ai_keywords and ent_keywords if they're stored as strings
    import ast
    if 'ai_keywords' in ai_terms.columns and ai_terms['ai_keywords'].dtype == 'object':
        ai_terms['ai_keywords'] = ai_terms['ai_keywords'].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else {}
        )
    if 'ent_keywords' in ent_terms.columns and ent_terms['ent_keywords'].dtype == 'object':
        ent_terms['ent_keywords'] = ent_terms['ent_keywords'].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else {}
        )

    if output_dir is None:
        output_dir = run_dir

    return filter_assigned_papers(
        doc_topics_df=doc_topics,
        doc_index_df=doc_index,
        ai_terms_df=ai_terms,
        ent_terms_df=ent_terms,
        out_dir=output_dir,
        min_ai_terms=min_ai_terms,
        min_ent_terms=min_ent_terms
    )
