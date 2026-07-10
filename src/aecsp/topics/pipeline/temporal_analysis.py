"""
temporal_analysis.py

Temporal and VOS cluster analysis of topic evolution.

KEY FEATURES:
- Topics over time: Track emergence, growth, and decline (2000-2024)
- VOS community analysis: Topics per bibliometric cluster
- Replaces STM year covariate with BERTopic's built-in temporal support

USAGE:
    from theory_elaboration.topic_modeling import analyze_topics_over_time

    topics_over_time_df = analyze_topics_over_time(
        model=trained_model,
        documents=phrase_documents,
        doc_index_df=doc_index_df,
        df_work=df_work
    )
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from bertopic import BERTopic

logger = logging.getLogger(__name__)


def analyze_topics_over_time(
    model: BERTopic,
    documents: list[str],
    doc_index_df: pd.DataFrame,
    df_work: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """
    Analyze topic evolution over time using BERTopic's built-in temporal analysis.

    NEW FEATURE (replaces STM's year covariate):
    - Shows which topics emerge, grow, or decline over time
    - Tracks topic trends 2000-2024
    - Enables temporal theory elaboration

    Args:
        model: Trained BERTopic model
        documents: List of documents (phrase-enhanced)
        doc_index_df: Document index with doc_local_index → original_index mapping
        df_work: Working DataFrame with __YEAR__ column

    Returns:
        DataFrame with temporal analysis results, or None if insufficient data
        Columns: Timestamp, Topic, Words, Frequency
    """
    logger.info("\n" + "=" * 80)
    logger.info("TEMPORAL ANALYSIS: Topics Over Time")
    logger.info("=" * 80)

    # Extract year timestamps from doc_index
    timestamps = []
    for idx in range(len(documents)):
        doc_idx_row = doc_index_df[doc_index_df['doc_local_index'] == idx]
        if doc_idx_row.empty:
            timestamps.append("Unknown")
            continue

        orig_idx = doc_idx_row.iloc[0]['original_index']
        year = df_work.iloc[orig_idx].get("__YEAR__", "Unknown")
        timestamps.append(str(year) if year else "Unknown")

    # Count valid years
    valid_years = [t for t in timestamps if t != "Unknown" and t.isdigit()]
    logger.info(f"Documents with valid years: {len(valid_years):,} / {len(documents):,}")

    if len(valid_years) < 10:
        logger.warning("   Insufficient temporal data for analysis (< 10 docs with years)")
        return None

    logger.info(f"Year range: {min(valid_years)} - {max(valid_years)}")
    logger.info("Computing topics over time...")

    try:
        topics_over_time = model.topics_over_time(documents, timestamps, nr_bins=20)

        logger.info(f"   Temporal analysis complete")
        logger.info(f"  → Time periods: {topics_over_time['Timestamp'].nunique()}")
        logger.info(f"  → Topics tracked: {topics_over_time['Topic'].nunique()}")
        logger.info("=" * 80)

        return topics_over_time
    except Exception as e:
        logger.warning(f"   Temporal analysis failed: {e}")
        return None


def analyze_topics_by_vos(
    topics_df: pd.DataFrame,
    doc_topics_df: pd.DataFrame,
    doc_index_df: pd.DataFrame,
    df_work: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """
    Analyze topic distribution across VOS bibliometric communities.

    Shows which topics are concentrated in which bibliometric clusters,
    helping identify disciplinary boundaries and cross-community topics.

    Args:
        topics_df: Topic information DataFrame
        doc_topics_df: Document topic assignments
        doc_index_df: Document index with original indices
        df_work: Working DataFrame with VOS_cluster column

    Returns:
        DataFrame with VOS cluster analysis, or None if no VOS data
        Columns: VOS_cluster, topic_id, doc_count, percentage
    """
    logger.info("\n" + "=" * 80)
    logger.info("VOS CLUSTER ANALYSIS: Topics per Bibliometric Community")
    logger.info("=" * 80)

    # Check if VOS cluster data exists
    if 'VOS_cluster' not in df_work.columns:
        logger.warning("   No VOS_cluster column found in data")
        return None

    # Merge to get VOS clusters for each document
    merged = doc_topics_df[doc_topics_df['topic_id'].notna()].merge(
        doc_index_df[['doc_local_index', 'original_index']],
        on='doc_local_index'
    )

    # Add VOS cluster information
    vos_clusters = []
    for _, row in merged.iterrows():
        orig_idx = row['original_index']
        if orig_idx >= len(df_work):
            vos_clusters.append("Unknown")
            continue
        vos_cluster = df_work.iloc[orig_idx].get("VOS_cluster", "Unknown")
        vos_clusters.append(str(vos_cluster) if pd.notna(vos_cluster) else "Unknown")

    merged['VOS_cluster'] = vos_clusters

    # Count valid VOS clusters
    valid_vos = merged[merged['VOS_cluster'] != "Unknown"]
    logger.info(f"Documents with VOS clusters: {len(valid_vos):,} / {len(merged):,}")

    if len(valid_vos) < 10:
        logger.warning("   Insufficient VOS data for analysis (< 10 docs with clusters)")
        return None

    # Group by VOS cluster and topic
    vos_topic_counts = valid_vos.groupby(['VOS_cluster', 'topic_id']).size().reset_index(name='doc_count')

    # Calculate percentage within each VOS cluster
    cluster_totals = valid_vos.groupby('VOS_cluster').size().to_dict()
    vos_topic_counts['percentage'] = vos_topic_counts.apply(
        lambda row: (row['doc_count'] / cluster_totals[row['VOS_cluster']]) * 100,
        axis=1
    )

    # Sort by VOS cluster and doc count
    vos_topic_counts = vos_topic_counts.sort_values(['VOS_cluster', 'doc_count'], ascending=[True, False])

    logger.info(f"   VOS cluster analysis complete")
    logger.info(f"  → VOS clusters: {vos_topic_counts['VOS_cluster'].nunique()}")
    logger.info(f"  → Topics analyzed: {vos_topic_counts['topic_id'].nunique()}")

    # Show top topics per cluster
    logger.info("\n  Top 3 topics per VOS cluster:")
    for cluster in sorted(vos_topic_counts['VOS_cluster'].unique()):
        cluster_data = vos_topic_counts[vos_topic_counts['VOS_cluster'] == cluster].head(3)
        logger.info(f"\n  Cluster {cluster}:")
        for _, row in cluster_data.iterrows():
            logger.info(f"    Topic {int(row['topic_id']):2d}: {int(row['doc_count']):3d} docs ({row['percentage']:.1f}%)")

    logger.info("=" * 80)

    return vos_topic_counts
