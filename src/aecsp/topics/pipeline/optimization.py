"""Evaluate BERTopic configurations and recommend topic parameters.

Inputs: documents, embeddings, candidate settings, and optional trained models.
Outputs: hierarchical or grid-search metrics, recommendations, and diagnostics.
"""

from __future__ import annotations

import logging
import os
import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Tuple, Optional

import numpy as np

from aecsp.progress import ProgressReporter

if TYPE_CHECKING:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def optimize_topic_count_hierarchical(
    model: BERTopic,
    documents: List[str],
    embeddings: np.ndarray,
    min_topics: int = 10,
    max_topics: int = 50,
    method: str = "distance_threshold"
) -> Tuple[int, Dict]:
    """
    Automatically find optimal number of topics using hierarchical merging.

    This is the PRINCIPLED approach analogous to Top2Vec's automatic detection:
    1. Start with fine-grained topics (model already trained with small min_topic_size)
    2. Build hierarchical topic tree using BERTopic's hierarchical_topics()
    3. Find optimal merge level using distance threshold or coherence

    Args:
        model: Trained BERTopic model (with many fine-grained topics)
        documents: Original documents
        embeddings: Document embeddings
        min_topics: Minimum acceptable topics (default: 10)
        max_topics: Maximum acceptable topics (default: 50)
        method: "distance_threshold" or "coherence_max"

    Returns:
        optimal_nr_topics: Recommended number of topics
        metrics: Dictionary of quality metrics at each level
    """
    logger.info("\n" + "=" * 80)
    logger.info("AUTOMATIC TOPIC OPTIMIZATION: Hierarchical Merging")
    logger.info("=" * 80)
    logger.info(f"Initial topics: {len(set(model.topics_)) - (1 if -1 in model.topics_ else 0)}")
    logger.info(f"Method: {method}")
    logger.info(f"Acceptable range: {min_topics} - {max_topics} topics")
    logger.info("\nComputing hierarchical topic structure...")

    # Build hierarchical topic tree
    hierarchical_topics = model.hierarchical_topics(documents)

    # Extract unique topic counts at each merge level
    # Start with initial topic count and decrease as topics merge
    initial_topics = len(set(model.topics_)) - (1 if -1 in model.topics_ else 0)
    merge_sequence = []

    for idx, row in hierarchical_topics.iterrows():
        topics_remaining = initial_topics - idx - 1  # Each row represents one merge
        distance = row['Distance']
        merge_sequence.append((topics_remaining, distance))

    # Filter to acceptable range
    valid_merges = [(n, d) for n, d in merge_sequence if min_topics <= n <= max_topics]

    if not valid_merges:
        logger.warning(f"No merge levels in range {min_topics}-{max_topics}, using midpoint")
        optimal_nr_topics = (min_topics + max_topics) // 2
        metrics = {"method": method, "warning": "no_valid_merges"}
        return optimal_nr_topics, metrics

    topic_counts, distances = zip(*valid_merges)

    logger.info(f"Hierarchical levels in range: {len(valid_merges)}")

    if method == "distance_threshold":
        # Find elbow: largest distance jump indicates natural clustering boundary
        distance_diffs = [distances[i+1] - distances[i] for i in range(len(distances)-1)]
        if distance_diffs:
            elbow_idx = np.argmax(distance_diffs)
            optimal_nr_topics = topic_counts[elbow_idx]

            logger.info(f"\n Distance-based optimization:")
            logger.info(f"   Largest distance jump at {topic_counts[elbow_idx]} topics")
            logger.info(f"   Distance: {distances[elbow_idx]:.4f} → {distances[elbow_idx+1]:.4f}")
            logger.info(f"   Jump size: {distance_diffs[elbow_idx]:.4f}")
        else:
            optimal_nr_topics = topic_counts[0]
            logger.info(f"\n Distance-based optimization: Using first valid level ({optimal_nr_topics} topics)")

    else:  # coherence_max
        from sklearn.metrics import silhouette_score

        logger.info("\nComputing silhouette scores at each level...")
        silhouette_scores = []

        for nr_topics in topic_counts[:10]:  # Sample up to 10 levels to save time
            # Reduce model to this topic count
            reduced_model = model
            try:
                reduced_model.reduce_topics(documents, nr_topics=nr_topics)
                reduced_topics = reduced_model.topics_

                # Compute silhouette score
                if len(set(reduced_topics)) > 1:
                    sil_score = silhouette_score(
                        embeddings[:len(documents)],
                        reduced_topics[:len(documents)],
                        sample_size=min(1000, len(documents))
                    )
                    silhouette_scores.append((nr_topics, sil_score))
                    logger.info(f"   {nr_topics} topics → silhouette {sil_score:.4f}")
            except Exception as e:
                logger.warning(f"   {nr_topics} topics → failed ({str(e)[:50]})")
                continue

        if silhouette_scores:
            optimal_nr_topics, best_score = max(silhouette_scores, key=lambda x: x[1])
            logger.info(f"\n Coherence-based optimization:")
            logger.info(f"   Best silhouette score: {best_score:.4f}")
            logger.info(f"   → Optimal topics: {optimal_nr_topics}")
        else:
            optimal_nr_topics = topic_counts[len(topic_counts)//2]  # Fallback to middle
            logger.warning(f"Coherence computation failed, using middle value: {optimal_nr_topics}")

    metrics = {
        "method": method,
        "initial_topics": initial_topics,
        "optimal_topics": optimal_nr_topics,
        "merge_levels_analyzed": len(valid_merges),
        "topic_range": f"{min_topics}-{max_topics}"
    }

    logger.info(f"\n Recommended topic count: {optimal_nr_topics}")
    logger.info("=" * 80)

    return optimal_nr_topics, metrics


def optimize_topic_count_grid_search(
    documents: List[str],
    embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    min_topic_sizes: Optional[List[int]] = None,
    seed: int = 42,
    target_topic_range: Optional[Tuple[int, int]] = None,
    plot_metrics: bool = True,
    out_dir: Optional[Path] = None,
    custom_tokenizer: Optional[callable] = None
) -> Tuple[int, Dict]:
    """
    Find optimal min_topic_size using grid search with multiple quality metrics.

    This is analogous to STM's searchK() function that tries different topic counts
    and evaluates held-out likelihood, residuals, semantic coherence, etc.

    Metrics computed:
    - Silhouette score (cluster separation in embedding space)
    - Topic diversity (unique top words across topics)
    - Outlier rate (lower is better, but not too low)
    - Average topic size variance (more balanced is better)

    Composite scoring (weights optimized for theory elaboration):
    - 35% outlier rate (minimize unassigned papers - CRITICAL)
    - 25% silhouette (cluster quality)
    - 25% diversity (topic distinctiveness)
    - 15% size CV (topic balance)

    Args:
        documents: List of documents
        embeddings: Document embeddings
        embedding_model: SentenceTransformer model
        min_topic_sizes: List of min_topic_size values to try
        seed: Random seed
        target_topic_range: Optional (min_topics, max_topics) review range. The
            grid is never forced into a range when this is omitted.
        plot_metrics: Whether to generate matplotlib plots
        out_dir: Output directory for plots
        custom_tokenizer: Optional tokenizer function (preserves underscores)

    Returns:
        optimal_min_topic_size: Best min_topic_size value
        results: Dictionary with metrics for each configuration
    """
    if min_topic_sizes is None:
        min_topic_sizes = [20, 30, 40, 50, 75, 100]

    if not min_topic_sizes or any(value < 2 for value in min_topic_sizes):
        raise ValueError("min_topic_sizes must contain integers >= 2")

    # Import custom tokenizer if not provided
    if custom_tokenizer is None:
        from .phrase_detection import custom_tokenizer_preserve_underscores
        custom_tokenizer = custom_tokenizer_preserve_underscores

    from .compat import patch_hdbscan_sklearn_compat

    patch_hdbscan_sklearn_compat()
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import silhouette_score
    from umap import UMAP

    logger.info("\n" + "=" * 80)
    logger.info("AUTOMATIC TOPIC OPTIMIZATION: Grid Search")
    logger.info("=" * 80)
    logger.info(f"Testing min_topic_size values: {min_topic_sizes}")
    logger.info(f"Documents: {len(documents)}")
    logger.info(
        "Target topic range: %s",
        f"{target_topic_range[0]}-{target_topic_range[1]}" if target_topic_range else "not forced",
    )
    logger.info("\nThis may take 5-10 minutes...\n")

    results = []
    grid_progress = ProgressReporter("Topic grid", len(min_topic_sizes), every=1)

    for configuration_number, min_size in enumerate(min_topic_sizes, start=1):
        logger.info(f"[{min_size}] Training with min_topic_size={min_size}...")

        # Train model
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=seed
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=min_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        vectorizer_model = CountVectorizer(
            tokenizer=custom_tokenizer,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2
        )

        temp_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=False,  # Faster
            verbose=False
        )

        topics, _ = temp_model.fit_transform(documents, embeddings)

        # Compute metrics
        n_topics = len(set(topics)) - (1 if -1 in topics else 0)
        n_outliers = sum(1 for t in topics if t == -1)
        outlier_rate = n_outliers / len(topics) if len(topics) > 0 else 0

        # Silhouette score (cluster quality)
        labels = np.asarray(topics)
        assigned_mask = labels != -1
        if n_topics > 1 and assigned_mask.sum() > n_topics:
            try:
                sil_score = silhouette_score(
                    embeddings[assigned_mask],
                    labels[assigned_mask],
                    sample_size=min(2000, int(assigned_mask.sum())),
                    random_state=seed,
                )
            except Exception:
                sil_score = 0.0
        else:
            sil_score = 0.0

        # Topic diversity (unique top words)
        topic_info = temp_model.get_topic_info()
        all_top_words = set()
        for topic_id in range(n_topics):
            topic_words = temp_model.get_topic(topic_id)
            if topic_words:
                all_top_words.update([w for w, _ in topic_words[:10]])
        diversity = len(all_top_words) / (n_topics * 10) if n_topics > 0 else 0.0

        # Topic size balance (coefficient of variation)
        topic_sizes = topic_info[topic_info['Topic'] != -1]['Count'].values
        size_cv = topic_sizes.std() / topic_sizes.mean() if len(topic_sizes) > 0 and topic_sizes.mean() > 0 else 0.0

        # Composite score (higher is better)
        # Normalize each metric to 0-1 range, then combine
        # - Silhouette: higher is better (already -1 to 1, shift to 0-1)
        # - Diversity: higher is better (already 0-1)
        # - Outlier rate: lower is better (invert: 1 - rate) - HEAVILY WEIGHTED for theory elaboration
        # - Size CV: lower is better (invert: 1 / (1 + cv))
        score = (
            (sil_score + 1) / 2 * 0.25 +  # 25% weight (cluster quality)
            diversity * 0.25 +               # 25% weight (topic distinctiveness)
            (1 - outlier_rate) * 0.35 +      # 35% weight (minimize unassigned papers - CRITICAL)
            (1 / (1 + size_cv)) * 0.15       # 15% weight (topic balance)
        )

        results.append({
            "min_topic_size": min_size,
            "n_topics": n_topics,
            "outlier_rate": outlier_rate,
            "silhouette": sil_score,
            "diversity": diversity,
            "size_cv": size_cv,
            "composite_score": score
        })

        logger.info(f"   Topics: {n_topics}, Outliers: {outlier_rate:.1%}, "
                   f"Silhouette: {sil_score:.3f}, Diversity: {diversity:.3f}, "
                   f"Score: {score:.3f}")
        grid_progress.update(
            configuration_number,
            detail=f"min_size={min_size}, topics={n_topics}, outliers={outlier_rate:.1%}",
        )

    # A range is a review aid, not an implicit methodological constraint.
    filtered_results = results
    if target_topic_range is not None:
        min_topics, max_topics = target_topic_range
        in_range = [r for r in results if min_topics <= r['n_topics'] <= max_topics]
        if in_range:
            filtered_results = in_range
            logger.info(
                "\nFiltered to %d/%d configurations in review range %d-%d topics",
                len(in_range), len(results), min_topics, max_topics,
            )
        else:
            logger.warning(
                "No configurations fall in review range %d-%d; recommendation uses the full grid",
                min_topics, max_topics,
            )

    # Plotly HTML is the guaranteed graph format (already required by BERTopic).
    if plot_metrics and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        from plotly import graph_objects as go
        from plotly.subplots import make_subplots

        min_sizes = [r['min_topic_size'] for r in results]
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Silhouette", "Topic diversity", "Outlier rate", "Topic-size CV"),
        )
        fig.add_trace(go.Scatter(x=min_sizes, y=[r['silhouette'] for r in results], mode="lines+markers"), row=1, col=1)
        fig.add_trace(go.Scatter(x=min_sizes, y=[r['diversity'] for r in results], mode="lines+markers"), row=1, col=2)
        fig.add_trace(go.Scatter(x=min_sizes, y=[r['outlier_rate'] for r in results], mode="lines+markers"), row=2, col=1)
        fig.add_trace(go.Scatter(x=min_sizes, y=[r['size_cv'] for r in results], mode="lines+markers"), row=2, col=2)
        fig.update_layout(title="BERTopic Grid Search Diagnostics", showlegend=False)
        fig.write_html(str(out_dir / "grid_search_metrics.html"), include_plotlyjs="cdn")

        fig = go.Figure(go.Scatter(
            x=min_sizes,
            y=[r['n_topics'] for r in results],
            mode="lines+markers+text",
            text=[str(r['n_topics']) for r in results],
            textposition="top center",
        ))
        fig.update_layout(
            title="Topic Count Sensitivity",
            xaxis_title="min_topic_size",
            yaxis_title="Topics discovered",
        )
        fig.write_html(str(out_dir / "topic_count_sensitivity.html"), include_plotlyjs="cdn")

        fig = go.Figure(go.Scatter(
            x=min_sizes,
            y=[r['composite_score'] for r in results],
            mode="lines+markers",
        ))
        fig.update_layout(
            title="Configuration Comparison (Recommendation Only)",
            xaxis_title="min_topic_size",
            yaxis_title="Diagnostic composite score",
        )
        fig.write_html(str(out_dir / "configuration_scores.html"), include_plotlyjs="cdn")

        # Optional static copies for manuscript workflows.
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('BERTopic Grid Search: Quality Metrics vs min_topic_size', fontsize=16, fontweight='bold')

            min_sizes = [r['min_topic_size'] for r in results]

            # Plot 1: Silhouette Score (cluster separation)
            ax1 = axes[0, 0]
            silhouettes = [r['silhouette'] for r in results]
            ax1.plot(min_sizes, silhouettes, 'o-', color='#2ecc71', linewidth=2, markersize=8)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax1.set_xlabel('min_topic_size', fontsize=12)
            ax1.set_ylabel('Silhouette Score', fontsize=12)
            ax1.set_title('Cluster Separation (Higher = Better)', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            if silhouettes:
                ax1.set_ylim([min(silhouettes) - 0.1, max(silhouettes) + 0.1])

            # Plot 2: Topic Diversity (unique top words)
            ax2 = axes[0, 1]
            diversities = [r['diversity'] for r in results]
            ax2.plot(min_sizes, diversities, 'o-', color='#3498db', linewidth=2, markersize=8)
            ax2.set_xlabel('min_topic_size', fontsize=12)
            ax2.set_ylabel('Topic Diversity', fontsize=12)
            ax2.set_title('Topic Distinctiveness (Higher = Better)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 1])

            # Plot 3: Outlier Rate (lower = better, but not 0)
            ax3 = axes[1, 0]
            outliers = [r['outlier_rate'] * 100 for r in results]
            ax3.plot(min_sizes, outliers, 'o-', color='#e74c3c', linewidth=2, markersize=8)
            ax3.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='30% threshold')
            ax3.set_xlabel('min_topic_size', fontsize=12)
            ax3.set_ylabel('Outlier Rate (%)', fontsize=12)
            ax3.set_title('Papers Without Topic (Lower = Better)', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend()

            # Plot 4: Size Coefficient of Variation (balance)
            ax4 = axes[1, 1]
            size_cvs = [r['size_cv'] for r in results]
            ax4.plot(min_sizes, size_cvs, 'o-', color='#9b59b6', linewidth=2, markersize=8)
            ax4.set_xlabel('min_topic_size', fontsize=12)
            ax4.set_ylabel('Size CV', fontsize=12)
            ax4.set_title('Topic Size Balance (Lower = Better)', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = out_dir / 'grid_search_metrics.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"\n Saved metrics plot: {plot_path.name}")

            fig, ax = plt.subplots(figsize=(9, 5))
            topic_counts = [r['n_topics'] for r in results]
            ax.plot(min_sizes, topic_counts, 'o-', color='#2c3e50', linewidth=2)
            for x, y in zip(min_sizes, topic_counts):
                ax.annotate(str(y), (x, y), xytext=(0, 7), textcoords='offset points', ha='center')
            if target_topic_range is not None:
                ax.axhspan(target_topic_range[0], target_topic_range[1], color='#3498db', alpha=0.1)
            ax.set_xlabel('min_topic_size')
            ax.set_ylabel('Topics discovered')
            ax.set_title('Topic Count Sensitivity')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / 'topic_count_sensitivity.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(9, 5))
            scores = [r['composite_score'] for r in results]
            ax.plot(min_sizes, scores, 'o-', color='#8e44ad', linewidth=2)
            ax.set_xlabel('min_topic_size')
            ax.set_ylabel('Diagnostic composite score')
            ax.set_title('Configuration Comparison (Recommendation Only)')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / 'configuration_scores.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            logger.warning(f"\n  Could not create plots: {str(e)}")

    # Find optimal configuration using composite score
    best_idx = max(range(len(filtered_results)), key=lambda i: filtered_results[i]["composite_score"])
    optimal_min_topic_size = filtered_results[best_idx]["min_topic_size"]

    logger.info(f"\n OPTIMAL SELECTION (Highest Composite Score):")
    logger.info(f"   Selected: min_topic_size={optimal_min_topic_size}")
    logger.info(f"   Composite score: {filtered_results[best_idx]['composite_score']:.3f}")
    logger.info(f"   → Silhouette: {filtered_results[best_idx]['silhouette']:.3f} (25% weight)")
    logger.info(f"   → Diversity: {filtered_results[best_idx]['diversity']:.3f} (25% weight)")
    logger.info(f"   → Outlier rate: {filtered_results[best_idx]['outlier_rate']:.1%} (35% weight - prioritized)")
    logger.info(f"   → Size CV: {filtered_results[best_idx]['size_cv']:.3f} (15% weight)")

    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 80)
    logger.info(f" Optimal min_topic_size: {optimal_min_topic_size}")
    logger.info(f"  → Topics: {filtered_results[best_idx]['n_topics']}")
    logger.info(f"  → Outliers: {filtered_results[best_idx]['outlier_rate']:.1%}")
    logger.info(f"  → Silhouette: {filtered_results[best_idx]['silhouette']:.3f}")
    logger.info(f"  → Diversity: {filtered_results[best_idx]['diversity']:.3f}")
    logger.info(f"  → Composite score: {filtered_results[best_idx]['composite_score']:.3f}")
    if target_topic_range is not None:
        logger.info(f"  → Review range: {target_topic_range[0]}-{target_topic_range[1]} topics")
    logger.info("\nAll tested configurations:")
    for r in results:
        marker = " ← BEST" if r["min_topic_size"] == optimal_min_topic_size else ""
        logger.info(f"  min_topic_size={r['min_topic_size']:3d}: "
                   f"{r['n_topics']:2d} topics, "
                   f"outliers={r['outlier_rate']:.1%}, "
                   f"score={r['composite_score']:.3f}{marker}")
    logger.info("=" * 80)

    payload = {
        "selection_status": "recommendation_requires_human_approval",
        "grid_search": results,
        "eligible": filtered_results,
        "recommended": filtered_results[best_idx],
        "target_topic_range": list(target_topic_range) if target_topic_range else None,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "grid_search_results.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        import pandas as pd
        pd.DataFrame(results).to_csv(out_dir / "grid_search_results.csv", index=False)
    return optimal_min_topic_size, payload


def export_model_diagnostics(model: BERTopic, out_dir: Path) -> dict[str, str]:
    """Write inspectable plots for an approved final BERTopic model.

    Interactive BERTopic plots are HTML so labels and relationships remain
    readable for large models. The topic-size distribution is also exported as
    a static PNG suitable for reports.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}

    topic_info = model.get_topic_info()
    sizes = topic_info.loc[topic_info["Topic"] != -1, "Count"].sort_values(
        ascending=False
    )
    try:
        from plotly import graph_objects as go

        path = out_dir / "topic_size_distribution.html"
        fig = go.Figure(go.Bar(x=list(range(len(sizes))), y=sizes.tolist()))
        fig.update_layout(
            title="Final Topic Size Distribution",
            xaxis_title="Topics ranked by size",
            yaxis_title="Papers",
        )
        fig.write_html(str(path), include_plotlyjs="cdn")
        status["topic_size_distribution"] = str(path)
    except Exception as exc:
        status["topic_size_distribution_error"] = str(exc)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(range(len(sizes)), sizes, color="#3498db")
        ax.set_xlabel("Topics ranked by size")
        ax.set_ylabel("Papers")
        ax.set_title("Final Topic Size Distribution")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = out_dir / "topic_size_distribution.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        status["topic_size_distribution_png"] = str(path)
    except Exception as exc:
        status["topic_size_distribution_png_error"] = str(exc)

    interactive = {
        "intertopic_distance": lambda: model.visualize_topics(),
        "topic_hierarchy": lambda: model.visualize_hierarchy(),
        "topic_terms": lambda: model.visualize_barchart(top_n_topics=50),
    }
    for name, build in interactive.items():
        try:
            path = out_dir / f"{name}.html"
            build().write_html(str(path), include_plotlyjs="cdn")
            status[name] = str(path)
        except Exception as exc:
            status[f"{name}_error"] = str(exc)

    with open(out_dir / "diagnostics_status.json", "w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2)
    return status
