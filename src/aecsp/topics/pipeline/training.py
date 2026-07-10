"""
training.py

BERTopic model training with BERT embeddings, UMAP dimensionality reduction, and HDBSCAN clustering.

KEY IMPROVEMENTS OVER TOP2VEC:
- Better embeddings: BERT (all-MiniLM-L6-v2) > Doc2Vec for semantic quality
- More control: Explicit UMAP + HDBSCAN configuration
- Temporal support: Built-in topics_over_time() method
- Metadata support: Can analyze topics by groups (VOS clusters, journals, etc.)
- Reproducible: Fixed random seeds for deterministic results

USAGE:
    from theory_elaboration.topic_modeling import train_bertopic

    model = train_bertopic(
        documents=phrase_documents,
        embedding_model=sentence_transformer,
        nr_topics="auto",  # or specific number
        min_topic_size=15
    )
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Union

import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from .compat import patch_hdbscan_sklearn_compat

patch_hdbscan_sklearn_compat()

from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)


def train_bertopic(
    documents: List[str],
    embedding_model: SentenceTransformer,
    nr_topics: Union[str, int],
    min_topic_size: int,
    custom_tokenizer: callable = None,
    calculate_probabilities: bool = True,
    seed: int = 42,
    embeddings: Optional[np.ndarray] = None,
) -> BERTopic:
    """
    Train BERTopic with pre-computed BERT embeddings, UMAP dimensionality reduction, and HDBSCAN clustering.

    KEY IMPROVEMENTS OVER TOP2VEC:
    - Better embeddings: BERT (all-MiniLM-L6-v2) > Doc2Vec
    - More control: Explicit UMAP + HDBSCAN configuration
    - Temporal support: Built-in topics_over_time() method
    - Metadata support: Can analyze topics by groups (VOS clusters, journals, etc.)

    Args:
        documents: List of documents (phrase-enhanced recommended)
        embedding_model: SentenceTransformer model for encoding
        nr_topics: Number of topics ("auto" for automatic, or specific integer)
        min_topic_size: Minimum documents per topic (HDBSCAN min_cluster_size)
        custom_tokenizer: Tokenizer function (preserves underscores)
        calculate_probabilities: Whether to compute topic probabilities (slower but useful)
        seed: Random seed for reproducibility
        embeddings: Optional precomputed document embeddings. Reusing these is
            required when a grid search precedes final training.

    Returns:
        Trained BERTopic model
    """
    # Import custom tokenizer if not provided
    if custom_tokenizer is None:
        from .phrase_detection import custom_tokenizer_preserve_underscores
        custom_tokenizer = custom_tokenizer_preserve_underscores

    # Set random seeds for reproducibility
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = "0"

    logger.info("=" * 80)
    logger.info("TRAINING BERTOPIC MODEL")
    logger.info("=" * 80)
    logger.info(f"Documents: {len(documents):,}")
    logger.info(f"Embedding model: {embedding_model}")
    logger.info(f"Number of topics: {nr_topics}")
    logger.info(f"Min topic size: {min_topic_size}")
    logger.info(f"Calculate probabilities: {calculate_probabilities}")
    logger.info(f"Random seed: {seed}")

    # Step 1: Generate embeddings
    if embeddings is None:
        logger.info("\n[1/4] Generating BERT embeddings...")
        logger.info("       This may take 5-10 minutes depending on corpus size...")
        embeddings = embedding_model.encode(documents, show_progress_bar=True)
        logger.info(f"        Generated embeddings: shape {embeddings.shape}")
    else:
        if len(embeddings) != len(documents):
            raise ValueError("precomputed embeddings must align with documents")
        logger.info("\n[1/4] Reusing precomputed embeddings: shape %s", embeddings.shape)

    # Step 2: Configure UMAP for dimensionality reduction
    logger.info("\n[2/4] Configuring UMAP dimensionality reduction...")
    logger.info(f"       n_neighbors=15, n_components=5, metric='cosine'")
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=seed
    )

    # Step 3: Configure HDBSCAN for clustering
    logger.info("\n[3/4] Configuring HDBSCAN clustering...")
    logger.info(f"       min_cluster_size={min_topic_size}, cluster_selection_method='eom'")
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    # Step 4: Configure vectorizer to preserve underscored phrases
    logger.info("\n[4/4] Configuring CountVectorizer for topic representation...")
    logger.info(f"       ngram_range=(1, 3), min_df=2, preserves underscores")
    vectorizer_model = CountVectorizer(
        tokenizer=custom_tokenizer,
        stop_words='english',
        ngram_range=(1, 3),
        min_df=2
    )

    # Train BERTopic
    logger.info("\nTraining BERTopic (UMAP + HDBSCAN + c-TF-IDF)...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=nr_topics if nr_topics != "auto" else None,
        calculate_probabilities=calculate_probabilities,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(documents, embeddings)

    logger.info("=" * 80)
    logger.info("BERTOPIC TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Topics discovered: {len(set(topics)) - (1 if -1 in topics else 0)}")
    logger.info(f"Outliers (topic -1): {sum(1 for t in topics if t == -1)} ({sum(1 for t in topics if t == -1) / len(topics) * 100:.1f}%)")

    # Show top topics
    logger.info("\nTop 5 topics by size:")
    topic_info = topic_model.get_topic_info()
    for idx, row in topic_info[topic_info['Topic'] != -1].head(5).iterrows():
        topic_id = row['Topic']
        topic_size = row['Count']
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            words = ', '.join([w for w, s in topic_words[:5]])
            logger.info(f"  Topic {topic_id}: {topic_size:4d} docs - {words}")

    logger.info("=" * 80)

    return topic_model
