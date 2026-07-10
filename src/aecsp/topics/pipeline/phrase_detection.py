"""
phrase_detection.py

Hybrid phrase detection combining semantic (KeyBERT + YAKE) and statistical (gensim Phrases)
approaches with seed-guided filtering for AI and entrepreneurship term preservation.

KEY FEATURES:
- Semantic extraction: KeyBERT + YAKE for important multi-word terms
- Seed-guided filtering: 0.65 similarity threshold to AI/entrepreneurship seeds
- Statistical collocations: gensim Phrases (NPMI scoring)
- Preserves domain-relevant phrases: "machine_learning", "venture_capital", etc.
- Filters noise: Generic terms like "data collection", "sample size"

USAGE:
    from theory_elaboration.topic_modeling import detect_phrases

    bigram_model, trigram_model, phrase_docs, stats = detect_phrases(
        documents=documents,
        seed_model=sentence_transformer,
        seed_embeddings=seed_embeddings,
        ai_seeds=ai_seed_list,
        ent_seeds=ent_seed_list,
        all_seeds=combined_seed_list
    )
"""

from __future__ import annotations

import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
import yake
from keybert import KeyBERT
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from aecsp.progress import ProgressReporter

logger = logging.getLogger(__name__)


def custom_tokenizer_preserve_underscores(doc: str) -> List[str]:
    """
    Tokenize text while preserving underscores in multi-word phrases.

    This is critical for maintaining semantic units like:
    - "machine_learning" (not "machine" + "learning")
    - "venture_capital" (not "venture" + "capital")
    - "neural_networks" (not "neural" + "networks")

    Args:
        doc: Input text to tokenize

    Returns:
        List of cleaned tokens with underscores preserved
    """
    tokens = doc.lower().strip().split()
    cleaned_tokens: List[str] = []
    for token in tokens:
        cleaned = token.strip('.,;:!?()[]{}"\'-')
        if cleaned and len(cleaned) >= 2 and any(c.isalnum() or c == '_' for c in cleaned):
            cleaned_tokens.append(cleaned)
    return cleaned_tokens


def detect_phrases(
    documents: List[str],
    seed_model: Optional[SentenceTransformer] = None,
    seed_embeddings: Optional[np.ndarray] = None,
    ai_seeds: Optional[List[str]] = None,
    ent_seeds: Optional[List[str]] = None,
    all_seeds: Optional[List[str]] = None
) -> Tuple[Phrases, Phrases, List[str], Dict]:
    """
    HYBRID + SEED-GUIDED approach: KeyBERT + YAKE (semantic) + gensim Phrases (statistical).

    5-STEP PROCESS:
    1. Extract important keyphrases from ALL documents using KeyBERT + YAKE
    2. FILTER keyphrases using semantic similarity to seeds (threshold: 0.65)
    3. Protect filtered phrases during tokenization (replace spaces with underscores)
    4. Apply gensim Phrases for statistical collocations
    5. Merge both sets for comprehensive phrase coverage

    This preserves:
    - AI-relevant terms: "generative AI", "neural networks", "machine learning"
    - Entrepreneurship terms: "venture capital", "startup funding", "business model"
    - AI × Entrepreneurship: "crowdfunding success", "entrepreneurial ai adoption"
    - Drops generic noise: "quarterly earnings", "sample size", "data collection"

    Args:
        documents: List of document strings
        seed_model: SentenceTransformer for computing similarities
        seed_embeddings: Pre-computed embeddings for seed terms
        ai_seeds: List of AI seed terms
        ent_seeds: List of entrepreneurship seed terms
        all_seeds: Combined list of all seed terms

    Returns:
        Tuple of:
        - bigram_model: Trained gensim Phrases model for bigrams
        - trigram_model: Trained gensim Phrases model for trigrams
        - phrase_documents: List of phrase-enhanced documents
        - keyphrase_stats: Dictionary with domain-classified phrases and statistics
    """
    logger.info("=" * 80)
    logger.info("HYBRID + SEED-GUIDED PHRASE DETECTION")
    logger.info("KeyBERT + YAKE + Semantic Filtering + gensim Phrases")
    logger.info("=" * 80)

    # STEP 1: Extract semantic keyphrases with KeyBERT + YAKE
    logger.info("\n[1/5] Extracting semantic keyphrases with KeyBERT + YAKE...")
    logger.info(f"       Processing {len(documents):,} documents (est. 10-15 min)...")

    keybert_model = KeyBERT()
    yake_extractor = yake.KeywordExtractor(
        lan="en",
        n=3,  # Max trigrams (1-3 grams)
        dedupLim=0.9,
        windowsSize=2,
        top=10,
        features=None
    )

    important_phrases = set()
    document_progress = ProgressReporter(
        "Hybrid phrase extraction", len(documents), every=100
    )
    for idx, doc in enumerate(documents):
        # KeyBERT extraction (1-4 grams)
        try:
            kb_kw = keybert_model.extract_keywords(
                doc,
                keyphrase_ngram_range=(1, 4),
                stop_words='english',
                top_n=15,
                use_mmr=True,
                diversity=0.6
            )
            for phrase, score in kb_kw:
                if score > 0.3:  # Quality threshold
                    important_phrases.add(phrase.lower().strip())
        except:
            pass

        # YAKE extraction (statistical, multi-word only)
        try:
            yake_kw = yake_extractor.extract_keywords(doc)
            for phrase, score in yake_kw:
                if score < 0.1 and len(phrase.split()) >= 2:  # YAKE: lower is better
                    important_phrases.add(phrase.lower().strip())
        except:
            pass

        document_progress.update(
            idx + 1,
            detail=f"phrases={len(important_phrases):,}",
        )

    logger.info(f"        Extracted {len(important_phrases):,} unique semantic keyphrases (pre-filtering)")

    # STEP 2: Filter keyphrases using semantic similarity to seeds
    logger.info("\n[2/5] Filtering keyphrases by semantic similarity to seed terms...")

    if seed_embeddings is not None and len(all_seeds) > 0:
        filtered_phrases = set()
        domain_matches = {'ai': set(), 'entrepreneurship': set(), 'both': set(), 'unmatched': set()}

        # Encode extracted phrases
        phrase_list = list(important_phrases)
        phrase_embeddings = seed_model.encode(phrase_list, batch_size=32, show_progress_bar=False)

        # Compute similarities to all seeds
        similarities = cosine_similarity(phrase_embeddings, seed_embeddings)

        for i, phrase in enumerate(phrase_list):
            max_sim = similarities[i].max()

            if max_sim >= 0.65:  # Threshold from config
                filtered_phrases.add(phrase)
                best_match_idx = similarities[i].argmax()
                best_match_seed = all_seeds[best_match_idx]

                # Track domain (AI, Entrepreneurship, or Both)
                is_ai = best_match_seed in ai_seeds
                is_ent = best_match_seed in ent_seeds

                if is_ai and is_ent:
                    domain_matches['both'].add(phrase)
                elif is_ai:
                    domain_matches['ai'].add(phrase)
                elif is_ent:
                    domain_matches['entrepreneurship'].add(phrase)
            else:
                domain_matches['unmatched'].add(phrase)

        logger.info(f"        Filtered to {len(filtered_phrases):,} domain-relevant keyphrases")
        logger.info(f"       → AI domain: {len(domain_matches['ai']):,} phrases")
        logger.info(f"       → Entrepreneurship domain: {len(domain_matches['entrepreneurship']):,} phrases")
        logger.info(f"       → Both domains: {len(domain_matches['both']):,} phrases")
        logger.info(f"       → Dropped (noise): {len(domain_matches['unmatched']):,} phrases")
        logger.info(f"       Examples (AI): {list(domain_matches['ai'])[:5]}")
        logger.info(f"       Examples (Ent): {list(domain_matches['entrepreneurship'])[:5]}")
        logger.info(f"       Examples (Both): {list(domain_matches['both'])[:5]}")

        important_phrases = filtered_phrases
    else:
        logger.info(f"        No filtering applied (seeds not loaded)")
        logger.info(f"       → Keeping all {len(important_phrases):,} extracted phrases")
        domain_matches = {'ai': set(), 'entrepreneurship': set(), 'both': set(), 'unmatched': important_phrases}

    # STEP 3: Pre-process documents to protect semantic keyphrases
    logger.info("\n[3/5] Protecting semantic keyphrases (replace spaces → underscores)...")
    protected_docs = []
    for doc in documents:
        doc_lower = doc.lower()
        # Replace semantic phrases with underscore versions
        for phrase in important_phrases:
            if phrase in doc_lower:
                # Replace in original doc (preserving case for first occurrence)
                doc = doc.replace(phrase, phrase.replace(' ', '_'))
                doc = doc.replace(phrase.title(), phrase.replace(' ', '_'))
        protected_docs.append(doc)

    logger.info(f"        Protected {len(important_phrases):,} phrases in {len(protected_docs):,} documents")

    # STEP 4: Statistical phrase detection with gensim Phrases
    logger.info("\n[4/5] Detecting statistical collocations with gensim Phrases...")
    tokenized_docs = [custom_tokenizer_preserve_underscores(doc) for doc in protected_docs]

    # Train bigram detector
    bigram_model = Phrases(
        tokenized_docs,
        min_count=5,
        threshold=0.5,  # NPMI compatible (moderate threshold)
        connector_words=ENGLISH_CONNECTOR_WORDS,
        scoring='npmi'
    )

    # Apply bigrams
    bigram_docs = [bigram_model[doc] for doc in tokenized_docs]

    # Train trigram detector
    trigram_model = Phrases(
        bigram_docs,
        min_count=5,
        threshold=0.5,  # NPMI compatible
        connector_words=ENGLISH_CONNECTOR_WORDS,
        scoring='npmi'
    )

    # Apply both bigram and trigram
    phrase_docs = [trigram_model[bigram_model[doc]] for doc in tokenized_docs]

    # STEP 5: Final phrase documents
    logger.info("\n[5/5] Finalizing phrase-enhanced documents...")
    phrase_documents = [' '.join(doc) for doc in phrase_docs]

    # Log statistics
    bigram_phrases_dict = bigram_model.export_phrases()
    trigram_phrases_dict = trigram_model.export_phrases()

    gensim_bigrams = [p for p in bigram_phrases_dict.keys() if '_' in str(p)]
    gensim_trigrams = [p for p in trigram_phrases_dict.keys() if '_' in str(p)]

    logger.info("")
    logger.info("=" * 80)
    logger.info("PHRASE DETECTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Domain seeds loaded:                {len(all_seeds) if all_seeds else 0:,} (AI + Entrepreneurship)")
    logger.info(f"Semantic keyphrases (filtered):     {len(important_phrases):,}")
    if seed_embeddings is not None:
        logger.info(f"  → AI domain: {len(domain_matches['ai']):,}")
        logger.info(f"  → Entrepreneurship domain: {len(domain_matches['entrepreneurship']):,}")
        logger.info(f"  → Both domains: {len(domain_matches['both']):,}")
    logger.info(f"  Examples: {list(important_phrases)[:10]}")
    logger.info(f"\nStatistical bigrams (gensim):       {len(gensim_bigrams):,}")
    logger.info(f"  Examples: {gensim_bigrams[:10]}")
    logger.info(f"\nStatistical trigrams (gensim):      {len(gensim_trigrams):,}")
    logger.info(f"  Examples: {gensim_trigrams[:10]}")
    logger.info(f"\nTotal unique phrases preserved:     ~{len(important_phrases) + len(gensim_bigrams) + len(gensim_trigrams):,}")
    logger.info(f"Noise filtered out:                 ~{len(domain_matches['unmatched']) if seed_embeddings is not None else 0:,}")
    logger.info(f"\nSample phrase-enhanced document:")
    logger.info(f"  {' '.join(phrase_docs[0][:50])}...")
    logger.info("=" * 80)

    # Prepare keyphrase statistics for export
    keyphrase_stats = {
        'semantic_phrases': {
            'ai': sorted(list(domain_matches.get('ai', set()))),
            'entrepreneurship': sorted(list(domain_matches.get('entrepreneurship', set()))),
            'both': sorted(list(domain_matches.get('both', set()))),
            'unmatched': sorted(list(domain_matches.get('unmatched', set())))
        },
        'statistical_phrases': {
            'bigrams': sorted(gensim_bigrams),
            'trigrams': sorted(gensim_trigrams)
        },
        'summary': {
            'total_semantic': len(important_phrases),
            'total_statistical': len(gensim_bigrams) + len(gensim_trigrams),
            'total_noise_filtered': len(domain_matches.get('unmatched', set()))
        }
    }

    return bigram_model, trigram_model, phrase_documents, keyphrase_stats
