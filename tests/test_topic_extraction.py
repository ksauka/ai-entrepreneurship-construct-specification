"""Regression tests for domain-term extraction."""

from time import perf_counter

from aecsp.topics.pipeline.extraction import _build_phrase_index, _phrase_counts


def test_phrase_counts_supports_spaces_underscores_and_overlaps():
    phrases = {"machine_learning", "learning"}
    counts = _phrase_counts(
        "Machine learning improves learning; machine_learning scales.", phrases
    )

    assert counts == {"machine_learning": 2, "learning": 3}


def test_phrase_counts_requires_uppercase_for_standalone_ai():
    phrases = {"ai", "explainable ai"}
    counts = _phrase_counts(
        "AI supports founders, but ai and explainable AI differ.",
        phrases,
        uppercase_ai_only=True,
    )

    assert counts == {"ai": 2, "explainable ai": 1}


def test_precompiled_phrase_index_avoids_phrase_cartesian_product():
    phrases = {f"synthetic phrase {index}" for index in range(10_000)}
    phrases.add("machine learning")
    index = _build_phrase_index(phrases)

    started = perf_counter()
    for _ in range(100):
        counts = _phrase_counts(
            "machine learning enables new ventures", phrases, phrase_index=index
        )
    elapsed = perf_counter() - started

    assert counts == {"machine learning": 1}
    assert elapsed < 0.5
