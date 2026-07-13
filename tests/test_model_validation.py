"""Tests for model-validation statistics."""

from aecsp.analytics.model_validation import (
    multirater_summary,
    normalized_entropy,
    pairwise_with_bootstrap,
)


def test_pairwise_bootstrap_and_weighted_agreement():
    result = pairwise_with_bootstrap(
        ["a", "a", "b", ""], ["a", "b", "b", "a"],
        weights=[1, 10, 1, 1], repetitions=50, seed=7,
    )
    assert result["comparable"] == 3
    assert result["percent_agreement"] == 2 / 3
    assert result["weighted_percent_agreement"] == 2 / 12
    assert result["agreement_ci_low"] <= result["percent_agreement"] <= result["agreement_ci_high"]


def test_multirater_and_entropy():
    result = multirater_summary([["a", "a", "a"], ["a", "b", "b"]])
    assert result["comparable_units"] == 2
    assert result["unanimous_share"] == 0.5
    assert normalized_entropy(["a", "b"]) == 1.0
    assert normalized_entropy(["a", "a"]) == 0.0
