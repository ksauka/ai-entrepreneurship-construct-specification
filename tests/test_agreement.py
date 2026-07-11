"""Tests for locked nominal agreement metrics."""

import pytest

from aecsp.analytics.agreement import (
    krippendorff_alpha_nominal,
    pairwise_percent_agreement,
)


def test_pairwise_agreement_excludes_missing_values():
    result = pairwise_percent_agreement(["a", "b", "", None], ["a", "c", "b", "a"])
    assert result.comparable == 2
    assert result.agreements == 1
    assert result.percent_agreement == 0.5


def test_pairwise_agreement_requires_alignment():
    with pytest.raises(ValueError):
        pairwise_percent_agreement(["a"], ["a", "b"])


def test_nominal_alpha_is_one_for_unanimous_ratings():
    assert krippendorff_alpha_nominal([["a", "a"], ["b", "b"]]) == 1.0


def test_nominal_alpha_detects_systematic_disagreement():
    alpha = krippendorff_alpha_nominal([["a", "b"], ["a", "b"]])
    assert alpha is not None
    assert alpha < 0


def test_nominal_alpha_returns_none_without_pairs():
    assert krippendorff_alpha_nominal([["a", None], ["b"]]) is None
