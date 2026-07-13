"""Tests for deterministic stratified reliability stability simulation."""

import pandas as pd

from aecsp.analytics.sampling_stability import (
    draw_stratified_sample,
    proportional_allocation,
    recommend_fraction,
    simulate_stability,
)


def test_proportional_allocation_is_exact_and_covers_strata():
    result = proportional_allocation(pd.Series({"a": 70, "b": 20, "c": 10}), 20)
    assert result.sum() == 20
    assert (result >= 1).all()
    assert (result <= pd.Series({"a": 70, "b": 20, "c": 10})).all()


def test_simulation_is_reproducible_and_reports_metrics():
    rows = []
    for index in range(60):
        rows.append(
            {
                "paper_id": str(index),
                "sampling_stratum": "a" if index < 30 else "b",
                "code_mini": "x" if index % 3 else "y",
                "code_nano": "x" if index % 4 else "y",
            }
        )
    frame = pd.DataFrame(rows)
    first = simulate_stability(frame, ["code"], fractions=(0.25,), replicates=10, seed=7)
    second = simulate_stability(frame, ["code"], fractions=(0.25,), replicates=10, seed=7)
    pd.testing.assert_frame_equal(first[0], second[0])
    assert set(first[1]["metric"]) == {"percent_agreement", "krippendorff_alpha"}
    assert set(first[3]["sampling_stratum"]) == {"a", "b"}


def test_recommendation_returns_smallest_complete_pass():
    summary = pd.DataFrame(
        {
            "fraction": [0.1, 0.1, 0.25, 0.25],
            "precision_passed": [True, False, True, True],
        }
    )
    assert recommend_fraction(summary) == 0.25


def test_target_population_controls_fraction_sample_size():
    frame = pd.DataFrame(
        {
            "paper_id": [str(index) for index in range(20)],
            "sampling_stratum": ["a"] * 20,
            "code_mini": ["x"] * 20,
            "code_nano": ["x"] * 20,
        }
    )
    _, summary, _, _ = simulate_stability(
        frame,
        ["code"],
        fractions=(0.10,),
        replicates=2,
        target_population_size=21,
    )
    assert set(summary["sample_size"]) == {3}


def test_stratified_draw_is_exact_reproducible_and_weighted():
    frame = pd.DataFrame(
        {
            "paper_id": [str(index) for index in range(100)],
            "sampling_stratum": ["a"] * 70 + ["b"] * 30,
        }
    )
    first, allocation = draw_stratified_sample(frame, 20, seed=9)
    second, _ = draw_stratified_sample(frame, 20, seed=9)
    assert first["paper_id"].tolist() == second["paper_id"].tolist()
    assert len(first) == 20
    assert first["paper_id"].nunique() == 20
    assert allocation["sample_n"].sum() == 20
    assert (first["sampling_weight"] > 0).all()
