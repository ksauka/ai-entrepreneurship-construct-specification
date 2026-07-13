"""Evaluate reliability-estimate stability under stratified probability subsets.

Inputs: aligned paper-level ratings and corpus metadata. Outputs: repeated-
sampling estimates, precision summaries, rare-category coverage, and a
deterministic sample-size recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StabilityThresholds:
    agreement_bias: float = 0.01
    agreement_width: float = 0.05
    alpha_bias: float = 0.02
    alpha_width: float = 0.10


def add_sampling_strata(corpus: pd.DataFrame) -> pd.DataFrame:
    """Create mutually exclusive, model-independent sampling strata."""

    frame = corpus.copy()
    years = pd.to_numeric(frame["Year"], errors="coerce")
    frame["sampling_era"] = pd.cut(
        years,
        bins=[-np.inf, 2015, 2020, np.inf],
        labels=["through_2015", "2016_2020", "2021_plus"],
    ).astype(str)
    lengths = frame["Abstract"].fillna("").str.split().str.len()
    frame["sampling_abstract_length"] = pd.qcut(
        lengths.rank(method="first"), 3, labels=["short", "medium", "long"]
    ).astype(str)
    query_columns = [f"in_query_{index}" for index in range(1, 5)]
    frame["sampling_query_signature"] = frame[query_columns].apply(
        lambda row: "".join(
            str(int(float(value))) if str(value).strip() else "0" for value in row
        ),
        axis=1,
    )
    journal_sizes = frame["Source title"].map(frame["Source title"].value_counts())
    frame["sampling_journal_band"] = pd.cut(
        journal_sizes,
        bins=[0, 9, 49, np.inf],
        labels=["small", "medium", "large"],
        include_lowest=True,
    ).astype(str)
    frame["sampling_metadata"] = np.where(
        frame["Author Keywords"].fillna("").str.strip().eq(""),
        "keywords_missing",
        "keywords_present",
    )
    columns = [
        "sampling_era",
        "sampling_query_signature",
        "sampling_abstract_length",
        "sampling_journal_band",
        "sampling_metadata",
    ]
    frame["sampling_stratum"] = frame[columns].agg("|".join, axis=1)
    return frame


def proportional_allocation(stratum_sizes: pd.Series, sample_size: int) -> pd.Series:
    """Allocate an exact sample proportionally, with one unit per stratum."""

    sizes = stratum_sizes.astype(int).sort_index()
    if sample_size < len(sizes):
        raise ValueError("sample size must cover every non-empty stratum")
    if sample_size > int(sizes.sum()):
        raise ValueError("sample size exceeds population")
    ideal = sizes * sample_size / sizes.sum()
    allocation = np.floor(ideal).astype(int).clip(lower=1)
    allocation = pd.Series(np.minimum(allocation, sizes), index=sizes.index)
    while int(allocation.sum()) < sample_size:
        capacity = sizes - allocation
        candidates = capacity[capacity > 0].index
        priority = (ideal - allocation).loc[candidates]
        chosen = priority.idxmax()
        allocation.loc[chosen] += 1
    while int(allocation.sum()) > sample_size:
        candidates = allocation[allocation > 1].index
        priority = (allocation - ideal).loc[candidates]
        chosen = priority.idxmax()
        allocation.loc[chosen] -= 1
    return allocation.astype(int)


def draw_stratified_sample(
    frame: pd.DataFrame, sample_size: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw an exact reproducible sample and attach design weights."""

    if "sampling_stratum" not in frame:
        raise KeyError("sampling_stratum is required")
    sizes = frame["sampling_stratum"].value_counts().sort_index()
    allocation = proportional_allocation(sizes, sample_size)
    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []
    for stratum, count in allocation.items():
        candidates = frame.index[frame["sampling_stratum"] == stratum].to_numpy()
        selected_indices.extend(rng.choice(candidates, size=count, replace=False).tolist())
    selected = frame.loc[selected_indices].copy()
    selected["stratum_population_n"] = selected["sampling_stratum"].map(sizes)
    selected["stratum_sample_n"] = selected["sampling_stratum"].map(allocation)
    selected["selection_probability"] = (
        selected["stratum_sample_n"] / selected["stratum_population_n"]
    )
    selected["sampling_weight"] = 1.0 / selected["selection_probability"]
    selected["sampling_random_order"] = rng.permutation(len(selected)) + 1
    selected = selected.sort_values("sampling_random_order").reset_index(drop=True)
    allocation_frame = pd.DataFrame(
        {
            "sampling_stratum": allocation.index,
            "population_n": sizes.loc[allocation.index].to_numpy(),
            "sample_n": allocation.to_numpy(),
        }
    )
    allocation_frame["selection_probability"] = (
        allocation_frame["sample_n"] / allocation_frame["population_n"]
    )
    allocation_frame["sampling_weight"] = 1.0 / allocation_frame["selection_probability"]
    return selected, allocation_frame


def _pair_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    agreement = float(np.mean(left == right))
    pooled = np.concatenate((left, right))
    _, counts = np.unique(pooled, return_counts=True)
    total = len(pooled)
    expected_disagreement = 1.0 - float(
        np.sum(counts * (counts - 1)) / (total * (total - 1))
    )
    observed_disagreement = 1.0 - agreement
    alpha = (
        1.0
        if expected_disagreement == 0 and observed_disagreement == 0
        else 1.0 - observed_disagreement / expected_disagreement
        if expected_disagreement > 0
        else np.nan
    )
    return agreement, float(alpha)


def simulate_stability(
    aligned: pd.DataFrame,
    dimensions: list[str],
    fractions: tuple[float, ...] = (0.10, 0.25, 0.40),
    replicates: int = 1000,
    seed: int = 20260711,
    thresholds: StabilityThresholds = StabilityThresholds(),
    target_population_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Repeatedly sample strata and summarize agreement-estimate stability."""

    required = {"paper_id", "sampling_stratum"} | {
        f"{dimension}_{side}" for dimension in dimensions for side in ("mini", "nano")
    }
    missing = required - set(aligned.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")
    frame = aligned.reset_index(drop=True)
    size_basis = target_population_size or len(frame)
    if size_basis < len(frame):
        raise ValueError("target population cannot be smaller than the aligned frame")
    groups = {
        key: values.index.to_numpy()
        for key, values in frame.groupby("sampling_stratum", sort=True)
    }
    sizes = pd.Series({key: len(indices) for key, indices in groups.items()})
    rng = np.random.default_rng(seed)
    full = {
        dimension: _pair_metrics(
            frame[f"{dimension}_mini"].astype(str).to_numpy(),
            frame[f"{dimension}_nano"].astype(str).to_numpy(),
        )
        for dimension in dimensions
    }
    rare = {}
    for dimension in dimensions:
        entries = []
        for side in ("mini", "nano"):
            counts = frame[f"{dimension}_{side}"].astype(str).value_counts()
            entries.extend((side, value) for value, count in counts.items() if count / len(frame) < 0.01)
        rare[dimension] = entries

    rows: list[dict] = []
    coverage_rows: list[dict] = []
    allocation_rows: list[dict] = []
    for fraction in fractions:
        sample_size = int(np.ceil(size_basis * fraction))
        allocation = proportional_allocation(sizes, sample_size)
        allocation_rows.extend(
            {
                "fraction": fraction,
                "sample_size": sample_size,
                "sampling_stratum": stratum,
                "population_n": int(sizes[stratum]),
                "sample_n": int(count),
                "selection_probability": float(count / sizes[stratum]),
                "sampling_weight": float(sizes[stratum] / count),
            }
            for stratum, count in allocation.items()
        )
        for replicate in range(replicates):
            indices = np.concatenate(
                [rng.choice(groups[key], size=count, replace=False) for key, count in allocation.items()]
            )
            sample = frame.iloc[indices]
            for dimension in dimensions:
                agreement, alpha = _pair_metrics(
                    sample[f"{dimension}_mini"].astype(str).to_numpy(),
                    sample[f"{dimension}_nano"].astype(str).to_numpy(),
                )
                rows.append(
                    {
                        "fraction": fraction,
                        "sample_size": sample_size,
                        "replicate": replicate,
                        "dimension": dimension,
                        "percent_agreement": agreement,
                        "krippendorff_alpha": alpha,
                    }
                )
                rare_entries = rare[dimension]
                observed = sum(
                    bool((sample[f"{dimension}_{side}"].astype(str) == value).any())
                    for side, value in rare_entries
                )
                coverage_rows.append(
                    {
                        "fraction": fraction,
                        "sample_size": sample_size,
                        "replicate": replicate,
                        "dimension": dimension,
                        "rare_categories": len(rare_entries),
                        "rare_categories_observed": observed,
                        "rare_category_coverage": observed / len(rare_entries) if rare_entries else 1.0,
                    }
                )

    estimates = pd.DataFrame(rows)
    summaries = []
    for (fraction, sample_size, dimension), group in estimates.groupby(
        ["fraction", "sample_size", "dimension"], sort=True
    ):
        for metric, full_index in (("percent_agreement", 0), ("krippendorff_alpha", 1)):
            values = group[metric].dropna()
            full_value = full[dimension][full_index]
            p025, p975 = values.quantile([0.025, 0.975])
            bias = float(values.mean() - full_value)
            width = float(p975 - p025)
            bias_limit = thresholds.agreement_bias if metric == "percent_agreement" else thresholds.alpha_bias
            width_limit = thresholds.agreement_width if metric == "percent_agreement" else thresholds.alpha_width
            summaries.append(
                {
                    "fraction": fraction,
                    "sample_size": sample_size,
                    "dimension": dimension,
                    "metric": metric,
                    "full_corpus_value": full_value,
                    "simulation_mean": float(values.mean()),
                    "bias": bias,
                    "absolute_bias": abs(bias),
                    "standard_deviation": float(values.std(ddof=1)),
                    "rmse": float(np.sqrt(np.mean((values - full_value) ** 2))),
                    "empirical_p025": float(p025),
                    "empirical_p975": float(p975),
                    "empirical_95_width": width,
                    "bias_limit": bias_limit,
                    "width_limit": width_limit,
                    "precision_passed": abs(bias) <= bias_limit and width <= width_limit,
                }
            )
    return estimates, pd.DataFrame(summaries), pd.DataFrame(coverage_rows), pd.DataFrame(allocation_rows)


def recommend_fraction(summary: pd.DataFrame) -> float | None:
    """Return the smallest fraction passing every metric and dimension."""

    decisions = summary.groupby("fraction")["precision_passed"].all().sort_index()
    passed = decisions[decisions].index
    return float(passed[0]) if len(passed) else None
