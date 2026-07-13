"""Run Mini-Nano stratified subset stability simulations and write visuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from aecsp.analytics.sampling_stability import (  # noqa: E402
    add_sampling_strata,
    recommend_fraction,
    simulate_stability,
)
from aecsp.specification.analysis_columns import enrich_for_analysis  # noqa: E402

DIMENSIONS = [
    "ai_method_or_phenomenon",
    "ai_role_function",
    "ai_type_form",
    "ai_mechanism_analysis",
    "level_of_analysis",
    "entrepreneurial_process_stage",
    "scope_conditions",
    "definition_construct_clarity",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def svg_chart(summary: pd.DataFrame, metric: str, value: str, title: str, output: Path) -> None:
    data = summary[summary["metric"] == metric].copy()
    dimensions = DIMENSIONS
    fractions = sorted(data["fraction"].unique())
    width, height = 1200, 620
    left, top, plot_w, plot_h = 250, 80, 880, 440
    maximum = max(float(data[value].max()) * 1.12, 0.01)
    colors = ["#2563eb", "#059669", "#d97706"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold">{title}</text>']
    group_h = plot_h / len(dimensions)
    bar_h = group_h / (len(fractions) + 1)
    for index, dimension in enumerate(dimensions):
        y0 = top + index * group_h
        label = dimension.replace("_", " ")
        parts.append(f'<text x="240" y="{y0 + group_h/2:.1f}" text-anchor="end" dominant-baseline="middle" font-family="sans-serif" font-size="13">{label}</text>')
        for f_index, fraction in enumerate(fractions):
            row = data[(data["dimension"] == dimension) & (data["fraction"] == fraction)].iloc[0]
            amount = float(row[value]); bar_w = plot_w * amount / maximum
            y = y0 + 4 + f_index * bar_h
            parts.append(f'<rect x="{left}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h-2:.1f}" fill="{colors[f_index]}"/>')
            parts.append(f'<text x="{left+bar_w+5:.1f}" y="{y+(bar_h-2)/2:.1f}" dominant-baseline="middle" font-family="sans-serif" font-size="11">{amount:.3f}</text>')
    for i, fraction in enumerate(fractions):
        x = left + i * 160
        parts.extend([f'<rect x="{x}" y="555" width="18" height="12" fill="{colors[i]}"/>', f'<text x="{x+25}" y="566" font-family="sans-serif" font-size="13">{fraction:.0%}</text>'])
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/processed/analysis/sampling_stability")
    args = parser.parse_args()
    spec = PROJECT_ROOT / "data/processed/specification"
    mini_path = spec / "paper_specifications_gpt-5.4-mini-2026-03-17_spec-v3.csv"
    nano_path = spec / "paper_specifications_gpt-4.1-nano-2025-04-14_spec-v3.csv"
    corpus_path = PROJECT_ROOT / "data/processed/master_corpus.csv"
    mini = pd.read_csv(mini_path, dtype=str, keep_default_na=False, low_memory=False)
    nano = enrich_for_analysis(pd.read_csv(nano_path, dtype=str, keep_default_na=False, low_memory=False))
    corpus = add_sampling_strata(pd.read_csv(corpus_path, dtype=str, keep_default_na=False, low_memory=False))
    aligned = corpus[["paper_id", "sampling_stratum"]].merge(mini[["paper_id", *DIMENSIONS]], on="paper_id").merge(nano[["paper_id", *DIMENSIONS]], on="paper_id", suffixes=("_mini", "_nano"), validate="one_to_one")
    estimates, summary, coverage, allocation = simulate_stability(
        aligned,
        DIMENSIONS,
        replicates=args.replicates,
        seed=args.seed,
        target_population_size=len(corpus),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(args.output_dir / "replicate_estimates.csv", index=False)
    summary.to_csv(args.output_dir / "stability_summary.csv", index=False)
    coverage.groupby(["fraction", "sample_size", "dimension"], as_index=False).agg(mean_rare_category_coverage=("rare_category_coverage", "mean"), minimum_rare_category_coverage=("rare_category_coverage", "min"), rare_categories=("rare_categories", "max")).to_csv(args.output_dir / "rare_category_coverage.csv", index=False)
    allocation.to_csv(args.output_dir / "stratum_allocation.csv", index=False)
    recommendation = recommend_fraction(summary)
    manifest = {"seed": args.seed, "replicates": args.replicates, "fractions": [0.10, 0.25, 0.40], "population_n": len(corpus), "intersection_n": len(aligned), "recommended_fraction": recommendation, "inputs": {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in (corpus_path, mini_path, nano_path)}}
    (args.output_dir / "simulation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    svg_chart(summary, "krippendorff_alpha", "absolute_bias", "Absolute bias of Krippendorff alpha", args.output_dir / "alpha_bias.svg")
    svg_chart(summary, "krippendorff_alpha", "empirical_95_width", "Empirical 95% interval width for Krippendorff alpha", args.output_dir / "alpha_interval_width.svg")
    rare_summary = pd.read_csv(args.output_dir / "rare_category_coverage.csv")
    plot_ready = rare_summary.rename(columns={"mean_rare_category_coverage": "coverage"})
    plot_ready["metric"] = "rare_coverage"
    svg_chart(plot_ready, "rare_coverage", "coverage", "Mean rare-category coverage", args.output_dir / "rare_category_coverage.svg")
    decision = "no tested fraction" if recommendation is None else f"{recommendation:.0%}"
    print(f"Simulated {args.replicates:,} samples at each fraction on {len(aligned):,} aligned papers.")
    print(f"Smallest fraction passing all precision thresholds: {decision}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
