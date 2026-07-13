"""Create and audit the frozen proprietary-rater probability sample."""

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
    draw_stratified_sample,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison(population: pd.Series, sample: pd.Series, variable: str) -> pd.DataFrame:
    pop = population.astype(str).value_counts().rename("population_n")
    sub = sample.astype(str).value_counts().rename("sample_n")
    result = pd.concat([pop, sub], axis=1).fillna(0).reset_index(names="category")
    result.insert(0, "variable", variable)
    result["population_share"] = result["population_n"] / len(population)
    result["sample_share"] = result["sample_n"] / len(sample)
    result["share_difference"] = result["sample_share"] - result["population_share"]
    return result.sort_values(["variable", "population_share"], ascending=[True, False])


def query_comparison(population: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index in range(1, 5):
        column = f"in_query_{index}"
        pop = pd.to_numeric(population[column], errors="coerce").fillna(0).eq(1)
        sub = pd.to_numeric(sample[column], errors="coerce").fillna(0).eq(1)
        rows.append(
            {
                "variable": "overlapping_query_membership",
                "category": f"Query {index}",
                "population_n": int(pop.sum()),
                "sample_n": int(sub.sum()),
                "population_share": float(pop.mean()),
                "sample_share": float(sub.mean()),
                "share_difference": float(sub.mean() - pop.mean()),
            }
        )
    return pd.DataFrame(rows)


def svg_grouped_bars(data: pd.DataFrame, title: str, output: Path) -> None:
    data = data.sort_values("population_share", ascending=True).tail(20)
    width, height = 1050, max(360, 70 + len(data) * 34)
    left, top, plot_w = 280, 60, 680
    maximum = max(float(data[["population_share", "sample_share"]].max().max()) * 1.12, 0.01)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="21" font-weight="bold">{title}</text>']
    for index, row in enumerate(data.itertuples(index=False)):
        y = top + index * 34
        label = str(row.category).replace("&", "&amp;")
        pop_w = plot_w * float(row.population_share) / maximum
        sub_w = plot_w * float(row.sample_share) / maximum
        parts.extend([
            f'<text x="{left-10}" y="{y+15}" text-anchor="end" font-family="sans-serif" font-size="12">{label}</text>',
            f'<rect x="{left}" y="{y+2}" width="{pop_w:.1f}" height="11" fill="#94a3b8"/>',
            f'<rect x="{left}" y="{y+16}" width="{sub_w:.1f}" height="11" fill="#2563eb"/>',
            f'<text x="{left+max(pop_w,sub_w)+6:.1f}" y="{y+20}" font-family="sans-serif" font-size="10">{float(row.sample_share):.1%}</text>',
        ])
    parts.extend([
        f'<rect x="{left}" y="{height-32}" width="16" height="10" fill="#94a3b8"/><text x="{left+22}" y="{height-23}" font-family="sans-serif" font-size="12">Population</text>',
        f'<rect x="{left+130}" y="{height-32}" width="16" height="10" fill="#2563eb"/><text x="{left+152}" y="{height-23}" font-family="sans-serif" font-size="12">Sample</text>',
        '</svg>',
    ])
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=2235)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/interim/proprietary_validation")
    args = parser.parse_args()
    corpus_path = PROJECT_ROOT / "data/processed/master_corpus.csv"
    corpus = add_sampling_strata(pd.read_csv(corpus_path, dtype=str, keep_default_na=False, low_memory=False))
    selected, allocation = draw_stratified_sample(corpus, args.size, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    design_columns = [
        "paper_id", "sampling_random_order", "sampling_stratum", "sampling_era",
        "sampling_query_signature", "sampling_abstract_length", "sampling_journal_band",
        "sampling_metadata", "stratum_population_n", "stratum_sample_n",
        "selection_probability", "sampling_weight",
    ]
    manifest_path = args.output_dir / "proprietary_probability_sample_2235.csv"
    selected[design_columns].to_csv(manifest_path, index=False)
    paper_columns = [
        *design_columns,
        "Title", "Abstract", "Author Keywords", "Source title", "Year",
        "in_query_1", "in_query_2", "in_query_3", "in_query_4",
    ]
    papers_path = args.output_dir / "proprietary_probability_sample_2235_papers.csv"
    selected[paper_columns].to_csv(papers_path, index=False, encoding="utf-8-sig")
    allocation.to_csv(args.output_dir / "proprietary_sample_stratum_allocation.csv", index=False)

    comparisons = [
        comparison(corpus["sampling_era"], selected["sampling_era"], "publication_era"),
        query_comparison(corpus, selected),
        comparison(corpus["sampling_abstract_length"], selected["sampling_abstract_length"], "abstract_length"),
        comparison(corpus["sampling_journal_band"], selected["sampling_journal_band"], "journal_size_band"),
        comparison(corpus["sampling_metadata"], selected["sampling_metadata"], "metadata_completeness"),
        comparison(corpus["Source title"], selected["Source title"], "journal"),
    ]
    audit = pd.concat(comparisons, ignore_index=True)
    audit.to_csv(args.output_dir / "proprietary_sample_distribution_audit.csv", index=False)
    titles = {
        "publication_era": "Publication-era distribution",
        "overlapping_query_membership": "Overlapping query membership",
        "abstract_length": "Abstract-length distribution",
        "journal_size_band": "Journal-size-band distribution",
        "metadata_completeness": "Metadata-completeness distribution",
        "journal": "Leading-journal distribution",
    }
    for variable, title in titles.items():
        svg_grouped_bars(audit[audit["variable"] == variable], title, args.output_dir / f"distribution_{variable}.svg")

    human_key = PROJECT_ROOT / "data/interim/human_validation/private_sample_key.csv"
    human_ids = set(pd.read_csv(human_key, dtype=str)["paper_id"]) if human_key.exists() else set()
    target_ids = list(selected["paper_id"])
    target_ids.extend(sorted(human_ids - set(target_ids)))
    target = corpus.set_index("paper_id").loc[target_ids].reset_index()
    probability_ids = set(selected["paper_id"])
    target.insert(1, "in_probability_sample", target["paper_id"].isin(probability_ids).astype(int))
    target.insert(2, "in_human_anchor", target["paper_id"].isin(human_ids).astype(int))
    target.insert(3, "provider_target_order", range(1, len(target) + 1))
    target_columns = [
        "paper_id", "provider_target_order", "in_probability_sample", "in_human_anchor",
        "Title", "Abstract", "Author Keywords", "Source title", "Year",
        "in_query_1", "in_query_2", "in_query_3", "in_query_4",
    ]
    target_path = args.output_dir / "proprietary_rater_target_2276_papers.csv"
    target[target_columns].to_csv(target_path, index=False, encoding="utf-8-sig")
    metadata = {
        "created_at": pd.Timestamp.now(tz="Europe/Amsterdam").isoformat(),
        "sampling_frame": str(corpus_path.relative_to(PROJECT_ROOT)),
        "sampling_frame_sha256": sha256(corpus_path),
        "population_n": len(corpus),
        "sample_n": len(selected),
        "sample_fraction": len(selected) / len(corpus),
        "seed": args.seed,
        "strata": ["publication era", "query signature", "abstract-length tertile", "journal-size band", "keyword completeness"],
        "model_outputs_used_for_selection": False,
        "human_anchor_overlap_n": len(set(selected["paper_id"]) & human_ids),
        "manifest": manifest_path.name,
        "manifest_sha256": sha256(manifest_path),
        "paper_dataset": papers_path.name,
        "paper_dataset_sha256": sha256(papers_path),
        "provider_target": target_path.name,
        "provider_target_n": len(target),
        "provider_target_sha256": sha256(target_path),
    }
    (args.output_dir / "proprietary_sample_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Selected {len(selected):,}/{len(corpus):,} papers ({len(selected)/len(corpus):.2%}).")
    print(f"Manifest SHA256: {metadata['manifest_sha256']}")
    print(f"Maximum absolute distribution-share difference: {audit.share_difference.abs().max():.4f}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
