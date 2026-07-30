"""Generate reproducible multi-model validation tables and figures.

Inputs: frozen spec-v3 model exports, probability/human manifests, master
corpus, and model caches. Outputs: non-mutating CSV/JSON/SVG analysis artifacts
and a Markdown summary under data/processed/analysis and reports/analysis.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aecsp.analytics.model_validation import (  # noqa: E402
    CORE_DIMENSIONS,
    DIMENSIONS,
    EXPLORATORY_DIMENSIONS,
    SUPPLEMENTARY_DIAGNOSTIC_DIMENSIONS,
    multirater_summary,
    normalized_entropy,
    pairwise_with_bootstrap,
    stable_seed,
)
from aecsp.specification.analysis_columns import enrich_for_analysis  # noqa: E402

SEED = 20_260_711
REPETITIONS = 2_000
PROCESSED = PROJECT_ROOT / "data/processed"
OUTPUT = PROCESSED / "analysis/model_validation"
FIGURES = PROJECT_ROOT / "reports/analysis/figures/model_validation"
TABLES = PROJECT_ROOT / "reports/analysis/tables/model_validation"
SUMMARY = PROJECT_ROOT / "reports/analysis/model_validation_summary.md"
PROBABILITY_SAMPLE = PROJECT_ROOT / "data/interim/proprietary_validation/proprietary_probability_sample_2235.csv"
HUMAN_ANCHOR = PROJECT_ROOT / "data/interim/human_validation/private_sample_key.csv"

MODEL_FILES = {
    "Mini": PROCESSED / "specification/paper_specifications_gpt-5.4-mini-2026-03-17_spec-v3.csv",
    "Nano": PROCESSED / "specification/paper_specifications_gpt-4.1-nano-2025-04-14_spec-v3.csv",
    "Claude": PROCESSED / "specification/paper_specifications_claude-sonnet-5_spec-v3.csv",
    "Gemini": PROCESSED / "specification/paper_specifications_gemini-3.1-pro-preview_spec-v3.csv",
}
LOCAL_CACHE_MODELS = {
    "Llama": PROJECT_ROOT / "data/interim/spec_cache/spec-v3/llama3.2",
    "Gemma": PROJECT_ROOT / "data/interim/spec_cache/spec-v3/gemma4_31b",
}
REPRESENTATIVE_MODELS = ("Mini", "Nano", "Claude", "Gemini")
DIMENSION_LABELS = {
    "ai_method_or_phenomenon": "AI positioning",
    "ai_type_form": "Technical AI type/form",
    "ai_role_function": "AI role/function",
    "ai_mechanism": "AI mechanism",
    "level_of_analysis": "Level of analysis",
    "entrepreneurial_process_stage": "Entrepreneurial process stage",
    "scope_conditions": "Scope conditions",
    "definition_construct_clarity": "Definition clarity",
    "ai_definition_present": "AI definition present",
    "ai_distinction_present": "AI distinction present",
    "ai_mechanism_analysis": "AI mechanism analysis",
    "process_sequence_specified": "Process sequence specified",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_models() -> dict[str, pd.DataFrame]:
    frames = {}
    for name, path in MODEL_FILES.items():
        frame = enrich_for_analysis(pd.read_csv(path, dtype=str, keep_default_na=False))
        if frame["paper_id"].duplicated().any():
            raise SystemExit(f"{name} has duplicate paper IDs")
        frames[name] = frame.set_index("paper_id", drop=False)
    for name, directory in LOCAL_CACHE_MODELS.items():
        records = []
        for path in directory.glob("*.json"):
            if path.name == "protocol_manifest.json":
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if record.get("paper_id"):
                records.append(record)
        frame = enrich_for_analysis(pd.DataFrame(records))
        frame = frame.drop_duplicates("paper_id", keep="last")
        frames[name] = frame.set_index("paper_id", drop=False)
    return frames


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def markdown_2dp(frame: pd.DataFrame) -> str:
    """Render floating-point report values consistently to two decimals."""

    displayed = frame.copy()
    for column in displayed.select_dtypes(include=["floating"]).columns:
        displayed[column] = displayed[column].map(
            lambda value: "—" if pd.isna(value) else f"{value:.2f}"
        )
    return displayed.to_markdown(index=False, disable_numparse=True)


def bar_svg(rows: list[tuple[str, float]], title: str, path: Path, *, maximum: float = 1.0) -> None:
    width, left, right, top, row_h = 980, 300, 60, 75, 28
    height = top + len(rows) * row_h + 55
    plot_w = width - left - right
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">{esc(title)}</text>']
    for index, (label, value) in enumerate(rows):
        y = top + index * row_h
        bar_w = max(0, min(plot_w, plot_w * value / maximum))
        parts += [f'<text x="{left-10}" y="{y+17}" text-anchor="end" font-family="sans-serif" font-size="12">{esc(label)}</text>',
                  f'<rect x="{left}" y="{y+3}" width="{plot_w}" height="18" fill="#eef2f7"/>',
                  f'<rect x="{left}" y="{y+3}" width="{bar_w}" height="18" fill="#3973ac"/>',
                  f'<text x="{left+bar_w+7}" y="{y+17}" font-family="sans-serif" font-size="12">{value:.2f}</text>']
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def heatmap_svg(frame: pd.DataFrame, value: str, title: str, path: Path) -> None:
    models = ["Mini", "Nano", "Claude", "Gemini"]
    lookup = {(row.left_model, row.right_model): getattr(row, value) for row in frame.itertuples()}
    size, left, top = 120, 170, 90
    width, height = left + size * 4 + 40, top + size * 4 + 50
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="19" font-weight="bold">{esc(title)}</text>']
    for index, model in enumerate(models):
        parts.append(f'<text x="{left+index*size+size/2}" y="70" text-anchor="middle" font-family="sans-serif" font-size="13">{model}</text>')
        parts.append(f'<text x="{left-12}" y="{top+index*size+size/2+5}" text-anchor="end" font-family="sans-serif" font-size="13">{model}</text>')
    for row, left_model in enumerate(models):
        for col, right_model in enumerate(models):
            if row == col:
                metric = 1.0
            else:
                metric = lookup.get((left_model, right_model), lookup.get((right_model, left_model), np.nan))
            intensity = 0 if pd.isna(metric) else max(0, min(1, float(metric)))
            blue = int(245 - 150 * intensity)
            fill = f'rgb({blue},{blue+5},{245})'
            x, y = left + col * size, top + row * size
            label = "—" if pd.isna(metric) else f"{metric:.2f}"
            parts += [f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="{fill}" stroke="white"/>',
                      f'<text x="{x+size/2}" y="{y+size/2+5}" text-anchor="middle" font-family="sans-serif" font-size="15">{label}</text>']
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(value).lower())).strip()


def main() -> None:
    for directory in (OUTPUT, FIGURES, TABLES):
        directory.mkdir(parents=True, exist_ok=True)
    frames = load_models()
    probability = pd.read_csv(PROBABILITY_SAMPLE, dtype=str, keep_default_na=False)
    human = pd.read_csv(HUMAN_ANCHOR, dtype=str, keep_default_na=False)
    master = pd.read_csv(PROCESSED / "master_corpus.csv", dtype=str, keep_default_na=False).set_index("paper_id")
    probability_ids = probability["paper_id"].tolist()
    weights = probability.set_index("paper_id")["sampling_weight"].astype(float)

    coverage_rows = []
    for name, frame in frames.items():
        probability_success = len(set(frame.index) & set(probability_ids))
        coverage_rows.append({"model": name, "all_successful_records": len(frame),
                              "probability_target": len(probability_ids),
                              "probability_successful": probability_success,
                              "probability_nonresponse": len(probability_ids) - probability_success,
                              "probability_coverage": probability_success / len(probability_ids)})
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUTPUT / "execution_coverage.csv", index=False)

    prevalence_rows = []
    for name, frame in frames.items():
        ids = [paper_id for paper_id in probability_ids if paper_id in frame.index]
        sample_weights = weights.reindex(ids).to_numpy()
        for dimension in DIMENSIONS:
            values = frame.loc[ids, dimension].astype(str)
            valid = values.str.strip() != ""
            for category, count in values[valid].value_counts().items():
                category_mask = values == category
                prevalence_rows.append({
                    "model": name, "dimension": dimension, "category": category,
                    "count": int(count), "denominator": int(valid.sum()),
                    "prevalence": count / valid.sum() if valid.sum() else None,
                    "weighted_prevalence": float(sample_weights[category_mask.to_numpy()].sum() / sample_weights[valid.to_numpy()].sum()),
                    "normalized_entropy": normalized_entropy(values.tolist(), sample_weights),
                })
    prevalence = pd.DataFrame(prevalence_rows)
    prevalence.to_csv(OUTPUT / "dimension_prevalence.csv", index=False)

    agreement_rows = []
    comparison_sets = [("probability_sample", REPRESENTATIVE_MODELS)]
    for set_name, model_names in comparison_sets:
        for left, right in combinations(model_names, 2):
            ids = [paper_id for paper_id in probability_ids if paper_id in frames[left].index and paper_id in frames[right].index]
            for dimension in DIMENSIONS:
                result = pairwise_with_bootstrap(
                    frames[left].loc[ids, dimension].tolist(),
                    frames[right].loc[ids, dimension].tolist(),
                    weights=weights.reindex(ids).tolist(), repetitions=REPETITIONS,
                    seed=stable_seed(SEED, set_name, left, right, dimension),
                )
                agreement_rows.append({"comparison_set": set_name, "left_model": left,
                                       "right_model": right, "dimension": dimension, **result})
    for left, right in combinations(frames, 2):
        if left in REPRESENTATIVE_MODELS and right in REPRESENTATIVE_MODELS:
            continue
        ids = [paper_id for paper_id in probability_ids if paper_id in frames[left].index and paper_id in frames[right].index]
        if len(ids) < 2:
            continue
        for dimension in DIMENSIONS:
            result = pairwise_with_bootstrap(
                frames[left].loc[ids, dimension].tolist(), frames[right].loc[ids, dimension].tolist(),
                weights=weights.reindex(ids).tolist(), repetitions=REPETITIONS,
                seed=stable_seed(SEED, "supplementary_local_intersection", left, right, dimension),
            )
            agreement_rows.append({"comparison_set": "supplementary_local_intersection",
                                   "left_model": left, "right_model": right,
                                   "dimension": dimension, **result})
    full_ids = sorted(set(frames["Mini"].index) & set(frames["Nano"].index))
    for dimension in DIMENSIONS:
        result = pairwise_with_bootstrap(
            frames["Mini"].loc[full_ids, dimension].tolist(),
            frames["Nano"].loc[full_ids, dimension].tolist(),
            repetitions=REPETITIONS, seed=stable_seed(SEED, "full", dimension),
        )
        agreement_rows.append({"comparison_set": "full_mini_nano", "left_model": "Mini",
                               "right_model": "Nano", "dimension": dimension, **result})
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(OUTPUT / "agreement_pairwise.csv", index=False)

    common = [paper_id for paper_id in probability_ids if all(paper_id in frames[name].index for name in REPRESENTATIVE_MODELS)]
    multirater_rows = []
    for dimension in DIMENSIONS:
        units = [[frames[name].at[paper_id, dimension] for name in REPRESENTATIVE_MODELS] for paper_id in common]
        multirater_rows.append({"comparison_set": "probability_sample_common4", "dimension": dimension,
                                "models": "Mini|Nano|Claude|Gemini", **multirater_summary(units)})
    multirater = pd.DataFrame(multirater_rows)
    multirater.to_csv(OUTPUT / "agreement_multirater.csv", index=False)

    confusion_rows = []
    for left, right in combinations(frames, 2):
        ids = [paper_id for paper_id in probability_ids if paper_id in frames[left].index and paper_id in frames[right].index]
        for dimension in DIMENSIONS:
            table = pd.crosstab(frames[left].loc[ids, dimension], frames[right].loc[ids, dimension])
            for left_code, row in table.iterrows():
                for right_code, count in row.items():
                    confusion_rows.append({"left_model": left, "right_model": right,
                                           "dimension": dimension, "left_code": left_code,
                                           "right_code": right_code, "count": int(count)})
    pd.DataFrame(confusion_rows).to_csv(OUTPUT / "confusion_matrices_long.csv", index=False)

    grounding_rows = []
    evidence_sources = {
        dimension: ("ai_mechanism" if dimension == "ai_mechanism_analysis" else dimension)
        for dimension in DIMENSIONS
        if f"{'ai_mechanism' if dimension == 'ai_mechanism_analysis' else dimension}_evidence"
        in next(iter(frames.values())).columns
    }
    for name, frame in frames.items():
        ids = [paper_id for paper_id in probability_ids if paper_id in frame.index]
        for dimension, source_dimension in evidence_sources.items():
            eligible = matches = 0
            for paper_id in ids:
                if frame.at[paper_id, f"{source_dimension}_evidence_type"] != "stated":
                    continue
                evidence = normalize_text(frame.at[paper_id, f"{source_dimension}_evidence"])
                if not evidence:
                    continue
                source = normalize_text(" ".join(master.loc[paper_id, ["Title", "Abstract", "Author Keywords"]].tolist()))
                eligible += 1
                matches += evidence in source
            grounding_rows.append({"model": name, "dimension": dimension,
                                   "stated_evidence_n": eligible, "exact_match_n": matches,
                                   "exact_match_share": matches / eligible if eligible else None})
    grounding = pd.DataFrame(grounding_rows)
    grounding.to_csv(OUTPUT / "evidence_grounding.csv", index=False)

    quality_rows = []
    evidence_dimensions = evidence_sources
    for name, frame in frames.items():
        ids = [paper_id for paper_id in probability_ids if paper_id in frame.index]
        for dimension, source_dimension in evidence_dimensions.items():
            types = frame.loc[ids, f"{source_dimension}_evidence_type"].astype(str)
            confidence = pd.to_numeric(frame.loc[ids, f"{source_dimension}_confidence"], errors="coerce")
            quality_rows.append({
                "model": name, "dimension": dimension, "papers": len(ids),
                "mean_confidence": confidence.mean(), "median_confidence": confidence.median(),
                "stated_share": (types == "stated").mean(),
                "inferred_share": (types == "inferred").mean(),
                "absent_share": (types == "absent").mean(),
            })
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(OUTPUT / "confidence_evidence_diagnostics.csv", index=False)

    intersections = []
    for size in range(2, len(REPRESENTATIVE_MODELS) + 1):
        for names in combinations(REPRESENTATIVE_MODELS, size):
            ids = set.intersection(*(set(frames[name].index) for name in names)) & set(probability_ids)
            intersections.append({"models": "|".join(names), "model_count": size, "intersection_n": len(ids)})
    pd.DataFrame(intersections).to_csv(OUTPUT / "agreement_intersections.csv", index=False)

    probability_agreement = agreement[agreement.comparison_set == "probability_sample"]
    core_probability_agreement = probability_agreement[
        probability_agreement.dimension.isin(CORE_DIMENSIONS)
    ]
    pair_macro = core_probability_agreement.groupby(
        ["left_model", "right_model"], as_index=False
    ).agg(
        percent_agreement=("percent_agreement", "mean"),
        krippendorff_alpha=("krippendorff_alpha", "mean"),
    )
    pair_macro.insert(0, "dimension_set", "six_core_dimensions")
    pair_macro.to_csv(TABLES / "pairwise_macro_agreement.csv", index=False)
    local_macro = agreement[
        (agreement.comparison_set == "supplementary_local_intersection")
        & agreement.dimension.isin(CORE_DIMENSIONS)
    ].groupby(
        ["left_model", "right_model"], as_index=False
    ).agg(percent_agreement=("percent_agreement", "mean"),
          krippendorff_alpha=("krippendorff_alpha", "mean"),
          minimum_comparable=("comparable", "min"))
    local_macro.to_csv(TABLES / "supplementary_local_macro_agreement.csv", index=False)
    heatmap_svg(pair_macro, "percent_agreement", "Mean exact agreement across six core dimensions", FIGURES / "pairwise_agreement_heatmap.svg")
    heatmap_svg(pair_macro, "krippendorff_alpha", "Mean nominal Krippendorff alpha across six core dimensions", FIGURES / "pairwise_alpha_heatmap.svg")
    bar_svg([(row.model, row.probability_coverage) for row in coverage.itertuples()], "Probability-sample model coverage", FIGURES / "model_coverage.svg")
    grounding_macro = grounding.groupby("model", as_index=False).agg(exact_match_share=("exact_match_share", "mean"))
    bar_svg([(row.model, row.exact_match_share) for row in grounding_macro.itertuples()], "Exact stated-evidence grounding (diagnostic)", FIGURES / "evidence_grounding.svg")
    strong_pair = probability_agreement[(probability_agreement.left_model == "Claude") & (probability_agreement.right_model == "Gemini")]
    bar_svg(
        [
            (DIMENSION_LABELS.get(row.dimension, row.dimension), row.percent_agreement)
            for row in strong_pair.itertuples()
        ],
        "Claude-Gemini agreement by dimension",
        FIGURES / "claude_gemini_dimension_agreement.svg",
    )
    for dimension in DIMENSIONS:
        subset = prevalence[prevalence.dimension == dimension]
        categories = subset.groupby("category")["count"].sum().sort_values(ascending=False).head(12).index
        rows = []
        for category in categories:
            for model in frames:
                match = subset[(subset.model == model) & (subset.category == category)]
                rows.append((f"{category} · {model}", float(match.weighted_prevalence.iloc[0]) if len(match) else 0.0))
        bar_svg(
            rows,
            f"Weighted distribution: {DIMENSION_LABELS.get(dimension, dimension)}",
            FIGURES / f"distribution_{dimension}.svg",
        )

    generated_at = datetime.now().isoformat()
    local_snapshot = {
        name: {
            "all_successful_records": int(
                coverage.loc[coverage.model == name, "all_successful_records"].iloc[0]
            ),
            "probability_successful": int(
                coverage.loc[coverage.model == name, "probability_successful"].iloc[0]
            ),
            "probability_target": int(
                coverage.loc[coverage.model == name, "probability_target"].iloc[0]
            ),
        }
        for name in LOCAL_CACHE_MODELS
    }
    manifest = {
        "generated_at": generated_at, "analysis_seed": SEED,
        "bootstrap_repetitions": REPETITIONS, "probability_sample": str(PROBABILITY_SAMPLE.relative_to(PROJECT_ROOT)),
        "probability_sample_sha256": sha256(PROBABILITY_SAMPLE), "human_anchor_n": len(human),
        "inputs": {name: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256(path)} for name, path in MODEL_FILES.items()},
        "local_cache_inputs": {name: str(path.relative_to(PROJECT_ROOT)) for name, path in LOCAL_CACHE_MODELS.items()},
        "local_cache_snapshot": local_snapshot,
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip(),
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "notes": ["Agreement is diagnostic; no model is a gold standard.",
                  "Macro pairwise heatmaps average the six core dimensions only.",
                  f"Exploratory displayed dimensions: {', '.join(EXPLORATORY_DIMENSIONS)}.",
                  f"Supplementary binary diagnostics: {', '.join(SUPPLEMENTARY_DIAGNOSTIC_DIMENSIONS)}.",
                  "Exact evidence matching is conservative and does not detect paraphrases.",
                  "Llama and Gemma cache counts are point-in-time partial snapshots and may increase after this analysis.",
                  "Human coding fields were empty at generation time; human-model IRR is pending."],
    }
    (OUTPUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    best = pair_macro.sort_values("percent_agreement", ascending=False).iloc[0]
    summary = f"""# Model-validation analysis\n\nGenerated {manifest['generated_at']}. All outputs are non-mutating and use the frozen 2,235-paper probability sample unless labelled full Mini-Nano.\n\n## Coverage\n\n{markdown_2dp(coverage)}\n\n## Pairwise macro agreement: six core dimensions\n\n{markdown_2dp(pair_macro.sort_values('percent_agreement', ascending=False))}\n\nThe macro heatmaps average only AI positioning, technical type, AI role, mechanism, level, and scope. Process stage and definition form are displayed as exploratory dimensions; the three binary diagnostics are supplementary. The strongest representative pair is **{best.left_model}-{best.right_model}** with mean exact agreement **{best.percent_agreement:.2f}**. Agreement is not accuracy; blinded human coding remains the accuracy anchor.\n\n## Supplementary local-model intersections\n\n{markdown_2dp(local_macro.sort_values('percent_agreement', ascending=False))}\n\nLocal estimates are supplementary because successful local records cover only part of the probability sample and their non-response may be selective.\n\n## Four-model agreement\n\n{markdown_2dp(multirater)}\n\n## Interpretation controls\n\n- Mini remains the primary full-corpus rater; Nano is a full-corpus sensitivity baseline.\n- Claude and Gemini validate Mini on a model-independent probability sample.\n- Llama and Gemma comparisons are separately labelled supplementary intersections.\n- Macro heatmaps use the six core dimensions; process stage and definition form remain exploratory.\n- Binary fields with high raw agreement but low alpha are prevalence-sensitive and require both statistics.\n- Exact evidence grounding is a conservative lexical diagnostic, not a hallucination rate.\n- Human-model IRR will be added only after blinded human fields are completed.\n"""
    summary = summary.replace(
        "\n\n## Coverage",
        (
            "\n\nLlama and Gemma rows are point-in-time supplementary cache "
            "snapshots. Their coverage is incomplete and must be regenerated "
            "before later reporting of local-model results.\n\n## Coverage"
        ),
        1,
    )
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(summary, encoding="utf-8")
    print(f"Analysis tables -> {OUTPUT}")
    print(f"Figures -> {FIGURES}")
    print(f"Summary -> {SUMMARY}")
    print(f"Probability common-four intersection: {len(common):,}/{len(probability_ids):,}")


if __name__ == "__main__":
    main()
