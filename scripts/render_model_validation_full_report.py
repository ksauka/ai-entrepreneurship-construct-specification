"""Render the full narrative model-validation results report.

Inputs are the reproducible model-validation CSV outputs. The output is a
detailed Markdown text document with per-dimension prevalence, reliability,
grounding, interpretation, implications, decisions, and next actions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data/processed/analysis/model_validation"
TABLES = PROJECT_ROOT / "reports/analysis/tables/model_validation"
OUTPUT = PROJECT_ROOT / "reports/analysis/MODEL_VALIDATION_FULL_RESULTS.md"
MODELS = ["Mini", "Nano", "Claude", "Gemini"]

DIMENSIONS = {
    "ai_method_or_phenomenon": {
        "title": "AI as method or phenomenon",
        "question": "Is AI the substantive object of study, a research method, both, or unclear?",
        "insight": "The distinction separates scholarship about AI-enabled entrepreneurship from scholarship that merely uses AI to analyse another phenomenon. Claude is more method-oriented, whereas Mini and Nano classify more papers as substantive AI phenomena. The disagreement reflects a consequential boundary decision rather than random noise.",
        "implication": "This dimension can support a central argument about whether AI has become an entrepreneurship construct or remains primarily an analytical instrument. Results must retain the four categories; collapsing `both` and `unclear` would conceal meaningful ambiguity.",
        "decision": "Primary dimension. Report Mini full-corpus prevalence, Claude/Gemini weighted sensitivity estimates, and method-by-technical-type and method-by-role contrasts.",
    },
    "ai_role_function": {
        "title": "AI role/function",
        "question": "What explanatory or methodological role does AI occupy?",
        "insight": "Mini and Gemini most often position AI as a tool, while Claude more often identifies a research-method role. This shows why technical AI usage in the method cannot automatically be interpreted as a substantive entrepreneurial capability or actor.",
        "implication": "The role dimension reveals whether AI is treated as a tool, actor, capability, infrastructure, context, label, or research method. It provides a stronger theoretical diagnosis than simple AI mention counts.",
        "decision": "Primary dimension with sensitivity reporting. Examine role-by-method/phenomenon and role-by-mechanism tables, and qualitatively audit consequential actor/capability classifications.",
    },
    "ai_type_form": {
        "title": "Technical AI type/form",
        "question": "Which technical form of AI is actually identified?",
        "insight": "This is the most reliable substantive dimension. Mini, Claude and Gemini converge on unspecified AI and machine learning as the largest categories. The high unspecified share demonstrates that many papers group heterogeneous technologies under a generic AI label.",
        "implication": "Technical under-specification is a direct empirical foundation for the construct-specification argument because machine learning, NLP, computer vision, generative AI and automation are not theoretically interchangeable.",
        "decision": "Primary dimension. Use full Mini prevalence and strong-rater sensitivity; build type-by-role, type-by-mechanism, and method/phenomenon-by-type contrasts.",
    },
    "ai_mechanism_analysis": {
        "title": "Observable AI mechanism",
        "question": "Through what observable logic is AI claimed to affect an outcome?",
        "insight": "Every model identifies substantial mechanism non-observability, although the estimated prevalence varies. Prediction is the most consistently recovered substantive mechanism. Variation primarily concerns how much inference a model permits when the abstract implies but does not state a mechanism.",
        "implication": "The research can claim that many abstracts invoke AI without making an operative mechanism observable. It cannot infer that the unobserved full paper contains no mechanism. Raw and corrected mechanism fields must remain separate.",
        "decision": "Theory-bearing primary dimension, restricted to observable evidence. Report stated mechanisms, inferred mechanisms and mechanism-missing separately; use Mini as primary and Claude/Gemini as a sensitivity range.",
    },
    "level_of_analysis": {
        "title": "Level of analysis",
        "question": "At what analytical level is the AI-related claim located?",
        "insight": "Mini, Claude and Gemini locate much of the literature at firm level, whereas Nano frequently uses venture. The broader organizational-level dominance is more stable than the precise firm-versus-venture boundary.",
        "implication": "Level specification permits diagnosis of cross-level slippage, such as individual-level evidence supporting firm- or ecosystem-level claims.",
        "decision": "Primary dimension. Report organizational, individual and system-level patterns and inspect level-by-mechanism contrasts. Treat venture-versus-firm differences as model-sensitive.",
    },
    "entrepreneurial_process_stage": {
        "title": "Entrepreneurial process stage",
        "question": "Which stage or temporal position in the entrepreneurial process is specified?",
        "insight": "The models apply fundamentally different thresholds. Claude favours static input, Gemini overwhelmingly uses process unspecified, and Nano assigns many substantive stages. Very low alpha shows that the current categories are not consistently operationalized from abstracts.",
        "implication": "A central process-stage claim would be driven by rater interpretation rather than stable measurement.",
        "decision": "Exploratory only. Do not use as a headline construct dimension unless blinded human coding establishes a defensible threshold. Restrict any analysis to explicitly stated stage language.",
    },
    "scope_conditions": {
        "title": "Scope conditions",
        "question": "Under what sectoral, national, organizational or technological boundaries does the claim apply?",
        "insight": "Sector-specific scope dominates across raters, with country-specific scope a smaller but persistent category. Models differ mainly in whether an unbounded claim is coded as missing scope or generalized without scope.",
        "implication": "The dimension supports a critique of generalizing findings obtained in specific sectors, countries or organizational populations to entrepreneurship broadly.",
        "decision": "Theory-bearing primary dimension. Report explicit scope categories and distinguish missing scope from generalized claims; examine scope-by-mechanism patterns.",
    },
    "definition_construct_clarity": {
        "title": "Definition and construct clarity",
        "question": "Is an explicit and claim-fitting definition observable?",
        "insight": "All raters find explicit definitions uncommon, but Gemini applies the strictest criterion. Raw agreement is high because `no definition` dominates, while alpha remains low.",
        "implication": "The data support a statement that explicit definitions are uncommon in titles, abstracts and keywords, not that full papers fail to define AI.",
        "decision": "Supplementary descriptive diagnostic. Report category prevalence and the evidentiary boundary; do not use it as a full-paper clarity score or combine it into a composite misspecification index.",
    },
    "process_sequence_specified": {
        "title": "Process sequence specified",
        "question": "Does the metadata state an ordered temporal or causal sequence?",
        "insight": "Mini and Nano identify sequences frequently, while Claude and Gemini almost never do. The divergence shows incompatible thresholds for what constitutes an explicit sequence.",
        "implication": "High agreement within the strong pair is driven by a dominant `no` category and does not resolve which threshold matches the coding manual.",
        "decision": "Supplementary and human-dependent. Do not use for a central process claim before human adjudication.",
    },
    "ai_definition_present": {
        "title": "AI definition present",
        "question": "Is an explicit AI definition observable in the supplied metadata?",
        "insight": "All models find definitions rare; Mini and Nano use a broader criterion than Claude and Gemini. Dominant `no` prevalence inflates raw agreement.",
        "implication": "The stable broad conclusion is that abstract-level definitions are uncommon. Exact prevalence depends on whether examples and technical descriptions qualify as definitions.",
        "decision": "Supplementary binary diagnostic. Always report yes/no prevalence alongside alpha and avoid claims about unobserved full texts.",
    },
    "ai_distinction_present": {
        "title": "AI distinction present",
        "question": "Does the paper explicitly distinguish AI from neighbouring constructs or technologies?",
        "insight": "Mini applies a much broader interpretation than Nano, Claude and Gemini. Negative four-model alpha indicates that the variable is not reliably operationalized across raters.",
        "implication": "Any substantive result would currently reflect coder threshold choice more than stable evidence.",
        "decision": "Exclude from the main empirical argument unless human coding supplies a clear operational rule. Retain raw output for transparency and diagnostic analysis.",
    },
}


def percent(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.1%}"


def number(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.3f}"


def prevalence_table(prevalence: pd.DataFrame, dimension: str) -> str:
    subset = prevalence[(prevalence.dimension == dimension) & prevalence.model.isin(MODELS)]
    pivot = subset.pivot(index="category", columns="model", values="weighted_prevalence").fillna(0)
    pivot = pivot.reindex(columns=MODELS)
    pivot["_order"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("_order", ascending=False).drop(columns="_order")
    return pivot.map(percent).reset_index().to_markdown(index=False)


def agreement_table(agreement: pd.DataFrame, dimension: str) -> str:
    subset = agreement[(agreement.comparison_set == "probability_sample") & (agreement.dimension == dimension)].copy()
    subset["pair"] = subset.left_model + "–" + subset.right_model
    subset["agreement (95% CI)"] = subset.apply(
        lambda row: f"{row.percent_agreement:.3f} [{row.agreement_ci_low:.3f}, {row.agreement_ci_high:.3f}]", axis=1
    )
    subset["alpha (95% CI)"] = subset.apply(
        lambda row: f"{row.krippendorff_alpha:.3f} [{row.alpha_ci_low:.3f}, {row.alpha_ci_high:.3f}]", axis=1
    )
    return subset[["pair", "comparable", "agreement (95% CI)", "alpha (95% CI)"]].sort_values("pair").to_markdown(index=False)


def grounding_table(grounding: pd.DataFrame, dimension: str) -> str:
    subset = grounding[(grounding.dimension == dimension) & grounding.model.isin(MODELS)].copy()
    if subset.empty:
        return "No dedicated evidence triplet exists for this derived or binary field."
    subset["exact stated-evidence match"] = subset.exact_match_share.map(percent)
    return subset[["model", "stated_evidence_n", "exact_match_n", "exact stated-evidence match"]].to_markdown(index=False)


def main() -> None:
    prevalence = pd.read_csv(DATA / "dimension_prevalence.csv")
    agreement = pd.read_csv(DATA / "agreement_pairwise.csv")
    multirater = pd.read_csv(DATA / "agreement_multirater.csv")
    coverage = pd.read_csv(DATA / "execution_coverage.csv")
    grounding = pd.read_csv(DATA / "evidence_grounding.csv")
    quality = pd.read_csv(DATA / "confidence_evidence_diagnostics.csv")
    macro = pd.read_csv(TABLES / "pairwise_macro_agreement.csv")
    local = pd.read_csv(TABLES / "supplementary_local_macro_agreement.csv")

    sections = []
    for dimension, content in DIMENSIONS.items():
        four = multirater[multirater.dimension == dimension].iloc[0]
        graph = f"figures/model_validation/distribution_{dimension}.svg"
        sections.append(f"""## {content['title']}

**Measurement question.** {content['question']}

**Weighted probability-sample distribution**

{prevalence_table(prevalence, dimension)}

**Pairwise reliability**

{agreement_table(agreement, dimension)}

The four-model common-intersection result is **alpha = {four.krippendorff_alpha:.3f}**, with **{four.unanimous_share:.1%}** unanimous classifications across {int(four.comparable_units):,} comparable papers.

**Evidence-grounding diagnostic**

{grounding_table(grounding, dimension)}

**Interpretation.** {content['insight']}

**Research implication.** {content['implication']}

**Analytical decision and way forward.** {content['decision']}

**Associated figure:** [{Path(graph).name}]({graph})
""")

    sections_text = "\n\n".join(sections)
    report = f"""# Full multi-model validation results and research implications

## Purpose

This document reports the complete model-validation results for the AI–entrepreneurship construct-specification study. It is not a summary. It presents the denominator, coverage, model roles, weighted category distributions, pairwise exact agreement, nominal Krippendorff alpha, 2,000-resample percentile confidence intervals, four-model reliability, evidence-grounding diagnostics, implications for the research argument, and explicit decisions for every coded dimension.

The analysis is reproducible from `scripts/analyze_model_validation.py`. Numeric tables come from `data/processed/analysis/model_validation/`; figures come from `reports/analysis/figures/model_validation/`.

## Evidentiary and inferential boundary

The coding evidence consists only of titles, abstracts and author keywords. Therefore, absence codes mean “not observable in the supplied metadata,” not “absent from the full paper.” Agreement measures reliability, not truth. Mini remains the preselected primary full-corpus rater; Nano is a complete sensitivity baseline; Claude and Gemini are independent strong raters on a frozen probability sample; Llama and Gemma are supplementary local stress tests. No model is treated as an infallible gold standard.

## Analysis populations and coverage

The representative validation population is the frozen 2,235-paper stratified probability sample. The common Mini–Nano–Claude–Gemini intersection contains 2,233 papers. The additional human-anchor-only papers are excluded from representative model prevalence and IRR. Sampling weights are used for category prevalence; agreement is calculated on exact paper-ID intersections.

{coverage.to_markdown(index=False)}

Mini and Gemini cover the complete probability sample. Nano and Claude each have one non-response, leaving a 2,233-paper common-four intersection. Llama and Gemma cover only selective subsets and are not pooled into representative four-model reliability.

## How to interpret the reliability statistics

Exact agreement is the proportion of papers receiving identical codes. Krippendorff alpha adjusts for agreement expected from the category distribution. High agreement with low alpha usually indicates a prevalence problem: raters repeatedly select the same dominant category but do not reliably discriminate minority categories. Confidence intervals quantify paper-sampling uncertainty; they do not include uncertainty caused by prompt choice, model version or the abstract-only evidentiary boundary.

The average values below are orientation summaries only. A mean alpha across heterogeneous dimensions is not a formal overall reliability coefficient; all substantive decisions use dimension-level estimates.

{macro.sort_values('percent_agreement', ascending=False).to_markdown(index=False)}

Claude–Gemini is the most convergent representative pair. Mini is closer to each strong sampled rater than Nano is, supporting Mini's continued role as the primary production rater while preserving model sensitivity as an explicit limitation.

## Interpretation of the global figures

- [Pairwise agreement heatmap](figures/model_validation/pairwise_agreement_heatmap.svg): orientation view of mean exact agreement. It shows relative model convergence but must not replace dimension-level tables.
- [Pairwise alpha heatmap](figures/model_validation/pairwise_alpha_heatmap.svg): prevalence-adjusted orientation view. Lower values than raw agreement reveal dominant-category effects.
- [Claude–Gemini agreement by dimension](figures/model_validation/claude_gemini_dimension_agreement.svg): identifies the strongest and weakest dimension-specific measurements.
- [Model coverage](figures/model_validation/model_coverage.svg): demonstrates why proprietary comparisons are representative and local comparisons are supplementary.
- [Evidence grounding](figures/model_validation/evidence_grounding.svg): conservative exact-text traceability diagnostic; it is not a hallucination rate.

# Results by dimension

{sections_text}

# Cross-model patterns

## Strong-rater convergence

Claude and Gemini agree most strongly on technical AI type/form and method-versus-phenomenon status. Their moderate agreement on role, mechanism, level and scope indicates a stable common signal accompanied by meaningful threshold variation. Their convergence does not prove accuracy, but it narrows the range of defensible model sensitivity.

## Mini as the primary full-corpus rater

Mini has complete 22,345-paper coverage and is closer to Claude and Gemini than Nano is. Its primary estimates should remain the headline results because the rater was selected before the validation comparison and covers the entire population. Claude/Gemini results should be used to identify robust dimensions, quantify sensitivity ranges and flag categories requiring qualitative audit—not to select a post-hoc winning model.

## Nano as a sensitivity baseline

Nano's lower agreement with every strong rater demonstrates capability-related measurement sensitivity. It remains valuable because it shows which conclusions depend on using a stronger model. Nano distributions should be reported as sensitivity results rather than averaged with Mini.

## Supplementary local-model intersections

{local.sort_values('percent_agreement', ascending=False).to_markdown(index=False)}

Llama and Gemma comparisons use small successful intersections and may be affected by selective non-response. Their high literal evidence overlap cannot compensate for low or selective coverage. These estimates are diagnostic stress tests and must not be generalized to the corpus.

## Confidence and evidence labels

Model confidence values are not calibrated on a shared scale. A Claude confidence of 0.6 and Gemini confidence of 0.9 cannot be interpreted as equivalent probabilities. Confidence and evidence-type fields remain useful within each model for sensitivity filters—for example, comparing all Mini results with Mini results having stated evidence and confidence at least 0.8.

The complete confidence/evidence table is available at `data/processed/analysis/model_validation/confidence_evidence_diagnostics.csv`.

# Implications for the research argument

## Defensible central dimensions

The research-grade core should contain:

1. AI as method or phenomenon.
2. Technical AI type/form.
3. AI role/function.
4. Observable AI mechanism.
5. Level of analysis.
6. Scope conditions.

Together they answer whether AI is the object or method of study, what technology is involved, what role AI occupies, how it is claimed to matter, where the claim is located and under what boundaries it applies.

## Central substantive argument

The evidence supports the argument that entrepreneurship research does not employ one consistently specified construct called AI. Papers vary in study status, technical identity, explanatory role, observable mechanism, analytical level and scope. Generic AI labels obscure theoretically different technologies and roles, while mechanism and scope are frequently not observable in abstracts. This heterogeneity matters because findings about machine learning as a research method are not theoretically equivalent to claims about generative AI as an entrepreneurial actor or firm capability.

## Dimensions unsuitable for headline claims

Entrepreneurial process stage is not sufficiently reliable for a central argument. Process sequence and AI distinction use incompatible thresholds across raters. Definition fields support only abstract-level descriptive claims. These outputs remain valuable for transparency and targeted human adjudication but should not be folded into an overall misspecification score.

# Way forward

## Immediate analytical steps

1. Preserve Mini `spec-v3` as the primary 22,345-paper study measurement.
2. Use the six core dimensions for primary descriptive and construct-contrast analysis.
3. Report Claude/Gemini weighted distributions and pairwise reliability as sensitivity evidence.
4. **Next executable stage:** run `python scripts/run_topics.py --optimize-only` and review all five scopes' diagnostics.
5. Explicitly approve the topic parameters before running `python scripts/run_topics.py --use-optimized`.
6. Join approved topic assignments to `primary_analysis_dataset.csv`.
7. Complete blinded human coding in parallel and add human–Mini, human–Claude and human–Gemini reliability without changing the frozen prompt.
8. Generate the locked contrasts: status×type, status×role, type×role, type×mechanism, level×mechanism and process-stage×mechanism, with the final process-stage contrast labelled exploratory.
9. Generate era, journal and query-view distributions with explicit denominators and query-overlap warnings.
10. Qualitatively audit high-frequency, theoretically consequential and model-disagreement cells using the retained evidence text.

## Reporting rules

- Say “observable in the title, abstract and author keywords,” not “absent from the paper.”
- Report exact agreement and alpha together.
- Report sampling and non-response denominators.
- Do not average rater outputs into a synthetic consensus code without a predeclared adjudication rule.
- Do not choose a post-hoc model winner based on the validation sample.
- Do not create a composite misspecification score.
- Treat human coding as an accuracy anchor, not an infallible gold standard.
- Preserve all raw codes, evidence, confidence, failures and protocol fingerprints.

# Limitations

The validation estimates describe abstract-level coding. They do not establish full-text construct specification. Strong-model convergence may reflect shared training or shared conservative interpretation. Exact lexical evidence matching undercounts valid paraphrases. Local-model results are selectively observed. Alpha is sensitive to rare and dominant categories. The human coding phase is still incomplete, so current results establish model reliability and sensitivity but not criterion validity.

# Reproducibility and source files

- Full pairwise results: `data/processed/analysis/model_validation/agreement_pairwise.csv`
- Four-model results: `data/processed/analysis/model_validation/agreement_multirater.csv`
- Weighted prevalence: `data/processed/analysis/model_validation/dimension_prevalence.csv`
- Confusion matrices: `data/processed/analysis/model_validation/confusion_matrices_long.csv`
- Evidence grounding: `data/processed/analysis/model_validation/evidence_grounding.csv`
- Confidence/evidence diagnostics: `data/processed/analysis/model_validation/confidence_evidence_diagnostics.csv`
- Coverage: `data/processed/analysis/model_validation/execution_coverage.csv`
- Manifest: `data/processed/analysis/model_validation/analysis_manifest.json`

Regenerate the statistics and this document with:

```bash
python scripts/analyze_model_validation.py
python scripts/render_model_validation_full_report.py
```
"""
    OUTPUT.write_text(report, encoding="utf-8")
    print(f"Full results report -> {OUTPUT}")


if __name__ == "__main__":
    main()
