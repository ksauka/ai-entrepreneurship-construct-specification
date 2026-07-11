# Locked study and analysis protocol

Status: locked 2026-07-11, before the GPT-5.4 mini full-corpus run and before
inspection of Stage 4 substantive distributions.

The machine-readable source of truth is `configs/experiment_register.json`.
It records the corpus checksum, rater roles, analysis rules, scope hierarchy,
Git head, and the dirty-worktree diff checksum present at lock time.

## Study roles

- `gpt-5.4-mini-2026-03-17` is the authoritative primary full-corpus rater.
- Completed `gpt-4.1-nano-2025-04-14` results are the prospectively initiated
  baseline and full-corpus sensitivity dataset.
- Llama 3.2 and Gemma 4 31B are supplementary validation raters. Their
  successful records are retained; neither must finish the corpus.
- Qwen 3.5 27B is an excluded protocol-compatibility pilot.
- The 50-paper mini challenge set was enriched for nano difficulty and is
  evidence of protocol compliance, not an unbiased estimate of population
  accuracy. Its selection procedure and results must be disclosed.

## Evidentiary claims

The unit of measurement is what is observable in a paper's title, abstract,
and author keywords. Report “no mechanism was observable in the available
metadata,” never “the paper contains no mechanism.” `needs_full_text` records
metadata insufficiency; it is not evidence of theoretical failure.

The protocol lineage is instrument development: spec-v1 and spec-v2 are pilot
calibration stages on the target corpus; spec-v3 is the frozen study
instrument. Do not describe this project as prospectively preregistered. The
empty-mechanism-logic correction was locked before the completed full-corpus
analysis and is reported both raw and corrected.

## Human validation

A 50-paper sample is selected independently of model outputs using a fixed
seed and stratification across publication era, query provenance, and abstract
length. Human coders are blind to model identity and outputs. They code the
same abstract-level dimensions independently; disagreements are not silently
adjudicated. Report per-dimension agreement and error patterns against the
primary and baseline raters. This is a human validation anchor, not an
infallible gold standard.

## Scope hierarchy

`full_corpus` is the sole primary analytical population. Query 1–4 are
overlapping descriptive and sensitivity views, not independent samples.
Report overlap counts and avoid pooled tests that treat repeated papers as
independent observations. VOS connectivity filtering, if used, is a
sensitivity analysis only.

## Locked Stage 4 outputs

1. Coverage and non-response by rater, year, journal, query membership,
   abstract length, and available-metadata completeness.
2. Per-dimension category counts and prevalence for the primary rater.
3. Raw and corrected mechanism-observability prevalence.
4. Human–model and model–model agreement by dimension: percent agreement and
   Krippendorff's alpha for nominal codes, with bootstrap confidence intervals.
5. IRR intersection selection diagnostics against the full corpus. If the
   intersection is small or selected, characterize agreement as illustrative.
6. Descriptive journal, era, topic, and query-view contrasts with denominators
   and overlap disclosed.
7. Construct-contrast cross-tabs for type × role, type × mechanism, level ×
   mechanism, and process stage × mechanism, with traceable paper IDs.

Normalized entropy is labelled category concentration/dispersion only. It is
not construct clarity, theoretical agreement, convergence, fragmentation, or
evidence of misspecification. The model-generated 0–100 construct-clarity score
is diagnostic metadata and is excluded from primary claims unless the human
validation study supports it.

No composite misspecification index, significance test, topic-linked causal
claim, or additional correction rule may be introduced after inspecting the
full distributions without being labelled exploratory.
