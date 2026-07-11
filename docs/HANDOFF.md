# Handoff — AI-Entrepreneurship Construct Specification Platform (aecsp)

## Where we work
- We are working in **WSL**. Live project path: `~/projects/ETV_V2`
  (UNC from Windows: `\\wsl.localhost\Ubuntu-22.04\home\suvh\projects\ETV_V2`).
- Python package is `aecsp` (renamed from `etv_v2`). Conda env `graphrag`. GPU: RTX 5070 (CUDA available).
- GitHub remote: `https://github.com/ksauka/ai-entrepreneurship-construct-specification.git`.
- Working style: Claude EDITS files; the USER RUNS all terminal commands in WSL. Announce each file,
  keep changes small, no em dashes or emojis in product text, ask before methodological choices.

## What is done and working
- **Stages 0 to 1B** (`scripts/build_corpus.py`): merge + dedup + query provenance, source-title
  validation, AI x business/entrepreneurship relevance filter, five VOSviewer exports.
  `data/processed/master_corpus.csv` = **22,345 papers** (`corpus_relevant`); strict subset
  `ai_ent_relevant` = **2,509**. Query views `query_1..4.csv`.
- **Stage 2A.5 spec coder** (REBUILT 2026-07-10, robust version): `scripts/run_specification.py` +
  `src/aecsp/specification/{schema.py,llm_coder.py,curation.py}`. Runs on `master_corpus.csv` directly
  (paper-level, query-invariant, independent of topics). Prompt is generated from the schema: each
  dimension carries its literature diagnosis (role ambiguity, black-box claims, static input, loose
  label...); coding discipline enforces evidence-before-code, stated/inferred/absent labelling,
  per-dimension confidence, needs_full_text flagging, and an adversarial self-check. Works against any
  OpenAI-compatible endpoint: `--local` = Ollama (free), or OpenAI via `.env`. Cache is per paper AND
  per protocol/model (`data/interim/spec_cache/spec-v3/<model>/`; CURRENT protocol spec-v3,
  2026-07-11: uniform 4,096-token ceiling for every rater plus the mechanism/logic coupling and
  needs_full_text discipline rules added after the spec-v2 audit; spec-v1* and spec-v2 caches are
  retired pilots). VERIFIED end to end on local llama3.2: schema
  enforced, honest confidence spread, evidence real quotes. The stopped continuation preserved 23
  valid llama3.2 cache files.
- **Stage 2A topic optimization and diagnostics** (REBUILT 2026-07-10):
  `run_topics.py --optimize-only` now evaluates `full_corpus` plus native Query 1-4 independently, persists CSV/JSON
  metrics and Plotly HTML graphs, and stops at a human approval boundary. `--use-optimized` is the
  explicit approval action. Phrase-enhanced documents and embeddings are checkpointed under
  `data/interim/topics/`; final global/native models write topic-size, intertopic-distance, hierarchy,
  and topic-term diagnostics. The old fixed-parameter 95-topic output is preserved but invalid as a
  final research result.
- **Stage 2A.5 curation** (BUILT 2026-07-10): `scripts/curate_specification.py` per-dimension
  human-in-the-loop review (modeled on the paper_screening reference). Auto-accept = 'stated' evidence
  + confidence >= 0.8; rest queues least-confident-first; d defers a dimension corpus-wide; decisions
  in `curation_overrides_<model>_spec-v3.json`, LLM cache never modified; `--export` writes
  `paper_specifications_curated_<model>_spec-v3.csv` with `<dim>_curation` status columns.
- **Stage 2B graph** (BUILT): `src/aecsp/knowledge_graph/{schema.py,builder.py,neo4j_loader.py}` +
  `scripts/build_graph.py` (`--export-csv` or `--load`). `docker-compose.yml` for Neo4j.
- **VOS cleaning filter** (BUILT): `src/aecsp/vos/filter.py` + `scripts/apply_vos_filter.py` +
  `tests/test_vos_filter.py`. Splits each scope into retained/dropped by DOI membership in the map.
- **Stage 3 app** (BUILT): `src/aecsp/api/` (`main.py`, `graph_service.py`, `report.py`) and
  `src/aecsp/api/static/` (`esd.css`, `index.html`, `knowledge_graph.html`, `assistant.html`, `citation.js`).
  Dashboard has performance analysis (Chart.js line graph) + in-text citations + DOI links. KG page rebuilt
  to esd standard (controls, fixed legend, info panel, stats bar, node-type toggles, edge labels, click an
  author to list their papers). Run: `uvicorn aecsp.api.main:app --reload --app-dir src --port 8321`.
- Tests: **57 passing** as of 2026-07-10.

## Architecture decisions (locked)
- **Five dataset scopes everywhere**: `full_corpus`, `query_1..4` (overlapping one-hot views, never
  re-deduplicated) via `aecsp.corpus.scopes.iter_scopes` / `scope_frame`. Bonus `strict_ai_ent`.
  Natural labels (Broad business and management journals, FT50 journals, Leading entrepreneurship
  journals, Additional entrepreneurship journals, Full corpus).
- **Knowledge graph**: ONE graph, filtered per scope by `in_query_*` flags. Agreed nodes: Publication,
  Author, Journal, Year, SearchQuery, Institution, Keyword, Reference, Topic, SpecificationProfile +
  AIRole/AIType/Mechanism/LevelOfAnalysis/ProcessStage/ScopeCondition/DefinitionClarity/
  SpecificationProblem. Relationships: WROTE, CO_AUTHORED_WITH, AFFILIATED_WITH, PUBLISHED_IN,
  PUBLISHED_IN_YEAR, CAPTURED_BY, HAS_KEYWORD, HAS_TOPIC, REFERENCES, CITES (internal, DOI match),
  HAS_SPECIFICATION + SPECIFIES_* / HAS_DEFINITION_CLARITY / HAS_SPECIFICATION_PROBLEM.
  Deviations from brief (agreed): NO VOSCluster (VOS is a cleaning step); store only REFERENCES + CITES
  and derive cited-by by inbound traversal; Keyword and Topic are separate node types (NO SKOS/RDF).
- **VOS = citation-connectivity cleaning filter**, not graph metadata. Maps in `data/vosdata/` as
  `master_corpus_vos.csv`, `query_1..4_vos.csv`. Optional, per scope, skipped if missing or older than
  `master_corpus.csv`. Papers in a scope's map = retained (connected); absent = dropped (zero TLS).
- **KG page colours**: single source of truth is the FRONTEND `LABEL_COLOURS` dict; do NOT pass `group`
  to vis-network (it applies its own palette and breaks legend correspondence).
- **Topics**: global BERTopic on full corpus + native per-query models. KeyBERT is NOT a naive
  afterthought; it feeds hybrid phrase detection. Every final model must follow grid diagnostics and
  explicit human approval. The five views receive independent optimization tables and graphs.
- **GPU/CPU split**: RTX 5070/CUDA accelerates PyTorch/SentenceTransformer embeddings and semantic
  phrase work. Standard UMAP, HDBSCAN, CountVectorizer/c-TF-IDF, YAKE, gensim, pandas, extraction,
  and grid metrics remain CPU-bound. High CPU use does not imply CUDA was unused. GPU UMAP/HDBSCAN
  via RAPIDS cuML is not implemented or validated. Do not run specification and topics concurrently.
- **Specification coding experiment (updated 2026-07-11)**: GPT-5.4 mini is
  the adopted primary full-corpus rater. Completed GPT-4.1 nano is the
  prospectively initiated baseline/sensitivity dataset. Llama 3.2 and Gemma 4
  31B are supplementary validation raters and may stop
  below full-corpus completion when runtime is infeasible. This is a prospective
  role assignment, not a benchmark-winner gate. Cache and exports remain isolated per
  model under `data/interim/spec_cache/spec-v3/<model>/` and
  `data/processed/specification/*_<model>_spec-v3.*`.
  Each model codes every paper once; its result is then analyzed across `full_corpus` and Query 1-4.
  The five scopes are five overlapping analytical datasets, not five duplicate LLM calls per model.
  The required model, prompt, schema, request-parameter, corpus-checksum, software, and hardware
  record is defined in `docs/LLM_REPRODUCIBILITY.md`.
- **No spec-v4; nano completes under spec-v3 (2026-07-11)**: the checkpoint audit showed rules 7/8
  bind Gemma (perfect) and Llama (mechanism side) but not nano, whose mechanism codes carry empty
  causal logic ~62% of the time. Decision: no further protocol revision; a PRE-REGISTERED analysis
  rule (in docs/LLM_REPRODUCIBILITY.md, Evidentiary boundary section) treats substantive mechanism
  codes with empty logic as "mechanism missing" at analysis time, applied to all raters equally,
  with raw and corrected prevalences both reported.
- **GPT-5.4 mini decision gate (2026-07-11, supersedes the Claude/DeepSeek candidate plans)**:
  the pinned `gpt-5.4-mini-2026-03-17` ran the deterministic 50-paper challenge set (nano failures,
  lowest-confidence, needs_full_text-heavy, seeded controls) under unchanged spec-v3 and PASSED:
  50/50 coded, ~$0.21, rule-7 empty-logic 0% (nano same papers: 74%), evidence grounding 92%
  (nano 70%), needs_full_text 1.5 vs 5.0 dims/paper, honest confidence spread, discriminant
  validity (20/20 hardest papers mechanism-missing vs 8/18 controls), both nano output-limit
  failures completed. It also accepted the full spec-v3 request shape (temperature 0, strict
  schema) with zero 400s. ADOPTED 2026-07-11 as the primary full-corpus rater. The Gemma
  convergence gate was not required for adoption. The challenge is explicitly
  disclosed as enriched for nano difficulty. Gemma's earlier ~50% circuit
  breaker was dominated by 93 tunnel connection failures and must not be
  represented as a model-quality estimate. Role assignment: mini = primary;
  nano = baseline/sensitivity; Llama/Gemma = supplementary validation.
- **Full-corpus transport = OpenAI Batch (built 2026-07-11)**: correction on record - the Batch
  queue limit caps tokens ENQUEUED AT ONE TIME (not per day), so Batch IS viable at the current
  tier via sequential chunks. Economics at observed tokens (~3,328 in / ~472 out per paper):
  Batch ~$54 (1-3 unattended days) vs live ~$103 (~7 h). `scripts/run_specification_openai_batch.py`
  (logic in `src/aecsp/specification/openai_batch.py`, 6 contract tests): byte-identical request
  body to the live path, same spec-v3 cache (transports resume each other), sequential
  submit/poll/fetch under a 1,850,000-token chunk budget, failures to the shared failures.jsonl,
  paid submission gated behind `--yes`. Claude/DeepSeek/Grok candidates are retired (mini adopted).
- **Evidentiary boundary (2026-07-11)**: abstract-level construct-specification coding only (title,
  abstract, author keywords). A blinded 50-paper human validation anchor is required; it is not
  described as an infallible gold standard. `needs_full_text` is the explicit
  insufficiency indicator (never a routine disclaimer); human curation covers ambiguous/low-confidence
  results; agreement is reported as reliability of abstract-level coding. Dimension observability
  tiers and the verbatim methods statement live in docs/LLM_REPRODUCIBILITY.md ("Evidentiary
  boundary" section).
- **IRR comparison sets**: human–mini and human–nano agreement on the blinded
  sample are the primary validation anchors. Model-only agreement uses exact
  successful paper-ID intersections, including
  `S_mini ∩ S_nano ∩ S_llama ∩ S_gemma` when sufficiently large. Never compare
  independent first-N cache files. Report completion/failure counts,
  intersection size, and journal/year/query coverage; diagnose selection
  against the full corpus and label small/selected intersections illustrative.
- **Working responsibility**: the USER runs all model, topic, graph, Docker, and app commands in WSL.
  Codex provides exact commands, analyzes logs/errors/results supplied by the user, and performs only
  requested code edits, fixes, refactors, tests, and documentation changes. Codex does not launch
  long-running research models unless the user explicitly asks it to do so.
- **Concurrent specification execution**: local Llama (`--workers 1`) and remote
  GPT-4.1 nano (`--workers 10`) may run together. Gemma 4 31B is the strongest
  practical workbench rater and uses the workbench through an SSH tunnel and
  `--base-url`. Its user-owned Ollama server exposes both RTX 4090s, pins context
  at 16,384 and parallelism at 1. `scripts/workbench.sh` is the source of truth.
- **Qwen excluded pilot (2026-07-11)**: `qwen3.5:27b` produced 4 successes and
  51 failures under the strict `spec-v3` schema, predominantly exhausting the
  4,096-token output ceiling without completed JSON. Its cache is retained for
  audit but Qwen is no longer an active/full rater.
- **Gemma context correction (2026-07-11)**: actual prompt usage was ~4,866
  tokens including the strict schema. The 8,192-context attempt could not hold
  prompt plus the 4,096 output ceiling and is archived under
  `spec-v3-gemma8192-pilot/`. The corrected full run starts from zero with
  `OLLAMA_CONTEXT_LENGTH=16384`, both GPUs visible and `100% GPU` required.
- **Protocol lineage (CURRENT: spec-v3, 2026-07-11)**: spec-v1 (1,200-token ceiling) truncated
  valid responses because successful codings cluster at ~900-1,200 output tokens; spec-v2 fixed the
  ceiling uniformly at 4,096 (per-model ceilings rejected: the ceiling is part of the instrument)
  and added per-record prompt_tokens/output_tokens. The 4,568-paper spec-v2 nano audit then exposed
  a MECHANISM LEAK: substantive mechanism codes with EMPTY ai_mechanism_logic (27%) while flagging
  'mechanism missing' as a problem anyway (52%) - the coded mechanism-missing rate (5%) understated
  the corrected rate (31%) six-fold, on the paper's core black-box diagnosis; needs_full_text was
  also flagged routinely (>90% of papers). spec-v3 = identical schema and decoding + discipline
  rule 7 (substantive mechanism REQUIRES non-empty causal logic; code and problem flag must agree)
  and rule 8 (needs_full_text only when the abstract is genuinely insufficient). All raters restart
  under `spec-v3/`; every earlier cache (root llama3.2, spec-v1*, spec-v2) is retained as pilot
  data, never mixed in. Truncated responses raise (finish_reason=length), are logged to
  failures.jsonl, are never cached, and the ceiling is never raised in response (decision rule in
  docs/LLM_REPRODUCIBILITY.md).

## END-TO-END OBJECTIVE (specification and contrasting - NOT the old propositions paper)
IMPORTANT: the workbook's propositions (P1-P4) belong to the OLD paper framing and are retired.
The current paper is SPECIFICATION AND CONTRASTING. Central claim: AI should not enter
entrepreneurship theory as a generic enabling variable; it must be specified as a role-,
mechanism-, level-, process-, and scope-dependent construct. Specification before accumulation.

The platform's job is to make the construct specification problem in AI entrepreneurship research
VISIBLE, INSPECTABLE, and ANALYTICALLY DEFENSIBLE for ETP readers - traceable evidence, not a
narrative review. Concretely it must show:

- Per paper, HOW AI is specified across the seven dimensions (role/function, type/form, mechanism,
  level of analysis, process/sequence, scope conditions, definition/construct clarity), each code
  carrying its evidence. A paper treating AI as a prediction tool, one treating it as a firm
  capability, and one treating it as an autonomous actor are not making the same theoretical claim
  even though all three say "AI" - conceptual precision means separating those meanings.
- WHERE the inconsistency occurs: which journals treat AI mainly as a tool, which topics treat it
  as capability or actor, which papers omit mechanisms, which fail to define AI, where scope
  conditions are missing - across the full corpus and across the Query 1-4 subsets.
- THEORY DIAGNOSIS, not topic description: where AI changes the assumptions of existing
  entrepreneurship theories - if AI expands opportunity search, changes judgement, automates
  evaluation, reshapes experimentation, or alters resource access, the issue is WHAT mechanism is
  changed, at WHAT level, in WHICH process stage, under WHAT scope conditions.
- CONSTRUCT CONTRAST: where papers converge on a similar specification of AI and where they
  diverge while using the same label. Divergence indicates the field uses one term for multiple
  constructs - identifying where future theory should split, refine, or bound the AI construct
  rather than accumulate findings under an unstable label.

The pipeline (query provenance, dedup, journal validation, VOS, topics, knowledge graph) exists so
that convergence, divergence, construct contrast, and misspecification are auditable end to end.

## WORKFLOW (status per stage)

    data/queries/SQ*.csv -> [0-1B build_corpus.py]                    DONE
                              |
                              v
    master_corpus.csv: 22,345 papers, 5 overlapping scopes
                              |
                              v
    [2A.5 full nano + multi-model validation]                        IN PROGRESS
    nano full corpus -> Llama/Gemma validation runs
    -> common paper-ID intersection -> IRR in all 5 scopes
                              |
                              v
    [2A topic optimization]                                          BUILT, NOT RUN
    checkpoint phrases/embeddings -> five grids + graphs -> human review
    -> approved global + native final models -> master_corpus_topics.csv
                              |
                    .---------+---------.
                    v                   v
    [VOS cleaning maps] OPTIONAL   [2A.5b full curation]
                    |                   |
                    '---------+---------'
                              v
    [2B build_graph.py]                                               BUILT, NOT RUN
                              |
                              v
    [3 FastAPI app]                                                   BUILT, KG UNCONFIRMED
                              |
                              v
    [4 specification + contrast analysis]                            NOT BUILT
                              |
                              v
    manuscript evidence and construct-contrast tables

## WHAT REMAINS TO REACH THE OBJECTIVE
1. Specification first: preserve completed GPT-4.1 nano as baseline, complete
   the adopted GPT-5.4 mini primary corpus, and continue Llama/Gemma only as
   feasible supplementary validation. Generate and independently code the
   blinded 50-paper human-validation sample. Preserve every cache and diagnose
   selection into every model-only intersection.
2. Topics second: run `scripts/run_topics.py --optimize-only` across `full_corpus` and Query 1-4,
   inspect the grid tables and graphs, then explicitly approve final parameters or use the reviewed
   recommendations. The earlier unvalidated 95-topic run was stopped and must not be treated as final.
3. Validation for the manuscript: hand-code a stratified sample (~50 papers) and compute agreement
   with the LLM per dimension (the evidence fields make this audit cheap). Needed for methods section.
4. VOS cleaning maps per scope (user runs VOSviewer) + apply_vos_filter.py; decide whether the
   final spec/analysis uses retained-only.
5. Stage 2B: --export-csv, docker compose up, --load; verify CITES + specification nodes in Neo4j.
6. App: visually confirm KG page (esd standard); wire the assistant to the live-app model
   (local Ollama or gpt-4.1-nano-2025-04-14); surface specification views once coded data exists.
7. ANALYSIS STAGE (biggest missing piece, the research deliverable itself): the specification and
   contrasting analytics - misspecification prevalence, trends, per-dimension convergence vs
   divergence under the same label, journal/topic/query-view contrasts, type x mechanism and
   level x mechanism cross-tabs - exportable as manuscript tables. Not designed yet; design
   against the specification-and-contrasting framing above (NOT the old P1-P4 propositions).
8. Housekeeping: commit the stage-2a batch in logical units; push to the GitHub remote.
9. Multi-model downstream plumbing: `build_graph.py` and `GraphService` still read the legacy generic
   `paper_specifications.csv`. Before Stage 2B/app specification views, add explicit model selection
   or multi-model comparison support. Do not silently use the most recently completed model.

## CURRENT STATE AND NEXT STEPS (2026-07-10)
No `run_topics.py` or `run_specification.py` process is currently running. All were terminated at the
user's request so the workflow can restart from documented commands. The earlier Stage 2A run produced an unvalidated 95-topic global
model from one fixed HDBSCAN configuration, not grid search. Those artifacts are diagnostic only.

Topic modeling now has a two-phase optimization boundary. Run five-scope optimization only after the
specification comparison, then review `data/processed/topics/optimization/`. The checkpoint mechanism
is implemented, but `data/interim/topics/` has NOT been created yet; the next optimization run must do
the first expensive phrase/embedding pass. Later final runs reuse it. The column contract remains preserved for
`build_graph.py` and the app: `bertopic_topic` (id), `bertopic_topic_label` (data-driven label),
`bertopic_topic_prob`, `bertopic_was_outlier`, `ai_terms`, `ai_term_count`, `ent_terms`,
`ent_term_count`, `keybert_phrases` as `term:count;...` (builder strips scores).

Spec coder was verified end to end on local llama3.2. Pilot caches are preserved for audit
(root-level `llama3.2/` and all `spec-v1*` directories) but are never reused. Standardized caches
live below `spec-v3/`: Llama uses `llama3.2/`, GPT-4.1 nano uses
`gpt-4.1-nano-2025-04-14/`, and corrected Gemma uses `gemma4_31b/`. Qwen's
`qwen3.5_27b/` is an excluded failed pilot; the 8k Gemma attempt is archived at
`spec-v3-gemma8192-pilot/gemma4_31b/`. Every retained experiment starts from zero
under the uniform 4,096-token protocol. Raw reports and curated CSV exports are
now model-specific so one full model run cannot overwrite another.

Next, in order (details in docs/RUNBOOK.md):
1. USER completes GPT-5.4 mini via the documented Batch workflow; nano remains
   the completed baseline/sensitivity corpus.
2. USER continues Llama and Gemma as supplementary validation raters as
   feasible; Gemma uses the helper/tunnel in `WORKBENCH.md`.
3. Generate the blinded human-validation sample, preserve independent coder
   files, then implement agreement and model-intersection selection diagnostics
   under `docs/METHODS_LOCK.md`. Qwen remains an excluded compatibility pilot.
6. USER runs `python scripts/run_topics.py --optimize-only`, inspects all five scopes' CSV/JSON/HTML
   diagnostics, then explicitly run `--use-optimized` or provide approved manual parameters.
7. Create VOS maps if used, run Stage 2B graph export/load, verify the KG page, and implement Stage 4.

## Also pending
- DONE 2026-07-10: superseded VOS files and `src/aecsp/topics/extract.py` deleted (were untracked;
  plain `rm`).
- User to confirm the rebuilt KG page meets the esd standard (author click -> papers, colours vs legend,
  edge labels). Last KG fixes (EDGE_NAMES, single-source colours, async info panel) are applied but not
  yet visually confirmed by the user.
- Reference implementations to keep matching: `source_projects/esd_platform/knowledge_graph_visualization.html`
  and `.../src/knowledge_graph/`; topic pipeline at
  `source_projects/AI-Entrepreneurship-SKOS-Ontology/src/theory_elaboration/topic_modeling/`;
  screening/curation UX at `.../src/theory_elaboration/paper_screening/`; annotation vocabulary at
  `docs/ETP Theory Elaboration  Writing Workbook.xlsx` (Paper annotation summary tab) - but its
  propositions (P1-P4) are the OLD paper framing; use it for annotation vocabulary only.
- Big uncommitted batch on branch `stage-2a`; commit in logical units. New since the last handoff:
  connected five-scope topic grid search, checkpoint reuse, behavior-preserving accelerated term
  extraction, required Plotly diagnostic graphs, explicit final-selection gate, specification-first
  execution order, full multi-model reliability design, updated tests, RUNBOOK, and HANDOFF.
