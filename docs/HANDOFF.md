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
  per model (`data/interim/spec_cache/<model>/`). VERIFIED end to end on local llama3.2: schema
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
  in `curation_overrides_<model>.json`, LLM cache never modified; `--export` writes
  `paper_specifications_curated.csv` with `<dim>_curation` status columns.
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
- **Specification coding model ladder (revised 2026-07-10)**: prove the pipeline on local Ollama,
  run an affordable paid pilot, then benchmark current affordable frontier candidates on the same
  stratified papers. Claude Sonnet 5 is a candidate, NOT a locked winner. Select the research-grade
  model using agreement, evidence grounding, structured-output reliability, latency, and projected
  full-run cost. Cache remains isolated per model under `data/interim/spec_cache/<model>/`, so model
  comparisons never contaminate one another. Code each paper once, then analyze the codes through
  all five overlapping dataset views; do not run five duplicate LLM coding jobs. Query 3 (Leading
  entrepreneurship journals) is the priority research-grade rollout, after which the same model cache
  can be extended to `full_corpus` without recoding Query 3 papers.

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
    [2A.5 specification model ladder]                                IN PROGRESS
    llama 25 -> Qwen 25 -> gpt-4.1-nano 25 -> curate/compare
    -> approve research model -> Query 3 -> extend cache to full corpus
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
1. Specification first: finish local/Qwen rehearsals, run the paid pilot, compare candidates on the
   same validation papers, approve the research model, then complete curation. Preserved cache state:
   `data/interim/spec_cache/llama3.2/` contains 23 papers after the stopped local run.
2. Topics second: run `scripts/run_topics.py --optimize-only` across `full_corpus` and Query 1-4,
   inspect the grid tables and graphs, then explicitly approve final parameters or use the reviewed
   recommendations. The earlier unvalidated 95-topic run was stopped and must not be treated as final.
3. Validation for the manuscript: hand-code a stratified sample (~50 papers) and compute agreement
   with the LLM per dimension (the evidence fields make this audit cheap). Needed for methods section.
4. VOS cleaning maps per scope (user runs VOSviewer) + apply_vos_filter.py; decide whether the
   final spec/analysis uses retained-only.
5. Stage 2B: --export-csv, docker compose up, --load; verify CITES + specification nodes in Neo4j.
6. App: visually confirm KG page (esd standard); wire the assistant to the live-app model
   (local Ollama or gpt-4.1-nano); surface specification views once coded data exists.
7. ANALYSIS STAGE (biggest missing piece, the research deliverable itself): the specification and
   contrasting analytics - misspecification prevalence, trends, per-dimension convergence vs
   divergence under the same label, journal/topic/query-view contrasts, type x mechanism and
   level x mechanism cross-tabs - exportable as manuscript tables. Not designed yet; design
   against the specification-and-contrasting framing above (NOT the old P1-P4 propositions).
8. Housekeeping: commit the stage-2a batch in logical units; push to the GitHub remote.

## CURRENT STATE AND NEXT STEPS (2026-07-10)
No `run_topics.py` or `run_specification.py` process is currently running. Both were gracefully
stopped before the pipeline changes. The earlier Stage 2A run produced an unvalidated 95-topic global
model from one fixed HDBSCAN configuration, not grid search. Those artifacts are diagnostic only.

Topic modeling now has a two-phase optimization boundary. Run five-scope optimization only after the
specification comparison, then review `data/processed/topics/optimization/`. The checkpoint mechanism
is implemented, but `data/interim/topics/` has NOT been created yet; the next optimization run must do
the first expensive phrase/embedding pass. Later final runs reuse it. The column contract remains preserved for
`build_graph.py` and the app: `bertopic_topic` (id), `bertopic_topic_label` (data-driven label),
`bertopic_topic_prob`, `bertopic_was_outlier`, `ai_terms`, `ai_term_count`, `ent_terms`,
`ent_term_count`, `keybert_phrases` as `term:count;...` (builder strips scores).

Spec coder was verified end to end on local llama3.2. The interrupted continuation left 23 valid
per-paper cache files in `data/interim/spec_cache/llama3.2/`; preserve them. The Qwen cache directory
will be `qwen3.5_9b-q4_K_M/`, the paid pilot `gpt-4.1-nano/`, and the eventual research model gets its
own sanitized directory name.

Next, in order (details in docs/RUNBOOK.md):
1. Complete the llama comparison set from 23 to 25 cached papers:
   `python scripts/run_specification.py --local --limit 2`.
2. Run the Qwen local rehearsal on the same first 25 papers:
   `python scripts/run_specification.py --local --model qwen3.5:9b-q4_K_M --limit 25`,
   then curate it: `python scripts/curate_specification.py --model qwen3.5_9b-q4_K_M`.
3. Paid pilot: `--model gpt-4.1-nano --limit 25`; curate and compare against Qwen/llama.
4. Approve a research-grade model from the benchmark; do not assume Sonnet 5 wins. Run Query 3
   first, then extend that same cache to `full_corpus`.
5. Run `python scripts/run_topics.py --optimize-only`, inspect all five scopes' CSV/JSON/HTML
   diagnostics, then explicitly run `--use-optimized` or provide approved manual parameters.
6. Create VOS maps if used, run Stage 2B graph export/load, verify the KG page, and implement Stage 4.

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
  execution order, benchmark-based research model selection, updated tests, RUNBOOK, and HANDOFF.
