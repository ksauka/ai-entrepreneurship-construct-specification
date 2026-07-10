# ETV_V2 — Project Instructions & Handover

Per-project instructions for Claude Code. A fresh session should read this first.

## What this project is (design, not motivation)
A knowledge-graph-based construct specification platform that diagnoses how "AI" is
specified in entrepreneurship research, for a PhD theory-elaboration paper. Motivation
(why AI is inconsistently specified) is in
`docs/ETV_V2_ESD_Platform_Project_Overview_UPDATED_scope_Q1_Q4.md` Section 2 — do not
confuse it with the design below.

## Design: five connected layers
1. Corpus layer — deduplicated Scopus records, authors, journals, years
2. Query provenance layer — Query 1-4 one-hot membership, query-specific views
3. Bibliometric/topic layer — VOSviewer clusters, BERTopic topics, KeyBERT keyphrases
4. AI specification layer — Stage 2A.5: every paper coded on SEVEN dimensions
   (role/function, type/form, mechanism, level of analysis, process/sequence, scope
   conditions, definition/construct clarity). Controlled vocabularies + brief-aligned
   column names live in `src/aecsp/specification/schema.py` (SOURCE OF TRUTH):
   ai_role_function, ai_type_form, ai_mechanism, level_of_analysis,
   entrepreneurial_process_stage, scope_conditions, definition_construct_clarity,
   specification_problem.
5. Research intelligence layer — Neo4j knowledge graph, convergence/divergence
   statistics, construct-contrast networks, interactive evidence visualisation.

## Design rule: five dataset scopes EVERYWHERE
Every output from Stage 2A onward is produced for ALL FIVE scopes: full_corpus,
query_1, query_2, query_3, query_4 (overlapping one-hot views, NEVER re-deduplicated).
Use `aecsp.corpus.scopes.iter_scopes(master)` / `scope_frame(master, scope_id)`.
Because every master paper is in >=1 query, the union of the four query views IS the
full corpus — so expensive per-paper steps (KeyBERT, LLM coding) run ONCE on
full_corpus and each result appears in every view the paper belongs to. Bonus scope:
strict_ai_ent (the 2,509-paper strict AI x entrepreneurship subset, a column flag).

## Pipeline status
DONE — stages 0 -> 1B (committed on main; `python scripts/build_corpus.py`, ~2 min):
- 0/0.5 merge + dedup + provenance: 32,718 raw -> 30,673 unique
- 1 source-title validation (704-journal universe from the saved queries)
- 1.5 relevance: corpus_relevant (master) -> 22,345; ai_ent_relevant (strict) -> 2,509
- 1.6 master_corpus.csv + query_1..4.csv
- 1B five VOSviewer exports in data/exports/vosviewer/

BUILT, awaiting your runs — stages 2A / 2A.5 (branch stage-2a):
- 2A `scripts/run_topics.py`: GLOBAL BERTopic (feeds KG, comparable) + NATIVE per-query
  models (each dataset's own topics) + KeyBERT on full_corpus. Embeddings computed once.
- 2A.5 `scripts/run_specification.py`: OpenAI GPT codes 7 dimensions/paper, JSON-schema
  structured output, per-paper cache in data/interim/spec_cache/, --dry-run cost preview.
  Needs .env with OPENAI_API_KEY.

TODO:
- 2B Neo4j knowledge graph — `scripts/build_graph.py`, docker-compose, loader, graph
  stats. (Being built now.) In-memory GraphDraft builder + node/rel schema already exist
  in `src/aecsp/knowledge_graph/`.
- 3 FastAPI analytics + visualisation with evidence traceability (every statistic
  clickable back to its paper list, per scope).

## Environment
- Repo: ~/projects/ETV_V2 (WSL Ubuntu), remote https://github.com/ksauka/ETV_V2.git
- Python: conda env `graphrag` (pandas, pytest, pyyaml, bertopic 0.17.4, keybert 0.9.0,
  openai). `pytest` should stay green.
- Raw Scopus CSVs: data/queries/SQ*.csv (gitignored); paths in configs/data_sources.yaml
- Old reference code (esd_platform FastAPI/Neo4j/viz) is NOT in this repo; it is in the
  Windows copy's source_projects/. Ask before assuming it is available.

## Working style (IMPORTANT — see also the user's global preferences)
- Claude EDITS files (user reviews diffs in VS Code); the USER RUNS all terminal
  commands in WSL. Give paste-ready command blocks; do not wrap WSL in PowerShell.
- Announce each file before touching it; keep changes small; commit in logical units.
- Never commit data/queries/*.csv, generated outputs, .env, or spec_cache.
- ASK before methodological decisions (corpus scope, coding method, thresholds).
  Proceed freely on plumbing/refactoring.
