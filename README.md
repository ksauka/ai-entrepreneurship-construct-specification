# ETV_V2

ETV_V2 is the clean hybrid project for a theory-elaboration evidence platform that diagnoses how Artificial Intelligence is specified in entrepreneurship research. It borrows from two existing projects but gives the work a new home, a new package, and a schema built around construct specification, convergence, divergence, and construct contrast.

## Source Projects

The two original projects are preserved intact under `source_projects/`:

```text
source_projects/
  AI-Entrepreneurship-SKOS-Ontology/
  esd_platform/
```

Their role is now reference and migration material. New implementation should happen in the clean ETV_V2 structure, then old code can be copied, adapted, or trimmed deliberately.

## Project Purpose

ETV_V2 represents each paper as a query-aware, topic-aware, and specification-coded node in a knowledge graph. The platform should let a researcher inspect how AI is specified across journals, authors, topics, search-query subsets, VOS clusters, years, and individual papers, then identify convergence, divergence, construct contrast, and recurring specification failures. The July 2026 source searches are recorded as Query 1 = 29,294 records, Query 2 = 818 records, Query 3 = 1,097 records, and Query 4 = 1,509 records before merge and deduplication.

Type/form uses two separate paper-level columns. `ai_method_or_phenomenon`
identifies whether AI is the phenomenon studied, a research method, both, or
unclear; `ai_type_form` identifies the technical form. Both are preserved in
combined and curated datasets.

This is not the old SKOS ontology pipeline. It is also not only the old ESD dashboard. It is a theory-elaboration evidence system built from bibliometric data.

## Clean Structure

```text
ETV_V2/
  configs/                 # New platform configuration files.
  data/
    raw/                   # Original Scopus and VOS inputs.
    interim/               # Deduplicated and enriched working datasets.
    processed/             # Master analytical CSVs and graph-ready exports.
    exports/vosviewer/     # VOSviewer files for full corpus and query subsets.
  docs/                    # New architecture and platform documentation.
  scripts/                 # Thin command-line entrypoints.
  source_projects/         # Full copied source projects, kept as reference.
  src/aecsp/
    api/                   # Future FastAPI surface borrowed from esd_platform.
    analytics/             # Construct convergence, divergence, and contrast.
    corpus/                # Query import, merge, dedup, validation, provenance.
    knowledge_graph/       # New KG schema and builders.
    pipeline/              # Stage registry and orchestration.
    specification/         # Stage 2A.5 AI specification framework.
    topics/                # BERTopic, KeyBERT, and keyphrase extraction.
    visualization/         # Graph/dashboard views.
    vos/                   # VOSviewer export and cluster integration.
  tests/                   # Contract and migration tests.
```

## Pipeline

```text
Stage 0     - Import Scopus exports from Query 1-4
Stage 0.5   - Merge, deduplicate, and preserve query provenance
Stage 1     - Validate journals/source titles
Stage 1.5   - Filter for AI x entrepreneurship relevance
Stage 1.6   - Create one-hot query columns and query-specific views
Stage 1B    - Export VOSviewer files for full corpus and Query 1-4 subsets
Stage 2A.5  - Run full multi-model specification coding for reliability
Stage 2A    - Grid-search, review, then run BERTopic and keyphrase extraction
Stage 2B    - Build knowledge graph for theory elaboration
Stage 3     - Serve analytics and visualization
```

## Core Knowledge Graph

The MVP keeps the existing workable graph path:

```text
(:Author)-[:WROTE]->(:Publication)
(:Publication)-[:HAS_TOPIC]->(:Topic)
```

It extends that path into the target ETV_V2 graph:

```text
(:Publication)-[:CAPTURED_BY]->(:SearchQuery)
(:Publication)-[:PUBLISHED_IN]->(:Journal)
(:Publication)-[:PUBLISHED_IN_YEAR]->(:Year)
(:Publication)-[:HAS_TOPIC]->(:Topic)
(:Publication)-[:IN_VOS_CLUSTER]->(:VOSCluster)
(:Publication)-[:HAS_SPECIFICATION]->(:SpecificationProfile)

(:SpecificationProfile)-[:SPECIFIES_ROLE]->(:AIRole)
(:SpecificationProfile)-[:SPECIFIES_TYPE]->(:AIType)
(:SpecificationProfile)-[:SPECIFIES_MECHANISM]->(:Mechanism)
(:SpecificationProfile)-[:SPECIFIES_LEVEL]->(:LevelOfAnalysis)
(:SpecificationProfile)-[:SPECIFIES_PROCESS]->(:ProcessStage)
(:SpecificationProfile)-[:SPECIFIES_SCOPE]->(:ScopeCondition)
(:SpecificationProfile)-[:HAS_DEFINITION_CLARITY]->(:DefinitionClarity)
(:SpecificationProfile)-[:HAS_SPECIFICATION_PROBLEM]->(:SpecificationProblem)
```

BERTopic topics, KeyBERT phrases, and extracted keywords can be stored as `Topic` nodes with relationship metadata. AI specification values must stay in the `SpecificationProfile` layer because they are theoretical coding variables, not keyword outputs.

## Current Contracts

The first stable contracts live in:

- `src/aecsp/corpus/query_provenance.py`
- `src/aecsp/corpus/merge.py`
- `src/aecsp/specification/schema.py`
- `src/aecsp/knowledge_graph/schema.py`
- `src/aecsp/pipeline/stages.py`

Run the contract tests with:

```bash
pytest
```

## Current Status and Next Step

Corpus construction and full Mini/Nano coding are complete. Claude and Gemini
completed the frozen proprietary validation target, probability-sample IRR is
built, and the canonical full study dataset is
`data/processed/analysis/primary_analysis_dataset.csv` (22,345 papers).

The next executable stage is five-scope topic optimization:

```bash
python scripts/run_topics.py --optimize-only
```

Review and explicitly approve its recommendations before running
`python scripts/run_topics.py --use-optimized`. Approved topics are then joined
to the canonical dataset for Stage 4 contrasts, after which the graph/app are
run and verified. `docs/RUNBOOK.md` is the authoritative execution order.
