# ETV_V2 Pipeline

The ETV_V2 pipeline replaces the old SKOS-oriented downstream stages with a theory-elaboration graph and analytics workflow.

## Stages

```text
Stage 0     - Import Scopus exports from Query 1-4
Stage 0.5   - Merge, deduplicate, and preserve query provenance
Stage 1     - Validate journals/source titles
Stage 1.5   - Filter for AI x entrepreneurship relevance
Stage 1.6   - Create one-hot query columns and query-specific views
Stage 1B    - Export VOSviewer files for full corpus and Query 1-4 subsets
Stage 2A    - Run BERTopic and KeyBERT/keyphrase extraction
Stage 2A.5  - Code AI Specification Framework per paper
Stage 2B    - Build knowledge graph for theory elaboration
Stage 3     - Serve analytics and visualization
```

## Removed Old Downstream Meaning

Stage 2B no longer means LLM-assisted paper screening.

The removed old direction includes:

- SKOS ontology construction as the main platform output.
- AIO/CSO alignment as the main integration objective.
- SSSOM mappings as a primary deliverable.
- ontology validation as the core downstream task.
- ontology-first Hugging Face deployment.

## Query Source Counts

The July 2026 Scopus search counts recorded for the four final queries are:

```text
Query 1 - 29,294 records: broad curated 695-source-title query
Query 2 - 818 records: 2026 FT50 query
Query 3 - 1,097 records: leading entrepreneurship journals query based on Burnell et al. (2026)
Query 4 - 1,509 records: other entrepreneurship journals query
```

These are merged into one corpus. They are not treated as four disconnected datasets.

## Master Dataset Contract

The processed master CSV should eventually include:

- stable paper id
- title
- abstract
- authors
- year
- journal/source title
- document type
- DOI or EID where available
- `query_sources`
- `query_count`
- `in_query_1`
- `in_query_2`
- `in_query_3`
- `in_query_4`
- topic assignments
- KeyBERT/keyphrase assignments
- VOS cluster assignments
- seven AI specification columns
- `specification_problem`

The CSV stays important because it supports VOSviewer, Excel, R, validation, and manual review. The graph adds relational analytics and evidence navigation. Deduplication uses Scopus EID first, DOI second, then normalized title-year matching as a fallback. Query provenance must survive every deduplication step.
