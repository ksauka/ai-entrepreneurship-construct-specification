# ETV_V2 Architecture

ETV_V2 is organized as a clean project with the old systems kept as source material. The working code should live in `src/aecsp`; copied code from the old projects should be adapted into that package only when it matches the new platform contract.

## Source Roles

`AI-Entrepreneurship-SKOS-Ontology` contributes the bibliometric pipeline:

- Scopus Query 1-4 import assumptions.
- merge and deduplication logic.
- source-title validation universe.
- VOSviewer export workflow.
- BERTopic and KeyBERT/keyphrase extraction ideas.
- Stage 2A.5 AI specification framework direction.

`esd_platform` contributes the application and graph surface:

- FastAPI structure.
- pandas-backed local dataset analytics.
- Neo4j utilities and graph schema ideas.
- graph visualization pages.
- dashboard patterns.

## Target Layers

```text
Corpus layer
  Scopus records, deduplicated papers, authors, journals, years, document types.

Search provenance layer
  Query 1-4 one-hot membership and SearchQuery graph nodes.

Bibliometric layer
  VOSviewer exports, VOS clusters, citation and coupling fields.

Topic and keyword layer
  BERTopic topics, KeyBERT phrases, extracted keywords, topic probabilities.

AI specification layer
  Role/function, type/form, mechanism, level, process, scope, definition clarity,
  and specification problems.

Platform analytics layer
  Query-specific, topic-specific, journal-specific, author-specific, year-specific,
  and specification-problem statistics.
```

## Package Boundaries

`corpus/` owns import, merge, deduplication, query provenance, source-title validation, and the master analytical CSV.

`topics/` owns BERTopic, KeyBERT, and keyword/keyphrase outputs. For the MVP, these outputs become `Topic` nodes in the graph.

`specification/` owns Stage 2A.5. It defines the theoretical coding dimensions and the specification-problem vocabulary.

`knowledge_graph/` owns graph schema, export, and eventually Neo4j ingestion. It should not duplicate pandas analytics logic.

`analytics/` owns convergence, divergence, fragmentation, clarity, and contrast metrics.

`visualization/` owns graph and dashboard adapters. It should consume graph or analytics outputs rather than recreate the pipeline.

`api/` will later expose the platform through FastAPI.
