# Refactor Plan

ETV_V2 should be refactored by vertical slices, not by moving every old file at once.

## Slice 1: Corpus Contract

- migrate Query 1-4 import assumptions from `AI-Entrepreneurship-SKOS-Ontology`
- produce a deduplicated master CSV
- preserve `query_sources`
- add `in_query_1` to `in_query_4`
- add tests for query membership

## Slice 2: Topic Outputs

- migrate BERTopic and KeyBERT/keyphrase outputs
- normalize topic assignment columns
- map topic/keyphrase outputs to `Topic` graph nodes

## Slice 3: Specification Profiles

- implement Stage 2A.5 columns
- validate seven-dimension completeness
- add `SpecificationProblem` assignment

## Slice 4: Knowledge Graph Export

- build graph-ready node and relationship tables
- support CSV export first
- wire Neo4j ingestion after the export contract is stable

## Slice 5: Platform Views

- migrate the useful graph visualization from `esd_platform`
- point it at the ETV_V2 graph schema
- add paper, journal, topic, author, and specification views

## Working Rule

Keep `source_projects/` as reference material. New code should live in `src/etv_v2` unless there is a specific reason to preserve a legacy path.
