# ETV_V2 Scope Update Summary

Updated the project overview to make the dataset-scope rule explicit: downstream stages and analytics should be generated for five scopes, namely the full corpus plus Query 1, Query 2, Query 3, and Query 4.

Main changes:

- Added Section 4.1, Dataset Scope Principle for Stages 4-9.
- Updated the pipeline so Stages 1.5 through Stage 3 are framed as full-corpus plus Query 1-4 outputs.
- Clarified that query-specific datasets are overlapping views derived from one-hot query columns, not separate deduplicated corpora.
- Clarified that BERTopic should keep a common full-corpus topic space, with Query 1-4 topic summaries as comparable views.
- Updated visual analytics, filtering, outputs, and mental model sections so all key statistics and graph views can be scoped to full corpus or Query 1-4.
