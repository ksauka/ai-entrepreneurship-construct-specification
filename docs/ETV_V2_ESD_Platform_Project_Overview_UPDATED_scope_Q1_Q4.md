# ETV_V2 / ESD Platform Project Overview

This document explains the updated direction of the project. The project is no longer mainly a generic ESD bibliometric platform and it is no longer an ontology/SKOS construction project. It is now an integrated **AI × Entrepreneurship theory-elaboration platform** built to diagnose how Artificial Intelligence is specified, underspecified, contrasted, and fragmented across entrepreneurship research.

The working integration space is:

```text
/home/kudzai/projects/ETV_V2
```

The original `esd_platform` and `AI-Entrepreneurship-SKOS-Ontology` repositories should be treated as source projects. They provide working components, but new integration work should happen in `ETV_V2` first.

```text
ETV_V2/
  source_projects/
    AI-Entrepreneurship-SKOS-Ontology/
    esd_platform/
```

The project combines three things:

1. The CSV-first FastAPI research platform from `esd_platform`.
2. The query-aware Scopus pipeline, topic modelling, VOSviewer preparation, and AI specification logic from `AI-Entrepreneurship-SKOS-Ontology`.
3. A new knowledge-graph and visual analytics layer for construct specification, construct contrast, convergence, and divergence analysis.

The central goal is to make visible how AI is being used as a theoretical construct in entrepreneurship research. The platform must allow a researcher to inspect papers, topics, journals, authors, search queries, and specification dimensions, and then see where the literature converges, diverges, or fails to specify AI clearly.

---

## 1. Core Research Problem

The project addresses a construct specification problem in AI × entrepreneurship research. AI has entered entrepreneurship scholarship, but not as one stable construct. Across the corpus, AI may appear as:

- a tool entrepreneurs use,
- a capability firms develop,
- an actor or agent that shapes decisions,
- a context that changes entrepreneurial conditions,
- a method used by scholars to study entrepreneurship,
- a predictive technology,
- a generative technology,
- an unspecified label covering analytics, algorithms, automation, or digitalisation.

The platform therefore asks:

```text
Across AI × entrepreneurship research, where do papers converge on what AI is, what role it plays, and how it affects entrepreneurship, and where do they diverge, fragment, or contradict each other?
```

The platform must support two linked forms of analysis:

```text
1. Construct specification
How clearly is AI specified in each paper?

2. Construct contrasting
How do papers, authors, journals, topics, and query subsets specify AI differently?
```

The project is therefore not only a bibliometric review. It is a theory-elaboration evidence system. It gives structured evidence for the claim that AI in entrepreneurship research often suffers from construct ambiguity, black-box causal claims, weak scope conditions, and inconsistent theoretical roles.

---

## 2. Source Projects and What They Contribute

### 2.1 `esd_platform`

The current `esd_platform` repository is a local research analytics platform for bibliometric data. Its active runtime path is CSV-first. A user uploads or provides a Scopus-style CSV, the app loads it into an in-memory pandas DataFrame, then exposes search, analytics, vector search, AI assistant, graph analytics, and visualisation endpoints.

Useful components from `esd_platform` include:

- FastAPI application structure.
- CSV upload and dataset context handling.
- pandas-based analytics routes.
- FAISS vector search.
- AI assistant and HybridGraphRAG scaffolding.
- optional Neo4j integration.
- existing knowledge graph schema and graph ingestion utilities.
- static graph and analytics demo pages.
- WordPress connector and deployment assets.

The original ESD platform can still run as a local CSV app, but for this project it should be treated as infrastructure for the new AI entrepreneurship platform.

### 2.2 `AI-Entrepreneurship-SKOS-Ontology`

The second source project contributes the research pipeline. Some older parts of that repository were designed for SKOS ontology construction, but the current project no longer uses the SKOS output as the main goal. Useful parts include:

- Scopus search-query design.
- Query-aware corpus construction.
- source-title validation.
- deduplication logic.
- journal/source-title filtering.
- AI × entrepreneurship relevance filtering.
- VOSviewer export preparation.
- BERTopic and KeyBERT processing ideas.
- Stage 2A.5 AI specification framework.

The old SKOS stages should be removed from the active project direction. The useful contribution is the pipeline that creates a clean, query-aware, topic-aware, specification-coded corpus.

---

## 3. New Project Purpose

The new platform is an interactive research intelligence system for AI × entrepreneurship theory elaboration. It should allow the researcher to:

1. Merge Scopus exports from the four final search queries.
2. Deduplicate records while preserving query provenance.
3. Remove papers from journals/source titles outside the four-query universe.
4. Filter the corpus to papers that are genuinely about AI and entrepreneurship.
5. Create query one-hot columns for Query 1, Query 2, Query 3, and Query 4.
6. Generate five VOSviewer files: one full corpus file and one file per query.
7. Run BERTopic and keyword extraction to identify topic structure and keyphrases.
8. Code each paper through the AI Specification Framework.
9. Build a knowledge graph linking papers, authors, journals, years, queries, topics, VOS clusters, and specification dimensions.
10. Produce visual and statistical analyses of convergence, divergence, construct contrast, and specification problems.
11. Allow node-level inspection, so clicking a paper or specification dimension reveals connected papers, authors, journals, topics, frameworks, and evidence.

---

## 4. Four-Query Corpus Logic

The dataset is built from four final Scopus search queries:

```text
Query 1 — Broad curated 695 source-title query
Query 2 — 2026 FT50 query
Query 3 — Leading entrepreneurship journals as per Burnell et al. (2026)
Query 4 — Other entrepreneurship journals
```

The platform should not treat these as four isolated datasets. They should be merged into one master corpus, with query provenance preserved.

A paper can appear in more than one query. After deduplication it should appear once in the master dataset, but retain query membership through one-hot columns:

```text
in_query_1
in_query_2
in_query_3
in_query_4
query_count
query_sources
```

This enables the platform to filter every analysis by query subset while still preserving a single deduplicated master corpus.

---

## 4.1 Dataset Scope Principle for Stages 4-9

From the filtering point onward, the platform should treat every downstream analysis as a view over five dataset scopes:

```text
1. Full corpus
2. Query 1 subset
3. Query 2 subset
4. Query 3 subset
5. Query 4 subset
```

This means that the later analytical workflow, including AI × entrepreneurship filtering, query one-hot encoding, VOSviewer outputs, BERTopic modelling, AI specification coding, knowledge graph creation, and platform analytics, should be able to run or recalculate for the full corpus and for each query-specific subset. The query-specific datasets are not mutually exclusive partitions. They are overlapping views derived from the one-hot query columns.

The practical rule is:

```text
One master corpus is deduplicated once.
All query-specific outputs are generated by filtering the master corpus using in_query_1 to in_query_4.
Every later statistic, graph view, table, and visualisation must be reproducible for all five dataset scopes.
```

This scope principle is important because the project needs to compare how different search strategies shape the observed literature. The platform must therefore show whether specification problems, construct contrasts, topic structures, journal patterns, author patterns, and AI framing patterns are stable in the full corpus or are concentrated in one query subset.

---

## 5. Updated Pipeline

The updated pipeline is:

```text
Stage 0    — Import Scopus exports from Queries 1-4
Stage 0.5  — Merge, clean, deduplicate, and preserve query provenance
Stage 1    — Validate journals/source titles against the Query 1-4 source-title universe
Stage 1.5  — Filter for AI × entrepreneurship relevance across the full corpus and Query 1-4 views
Stage 1.6  — Create query one-hot encoding and query-specific dataset views
Stage 1B   — Generate VOSviewer files: full corpus + Query 1-4 subsets
Stage 2A   — BERTopic topic modelling and KeyBERT/keyphrase extraction for the full corpus and query views
Stage 2A.5 — AI Specification Framework, coded per paper and analysable across the full corpus and Query 1-4 views
Stage 2B   — Knowledge graph construction for theory-elaboration analysis across the full corpus and query views
Stage 3    — Interactive platform, visual analytics, and manuscript evidence outputs for the full corpus and Query 1-4 views
```

The old downstream stages are removed from the new project scope:

```text
Remove old Stage 2B LLM paper screening
Remove old Stage 2C theoretical theme profiling
Remove SKOS ontology construction
Remove AIO / CSO alignment
Remove SSSOM
Remove SKOS validation
Remove ontology-based Hugging Face deployment
```

The label `Stage 2B` can be reused, but it should now mean **knowledge graph construction**, not LLM-assisted paper screening.

---

## 6. Stage 0 to Stage 1.6 — Corpus Construction

### 6.1 Import and cleaning

All Scopus exports from the four search queries should be imported. Column names must be standardised, but original Scopus metadata should not be discarded. Important fields include:

```text
EID
DOI
Title
Abstract
Authors
Author Keywords
Index Keywords
Source title
Year
Document Type
Cited by
Affiliations
References
```

### 6.2 Deduplication

Deduplication should use:

```text
1. EID as the primary key
2. DOI as the fallback key
3. title + year as the final fallback
```

The key rule is that query provenance must be collected before duplicates are dropped. For example, if the same paper appears in Query 1 and Query 3, the final master row should contain:

```text
in_query_1 = 1
in_query_2 = 0
in_query_3 = 1
in_query_4 = 0
query_count = 2
query_sources = "Query 1; Query 3"
```

### 6.3 Journal/source-title validation

Every retained paper should be checked against the combined source-title universe from the four search queries. Papers outside that source-title universe should be flagged or removed.

This is important because Scopus source-title handling is sensitive. We already found that parenthetical qualifiers inside source-title filters can break manually pasted Scopus searches. For example, the working Scopus source-title form is:

```text
LIMIT-TO ( EXACTSRCTITLE , "Internet of Things" )
```

not:

```text
LIMIT-TO ( EXACTSRCTITLE , "Internet of Things (Netherlands)" )
```

The same issue applied to:

```text
Agricultural Economics
Corporate Governance
```

The platform should therefore store source-title normalisation rules and use them consistently in search documentation, validation, and filtering.

### 6.4 AI × entrepreneurship relevance filter

The relevance filter should remove records that are not substantively about both AI and entrepreneurship. It should retain papers where AI is part of the entrepreneurship argument and reject papers where AI or entrepreneurship appears only as a marginal or generic term.

The filter can combine:

```text
AI lexical seed terms
entrepreneurship lexical seed terms
semantic rescue for weak lexical matches
manual audit columns
optional LLM-assisted decision support, if needed later
```

The output of this stage is the clean AI × entrepreneurship analytical corpus. This filter should be auditable for the full corpus and for Query 1, Query 2, Query 3, and Query 4, so that the platform can show whether relevance loss differs by search-query strategy.

### 6.5 Query-specific dataset views

After filtering, the platform should generate five dataset views:

```text
dataset_all
dataset_query_1
dataset_query_2
dataset_query_3
dataset_query_4
```

These views should be used consistently across all downstream analytics. Query-specific subsets are derived from one-hot columns, not by re-importing or re-deduplicating separate files. The full corpus and Query 1-4 views should therefore be available as the default scope selector for VOSviewer export, BERTopic outputs, AI specification statistics, knowledge graph queries, visual analytics, and manuscript evidence tables.

---

## 7. Stage 1B — VOSviewer Outputs

The platform should create five VOSviewer-ready files:

```text
VOS_main_all_queries.csv
VOS_query_1.csv
VOS_query_2.csv
VOS_query_3.csv
VOS_query_4.csv
```

The main file contains the full deduplicated and relevance-filtered corpus. Each query file contains papers where the relevant one-hot query column equals 1.

A paper may appear in more than one query-specific VOS file if it was captured by more than one search query. This is correct because the query-specific VOS files represent query views, not mutually exclusive partitions.

The VOS outputs should preserve the fields required for bibliometric coupling, citation analysis, co-authorship, keyword co-occurrence, and source-title analysis.

---

## 8. Stage 2A — Topic and Keyword Layer

Stage 2A remains important. It provides the empirical topic structure of the corpus.

Expected outputs include:

```text
bertopic_topic_id
bertopic_topic_label
bertopic_probability
topic_keywords
topic_size
topic_year_distribution
topic_query_distribution
topic_journal_distribution
keybert_phrases
```

KeyBERT and other keyword outputs should be ingested into the knowledge graph as `Topic` nodes for the MVP because the current graph visualisation and analytics already expect:

```text
(:Publication)-[:HAS_TOPIC]->(:Topic)
```

The relationship should store extraction metadata:

```text
(:Publication)-[:HAS_TOPIC {
  extraction_method: "keybert",
  relevance_score: <score>,
  confidence: <score>,
  keyword_type: "keybert"
}]->(:Topic {text: <keyphrase>})
```

A future cleaner ontology layer can introduce explicit `Keyword` nodes, but the MVP should not break existing topic-based graph routes.

Stage 2A outputs should be generated in two ways: one topic model for the full cleaned corpus, and query-view summaries that report how each query subset is distributed across the full-corpus topics. If separate query-specific BERTopic runs are later useful, they should be marked as secondary sensitivity outputs, because the main platform needs a common topic space for comparing Query 1-4.

---

## 9. Stage 2A.5 — AI Specification Framework

Stage 2A.5 is the conceptual centre of the new project. It is a paper-level specification framework, not a topic-level framing report and not a SKOS concept generator.

Each paper should be coded across seven dimensions:

```text
1. Role/function
2. Type/form
3. Mechanism
4. Level of analysis
5. Process/sequence
6. Scope conditions
7. Definition/construct clarity
```

### 9.1 Role/function

Question:

```text
What role does AI play in the paper's argument?
```

Possible values include:

```text
AI as tool
AI as organisational capability
AI as actor/agent
AI as context or environmental condition
AI as research method
mixed or unclear role
```

Diagnosis:

```text
Role ambiguity
```

This dimension identifies whether AI is being theorised as something entrepreneurs use, something firms embed, something that acts, something that changes the entrepreneurial environment, or something scholars use to study entrepreneurship.

### 9.2 Type/form

Question:

```text
What kind of AI is actually being discussed?
```

Possible values include:

```text
predictive AI
generative AI
machine learning
deep learning
natural language processing
computer vision
recommender systems
large language models
robotics
automation
analytics
general AI
unspecified AI
```

Diagnosis:

```text
Type ambiguity
```

This dimension captures whether a paper distinguishes AI forms or treats AI as homogeneous.

### 9.3 Mechanism

Question:

```text
What exactly is AI doing that changes entrepreneurial outcomes?
```

Possible values include:

```text
reduces uncertainty
expands search
alters judgement
automates decisions
reshapes experimentation
supports learning
changes access to resources
transforms stakeholder interaction
supports prediction
supports recommendation
supports content generation
mechanism missing or unclear
```

Diagnosis:

```text
Mechanism opacity
```

This dimension identifies black-box causal claims where the paper says AI improves or changes entrepreneurship without specifying how.

### 9.4 Level of analysis

Question:

```text
At what level does the AI-related claim operate?
```

Possible values include:

```text
individual entrepreneur
founding team
venture
firm
platform
ecosystem
industry
national system
institutional environment
cross-level or unclear
```

Diagnosis:

```text
Level ambiguity or level mismatch
```

This dimension identifies whether papers mix levels or make claims at one level using evidence from another.

### 9.5 Process/sequence

Question:

```text
When and how does AI matter in the entrepreneurial process?
```

Possible values include:

```text
opportunity recognition
ideation
opportunity evaluation
venture creation
resource acquisition
experimentation
innovation
market entry
scaling
survival
exit
process not specified
static input only
```

Diagnosis:

```text
Process ambiguity
```

This dimension identifies whether AI is positioned in an entrepreneurial sequence or treated only as a static input.

### 9.6 Scope conditions

Question:

```text
Under what conditions does the AI-related claim hold?
```

Possible values include:

```text
early-stage ventures
established firms
SMEs
high-tech startups
solo entrepreneurs
digital platforms
ecosystems
specific industry
specific country or region
specific AI form
generalised or unclear scope
```

Diagnosis:

```text
Scope ambiguity
```

This dimension identifies whether the paper specifies boundary conditions or generalises beyond its evidence.

### 9.7 Definition/construct clarity

Questions:

```text
Does the paper define AI?
Does it distinguish AI from algorithms, analytics, automation, digitalisation, decision support, or information systems?
Does the definition fit the theoretical argument?
```

Possible values include:

```text
clear definition
partial definition
definition present but mismatched
AI not defined
AI used as loose label
AI conflated with analytics/automation/digitalisation
```

Diagnosis:

```text
Construct ambiguity
```

This dimension directly evaluates whether AI is a clearly specified construct.

### 9.8 Specification columns

The master dataset should include paper-level columns such as:

```text
ai_role_function
ai_type_form
ai_method_or_phenomenon
ai_mechanism
level_of_analysis
entrepreneurial_process_stage
process_sequence_specified
scope_conditions
ai_definition_present
ai_distinction_present
definition_clarity
construct_clarity_score
specification_problem
specification_problem_secondary
specification_notes
```

These columns should also be represented in the knowledge graph. The coding unit remains the paper, but every aggregate produced from these codes should be available for the full corpus and for Query 1-4. This allows the platform to compare whether a specification problem is general to the field or mainly produced by one search-query slice.

---

## 10. Knowledge Graph Design

The knowledge graph is the main analytical substrate for the platform. The current active graph path in `esd_platform` is simple:

```text
(:Author)-[:WROTE]->(:Publication)
(:Publication)-[:HAS_TOPIC]->(:Topic)
```

The formal schema has more node types, but many are not currently created by the ingestion path. The new platform should extend the active graph path rather than replace it.

### 10.1 Core bibliometric graph

```text
(:Author)-[:WROTE]->(:Publication)
(:Publication)-[:PUBLISHED_IN]->(:Journal)
(:Publication)-[:PUBLISHED_IN_YEAR]->(:Year)
(:Publication)-[:HAS_LANGUAGE]->(:Language)
(:Publication)-[:HAS_DOCUMENT_TYPE]->(:DocumentType)
(:Publication)-[:REFERENCES]->(:Reference)
(:Publication)-[:CITED_BY]->(:Publication)
```

### 10.2 Search provenance graph

```text
(:Publication)-[:CAPTURED_BY]->(:SearchQuery)
```

`SearchQuery` nodes should include:

```text
query_id
query_name
query_description
search_date
records_retrieved
source_title_count
```

Example query nodes:

```text
Query 1 — Broad curated 695 source-title query
Query 2 — 2026 FT50 query
Query 3 — Leading entrepreneurship journals
Query 4 — Other entrepreneurship journals
```

### 10.3 Topic and keyword graph

```text
(:Publication)-[:ASSIGNED_TO {
  probability: <topic_probability>,
  method: "bertopic"
}]->(:BERTopicTopic)

(:Publication)-[:HAS_TOPIC {
  extraction_method: "keybert",
  relevance_score: <score>,
  confidence: <score>,
  keyword_type: "keybert"
}]->(:Topic)
```

For the MVP, KeyBERT phrases, BERTopic labels, author keywords, and index keywords can all enter through `Topic` nodes with edge metadata. Later, a separate `Keyword` node path can be added.

### 10.4 VOSviewer graph

```text
(:Publication)-[:IN_VOS_CLUSTER]->(:VOSCluster)
```

`VOSCluster` nodes should store:

```text
cluster_id
cluster_label
dataset_view
size
total_link_strength
```

This allows topic structure and bibliometric structure to be compared.

### 10.5 AI specification graph

Each publication has one specification profile:

```text
(:Publication)-[:HAS_SPECIFICATION]->(:SpecificationProfile)
```

The specification profile connects to controlled specification nodes:

```text
(:SpecificationProfile)-[:SPECIFIES_ROLE]->(:AIRole)
(:SpecificationProfile)-[:SPECIFIES_TYPE]->(:AIType)
(:SpecificationProfile)-[:SPECIFIES_MECHANISM]->(:Mechanism)
(:SpecificationProfile)-[:SPECIFIES_LEVEL]->(:LevelOfAnalysis)
(:SpecificationProfile)-[:SPECIFIES_PROCESS]->(:ProcessStage)
(:SpecificationProfile)-[:SPECIFIES_SCOPE]->(:ScopeCondition)
(:SpecificationProfile)-[:HAS_DEFINITION_CLARITY]->(:DefinitionClarity)
(:SpecificationProfile)-[:HAS_SPECIFICATION_PROBLEM]->(:SpecificationProblem)
```

This should not be stored only as flat text in the `Publication` node. It should be represented as graph structure, because the platform needs to show connected papers, journals, authors, topics, and frameworks when a user clicks a specification node.

---

## 11. Construct Specification and Construct Contrast Analytics

The platform must compute statistics that expose specification problems.

### 11.1 Construct specification

Construct specification asks how clearly AI is specified in each paper. The platform should compute:

```text
definition clarity score
role clarity score
type clarity score
mechanism clarity score
level clarity score
process clarity score
scope clarity score
overall specification clarity score
```

A paper with AI role, AI type, mechanism, level, process, scope, and definition all specified has high specification clarity. A paper that only says “AI improves entrepreneurship” without defining AI, specifying AI form, naming a mechanism, or setting scope conditions has low specification clarity.

### 11.2 Construct contrast

Construct contrast asks how papers differ in their specification of AI. The platform should identify cases such as:

```text
same AI type, different role
same role, different mechanism
same mechanism, different level
same process stage, different AI type
same topic, different definition clarity
same journal, divergent AI roles
same author, shifting AI specification across papers
```

These contrasts are not errors by themselves. They become theoretically important because they show where AI is being used as multiple constructs under one label.

### 11.3 Convergence

Convergence occurs when papers in a group share similar specification profiles. Groups can be:

```text
BERTopic topic
VOS cluster
journal
author
search query
year period
entrepreneurial process stage
```

Example:

```text
Most papers in a topic treat AI as predictive AI, as a tool, operating through uncertainty reduction, at the venture level, during opportunity evaluation.
```

That topic shows high specification convergence.

### 11.4 Divergence

Divergence occurs when papers in the same group split across different AI roles, forms, mechanisms, levels, or scope conditions.

Example:

```text
A single topic contains papers where AI is a research method, firm capability, autonomous actor, and entrepreneurial tool.
```

That topic is semantically clustered but constructively fragmented.

### 11.5 Fragmentation and entropy

The platform should calculate fragmentation scores using distributional measures. For each group and each specification dimension, calculate whether one category dominates or whether the distribution is spread across many categories.

Example outputs:

```text
role_fragmentation_score
type_fragmentation_score
mechanism_fragmentation_score
scope_fragmentation_score
overall_construct_fragmentation_score
```

A high fragmentation score means the group uses AI in many different ways.

---

## 12. Visual Analytics Requirements

The platform must provide visual outputs that make construct specification visible.

### 12.1 Overview dashboard

The overview should first expose the selected dataset scope:

```text
full corpus
Query 1
Query 2
Query 3
Query 4
```

For the selected scope, the overview should show:

```text
total papers
papers by query
papers by year
papers by journal
papers by BERTopic topic
papers by VOS cluster
AI role distribution
AI type distribution
mechanism distribution
level distribution
process distribution
definition clarity distribution
specification problem distribution
overall specification clarity score
overall fragmentation score
```

### 12.2 Paper-centred graph view

When a user clicks a paper node, the side panel should show:

```text
title
authors
journal
year
query membership
abstract
BERTopic topic
VOS cluster
keywords
AI role/function
AI type/form
mechanism
level of analysis
process stage
scope conditions
definition clarity
specification problems
nearest convergent papers
nearest contrasting papers
connected authors
connected journals
connected frameworks or theoretical dimensions
```

The paper view should allow the researcher to move from one paper to related or contrasting papers.

### 12.3 Specification-dimension view

When a user clicks a specification node, such as `AI as actor`, `mechanism missing`, or `generative AI`, the platform should show:

```text
all connected papers
top journals
top authors
top BERTopic topics
top VOS clusters
year trend
co-occurring specification dimensions
definition clarity distribution
main specification problems
convergent papers
contrasting papers
```

This allows users to inspect one specification dimension as a construct area.

### 12.4 Journal specification view

For each journal, the platform should show:

```text
number of papers
query coverage
AI role distribution
AI type distribution
mechanism distribution
level distribution
process distribution
scope distribution
definition clarity distribution
most common specification problems
dominant topics
dominant VOS clusters
convergence score
divergence score
fragmentation score
paper list
```

This allows the platform to show whether a journal treats AI mainly as a tool, capability, method, actor, or context.

### 12.5 Author specification view

For each author, the platform should show:

```text
number of papers
co-author network
journals
topics
AI roles used
AI types used
mechanisms used
definition clarity
repeated specification problems
author convergence score
author divergence score
related authors
```

This is important because construct specification may vary by research community or author group.

### 12.6 Topic specification view

For each BERTopic topic, the platform should show:

```text
topic label
topic size
top terms
paper list
dominant AI role
dominant AI type
dominant mechanism
dominant level
dominant process stage
scope condition distribution
definition clarity distribution
specification problem distribution
convergent papers
divergent papers
contrasting papers
linked VOS clusters
linked journals
linked authors
```

This determines whether a topic is a stable construct area or only a keyword cluster.

### 12.7 Convergence-divergence matrix

The platform should include a matrix where rows are groups and columns are specification dimensions.

Possible rows:

```text
BERTopic topics
journals
authors
search queries
VOS clusters
years
```

Columns:

```text
role
type
mechanism
level
process
scope
definition clarity
```

Each cell should indicate whether that group is convergent, divergent, fragmented, or underspecified for that dimension.

### 12.8 Construct contrast network

The construct contrast network should show papers or groups that share one dimension but differ on another.

Examples:

```text
same AI type but different role
same role but different mechanism
same mechanism but different level
same topic but different definition clarity
same journal but different scope conditions
```

This visual is central for showing construct contrasting in the paper.

---

## 13. Platform Filtering Logic

Every analysis must be computed against one active dataset scope and then filterable within that scope. The default scopes are:

```text
full corpus
Query 1
Query 2
Query 3
Query 4
```

Within each active scope, every analysis must be filterable by:

```text
dataset view: full corpus, Query 1, Query 2, Query 3, Query 4
journal
author
year
BERTopic topic
VOS cluster
AI role/function
AI type/form
mechanism
level of analysis
process stage
scope condition
definition clarity
specification problem
```

Filtering should affect:

```text
counts
charts
graph nodes
graph edges
paper tables
VOS exports
topic summaries
journal summaries
author summaries
specification statistics
```

The user should be able to select a dataset view and immediately see all downstream outputs recalculated for that view.

---

## 14. Example Knowledge Graph Queries

### 14.1 Specification problems by query

```cypher
MATCH (p:Publication)-[:CAPTURED_BY]->(q:SearchQuery),
      (p)-[:HAS_SPECIFICATION]->(s:SpecificationProfile)-[:HAS_SPECIFICATION_PROBLEM]->(sp:SpecificationProblem)
RETURN q.name AS query, sp.name AS problem, count(DISTINCT p) AS papers
ORDER BY query, papers DESC
```

### 14.2 AI role by journal

```cypher
MATCH (p:Publication)-[:PUBLISHED_IN]->(j:Journal),
      (p)-[:HAS_SPECIFICATION]->(s:SpecificationProfile)-[:SPECIFIES_ROLE]->(r:AIRole)
RETURN j.name AS journal, r.name AS ai_role, count(DISTINCT p) AS papers
ORDER BY papers DESC
```

### 14.3 Mechanisms by topic

```cypher
MATCH (p:Publication)-[:ASSIGNED_TO]->(t:BERTopicTopic),
      (p)-[:HAS_SPECIFICATION]->(s:SpecificationProfile)-[:SPECIFIES_MECHANISM]->(m:Mechanism)
RETURN t.label AS topic, m.name AS mechanism, count(DISTINCT p) AS papers
ORDER BY papers DESC
```

### 14.4 Papers with missing mechanism but specified AI type

```cypher
MATCH (p:Publication)-[:HAS_SPECIFICATION]->(s:SpecificationProfile),
      (s)-[:SPECIFIES_TYPE]->(typ:AIType),
      (s)-[:HAS_SPECIFICATION_PROBLEM]->(sp:SpecificationProblem {name: "mechanism missing"})
RETURN p.title AS title, typ.name AS ai_type, p.year AS year
ORDER BY year DESC
```

### 14.5 Convergent papers sharing specification profile

```cypher
MATCH (p1:Publication)-[:HAS_SPECIFICATION]->(s1:SpecificationProfile),
      (p2:Publication)-[:HAS_SPECIFICATION]->(s2:SpecificationProfile)
WHERE p1.eid < p2.eid
WITH p1, p2, s1, s2
MATCH (s1)-[:SPECIFIES_ROLE]->(r)<-[:SPECIFIES_ROLE]-(s2)
MATCH (s1)-[:SPECIFIES_TYPE]->(t)<-[:SPECIFIES_TYPE]-(s2)
MATCH (s1)-[:SPECIFIES_MECHANISM]->(m)<-[:SPECIFIES_MECHANISM]-(s2)
RETURN p1.title AS paper_1, p2.title AS paper_2, r.name AS shared_role, t.name AS shared_type, m.name AS shared_mechanism
LIMIT 50
```

### 14.6 Construct contrast: same type, different role

```cypher
MATCH (p1:Publication)-[:HAS_SPECIFICATION]->(s1:SpecificationProfile)-[:SPECIFIES_TYPE]->(typ:AIType),
      (p2:Publication)-[:HAS_SPECIFICATION]->(s2:SpecificationProfile)-[:SPECIFIES_TYPE]->(typ)
WHERE p1.eid < p2.eid
MATCH (s1)-[:SPECIFIES_ROLE]->(r1:AIRole)
MATCH (s2)-[:SPECIFIES_ROLE]->(r2:AIRole)
WHERE r1.name <> r2.name
RETURN typ.name AS shared_type, p1.title AS paper_1, r1.name AS role_1, p2.title AS paper_2, r2.name AS role_2
LIMIT 100
```

---

## 15. Current Runtime Path from `esd_platform`

The current ESD runtime path is still useful for the MVP:

1. `src.api.main` starts the FastAPI application.
2. Startup loads configuration from `.env`.
3. If `data/dataset.csv` exists, `src.data.context.load_csv()` loads it into module-level pandas state.
4. `src.vector_store.initialize.initialize_vector_index()` loads or rebuilds the FAISS vector index.
5. Neo4j is checked through `src.knowledge_graph.enhanced_graph`, but remains optional.
6. The AI assistant is initialized through `src.chatbot.chatbot.ESDChatbot` if the LLM provider is configured.

The active dataset is read through:

```text
src/data/context.py
get_active_dataset()
set_active_filters()
clear_active_filters()
```

For the new project, this runtime path should be modified so the active dataset can represent:

```text
all papers
Query 1 subset
Query 2 subset
Query 3 subset
Query 4 subset
custom filtered subsets
```

---

## 16. API and Platform Extensions Needed

The current API exposes dataset, analytics, vector search, assistant, graph analytics, and visualisation routes. The new project requires additional endpoints.

### 16.1 Dataset and query endpoints

```text
GET  /api/corpus/status
GET  /api/corpus/query-views
POST /api/corpus/filter
GET  /api/corpus/paper/{paper_id}
```

### 16.2 Specification endpoints

```text
GET  /api/specification/overview
GET  /api/specification/dimensions
GET  /api/specification/problems
GET  /api/specification/by-query
GET  /api/specification/by-topic
GET  /api/specification/by-journal
GET  /api/specification/by-author
GET  /api/specification/paper/{paper_id}
```

### 16.3 Convergence/divergence endpoints

```text
GET  /api/construct/convergence
GET  /api/construct/divergence
GET  /api/construct/contrast
GET  /api/construct/fragmentation
```

### 16.4 Graph endpoints

```text
GET  /api/graph/paper/{paper_id}
GET  /api/graph/specification/{dimension}/{value}
GET  /api/graph/journal/{journal_id}
GET  /api/graph/author/{author_id}
GET  /api/graph/topic/{topic_id}
```

These endpoints should return both graph elements and summary statistics for side-panel visualisation.

---

## 17. Data Model Additions

The master analytical dataset should contain at least the following new columns.

### 17.1 Query provenance

```text
in_query_1
in_query_2
in_query_3
in_query_4
query_count
query_sources
```

### 17.2 Topic outputs

```text
bertopic_topic_id
bertopic_topic_label
bertopic_probability
bertopic_top_terms
keybert_phrases
keybert_scores
vos_cluster
vos_total_link_strength
```

### 17.3 AI specification dimensions

```text
ai_role_function
ai_type_form
ai_method_or_phenomenon
ai_mechanism
level_of_analysis
entrepreneurial_process_stage
process_sequence_specified
scope_conditions
ai_definition_present
ai_distinction_present
definition_clarity
construct_clarity_score
specification_problem
specification_problem_secondary
specification_notes
```

### 17.4 Construct analytics

```text
role_clarity_score
type_clarity_score
mechanism_clarity_score
level_clarity_score
process_clarity_score
scope_clarity_score
definition_clarity_score
overall_specification_score
construct_contrast_flag
fragmentation_group_flag
```

---

## 18. Outputs Required for the Manuscript

The platform should produce outputs that can feed the paper directly. Where possible, each output should be generated for five scopes: full corpus, Query 1, Query 2, Query 3, and Query 4. This applies especially to specification-problem statistics, convergence/divergence matrices, VOSviewer files, topic summaries, journal summaries, author summaries, and construct contrast evidence.

### 18.1 Tables

```text
Table: Query source counts before and after deduplication
Table: Journal/source-title coverage by query
Table: Final corpus filtering flow
Table: BERTopic topic profile
Table: AI specification dimensions and coding rules
Table: Specification problems by topic
Table: Specification problems by journal
Table: Specification convergence/divergence by topic
Table: Construct contrasts by AI type and role
```

### 18.2 Figures

```text
PRISMA-style search and filtering flow
Query overlap diagram
Topic map
VOSviewer network exports
AI specification distribution chart
Convergence-divergence matrix
Construct contrast network
Journal specification heatmap
Author specification network
Paper-centred knowledge graph view
```

### 18.3 Evidence files

```text
master_clean_corpus.csv
master_specification_dataset.csv
query_1_dataset.csv
query_2_dataset.csv
query_3_dataset.csv
query_4_dataset.csv
vos_main_all_queries.csv
vos_query_1.csv
vos_query_2.csv
vos_query_3.csv
vos_query_4.csv
kg_nodes.csv
kg_edges.csv
specification_problem_evidence.csv
construct_contrast_evidence.csv
```

---

## 19. Project Status Notes

Several parts are working but need alignment:

- The original `esd_platform` README describes a CSV-first app using `data/dataset.csv`.
- The repository also contains a richer SQLite database at `data/esd_platform.db`.
- Neo4j is optional in the current app, but required for the full graph analytics vision.
- The current active KG ingestion path creates only `Author`, `Publication`, and `Topic` nodes.
- The formal schema supports more nodes than the active ingestion path currently creates.
- `Keyword` exists in the schema, but should not be the MVP ingestion target unless visualisation and graph queries are updated.
- KeyBERT phrases should enter through `Topic` nodes first.
- The old ontology/SKOS language should be removed from active pipeline documentation unless clearly marked as deprecated.
- Stage 2A.5 should be treated as the AI Specification Framework and the main theoretical coding layer.
- `ETV_V2` is the integration workspace. Do new architecture work there first.

---

## 20. Updated Mental Model

Think of the project in six layers:

```text
1. Local platform layer
FastAPI, pandas, CSV upload, vector search, assistant, visualisation routes.

2. Query-aware corpus layer
Merge Scopus Query 1-4 exports, deduplicate, preserve query provenance, validate source titles.

3. Bibliometric and topic layer
VOSviewer exports, BERTopic topics, KeyBERT phrases, journal/year/author analytics.

4. AI specification layer
Paper-level coding across role/function, type/form, mechanism, level, process, scope, and definition clarity, with all aggregate statistics available for the full corpus and Query 1-4.

5. Knowledge graph layer
Publication, Author, Journal, SearchQuery, BERTopicTopic, VOSCluster, Topic, SpecificationProfile, and controlled specification nodes, with graph traversals scoped to the full corpus or any Query 1-4 subset.

6. Theory-elaboration analytics layer
Convergence, divergence, construct contrast, specification failures, evidence tables, and interactive visualisations, all reproducible for the full corpus and Query 1-4.
```

The final platform should allow a researcher to move from a high-level statistic to the exact papers behind it. If the platform reports that a topic has high construct fragmentation, the user must be able to click that result and see which papers caused the fragmentation, which specification dimensions diverge, and how those papers connect to authors, journals, queries, VOS clusters, and theoretical frameworks.

The project’s final contribution is therefore a working evidence system for AI construct specification in entrepreneurship research. It supports the manuscript by showing, in a reproducible and inspectable way, where AI is well specified, where it is used as a loose label, where papers converge, and where construct contrasts make cumulative theory development difficult.
