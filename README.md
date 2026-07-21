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

ETV_V2 represents each paper as a query-aware, topic-aware, and specification-coded node in a knowledge graph. The platform lets a researcher inspect how AI is specified across journals, authors, topics, search-query subsets, business domains, years, and individual papers, then examine construct composition and contrast with traceable evidence. The July 2026 source searches are recorded as Query 1 = 29,294 records, Query 2 = 818 records, Query 3 = 1,097 records, and Query 4 = 1,509 records before merge and deduplication.

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
  scripts/                 # Thin command-line entrypoints.
  source_projects/         # Full copied source projects, kept as reference.
  src/aecsp/
    api/                   # FastAPI routes, authentication, reports, and UI.
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

Operational research documentation, execution records, manuscript material,
and the detailed runbook are maintained locally under `docs/` and are
deliberately excluded from Git. The root README is the only tracked document.

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
Stage 4     - Classify domains and run theory-elaboration contrasts
```

## Core Knowledge Graph

The graph is built from the checksum-verified primary analysis dataset and uses
one database for every analytical scope. Query membership flags filter the same
publication nodes into `full_corpus`, `query_1` to `query_4`, and the optional
`strict_ai_ent` view. The locked core paths are:

```text
(:Author)-[:WROTE]->(:Publication)
(:Author)-[:CO_AUTHORED_WITH]->(:Author)
(:Author)-[:AFFILIATED_WITH]->(:Institution)
(:Publication)-[:CAPTURED_BY]->(:SearchQuery)
(:Publication)-[:PUBLISHED_IN]->(:Journal)
(:Publication)-[:PUBLISHED_IN_YEAR]->(:Year)
(:Publication)-[:HAS_TOPIC]->(:Topic)
(:Publication)-[:HAS_KEYWORD]->(:Keyword)
(:Publication)-[:REFERENCES]->(:Reference)
(:Publication)-[:CITES]->(:Publication)
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

BERTopic assignments use `Topic`. Author, index, and extracted keywords use the
separate `Keyword` label. AI specification values stay below
`SpecificationProfile` because they are theoretical coding variables, not topic
or keyword outputs. The contract contains no `VOSCluster` or SKOS nodes.

The `/knowledge-graph` interface starts from a bounded scoped seed and expands
lazily. Single-click focuses on direct neighbors; double-click expands one hop.
Node colors come only from the frontend `LABEL_COLOURS` dictionary. When Neo4j
is unavailable, the page shows a bounded dataframe seed rather than failing.

Neo4j loading uses administrative credentials, but the web application accepts
only separate `NEO4J_APP_USER` credentials. A genuine database-enforced reader
role requires Neo4j Enterprise; Community Edition has no roles. Exact setup,
load, verification, and fallback commands are retained in the local operational
runbook.

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

Five-scope topic optimization and final training are complete. Topic analysis uses the
53-topic Full Corpus model and the independently fitted 50-, 13-, 6-, and
8-topic Query 1-4 models. Official ASJC assignment, reviewed business-domain
membership, and the Construct Specification and Construct Contrasting platform
views are implemented. A tier-independent, sequential 11-shard Gemini
full-corpus plan is prepared offline; no full provider batch has been
submitted. The next release gates are full-corpus Claude/Gemini budget approval,
final theory-elaboration tables, human
annotation, and researcher interpretation of all 130 scope-topic labels.
Topic names
can be inspected, renamed, revised, and approved in the platform using their top
terms and centroid-nearest papers. The local operational runbook is the
authoritative execution order.

Saved labels update the Topic nodes in the existing Knowledge Graph without
changing the stable `scope:topic_id` identity. Select the same scope in the
Knowledge Graph and enable Topic nodes to inspect their connected papers. The
scope-specific figure axes also use the latest saved labels. The review page
downloads reviewed topic tables, figures, graph files, or a complete
checksummed release for the selected scope.

## Dashboard and Supervisor Sharing

The managed user services normally start with WSL. Retrieve the current public
Quick Tunnel URL with:

```bash
bash scripts/dashboard_url.sh 60
```

For manual local development only, run the environment-aware launcher:

```bash
bash scripts/serve_dashboard.sh
```

Open `http://127.0.0.1:8321`. The launcher binds to localhost by default and
does not expose the application publicly. The Construct Specification
view is available at `http://127.0.0.1:8321/composition`; it recalculates every
panel from the active dataset after applying the selected dataset scope and
study-status filter.

The methodological topic-review interface is available at
`http://127.0.0.1:8321/topic-review`. It presents the data-specific figures,
terms, representative papers and auditable label decisions without requiring
the researcher to edit repository files. Label writes require dashboard
authentication. Derived outputs are regenerated only after all 130 labels are
approved.

Deployment templates under `deploy/` mirror the Apartment Finder pattern:

- `etv-dashboard.service` keeps Uvicorn running as a user service.
- `etv-dashboard-tunnel-quick.service` creates a temporary public
  `trycloudflare.com` address that can be copied and shared directly. This is
  the same account-free design used by Apartment Finder: no domain, account or
  tunnel token is required. The dashboard applies HTTP Basic authentication to
  every page, asset and API endpoint before tunnel traffic is accepted. The
  random URL changes whenever the tunnel is recreated.

Create credentials outside the repository before installing the services:

```bash
mkdir -p ~/.config/etv-dashboard
chmod 700 ~/.config/etv-dashboard

cat > ~/.config/etv-dashboard/auth.env <<'EOF'
ETV_DASHBOARD_USERNAME=supervisor
ETV_DASHBOARD_PASSWORD=replace-with-a-long-random-password
EOF

chmod 600 ~/.config/etv-dashboard/auth.env
```

Install and start the account-free services with:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/etv-dashboard.service ~/.config/systemd/user/
cp deploy/etv-dashboard-tunnel-quick.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now etv-dashboard.service
systemctl --user enable --now etv-dashboard-tunnel-quick.service
```

Open the generated URL and enter the configured username and password in the
browser prompt. Share the link and password through separate channels. Quick
Tunnels are free and account-free but are intended for temporary demonstration
and development use; they provide no uptime guarantee.

Print the shareable URL with:

```bash
bash scripts/dashboard_url.sh
```

The helper waits up to 30 seconds and reports only the URL created during the
current WSL boot, preventing an obsolete URL from an earlier session from being
shared. After the services have been enabled once, systemd starts the dashboard
and creates a new Quick Tunnel whenever the WSL user-service manager starts.
The site remains available only while the host, WSL environment, dashboard
service, network connection, and tunnel are running. Laptop sleep, shutdown, or
WSL termination makes the local deployment unavailable; use a named tunnel on
a continuously running workbench or cloud VM when a permanent URL and persistent
availability are required.
