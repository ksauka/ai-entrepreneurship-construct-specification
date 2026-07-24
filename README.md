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
8-topic Query 1-4 models. Official ASJC assignment and an explicit ASJC-code-to-
business-domain aggregation are implemented. The ten selected business domains
cover 19,553 papers; the 2,792-paper residual retains official ASJC codes and
remains in the full-corpus comparison baseline. This residual is distinct from
the 20,713-paper `remaining_full_corpus` scope, now displayed as **Full corpus
excluding Combined entrepreneurship**, which is simply 22,345 minus the 1,632
Combined entrepreneurship papers. Analytics, Construct Specification, Knowledge
Graph, and Assistant use the same dataset-scope registry, including the ten
business domains and the disclosed residual. Topic Review remains limited to the
five separately fitted topic models, while Construct Contrasting displays domains
as matrix rows rather than treating them as competing page-level filters. A tier-independent, sequential 11-shard Gemini
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
study-status filter. Construct-specification charts retain each model's actual
successful-paper count. The model-IRR matrix instead uses one balanced paper-ID
intersection shared by all displayed models inside the registered Claude
reference cohort; the current four-model intersection is 21,930 papers.
Clicked Construct Specification and Construct Contrasting evidence can be
filtered to papers for which at least two models assign the exact selected
pattern. The preferred sweet spot requires agreement from GPT-5.4 Mini, Claude
Sonnet 5, and Gemini 3.1 Pro Preview. GPT-4.1 Nano remains visible in the
paper-level model comparison but does not veto that preferred three-model
criterion. Agreement indicates coding convergence on the selected pattern; it
does not by itself establish ground truth.

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

### Permanent reviewer hosting

The permanent deployment uses separate administrator and reviewer credentials
through the platform login page. The administrator account may write human
annotations and topic decisions. The reviewer account can inspect every
analysis, evidence panel, graph, report, and download but is rejected by every
research-record write endpoint. The active role and Sign out control appear at
the far-right of the platform header. Signing out invalidates the server session
and returns the browser to the login page; protected pages use no-store caching
and revalidate restored browser-history pages.

Copy `deploy/auth.env.example` to `~/.config/etv-dashboard/auth.env`, replace all
placeholder values, and protect it with mode `600`. After creating a
remotely-managed Cloudflare Tunnel and published application route, save the
tunnel token alone in `~/.config/etv-dashboard/tunnel.token` with mode `600`.
Neither file belongs in Git or a shared data bundle.

Install or refresh the services with:

```bash
bash scripts/configure_reviewer_credentials.sh reviewer
bash scripts/install_review_host.sh
bash scripts/review_host_status.sh
```

The credential helper preserves the existing administrator login, generates a
separate reviewer password, and writes a private receipt under
`~/.config/etv-dashboard/` without printing the password to logs.

Verify the permanent public deployment, including the reviewer write barrier:

```bash
bash scripts/smoke_review_site.sh https://aitheoryelaboration.org
```

For a WSL-hosted laptop, install the supplied Windows Startup launcher from the
Ubuntu terminal after the Linux services are healthy:

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File \
  "$(wslpath -w deploy/windows/install-etv-wsl-autostart.ps1)" \
  -Distro Ubuntu-22.04 -LinuxUser suvh
```

The installer places a launcher in the current Windows user's Startup folder,
removes the retired Scheduled Task implementation when present, and tests the
launcher immediately. It starts WSL and the enabled dashboard and named-tunnel
user services after every Windows sign-in. Windows sleep, shutdown, loss of
power, or loss of network still makes the origin temporarily unavailable.

If no named-tunnel token exists, installation leaves the existing Quick Tunnel
unchanged. Once the token exists, rerunning the installer enables the permanent
tunnel and disables the temporary tunnel. The same token and exact review
release can be installed on a standby machine, but only machines carrying the
same data release should run connectors concurrently.

### Linux desktop failover host

Git contains the application and deployment code, but it intentionally excludes
the large processed datasets, generated analytical tables, writable annotation
stores, credentials, and tunnel token. A desktop recovery therefore needs both
the Git commit and a private runtime bundle.

After committing the exact release, create the bundle on a flash drive or other
private destination:

```bash
bash scripts/backup_review_host.sh /path/to/private-drive
```

The script copies the required processed data, topic artifacts, and writable
SQLite stores; creates an offline Git bundle; records the exact commit; generates
SHA-256 checksums; and encrypts the administrator/reviewer credentials and
Cloudflare tunnel token with GPG/AES256. Use `--include-project-env` only if the
desktop also needs the private Neo4j or provider settings from `.env`. The GPG
passphrase must be stored separately. An encrypted bundle may be transferred
through Google Drive; unencrypted credentials and tunnel tokens must not be.

On the Linux desktop, place the checkout at `~/projects/ETV_V2` and reproduce
the serving environment:

```bash
conda create -n graphrag python=3.11 -y
~/miniconda3/envs/graphrag/bin/pip install -r deploy/review-host-requirements.txt
~/miniconda3/envs/graphrag/bin/pip install -e .
```

Install `cloudflared`, then restore and test the machine as a standby without
publishing it:

```bash
bash scripts/restore_review_host.sh \
  /path/to/etv-review-host-YYYYMMDDTHHMMSSZ \
  --standby

curl -u 'ADMINISTRATOR_USERNAME:ADMINISTRATOR_PASSWORD' \
  http://127.0.0.1:8321/api/health
```

For unattended startup on native Linux, enable the user service manager at boot
and disable automatic sleep in the desktop operating-system settings:

```bash
sudo loginctl enable-linger "$USER"
```

Cut over only after the desktop passes its local check. Stop the laptop
connector first, then activate the desktop connector:

```bash
# Laptop
systemctl --user disable --now etv-dashboard-tunnel-named.service

# Desktop
bash scripts/install_review_host.sh
bash scripts/review_host_status.sh
bash scripts/smoke_review_site.sh https://aitheoryelaboration.org
```

The two machines may briefly carry the same tunnel token, but unsynchronised
hosts must not remain active together. After any administrator writes on the
active host, create a new bundle before failing back to the other machine.
