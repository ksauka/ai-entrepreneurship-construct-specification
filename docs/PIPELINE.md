# ETV_V2 project status and research pipeline

Status checkpoint: 24 July 2026.

This document records what has been completed, what is operational, what is
intentionally hidden, and what remains before the manuscript and platform can
be frozen for review. The exact execution commands and safety gates remain in
[`RUNBOOK.md`](RUNBOOK.md). If a command in this summary differs from the
runbook, inspect the current script help and the runbook before running it.

## 1. Project purpose

ETV_V2 supports a theory-elaboration study of how artificial intelligence is
specified in entrepreneurship and business research. The project combines:

1. a deduplicated Scopus corpus;
2. eight abstract-level construct-specification dimensions;
3. independent coding by multiple language models;
4. data-specific topic models and VOSviewer maps;
5. business-domain and entrepreneurship-population contrasts;
6. systematic close reading and human interpretive triangulation; and
7. an interactive, paper-traceable methodological platform.

The platform is titled **AI-Entrepreneurship Construct Specification
Platform**. Its permanent review hostname is:

```text
https://aitheoryelaboration.org
```

## 2. Current top-level state

| Area | Current state |
|---|---|
| Corpus | Frozen at 22,345 unique papers |
| Primary coding | GPT-5.4 Mini complete for all 22,345 papers |
| Cross-provider coding | Nano 22,335; Claude 21,940; Gemini 22,345 |
| Four-model reliability | Built on the exact 21,930-paper intersection |
| Llama stress test | 21,136 usable papers; descriptive only and below the 21,930 IRR eligibility rule |
| Topic models | Five fitted models; 130 topics total |
| Topic-label review | 67 approved and 63 pending |
| Entrepreneurship analysis | Leading 646; Additional 986; Combined 1,632 |
| Business-domain analysis | Nine overlapping domains; 19,505 papers assigned |
| Systematic close reading | 136-paper ledger; 124 entrepreneurship papers plus 12 cross-domain contrasts |
| Human insight allocation | 14/14 agreement; Cohen's kappa = 1.00 |
| Blind eight-dimension human annotation | Platform ready; zero completed annotations |
| Targeted interpretation review | Implementation retained but hidden pending supervisor agreement |
| Knowledge Graph | Implementation retained but hidden because the release graph is not ready |
| Public hosting | Permanent Cloudflare hostname and authenticated reviewer/admin access configured |
| Desktop failover | Encrypted, checksum-verified recovery bundle created; desktop installation pending |
| Manuscript | Current author-edited draft and supplement remain under section-by-section review |

## 3. Executed pipeline stages

| Stage | Operation | Status |
|---|---|---|
| 0 | Import the four Scopus query exports | Complete |
| 0.5 | Merge, deduplicate, and preserve query provenance | Complete |
| 1 | Validate source titles and retrieval scopes | Complete |
| 1.5 | Apply the relevance workflow | Complete for the frozen corpus |
| 1.6 | Create query-membership views | Complete |
| 1B | Export VOSviewer inputs | Complete for the analytical query views |
| 2A | Optimize and fit the five data-specific BERTopic models | Complete |
| 2A.5 | Run construct-specification coding | Complete for Mini, Nano, Claude, and Gemini; Llama retained as a partial supplementary stress test |
| 2B | Implement publication-centred graph schema and builders | Code complete; current release export/load verification pending |
| 3 | Serve analytics, specification, contrasting, topics, evidence, annotation, reports, and downloads | Operational |
| 4 | Build ASJC domains and theory-elaboration contrasts | Implemented and rebuilt against the reviewed nine-domain aggregation |
| 5 | Systematic close reading and interpretive triangulation | Current 136-paper procedure and 14-paper insight allocation complete |
| 6 | Manuscript and supplementary exhibits | In progress; author review and final numerical audit remain |
| 7 | Review hosting and failover | Laptop hosting configured; encrypted desktop bundle verified; desktop cutover pending |

## 4. Corpus contract

The canonical corpus contains one deduplicated row per paper with:

- stable `paper_id`;
- title, abstract, author keywords, source title, year, DOI, and Scopus EID;
- citation and bibliographic metadata;
- source record links; and
- preserved membership in Queries 1-4.

Deduplication uses Scopus EID first, DOI second, and normalized title/year only
as a fallback. Query membership survives every downstream join.

The four retrievals overlap and are not independent samples. Before merge and
deduplication, the recorded query exports contained:

| Retrieval | Records |
|---|---:|
| Query 1 | 29,294 |
| Query 2 | 818 |
| Query 3 | 1,097 |
| Query 4 | 1,509 |

The frozen union contains **22,345 unique papers**. Corpus false positives that
survived the workflow are documented as a limitation rather than retrospectively
removed and propagated through every dependent artifact.

Primary corpus files:

```text
data/processed/master_corpus.csv
data/processed/analysis/primary_analysis_dataset.csv
data/processed/analysis/primary_analysis_dataset_with_topics.csv
```

## 5. Construct-specification instrument

Each model codes one paper from the title, abstract, and author keywords.
Source journal and publication year may accompany the record as contextual
metadata but are not admissible evidence. ASJC domain, query membership, topic
assignment, citation data, and other models' outputs are withheld from coding.

The eight dimensions are:

| Dimension | Status |
|---|---|
| Study status: method, phenomenon, both, or unclear | Core |
| AI role/function | Core |
| Technical AI type/form | Core |
| AI mechanism | Core |
| Level of analysis | Core |
| Scope conditions | Core |
| Entrepreneurial or organizational process stage | Exploratory |
| Definition clarity | Exploratory |

Core versus exploratory is an analytical-use distinction, not a missing-data
rule. Process stage and definition clarity remain visible and downloadable.
They are excluded from six-dimension summary averages and interpreted
cautiously because their category reliability is weak for different reasons.

For every dimension, the record retains:

- one selected category;
- supporting quotation or close paraphrase;
- `stated`, `inferred`, or `absent` evidence status;
- confidence;
- mechanism logic where relevant;
- full-text-review flags;
- specification-problem flags;
- named theories;
- an adversarial self-review; and
- model, protocol, cache, and paper identity.

Full and Observed are separate denominator views:

- **Full** retains every paper in the selected analytical population, including
  dimension-specific missing or unspecified categories.
- **Observed** removes only the unobserved/non-substantive values for the
  displayed outcome. It does not mean a different paper corpus.

## 6. Model coding and reliability

Current usable paper-level outputs:

| Model | Role | Usable papers | Treatment |
|---|---|---:|---|
| GPT-5.4 Mini | Prespecified primary coder | 22,345 | Population analysis |
| GPT-4.1 Nano | Full-corpus baseline/sensitivity coder | 22,335 | Reliability and model-specific distributions |
| Claude Sonnet 5 | Independent cross-provider coder | 21,940 | Reliability and robustness |
| Gemini 3.1 Pro Preview | Independent cross-provider coder | 22,345 | Reliability and robustness |
| Llama 3.2 3B | Local supplementary stress test | 21,136 | Model-specific distributions only |
| Gemma 4 31B | Partial local stress test | Partial legacy cache | Not a population estimator |

The legacy five-row `paper_specifications.csv` is not an analytical model
output.

The principal model-reliability release uses the exact **21,930-paper**
intersection shared by Mini, Nano, Claude, and Gemini. Four raters yield six
pairwise model combinations, not eight. Each pair is reported for all eight
dimensions using:

1. all-category exact agreement and pairwise nominal Krippendorff's alpha;
2. evidence-presence exact agreement and alpha; and
3. category agreement and alpha where both models found evidence.

The agreement analysis does not create a consensus code. Mini remains the
primary analytical record. Evidence lists may be filtered to exact agreement
between at least two models or the Mini-Claude-Gemini convergence set.

Llama is available as a model-specific descriptive selection because each
model's specification page reports its actual usable records. It is excluded
from the balanced IRR table because **21,136 is below the prespecified 21,930
eligibility threshold**. Remaining Llama failures may be rerun later, but the
current partial model must not be described as full-corpus or inserted into the
balanced four-model matrix.

## 7. Analytical populations

Entrepreneurship populations:

| Population | Papers | Interpretation |
|---|---:|---|
| Leading entrepreneurship journals | 646 | Query 3 retained view |
| Additional entrepreneurship journals | 986 | Query 4 retained view |
| Combined entrepreneurship | 1,632 | Exact non-overlapping union |
| FT50 restriction | 438 | Robustness and boundary view |

Some older internal files use `core_entrepreneurship` and
`other_entrepreneurship`. The public and manuscript-facing terminology is
**Leading** and **Additional**.

Papers are analyzed at the intersection of the selected dataset scope and the
selected coding model's successful records. A coding model does not artificially
inherit another model's coverage.

## 8. ASJC business-domain system

Domain assignments are rebuilt from official Scopus ASJC source classifications
for journals already represented in the corpus. No papers are retrieved to fill
a domain. Assignment is source-level and multi-label.

Current aggregation:

| Domain | Papers |
|---|---:|
| Management of Technology and Innovation | 3,101 |
| Strategy and Management | 7,074 |
| Marketing | 2,637 |
| Information systems | 5,969 |
| Finance | 1,179 |
| Management Science and Operations Research | 5,242 |
| Organization studies | 916 |
| Environmental and sustainability | 2,481 |
| Tourism, Leisure and Hospitality Management | 1,131 |

Validation:

- 22,345 papers have official ASJC assignments.
- 19,505 papers (87.29%) enter at least one selected analytical domain.
- 8,669 assigned papers enter more than one domain.
- 2,840 papers (12.71%) hold official codes outside the nine selected rows.
- The residual remains in the 22,345-paper full-corpus baseline.
- Domain counts overlap and must not be summed as independent populations.
- The former Ethics and corporate-social-responsibility overlay was removed.

Authoritative configuration and manifest:

```text
configs/asjc_business_domain_aggregation.yaml
data/processed/analysis/theory_elaboration/domains/business_domain_manifest.json
```

## 9. Theory-elaboration analyses

The platform operationalizes four Fisher-and-Aguinis theory-elaboration tactics:

1. **Construct specification:** marginal and conditioned distributions for all
   eight dimensions.
2. **Horizontal contrasting:** the same specification dimension across Leading,
   Additional, Combined, FT50, full-corpus, and selected business-domain views.
3. **Vertical contrasting:** any selected dimension crossed with level of
   analysis.
4. **Structuring:** role-mechanism, role-level, mechanism-level, role-scope, and
   other dynamically selected pairwise matrices.

Conditioning is dynamic. For example, a user can fix study status to phenomenon,
method, or both and recalculate every other dimension or matrix within the
selected population and model.

Full and observed denominators, counts, row/column percentages, matrix shares,
minimum-support thresholds, evidence status, and model agreement remain explicit.
Every displayed bar or cell can open its supporting papers and paper-level
evidence.

## 10. Topic modeling and review

Five data-specific BERTopic models were optimized and fitted separately:

| Scope | Topics | Review status |
|---|---:|---|
| Full corpus | 53 | 53 approved |
| Query 1 | 50 | 50 pending |
| Query 2 | 13 | 13 pending |
| Query 3 / Leading entrepreneurship | 6 | 6 approved |
| Query 4 / Additional entrepreneurship | 8 | 8 approved |
| **Total** | **130** | **67 approved; 63 pending** |

Topic identities are scope-specific. The stable key is `scope:topic_id`; topic
numbers must not be equated across scopes. Humanized names are display metadata,
not constructs.

The Topic Review workspace provides:

- top terms;
- centroid-nearest representative papers;
- all fitted papers;
- paper-level inspection;
- editable humanized labels;
- reviewer notes and approval states; and
- rebuilt topic figures and downloadable review artifacts.

Query 3 and Query 4 topic labels required for the entrepreneurship interpretation
are approved. Query 1 and Query 2 remain pending unless they are needed for the
final manuscript or supplementary release. A complete all-scope release remains
gated on all 130 labels or an explicit documented scope restriction.

## 11. Systematic close reading and interpretive triangulation

The current auditable reading ledger contains **136 papers**:

- 51 from Leading entrepreneurship journals;
- 73 from Additional entrepreneurship journals;
- 124 in the Combined entrepreneurship interpretation base; and
- 12 cross-domain contrasting papers.

All 124 entrepreneurship papers occur in the available Query 3 and Query 4
VOSviewer document maps.

Network-position audit:

- Leading reading papers have median total link strength 145, versus 68 for the
  complete map; 25 of 51 are in the highest quartile.
- Additional reading papers have median total link strength 181, versus 75 for
  the complete map; 37 of 73 are in the highest quartile.
- Represented clusters contain 99.2% of the Leading map and 98.4% of the
  Additional map.

The reading set supports interpretation and counterexample selection; it does
not estimate corpus prevalence.

The current configurational interpretation develops:

- bottleneck relocation as the central entrepreneurship insight;
- organizational embedding as a condition;
- domain-specific mechanism as a boundary; and
- agency allocation as an open theoretical frontier.

A second researcher independently allocated 14 papers to the three
narrative/insight families. Agreement was 14/14 and Cohen's kappa was 1.00.
This checks interpretive allocation only and does not validate the eight
construct-specification dimensions.

## 12. Human annotation

The Human Annotation workspace is implemented and uses the same fixed
eight-dimension instrument while withholding model outputs.

Available strata:

- 23 papers that occur naturally in the probability sample; and
- 113 separately identified interpretive papers.

Annotations are stored by annotator identity, are resumable, and can be exported.
Human-model reliability recalculates only on the exact completed intersection.

Current saved state:

```text
annotations = 0
annotation audit events = 0
```

Blind human annotation is therefore still pending and must not be described as
completed validation.

## 13. Platform workspaces

Currently active public workspaces:

- Analytics Dashboard;
- Construct Specification;
- Construct Contrasting;
- Topic Review;
- Human Annotation; and
- Construct Specification Assistant.

Implemented platform capabilities include:

- consistent dataset-scope information across pages;
- model-specific paper counts and distributions;
- study-status and other dimension/value filters;
- dynamic matrix axes and denominators;
- paper-level evidence panels and Scopus/DOI links;
- evidence-status and confidence display;
- model-convergence filters naming the agreeing models;
- pairwise nominal Krippendorff-alpha reliability tables;
- trend charts with annual count, cumulative count, and click-level growth
  details;
- year-level papers and keyword inspection;
- topic-label humanization;
- read-only reviewer sessions;
- separate administrator sessions;
- logout that invalidates the browser session; and
- generated reports, tables, downloads, and release manifests.

The platform uses one analytical service for browser panels, reports, and
downloads so displayed states remain traceable to the same source tables.

## 14. Intentionally hidden workspaces

### 14.1 Targeted interpretation review

The targeted-reading implementation and its stored data are retained, but
`/composition/targeted-reading` redirects to Construct Specification. It remains
hidden until the researcher and supervisor agree on the paper-level
interpretation form and its role relative to blind human validation.

To resume:

1. agree the fields and interpretive purpose with the supervisor;
2. preserve separation from the blind 23-paper validation stratum;
3. test write permissions and exports;
4. restore the page route and navigation tab;
5. update the methodology and supplementary workflow description; and
6. add a release test confirming reviewer/admin permissions.

### 14.2 Knowledge Graph

Decision on 24 July 2026: **hide the Knowledge Graph workspace until its release
graph is ready**.

What is retained:

- graph schema and node/relationship contracts;
- dataframe graph builder;
- Neo4j loader and reader;
- bounded seed and one-hop expansion logic;
- frontend explorer files;
- graph API/service methods;
- unit tests; and
- topic/specification graph relationships.

Why it is hidden:

- no current `data/processed/graph` release export is present;
- the current Neo4j application reader principal is not fully configured and
  verified;
- a release graph has not been loaded and reconciled to expected node and
  relationship counts;
- reviewer read-only behavior has not been verified end to end against the
  production graph;
- bounded-query performance and failure behavior need testing on the desktop
  host; and
- the current interface should not imply that an unfinished graph is part of
  the manuscript evidence chain.

Current hiding behavior:

- Knowledge Graph links are removed from active workspace navigation.
- `/knowledge-graph` redirects to the Analytics Dashboard.
- legacy `/graph` redirects to the Analytics Dashboard.
- direct `/static/knowledge_graph.html` access redirects to the dashboard.
- graph implementation files and APIs are not deleted.

#### Knowledge Graph reactivation gate

Do not restore the window until all of the following are complete:

1. Export the graph release from the exact frozen analytical dataset:

   ```bash
   ~/miniconda3/envs/graphrag/bin/python scripts/build_graph.py --export-csv
   ```

2. Record checksums and expected node/relationship counts.
3. Configure separate administrative and application reader credentials.
4. On Neo4j Enterprise, create and verify the database-enforced read-only
   application principal:

   ```bash
   ~/miniconda3/envs/graphrag/bin/python scripts/configure_neo4j_readonly.py
   ```

5. Load the release graph only after confirming whether the target database is
   empty. `--wipe` is destructive and requires an explicit backup and approval.
6. Verify the loaded database:

   ```bash
   ~/miniconda3/envs/graphrag/bin/python scripts/build_graph.py --verify
   ```

7. Run focused graph tests:

   ```bash
   pytest \
     tests/test_knowledge_graph_builder.py \
     tests/test_neo4j_reader.py \
     tests/test_api.py
   ```

8. Check every dataset scope, node filter, relationship filter, evidence link,
   fallback state, and reviewer permission locally.
9. Test bounded-query response time and memory use on the desktop host.
10. Restore `/knowledge-graph`, `/graph`, and the navigation links only after
    the checks pass.
11. Update this file, the platform methodology, screenshots, and review release
    manifest.

## 15. Manuscript and supplementary material

Current author-selected manuscript:

```text
docs/ETP draft - July2026ks.docx
```

Current supplementary material:

```text
docs/ETP supplementary material july2026 ks.docx
```

The manuscript is being reviewed section by section. Current narrative order is:

1. research design and corpus;
2. construct-specification instrument and reliability;
3. analytical populations and domain construction;
4. operationalization and systematic close reading;
5. platform implementation;
6. construct specification within entrepreneurship;
7. horizontal and vertical contrasting;
8. structuring and recurring configurations;
9. anchored interpretive insights; and
10. discussion of theoretical contribution and boundaries.

The manuscript must not use internal pipeline labels where ordinary academic
language is available. Numerical claims must be regenerated from the current
tables/platform state, not inferred from prose or earlier drafts.

Remaining manuscript work includes:

- complete author editing in the current draft;
- verify every table, figure, denominator, percentage, and appendix pointer;
- align the supplement with the nine-domain aggregation;
- ensure evidence papers use in-text citations and reference-list dates;
- decide the final treatment of exploratory dimensions;
- report the model-reliability bases and measures consistently;
- finish or explicitly bound human annotation;
- update screenshots after the final interface freeze; and
- make the hosted platform reference appropriate for confidential review.

## 16. Hosting, authentication, and failover

The permanent Cloudflare hostname is configured through the named tunnel:

```text
https://aitheoryelaboration.org
```

Access design:

- administrator credentials allow writes;
- reviewer credentials are read-only;
- credentials and tunnel token are stored outside Git;
- browser login creates a revocable session;
- logout returns to the login page and invalidates back-button access; and
- the public hostname points to a localhost dashboard through an outbound-only
  Cloudflare connector.

The verified desktop recovery bundle is:

```text
ETV_DESKTOP_BACKUP/etv-review-host-20260724T181710Z
```

Verified bundle contents:

- repository bundle;
- exact code commit;
- 2.62 GB portable runtime archive;
- encrypted administrator/reviewer credentials;
- encrypted named-tunnel token;
- encrypted project `.env`;
- transactionally consistent SQLite stores; and
- SHA-256 checksums.

The bundle records commit `a4e70c4`. It remains a valid recovery point, but it
predates the Knowledge Graph hiding change and later documentation commits.
After the current platform change is committed, refresh the bundle's code and
checksums with `--resume-incomplete`; the runtime archive does not need to be
copied again.

Desktop procedure:

```text
docs/DESKTOP_FAILOVER_SETUP.md
```

The desktop must first run in standby mode. Stop the laptop connector before
activating the desktop connector. Unsynchronized hosts must not remain publicly
active simultaneously.

## 17. Source-of-truth files

| Purpose | Source |
|---|---|
| Remaining execution sequence | `docs/RUNBOOK.md` |
| Detailed methodology | `docs/METHODOLOGY.md` |
| Frozen-method decisions | `docs/METHODS_LOCK.md` |
| Analysis design | `docs/ANALYSIS_PLAN.md` |
| Academic completion checklist | `docs/ACADEMIC_METHODOLOGY_THEORY_ELABORATION_TODO.md` |
| Platform/research configuration | `configs/theory_elaboration_analysis.yaml` |
| ASJC aggregation | `configs/asjc_business_domain_aggregation.yaml` |
| Domain manifest | `data/processed/analysis/theory_elaboration/domains/business_domain_manifest.json` |
| Primary corpus | `data/processed/analysis/primary_analysis_dataset.csv` |
| Model IRR tables | `reports/analysis/tables/model_validation/` |
| Contrasting tables | `reports/analysis/tables/contrasting/` |
| Topic-label review | `data/processed/analysis/stage4/topic_label_review.csv` |
| Close-reading ledger | `data/interim/theory_elaboration/` |
| Human annotation store | `data/interim/human_validation/human_annotations.sqlite3` |
| Desktop setup | `docs/DESKTOP_FAILOVER_SETUP.md` |

## 18. Ordered remaining work

1. Verify the Knowledge Graph is absent from every active navigation and direct
   page route.
2. Restart the authenticated dashboard and test administrator/reviewer access.
3. Commit and push the Knowledge Graph hiding and this status document.
4. Refresh the encrypted desktop bundle to the new code commit without
   rebuilding `runtime.tar`.
5. Install and test the Linux desktop in standby mode.
6. Complete the section-by-section manuscript and supplement audit.
7. Complete or explicitly bound the blind human-annotation stage.
8. Decide whether to rerun the remaining Llama failures; keep Llama outside
   balanced IRR unless it reaches the 21,930 threshold.
9. Resolve the 63 pending Query 1 and Query 2 topic labels or document their
   exclusion from the final release.
10. Freeze the final analytical release, screenshots, manifests, and checksums.
11. Tag the exact review release only after the manuscript, supplement, and
    hosted platform are synchronized.
12. Resume the Knowledge Graph later through the gate in Section 14.2.

