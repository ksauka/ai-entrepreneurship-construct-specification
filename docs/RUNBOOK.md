# Runbook: building and running the platform end to end

All commands run from the project root in WSL with the `graphrag` conda env active:

```bash
cd ~/projects/ETV_V2
conda activate graphrag
```

## Authoritative execution order

Run the remaining work in this order. Do not run specification coding and
topic modeling concurrently because both can compete for the RTX 5070:

1. Run every specification model over the complete master corpus, sequentially.
2. Curate every model and compute inter-rater reliability across all model results
   for `full_corpus` and Query 1-4.
3. Run five-scope topic optimization and inspect every metric table and graph.
4. Explicitly approve the topic recommendations, then run the final topic models.
5. Apply optional VOS filtering, build/load the graph, verify the app, and build
   the Stage 4 specification-and-contrast analysis.

Later stages read earlier outputs, although the app and graph degrade gracefully
when optional outputs are absent.

## 0. One-time housekeeping

Done 2026-07-10: the superseded VOS cluster files (`import_clusters.py`,
`scripts/import_vos.py`, `tests/test_vos_import.py`) and the thin
`src/aecsp/topics/extract.py` were deleted (never committed, so plain `rm`).

Confirm the test suite is green:

```bash
pytest -q
```

## 1. Stages 0 to 1B: corpus, provenance, VOSviewer exports

Reads `data/queries/SQ*.csv`, writes the master corpus, query views, and the
five VOSviewer export files. About two minutes.

```bash
python scripts/build_corpus.py
```

Outputs: `data/processed/master_corpus.csv`, `data/processed/query_1..4.csv`,
`data/exports/vosviewer/vos_*.csv`.

## 2. Stage 2A.5 first: full multi-model specification experiment

Specification coding is the conceptual centre and runs before topic modeling.
Code each paper once on the full corpus; the resulting paper-level codes are
then analyzed through all five overlapping views (`full_corpus`, `query_1..4`).
Do not pay to recode the same paper separately for each query.

Every study model must code the complete `master_corpus.csv`. The cache is
permanent model-specific audit state:

```text
data/interim/spec_cache/
    llama3.2/              # full llama3.2 experiment
    qwen3.5_9b-q4_K_M/     # full Qwen experiment
    gpt-4.1-nano/          # full gpt-4.1-nano experiment
    claude-sonnet-5/       # full research-grade experiment when implemented
    <additional-model>/    # any additional full study model
```

There is no benchmark winner and no model-selection gate. All model outputs are
research data used for inter-rater reliability and substantive comparison.
Limited runs may test plumbing, but they do not replace the required full run.
Run one model at a time because local models and topics share the RTX 5070.

```bash
# Existing cache files are reused automatically; do not delete them.
python scripts/run_specification.py --local --model llama3.2
python scripts/run_specification.py --local --model qwen3.5:9b-q4_K_M
python scripts/run_specification.py --model gpt-4.1-nano

# Run the research-grade provider/model over the same full corpus after its
# provider branch is implemented and its current model identifier is confirmed.
python scripts/run_specification.py --model <research-model>
```

Each command writes model-specific outputs and never overwrites another model:

```text
paper_specifications_<model>.csv
specification_report_<model>.json
curation_overrides_<model>.json
paper_specifications_curated_<model>.csv
```

For each model, the full paper-level result is analyzed as five overlapping
experiments: `full_corpus`, `query_1`, `query_2`, `query_3`, and `query_4`.
Coding remains query-invariant, so do not make five duplicate LLM calls per
model. The scope flags create the five datasets for reliability and analysis.

## 3. Curate and compare specification models

Curate every full model result and retain every model for inter-rater analysis.
Auto-accept requires `stated` evidence and confidence >= 0.8; everything else
queues least-confident-first. Human overrides remain separate from the immutable
per-model LLM cache.

```bash
python scripts/curate_specification.py --model llama3.2 --report
python scripts/curate_specification.py --model qwen3.5_9b-q4_K_M --report
python scripts/curate_specification.py --model gpt-4.1-nano --report
python scripts/curate_specification.py --model <research-model> --report

python scripts/curate_specification.py --model <model>
python scripts/curate_specification.py --model <model> --export
```

Review commands: Enter accepts, `1-N` overrides with a numbered value, `e`
shows the abstract, `s` skips, `d` defers the dimension corpus-wide, `B`
batch-accepts the remaining dimension queue, and `q` saves and quits. Decisions
save after every action and are resumable.

Outputs: `curation_overrides_<model>.json` and, with `--export`,
`paper_specifications_curated_<model>.csv` with per-dimension curation status
columns. The inter-rater reliability analysis across these model-specific files
and all five scopes is a required research stage and is not built yet.

## 4. Stage 2A: optimize, review, then run topics (GPU)

One-time deps (also in the pyproject `nlp` extras):

```bash
pip install -e '.[nlp]'
```

### GPU and CPU responsibilities

The RTX 5070 is used by PyTorch/SentenceTransformer for document embeddings,
seed embeddings, semantic phrase filtering, and KeyBERT when its underlying
SentenceTransformer selects CUDA. Confirm CUDA before a long run:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
nvidia-smi
```

The current BERTopic stack deliberately uses standard CPU implementations for:

- UMAP dimensionality reduction;
- HDBSCAN clustering and probability calculations;
- CountVectorizer and c-TF-IDF topic representation;
- YAKE and gensim phrase processing;
- pandas exports, term matching, metrics, and Plotly graph preparation.

Therefore a Stage 2A process can be CPU-heavy while CUDA was used successfully
for its embedding phases. The old PID 5019 log showed 699 embedding batches
finishing in 27 seconds, which is consistent with GPU acceleration; it was later
observed during a CPU-bound extraction/clustering phase. GPU UMAP/HDBSCAN would
require an explicit RAPIDS cuML implementation and separate output-equivalence
testing; it is not part of the validated environment today.

Monitor both resources during Stage 2A:

```bash
watch -n 2 nvidia-smi
htop
```

The real pipeline checkpoints hybrid phrase documents and embeddings under
`data/interim/topics/`, then evaluates HDBSCAN configurations independently for
the global corpus and Query 1-4. Optimization writes metrics and graphs but does
not silently approve its recommendation. No checkpoint exists yet, so the next
optimization run must perform the first phrase/embedding pass; subsequent final
runs reuse that checkpoint.

```bash
nohup python scripts/run_topics.py --optimize-only > topics_optimize.log 2>&1 &
tail -f topics_optimize.log
```

Review `data/processed/topics/optimization/`:

- `recommendations.json` summarizes all five views.
- Each scope has `grid_search_results.csv` and `.json`.
- `grid_search_metrics.html` compares silhouette, diversity, outliers, and balance.
- `topic_count_sensitivity.html` shows how topic count changes.
- `configuration_scores.html` shows the diagnostic recommendation score.

After human approval, either accept the reviewed recommendations or specify a
manual global value. The checkpoint prevents another phrase/embedding pass.

```bash
nohup python scripts/run_topics.py --use-optimized > topics_run.log 2>&1 &
tail -f topics_run.log

# Manual global override (native views retain their documented fallback sizes):
python scripts/run_topics.py --global-min-topic-size 50
```

Final diagnostics are written under every model's `diagnostics/` directory:
topic-size, intertopic-distance, hierarchy, and topic-term HTML graphs. Static
PNG copies are also written when Matplotlib is installed.

Outputs: `data/processed/master_corpus_topics.csv` (the graph and app pick
this up automatically in place of `master_corpus.csv`) and
`data/processed/topics/` (`global/` with topics, topic terms, doc topics,
AI/ent term tables and per-scope summaries; `native/<query>/` per query;
`keyphrases_detected.json`; `topics_report.json`).

## 5. VOS cleaning filter (optional, per scope)

Run VOSviewer separately on each `data/exports/vosviewer/vos_*.csv`, save each
returned map into `data/vosdata/` as `<scope>_vos.csv`
(`master_corpus_vos.csv`, `query_1_vos.csv` .. `query_4_vos.csv`). Then:

```bash
python scripts/apply_vos_filter.py
```

For every scope whose map is present and not older than the corpus, this writes
`data/processed/vos_filtered/<scope>_retained.csv` and `<scope>_dropped.csv`.
Scopes without a current map are skipped. Safe to re-run as each map finishes.

## 6. Stage 2B: knowledge graph

CSV export works now with no database:

```bash
python scripts/build_graph.py --export-csv
```

Output: `data/processed/graph/nodes.csv`, `relationships.csv`, plus a printed
summary of node and relationship counts.

To load into Neo4j, start the database first (see section 6), then:

```bash
python scripts/build_graph.py --load          # add --wipe to clear first
```

## 7. Neo4j setup (Docker)

Neo4j credentials come from `.env` (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`;
defaults are in `.env.example`). Start the container:

```bash
docker compose up -d
```

Browser UI: http://localhost:7474 (Bolt on 7687). Stop with
`docker compose down`; the data volume persists under `data/neo4j/`.

When Neo4j is running, the app connects to it automatically and the Knowledge
Graph page shows real internal citation (CITES) edges; otherwise it runs in CSV
mode from the processed files.

## 8. Run the app

```bash
pip install fastapi uvicorn neo4j httpx    # first time only
uvicorn aecsp.api.main:app --reload --app-dir src --port 8321
```

Open http://localhost:8321 for the Analytics Dashboard, `/graph` for the
Knowledge Graph, `/assistant` for the Assistant. The dashboard and graph work
after stage 1; specification views fill in after stage 2A.5; the graph gains
CITES edges once Neo4j is loaded.

## Quick dependency check

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import bertopic, keybert, fastapi, neo4j; print('deps ok')"
```
