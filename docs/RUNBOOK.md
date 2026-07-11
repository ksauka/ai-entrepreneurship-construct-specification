# Runbook: building and running the platform end to end

All commands run from the project root in WSL with the `graphrag` conda env active:

```bash
cd ~/projects/ETV_V2
conda activate graphrag
```

## Authoritative execution order

Run the remaining work in this order. Do not run specification coding and
topic modeling concurrently because both can compete for the RTX 5070:

1. Preserve the completed GPT-4.1 nano corpus as the baseline/sensitivity dataset.
2. Run the adopted GPT-5.4 mini primary rater over the complete master corpus;
   retain Llama and Gemma as supplementary validation raters.
3. Run five-scope topic optimization and inspect every metric table and graph.
4. Explicitly approve the topic recommendations, then run the final topic models.
5. Apply optional VOS filtering, build/load the graph, verify the app, and build
   the Stage 4 specification-and-contrast analysis.

Later stages read earlier outputs, although the app and graph degrade gracefully
when optional outputs are absent.

## Progress reporting

Long project scripts print flushed, log-safe progress lines containing a bar,
completed/total counts, percentage, elapsed time, processing rate, ETA, and
failure count. This applies to corpus stages, specification requests, topic
phrase extraction and optimization, VOS scopes, graph records, and Neo4j load
batches. Output remains visible with `tail -f` when a command is redirected to
a log. The human-in-the-loop curation command already reports its current and
total review position as `[current/total]` and saves after every decision.

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

## 2. Stage 2A.5 first: full mini study plus multi-model validation

### GPT-5.4 mini 50-paper decision pilot

Prepare the fixed challenge set offline (safe to repeat; no API request):

```bash
python scripts/prepare_gpt54mini_challenge.py
```

Confirm the exact scope and estimated standard-API cost without submitting:

```bash
python scripts/run_specification.py \
  --model gpt-5.4-mini-2026-03-17 \
  --paper-ids-file data/interim/spec_pilots/gpt54mini_challenge_50.csv \
  --workers 5 \
  --dry-run
```

Run the paid pilot manually after checking that `OPENAI_API_KEY` is loaded:

```bash
python scripts/run_specification.py \
  --model gpt-5.4-mini-2026-03-17 \
  --paper-ids-file data/interim/spec_pilots/gpt54mini_challenge_50.csv \
  --workers 5
```

The pilot uses the unchanged `spec-v3` prompt and writes to the isolated
`data/interim/spec_cache/spec-v3/gpt-5.4-mini-2026-03-17/` cache. Do not add
`--local`, delete the nano cache, or substitute GPT-5.4 nano.

### Blinded human validation anchor

Generate the fixed 50-paper sample before inspecting primary-rater
distributions. This is offline and does not call a model:

```bash
python scripts/prepare_human_validation.py
```

Give each independent human coder a separate copy of
`data/interim/human_validation/blind_coding_template.csv`. Keep
`private_sample_key.csv` unavailable to coders until coding is complete. Do
not show model outputs or identities during coding. Use the codebook values in
`src/aecsp/specification/schema.py`. Preserve each coder's original file; do
not overwrite disagreements with a consensus value.

Gate outcome (2026-07-11, recorded in LLM_REPRODUCIBILITY.md): 50/50 coded,
zero failures, ~$0.21; rule-7 empty-logic 0% (nano on the same papers: 74%),
evidence grounding 92% (nano 70%), needs_full_text 1.5 vs 5.0 dims/paper,
both nano output-limit failures completed. GPT-5.4 mini was subsequently
adopted as primary; Gemma remains a supplementary validation rater:

```bash
python scripts/run_specification.py --local --model gemma4:31b \
  --base-url http://127.0.0.1:11435/v1 --workers 1 \
  --paper-ids-file data/interim/spec_pilots/gpt54mini_challenge_50.csv
```

### Full-corpus GPT-5.4 mini via the OpenAI Batch API

The full run uses Batch at 50% of live prices: ~$54 vs ~$103 live at the
observed ~3,328 input / ~472 output tokens per paper. The Batch queue limit
caps tokens ENQUEUED AT ONE TIME (not per day), so the orchestrator submits
sequential chunks under a 1,850,000-token budget, polls each batch to a
terminal state, fetches results, and repeats - roughly 1-3 unattended days.
The request body is byte-identical to the live path (test-enforced) and
results land in the SAME `spec-v3/gpt-5.4-mini-2026-03-17/` cache, so live
and Batch runs resume each other and the pilot's 50 papers are never repaid.

Current launch preview (2026-07-11): 50 pilot papers are cached; 22,295 remain,
packed into 40 sequential Batch chunks at an estimated `$53.09`. Keep at least
`$60` available for estimation variance. Expected wall time is about 1–3 days
if chunks clear in the commonly observed sub-hour-to-few-hour range. The Batch
completion window is up to 24 hours per batch, so this is an operational
estimate, not a guarantee. The live API alternative has a theoretical
approximately 7.2-hour token-rate floor and should be budgeted as 8–10 hours
with backoff/retries, at roughly twice the Batch cost.

Before paid submission, commit the locked instrument and register so the run
is commit-addressable. Re-run `preview` and confirm protocol fingerprint
`04d00994822c239a35149a6ac4dcf46db9b803095f582a400bb133a5f1e4457c`.

```bash
python scripts/run_specification_openai_batch.py preview     # chunks + cost, no API call
python scripts/run_specification_openai_batch.py run --yes  # PAID; resumable, Ctrl+C safe
python scripts/run_specification_openai_batch.py status     # poll submitted batches
python scripts/run_specification_openai_batch.py fetch      # collect completed batches
python scripts/run_specification_openai_batch.py export     # assemble raw + derived analysis CSV
```

`run` refuses to submit without `--yes`. Truncations, content filters, and
missing result lines go to the shared `failures.jsonl`; failed papers are
never cached and are resubmitted automatically by the next `run`. The nano
live run may continue in parallel (separate per-model rate pools).

Specification coding is the conceptual centre and runs before topic modeling.
Code each paper once on the full corpus; the resulting paper-level codes are
then analyzed through all five overlapping views (`full_corpus`, `query_1..4`).
Do not pay to recode the same paper separately for each query.

GPT-5.4 mini supplies the authoritative full-corpus paper-level dataset.
Completed GPT-4.1 nano results are the prospectively initiated baseline and
full-corpus sensitivity dataset. Llama and Gemma are supplementary validation
raters; they need not finish the corpus. The full role and analysis decisions
are locked in `docs/METHODS_LOCK.md`. Every cache is permanent model-specific
audit state:

Validation raters may stop retrying a paper after two recorded model-content
failures. This leaves the frozen `spec-v3` prompt, schema, and 4,096-token
ceiling unchanged. Connection, timeout, and rate-limit events never count.
Skipped IDs and counts are recorded in the run manifest as model
non-responses, not successful codes. Resume Gemma with:

```bash
python scripts/run_specification.py \
  --local --model gemma4:31b \
  --base-url http://127.0.0.1:11435/v1 \
  --workers 1 \
  --max-content-failures 2
```

Before every full run, follow the model, parameter, software, hardware, corpus
checksum, and prompt/schema recording checklist in
[`LLM_REPRODUCIBILITY.md`](LLM_REPRODUCIBILITY.md). Do not change the prompt,
schema, model artifact, or request settings while a model's cache is in progress.

```text
data/interim/spec_cache/
    llama3.2/              # pre-standardization pilot; retained, not reused
    spec-v1/               # RETIRED pilot protocol (1,200-token ceiling)
    spec-v1-gemma2400/     # RETIRED Gemma ceiling pilot
    spec-v2/               # RETIRED pilot (mechanism leak found in audit)
    spec-v3/               # CURRENT frozen protocol: uniform 4,096-token
        llama3.2/          #   ceiling, temperature 0, top-p 1, seed 42,
        qwen3.5_27b/       #   excluded failed pilot; retained for audit
        gpt-4.1-nano-2025-04-14/  # and needs_full_text discipline rules
        gemma4_31b/        #   corrected full run at 16,384 context
    spec-v3-gemma8192-pilot/
        gemma4_31b/        #   archived undersized-context attempt
```

Protocol lineage: spec-v1's 1,200-token ceiling truncated valid responses;
spec-v2 fixed the ceiling uniformly at 4,096 but its 4,568-paper audit
exposed a mechanism leak (substantive mechanism codes with empty causal
logic, undercounting the black-box diagnosis 6x) and routine needs_full_text
flagging; spec-v3 adds two coding-discipline rules closing both, with schema
and decoding unchanged. All earlier directories are retained as pilots and
never combined with the standardized spec-v3 experiment. Full history:
docs/LLM_REPRODUCIBILITY.md.

### Primary dataset and IRR comparison set

Nano is not selected as a benchmark winner; it is designated prospectively as
the affordable full-corpus production rater. Its completed `spec-v3` output is
the study dataset used for prevalence, trends, scope comparisons, graph data,
and substantive tables. Llama and Gemma provide independent validation ratings.

The IRR base size is governed by the least-complete retained rater, but equal
file counts alone are not sufficient. Agreement must be calculated only on the
same papers. Define:

```text
S_nano   = successfully coded nano paper IDs
S_llama  = successfully coded Llama paper IDs
S_gemma  = successfully coded Gemma paper IDs
S_IRR    = S_mini ∩ S_nano ∩ S_llama ∩ S_gemma
N_IRR    = |S_IRR| <= min(|S_mini|, |S_nano|, |S_llama|, |S_gemma|)
```

Use all papers in `S_IRR` unless a smaller predeclared stratified sample is
needed. Never compare the first `N` files independently from each cache, because
they may represent different papers. Report each model's completed count, failed
count, intersection size, and coverage by journal, year, and Query 1-4 scope.
Model failures may be systematic, so compare `S_IRR` with the nano full corpus
and disclose any coverage differences before interpreting agreement.
Run only one model at a time on each GPU because local models and topics compete
for that device.

This restriction applies to processes using the same GPU. One local Ollama run
may run alongside a remote OpenAI run, and a second local model may run through
an SSH tunnel to a different workbench GPU.

### Workbench SSH setup and tunnel

The machine-wide helper and recovery reference is
[`WORKBENCH.md`](../WORKBENCH.md).

The workbench currently arrives through an ngrok TCP endpoint, so its hostname
or port may change when that tunnel is recreated. Keep the current values only
in the gitignored `.env`:

```text
WORKBENCH_SSH_HOST=<current ngrok host>
WORKBENCH_SSH_USER=<remote username>
WORKBENCH_SSH_PORT=<current ngrok port>
WORKBENCH_SSH_KEY=/home/suvh/.ssh/id_ed25519
WORKBENCH_OLLAMA_PORT=11434
WORKBENCH_MODEL=gemma4:31b
```

Do not source the complete `.env` because it may contain application settings
that are not shell syntax. Load only the workbench variables:

```bash
set -a
source <(grep -E '^WORKBENCH_[A-Z0-9_]+=' .env)
set +a
```

On first connection, verify the displayed SSH host fingerprint with the
workbench owner before accepting it. If passwordless access is not configured,
install the existing public key once:

```bash
ssh-copy-id \
  -p "$WORKBENCH_SSH_PORT" \
  -i "${WORKBENCH_SSH_KEY}.pub" \
  "$WORKBENCH_SSH_USER@$WORKBENCH_SSH_HOST"
```

Confirm the host, GPUs, Ollama, and installed models without a password prompt:

```bash
ssh \
  -o BatchMode=yes \
  -p "$WORKBENCH_SSH_PORT" \
  -i "$WORKBENCH_SSH_KEY" \
  "$WORKBENCH_SSH_USER@$WORKBENCH_SSH_HOST" \
  'hostname; nvidia-smi; ollama --version; ollama list'
```

The verified host is `Limmy`, with two RTX 4090 24 GiB GPUs and Ollama 0.20.7.
The selected workbench rater is `gemma4:31b` (Ollama ID `6316f0629137`, 19 GB).
Capture its complete Modelfile before the full experiment:

```bash
ssh \
  -o BatchMode=yes \
  -p "$WORKBENCH_SSH_PORT" \
  -i "$WORKBENCH_SSH_KEY" \
  "$WORKBENCH_SSH_USER@$WORKBENCH_SSH_HOST" \
  'ollama show gemma4:31b --modelfile'
```

Keep this tunnel open in its own terminal:

```bash
ssh \
  -N \
  -p "$WORKBENCH_SSH_PORT" \
  -i "$WORKBENCH_SSH_KEY" \
  -L "11435:127.0.0.1:$WORKBENCH_OLLAMA_PORT" \
  "$WORKBENCH_SSH_USER@$WORKBENCH_SSH_HOST"
```

No output is expected while the tunnel is healthy. Verify it from another
terminal:

```bash
curl -fsS http://127.0.0.1:11435/api/tags >/dev/null \
  && echo "Workbench Ollama reachable"
```

### Persistent tmux launch

Commit the frozen protocol before starting, then verify once per session that
the code, protocol, and caches are in the expected state:

```bash
cd ~/projects/ETV_V2
conda activate graphrag

pytest tests/test_llm_coder.py -q                 # includes the uniform-protocol tests
pgrep -af 'run_specification.py|run_topics.py'    # expect no stale runners

# Every dry run must print: Protocol 'spec-v3' ... Cache: data/interim/spec_cache/spec-v3/<model>
python scripts/run_specification.py --dry-run --local --model llama3.2
python scripts/run_specification.py --dry-run --model gpt-4.1-nano-2025-04-14
python scripts/run_specification.py --dry-run --local --model gemma4:31b \
  --base-url http://127.0.0.1:11435/v1
```

Use one tmux session with four fixed windows. This keeps the tunnel and runners
alive if the terminal or editor closes. Tmux does not prevent Windows/WSL from
sleeping, so Windows sleep must remain disabled while the experiment runs.

Create the session once. Do not repeat `new-window` after these windows exist:

```bash
tmux new-session -d -s etv-models -n tunnel
tmux new-window -t etv-models -n llama
tmux new-window -t etv-models -n nano
tmux new-window -t etv-models -n gemma

tmux set-window-option -t etv-models:0 automatic-rename off
tmux set-window-option -t etv-models:1 automatic-rename off
tmux set-window-option -t etv-models:2 automatic-rename off
tmux set-window-option -t etv-models:3 automatic-rename off
```

The fixed mapping is:

```text
0 tunnel   SSH forward to the Gemma workbench
1 llama    local Llama specification runner
2 nano     OpenAI nano specification runner
3 gemma    workbench Gemma specification runner
```

If the session already exists, inspect it instead of recreating windows:

```bash
tmux list-windows -t etv-models \
  -F 'index=#{window_index} name=#{window_name}'
tmux list-panes -a \
  -F 'window=#{window_index}:#{window_name} pane=#{pane_id} command=#{pane_current_command}'
```

Access a window from outside tmux with `tmux attach -t etv-models:<index>`.
When already inside tmux, use `tmux select-window -t etv-models:<index>`.
This conditional form works in either situation:

```bash
index=3  # change to 0, 1, 2, or 3
if [[ -n "${TMUX:-}" ]]; then
  tmux select-window -t "etv-models:$index"
else
  tmux attach -t "etv-models:$index"
fi
```

Window 0 — start the protected workbench tunnel:

```bash
cd ~/projects/ETV_V2
bash scripts/workbench.sh tunnel
```

Silence after the startup message is normal. Do not start a second tunnel on
port 11435.

Window 1 — local Llama on the RTX 5070:

```bash
cd ~/projects/ETV_V2
conda activate graphrag
python scripts/run_specification.py \
  --local --model llama3.2 --workers 1
```

Window 2 — remote GPT-4.1 nano with ten independent workers:

```bash
cd ~/projects/ETV_V2
conda activate graphrag
python scripts/run_specification.py \
  --model gpt-4.1-nano-2025-04-14 --workers 10
```

Window 3 — Gemma on the workbench GPUs:

```bash
cd ~/projects/ETV_V2
conda activate graphrag
python scripts/run_specification.py \
  --local \
  --model gemma4:31b \
  --base-url http://127.0.0.1:11435/v1 \
  --workers 1
```

Detach without stopping anything by holding `Ctrl`, pressing `b`, releasing
both keys, and then pressing `d`. Do not type `b` at the shell prompt. Reattach:

```bash
tmux attach -t etv-models
```

Verify the clean live state from any ordinary terminal:

```bash
tmux list-panes -a \
  -F 'window=#{window_index}:#{window_name} command=#{pane_current_command}'
```

Expected:

```text
window=0:tunnel command=ssh
window=1:llama command=python
window=2:nano command=python
window=3:gemma command=python
```

If accidental duplicate windows exist, first confirm they show `command=bash`,
then remove only an idle duplicate:

```bash
tmux kill-window -t etv-models:<duplicate-index>
```

Never kill a window reporting `python` or `ssh`.

Every command processes the complete corpus; no `--limit` is used. Existing
`spec-v3` paper caches are reused automatically; retired `spec-v1*`/`spec-v2`
pilot caches are never read. Nano's worker count is recorded in its manifest. HTTP
429 rate limits are waited out inside the run (jittered exponential backoff
holds the worker slot), so the pool self-paces at the account's TPM ceiling:
at 200,000 TPM and ~2,100 tokens per paper, expect roughly 90-95 papers per
minute and about 4 hours for the full corpus. Sustained 429 lines in
`failures.jsonl` after this change would mean the backoff budget itself is
exhausted; only then stop and resume with fewer workers.

The workbench is dedicated to Gemma. Qwen 3.5 27B was dropped after its
structured-output pilot produced 4 successful records and 51 failures; retain
that cache for audit but do not resume it. Both RTX 4090 GPUs are now visible to
the Gemma Ollama instance so Ollama can keep the model fully GPU-resident at the
required context length.

### Runtime context must be pinned on Ollama raters

Observed request usage is approximately 4,866 prompt tokens because strict
JSON-schema tokens count toward context, plus a uniform 4,096-token output
ceiling. The earlier 8,192-token Gemma instance could not contain that worst
case and produced truncation/malformed-JSON failures. Gemma is therefore pinned
at 16,384 tokens. `OLLAMA_NUM_PARALLEL=1` prevents context-memory multiplication,
and both GPUs are exposed because a 31B model plus the 16k KV cache may exceed
one 24 GiB card. The earlier `spec-v3/gemma4_31b` cache must be archived under
`spec-v3-gemma8192-pilot/` before the corrected full run starts from zero.

Local Llama (once, then restart the local Ollama service):

```bash
sudo systemctl edit ollama     # add under [Service]: Environment="OLLAMA_CONTEXT_LENGTH=8192"
sudo systemctl restart ollama
ollama ps                      # after the next request: CONTEXT column must show 8192
```

If the local Ollama is not a systemd service, launch it as
`OLLAMA_CONTEXT_LENGTH=8192 ollama serve` instead. An in-flight Llama request
fails once during the restart and is retried from cache automatically.

### Gemma-only workbench helper

Use the repository helper; it reads `.env`, manages only the user-owned server
on remote port 11437, and requires no sudo:

```bash
# Stop the local Gemma runner first, then preserve the 8k pilot once.
mkdir -p data/interim/spec_cache/spec-v3-gemma8192-pilot
mv data/interim/spec_cache/spec-v3/gemma4_31b \
  data/interim/spec_cache/spec-v3-gemma8192-pilot/

bash scripts/workbench.sh restart
bash scripts/workbench.sh status
bash scripts/workbench.sh tunnel   # foreground; keep this pane open
```

Resume Gemma in another pane:

```bash
python scripts/run_specification.py \
  --local --model gemma4:31b \
  --base-url http://127.0.0.1:11435/v1 --workers 1
```

After the first request begins, `bash scripts/workbench.sh status` must report
Gemma `CONTEXT 16384` and `PROCESSOR 100% GPU`. CPU offload is not acceptable.
Qwen remains stopped and its `spec-v3/qwen3.5_27b/` pilot is never combined
with completed-rater results.

### Stop and restart any model

Each rater stops and restarts independently. Every successful paper is cached
immediately, so a restart is free: relaunch the identical pane command and the
run resumes exactly where it stopped, skipping cached papers. Stopping one
model never affects the other two, topic modeling, or the Ollama service.

Stop: list the matching processes, send `SIGTERM`, and give the in-flight
request a few seconds to exit:

```bash
# Llama (local)
pgrep -af 'run_specification.py.*--model[ =]llama3\.2'
pkill -TERM -f '[r]un_specification.py.*--model[ =]llama3\.2'

# GPT-4.1 nano (OpenAI)
pgrep -af 'run_specification.py.*--model[ =]gpt-4\.1-nano'
pkill -TERM -f '[r]un_specification.py.*--model[ =]gpt-4\.1-nano'

# Gemma (workbench tunnel)
pgrep -af 'run_specification.py.*--model[ =]gemma4:31b'
pkill -TERM -f '[r]un_specification.py.*--model[ =]gemma4:31b'

# Qwen is excluded and should already be stopped. Verify only:
pgrep -af 'run_specification.py.*--model[ =]qwen3\.5:27b' \
  || echo "Qwen runner remains stopped"
```
If a listed process remains stuck, copy its PID from `pgrep` and force-stop
only that PID with `kill -KILL <stuck-pid>`.

Restart: the identical launch command per model; do not add `--limit` and do
not delete any cache:

```bash
# Llama (pane 2)
python scripts/run_specification.py \
  --local --model llama3.2 --workers 1

# GPT-4.1 nano (pane 3)
python scripts/run_specification.py \
  --model gpt-4.1-nano-2025-04-14 --workers 10

# Gemma (pane 4; requires the pane-1 tunnel)
python scripts/run_specification.py \
  --local \
  --model gemma4:31b \
  --base-url http://127.0.0.1:11435/v1 \
  --workers 1
```

Startup prints how many papers are already cached and excluded from the
remaining total; that number confirms the resume worked.

After stopping a local or workbench Ollama rater, optionally unload its model
from GPU memory (`ollama stop` unloads; it deletes nothing):

```bash
ollama stop llama3.2 && ollama ps            # local RTX 5070

bash scripts/workbench.sh stop               # workbench Gemma instance
```

Do not use `systemctl stop ollama` unless the entire Ollama service must be
taken offline.

Each model pane displays its own percentage, elapsed time, throughput, ETA,
failure count, and worker count. Monitor local and workbench allocation with:

```bash
ollama ps

bash scripts/workbench.sh status
```

Failures are appended immediately to
`data/interim/spec_cache/<protocol>/<model>/failures.jsonl`. The runner prints the
first five errors and every hundredth thereafter. A circuit breaker stops new
submissions when at least half of the first 20 or more attempts fail; diagnose
that log before resuming. Successful paper caches remain valid.

If the ngrok endpoint changes, stop the Gemma runner and tunnel, update and
reload the workbench `.env` values, recreate the tunnel, and rerun the identical
Gemma command. Completed paper caches remain valid. The Llama and nano processes
are independent of the workbench tunnel and may continue.

The root-level `llama3.2/` cache and all `spec-v1*`/`spec-v2` caches predate
the current frozen protocol. Do not delete them. Restarting the standard
commands uses `spec-v3/<model>/`, begins each standardized experiment at
zero, and leaves the pilot evidence available for audit.

In an interactive terminal, the runner rewrites one progress line in place:

```text
Specification [>-----------------------------] 24/22322 (0.11%) |
elapsed 1:24:03 | 0.005/s | ETA 1234:56:00 | failures 0 | START <paper_id>
```

`START` appears before the blocking model request, so a slow paper remains
visible while the model generates. The same line changes to `DONE` and advances
after the response is cached. Redirected output uses durable newline-based log
records instead. Startup reports cached papers excluded from the remaining total.

Each command writes model-specific outputs and never overwrites another model:

```text
paper_specifications_<model>_<protocol>.csv
specification_report_<model>_<protocol>.json
curation_overrides_<model>_<protocol>.json
paper_specifications_curated_<model>_<protocol>.csv
```

Nano's full paper-level result is analyzed as five overlapping views:
`full_corpus`, `query_1`, `query_2`, `query_3`, and `query_4`. Llama/Gemma IRR
uses the same scope flags within `S_IRR`. Coding remains query-invariant, so do
not make five duplicate LLM calls per model.

## 3. Curate and compare specification models

Curate nano's full result and the Llama/Gemma records in `S_IRR` for inter-rater
analysis. Preserve all other cached validation records for audit.
Auto-accept requires `stated` evidence and confidence >= 0.8; everything else
queues least-confident-first. Human overrides remain separate from the immutable
per-model LLM cache.

```bash
python scripts/curate_specification.py --model llama3.2 --report
python scripts/curate_specification.py --model gpt-4.1-nano-2025-04-14 --report
python scripts/curate_specification.py --model gemma4:31b --report

python scripts/curate_specification.py --model <model>
python scripts/curate_specification.py --model <model> --export
```

Review commands: Enter accepts, `1-N` overrides with a numbered value, `e`
shows the abstract, `s` skips, `d` defers the dimension corpus-wide, `B`
batch-accepts the remaining dimension queue, and `q` saves and quits. Decisions
save after every action and are resumable.

Outputs: `curation_overrides_<model>_spec-v3.json` and, with `--export`,
`paper_specifications_curated_<model>_spec-v3.csv` with per-dimension curation status
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
