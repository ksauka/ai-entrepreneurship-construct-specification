# LLM specification experiment reproducibility

This document records the model, request, prompt, schema, input, and execution
settings required to reproduce the paper-level specification experiment. The
implementation sources of truth are `scripts/run_specification.py`,
`src/aecsp/specification/llm_coder.py`, and
`src/aecsp/specification/schema.py`.

## Experimental unit and input

Each request codes one paper. The coder reads these columns from
`data/processed/master_corpus.csv`:

| Request field | Corpus column |
|---|---|
| Stable identifier | `paper_id` |
| Title | `Title` |
| Abstract | `Abstract` |
| Keywords | `Author Keywords` |
| Journal | `Source title` |
| Publication year | `Year` |

`Index Keywords`, full text, topic assignments, and query identity are not sent
to the model. A paper is coded once per model; Query 1-4 are overlapping
analytical views of the resulting paper-level codes.

The exact corpus file used for a completed experiment must be preserved or
recorded with a SHA-256 checksum:

```bash
sha256sum data/processed/master_corpus.csv
```

## Prompt and output contract

The system prompt is the `SYSTEM_PROMPT` constant in `llm_coder.py`. Its
dimension briefing is generated from `SPECIFICATION_DIMENSIONS` in `schema.py`.
The user prompt contains the title, journal, year, author keywords, and abstract
in a fixed template.

The response uses OpenAI-compatible structured output:

```text
response_format.type = json_schema
response_format.json_schema.name = ai_specification_profile
response_format.json_schema.strict = true
additionalProperties = false
```

The schema requires all seven specification dimensions, their evidence,
epistemic label, code, and confidence, plus the auxiliary construct fields,
specification problems, full-text flags, and adversarial review. Controlled
values come from `schema.py`; changing that file changes the coding instrument.

For each experiment, preserve the Git commit hash because it identifies the
exact prompt and schema:

```bash
git rev-parse HEAD
git status --short
```

A clean status is required for a fully commit-addressable instrument. If the
worktree is not clean, archive the diff with the experiment records.

## Evidentiary boundary: abstract-level construct-specification coding

Decision (2026-07-11): the study codes ONLY from title, abstract, and author
keywords. Outputs are abstract-level construct-specification measures, not
definitive full-text coding. A blinded, independently coded 50-paper human
sample supplies a validation anchor, not an infallible gold standard. The
defensible scope is:

- Treat every coded dimension as observable abstract-level specification.
- `needs_full_text` is the explicit indicator that a dimension cannot be
  established reliably from the available metadata; it must mark genuine
  insufficiency, never serve as a routine disclaimer (coding rule 8).
- Human curation handles ambiguous and lower-confidence results.
- GPT-5.4 mini supplies the primary full-corpus dataset; nano supplies the
  completed baseline/sensitivity dataset. Human–model validation and available
  model intersections supply agreement evidence.
- Model agreement is reported as reliability of abstract-level coding, not
  agreement about every detail contained in the full papers.

Dimension observability from abstracts (report alongside results):

| More observable | Often needs full text |
|---|---|
| AI role/function | exact construct definition |
| AI type/form | theoretical mechanism detail |
| level of analysis | boundary/scope conditions |
| broad process stage | process sequencing |
| explicitly named mechanisms | theory integration |

Pre-registered analysis rule (declared 2026-07-11, before run completion):
the spec-v3 checkpoint audit showed the nano rater codes substantive
mechanisms with an empty `ai_mechanism_logic` in ~62% of cases, violating
coding rule 7, while the 31B and 3B raters comply. Rather than a further
protocol revision, nano completes under spec-v3 and the following
deterministic rule applies at analysis time to every rater equally: a
substantive `ai_mechanism` code whose `ai_mechanism_logic` is empty is
analyzed as "mechanism missing" (no explicitly named mechanism observable at
abstract level). The recorded codes are never rewritten; the rule is applied
in the analysis layer and reported with both raw and corrected prevalences.
GPT-5.4 mini is the adopted mechanism-reliable primary rater; the challenge
pilot and its enriched selection are disclosed below.

### GPT-5.4 mini decision pilot

Before selecting another paid full-corpus rater, the pinned OpenAI model
`gpt-5.4-mini-2026-03-17` is evaluated on a fixed 50-paper challenge set under
the unchanged `spec-v3` instrument. No GPT-5.4 nano experiment is planned.
The challenge set contains nano output-limit failures, the lowest-confidence
completed nano records, records with the greatest `needs_full_text`
uncertainty, and a seeded completed-paper control sample. HTTP rate-limit
failures are excluded because they measure transport congestion rather than
paper-level coding difficulty.

The manifest is generated by `scripts/prepare_gpt54mini_challenge.py` and
stored at `data/interim/spec_pilots/gpt54mini_challenge_50.csv`. Its initial
SHA-256 is
`c2b5885cc9bd3a9f06c854dcf3f6071ff6fbc7382b12c43420a0c7ddddad4689`.
Adoption is based on valid-schema completion, output truncation, evidence
grounding, mechanism/logic consistency, confidence calibration, and agreement
with nano and available validation-rater results. Cost alone is not an
adoption criterion.

Gate results (run 2026-07-11, 50/50 coded, zero failures, ~$0.21): on the
identical challenge papers, gpt-5.4-mini vs nano -
rule-7 empty-logic on substantive mechanism codes 0% vs 74%; needs_full_text
1.5 vs 5.0 dimensions/paper; stated-evidence grounding (70% word overlap with
title+abstract) 92% vs 70%; confidence mean 0.67 with an honest low tail on
hard papers; discriminant validity: all 20 lowest-confidence papers coded
mechanism-missing vs 8/18 of the seeded controls; both nano output-limit
failures completed (max 612 output tokens under the 4,096 ceiling). Observed
tokens: ~3,328 input / ~472 output per paper.

Adoption (2026-07-11): gpt-5.4-mini-2026-03-17 is adopted as the primary
full-corpus rater. The 50-paper challenge is enriched for nano difficulty and
therefore establishes protocol compliance rather than population accuracy.
Gemma remains a supplementary validation rater. Its earlier 50%-failure
circuit-breaker event was dominated by a failed SSH tunnel (93 connection
errors), not a valid estimate of Gemma's JSON reliability; after tunnel repair
and a declared two-content-failure retry ceiling, it resumed with an initially
acceptable success rate. Llama and Gemma need not complete the full corpus.

Batch transport (correction recorded 2026-07-11): the OpenAI "Batch queue
limit" is the maximum input tokens ENQUEUED AT ONE TIME, not per day. An
earlier statement that Batch was "not viable" at the current tier was wrong.
Full-corpus economics for gpt-5.4-mini-2026-03-17 at observed tokens: live
~ $103 (~7 hours at 200,000 TPM with in-run backoff), Batch ~ $54 as
sequential chunks under the enqueued budget (~1-3 days, unattended).
`scripts/run_specification_openai_batch.py` implements the Batch transport
with a byte-identical request body to the live path (test-enforced), writes
into the SAME spec-v3 per-model cache (live and Batch runs resume each
other), records `provider_mode: openai_batch` and the per-chunk token budget
in the manifest, and refuses paid submission without an explicit `--yes`.

Methods statement (use verbatim):

> Coding was based on titles, abstracts, and author keywords. The resulting
> measures represent observable abstract-level construct specification rather
> than exhaustive full-text interpretation. Dimensions that could not be
> established reliably from the available metadata were marked as requiring
> full-text review and were not silently inferred.

## Request parameters

The current implementation sends the following settings to every provider:

| Parameter | Current value | Status |
|---|---:|---|
| Protocol ID | `spec-v3` (uniform for every rater) | Identifies the frozen request profile |
| `temperature` | `0.0` | Greedy/low-variance decoding |
| `top_p` | `1.0` | Neutral nucleus-sampling ceiling |
| `seed` | `42` | Best-effort repeatability on supported providers |
| output-token ceiling | `4096` (uniform) | `max_tokens` on Ollama; `max_completion_tokens` on OpenAI |
| frequency penalty | `0.0` | No frequency modification |
| presence penalty | `0.0` | No presence modification |
| completions (`n`) | `1` | One profile per paper and model |
| streaming | `false` | Parse one complete structured response |
| `response_format` | strict JSON schema | Controlled output contract |
| request timeout | `300` seconds | Explicit client timeout |
| SDK retries | `2` | Explicit client retry limit |
| HTTP 429 handling | in-run exponential backoff (jittered, capped 30s, max 12 attempts) | Execution-level pacing; 429 is transient transport state, never rater non-response, and reaches `failures.jsonl` only if the backoff budget is exhausted |
| project retry | next resumed run | Failed papers are not cached |
| validation content-failure ceiling | explicit `--max-content-failures`; recommended `2` | Repeated truncation/invalid-JSON papers become recorded model non-responses; transport failures never count |
| local workers | `1` by default | Protects single-GPU memory and throughput |
| remote workers | `10` by default | Concurrent independent API requests |
| Ollama runtime context | Gemma `16384`; local Llama recorded separately | Actual Gemma prompts are ~4,866 tokens including the strict schema, so 8,192 could not contain the prompt plus the 4,096 output ceiling. Both workbench GPUs are visible; parallelism remains 1 |

Temperature zero reduces sampling variation but does not guarantee identical
outputs across hardware, provider revisions, quantizations, or backend versions.
The seed is a best-effort control, not a guarantee of byte-identical responses.
Provider revisions, backend kernels, hardware, and quantization can still affect
outputs. These settings must not change within a declared protocol experiment.
Worker count is an execution setting recorded in the manifest; it does not
change any paper's prompt, schema, or decoding parameters.

`EST_INPUT_TOKENS=1400` and `EST_OUTPUT_TOKENS=700` are cost-estimation constants
only. They do not constrain the API request.

## Model and provider resolution

The command line has highest precedence, followed by `.env`, followed by the
code default.

| Mode | Endpoint | Model resolution | API key |
|---|---|---|---|
| OpenAI-compatible remote | `OPENAI_BASE_URL`, otherwise OpenAI | `--model`, `OPENAI_MODEL`, `gpt-4.1-nano-2025-04-14` | `OPENAI_API_KEY` |
| Local Ollama | `OLLAMA_BASE_URL`, otherwise `http://localhost:11434/v1` | `--model`, `OLLAMA_MODEL`, `llama3.2` | Literal placeholder `ollama` |

Planned study models currently named in the RUNBOOK are:

| Experiment label | Command model identifier | Provider status |
|---|---|---|
| Llama 3.2 | `llama3.2` | Local Ollama; standardized run restarts under `spec-v3` |
| Qwen 3.5 27B | `qwen3.5:27b` | Excluded after failed structured-output pilot: 4 successes, 51 failures; cache retained for audit |
| GPT-4.1 nano | `gpt-4.1-nano-2025-04-14` | Pinned OpenAI snapshot; restarts under `spec-v3` |
| Gemma 4 31B | `gemma4:31b` | Workbench Ollama; same uniform `spec-v3` protocol |

These identifiers are experiment labels until their immutable provider version
or local artifact digest is recorded. In particular, Ollama tags can be moved
or replaced without changing the tag text.

### Observed Llama environment on 2026-07-10

The diagnostic logs supplied during the active Llama run reported:

| Setting | Observed value |
|---|---|
| Ollama version | `0.17.6` |
| Requested tag | `llama3.2:latest` |
| Ollama model ID | `a80c4f17acd5` |
| Model | Llama 3.2 3B Instruct (`3.21 B` parameters) |
| Artifact format | GGUF V3 |
| Quantization | Q4_K Medium |
| Artifact size | `1.87 GiB` |
| Runtime context | `4096` tokens |
| GPU offload | `29/29` layers, reported as `100% GPU` |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU, 8 GiB |
| Driver / CUDA | driver `592.27`, CUDA `13.1` |
| Flash Attention | automatic, resolved to enabled |
| Ollama parallelism | `OLLAMA_NUM_PARALLEL=1` |
| Ollama keep-alive | `5m0s` |

This table documents the observed session, but the commands below must still be
captured with the final experiment archive because tags, software, and drivers
can change during a long-running study.

### Workbench research-grade model

The accessible workbench provides two RTX 4090 GPUs with 24 GiB each and has
`gemma4:31b` installed as a 19 GB Ollama artifact. Gemma 4 31B is selected as
the strongest practical workbench rater because it fits on one GPU and its
official model card reports stronger general instruction/reasoning performance
than Qwen3.5 27B. The older 42 GB Llama 3.3 70B artifact requires both GPUs and
is not selected merely on parameter count. Preserve the exact Gemma Ollama ID,
Modelfile, quantization, and runtime allocation before the full run.

Observed workbench inventory on 2026-07-10:

| Setting | Observed value |
|---|---|
| Host label | `Limmy` |
| GPUs | 2 × NVIDIA GeForce RTX 4090, 24 GiB each |
| NVIDIA driver / CUDA | `550.163.01` / `12.4` |
| Ollama | `0.20.7` |
| Gemma tag / Ollama ID | `gemma4:31b` / `6316f0629137` |
| Gemma artifact size | `19 GB` |
| Model maximum context | `262144` tokens |
| Failed Gemma pilot context | `8192` tokens |
| Corrected Gemma context | `16384` tokens; both GPUs visible, parallelism 1 |
| Model license | Apache License 2.0 |

The workbench is reached through an SSH-forwarded ngrok TCP endpoint. The
ephemeral ngrok hostname and port are execution metadata, not stable model
identity. Record them in the run log but never treat them as part of the coding
protocol. The SSH private key remains outside the repository; only its local
path may appear in the gitignored `.env`.

The full three-model execution uses independent resources concurrently:

| Model | Execution endpoint | Workers | Cache namespace |
|---|---|---:|---|
| Llama 3.2 | Local Ollama / RTX 5070 | 1 | `spec-v3/llama3.2/` |
| GPT-4.1 nano | OpenAI API | 10 | `spec-v3/gpt-4.1-nano-2025-04-14/` |
| Gemma 4 31B | Workbench pinned instance / both RTX 4090s visible | 1 | `spec-v3/gemma4_31b/` |
| Qwen 3.5 27B | Excluded pilot | 1 | `spec-v3/qwen3.5_27b/` retained, not completed |

### Full-study and IRR roles

GPT-5.4 mini is the primary full-corpus production rater. GPT-4.1 nano is the
completed prospectively initiated baseline and sensitivity dataset. Llama and
Gemma are supplementary validation raters whose completed records support
agreement analysis; they are not required to reach full-corpus completion.

Model-only IRR uses paper IDs successfully coded by all available raters:

```text
S_IRR = S_mini intersection S_nano intersection S_llama intersection S_gemma
N_IRR = size(S_IRR)
```

The least-complete model bounds `N_IRR`, but the actual intersection may be
smaller because failures differ across models. Agreement must never be computed
from separate equal-sized lists. Archive the ordered intersection IDs and report
completion/failure counts plus journal, year, and analytical-scope coverage.
Because non-response can be related to abstract length or output complexity,
compare the IRR intersection with nano's full dataset and disclose selection
differences as a limitation.

Concurrency changes wall-clock execution but not the per-paper prompt, schema,
decoding settings, or model-specific cache. Completion order is therefore not
an analytical variable; each cached record remains keyed by `paper_id`.

Gemma's first `spec-v3` attempt ran with an 8,192-token Ollama context. Actual
prompt usage was approximately 4,866 tokens once the strict JSON schema was
counted, leaving insufficient space for the uniform 4,096-token output ceiling.
That cache is archived under `spec-v3-gemma8192-pilot/`. The corrected full run
starts from zero at 16,384 context; prompt, schema and decoding parameters remain
unchanged. Qwen's 4-success/51-failure pilot is retained but excluded because
its responses repeatedly exhausted 4,096 output tokens without completing JSON.

Protocol history (why spec-v2 exists): spec-v1 set a 1,200-token output
ceiling. Otherwise valid structured Gemma responses exceeded 1,200 and then a
2,400-token pilot ceiling, and Llama subsequently produced runs of consecutive
1,200-token truncations. Successful codings from all three raters clustered at
roughly 900-1,200 output tokens, so the ceiling sat inside the natural length
distribution of the instrument (long abstracts produce longer evidence quotes
across seven dimensions). Model-specific ceilings (`spec-v1-gemma4096`) were
briefly introduced and then rejected because they weaken comparability:
the ceiling is part of the measurement instrument. spec-v2 therefore keeps the
prompt, schema, temperature, top-p, seed, penalties, timeout, and retry
settings identical to spec-v1 and raises the output ceiling to a uniform
4,096 for every rater. Each record since spec-v2 stores the observed
`prompt_tokens`/`output_tokens` so the final archive can report the true
output-length distribution and demonstrate the ceiling had headroom (spec-v2
audit: nano mean 522, p95 618, max 814; llama max 2,526).

spec-v3 (2026-07-11, current): a 4,568-paper audit of spec-v2 nano output
exposed a MECHANISM LEAK - raters coded substantive mechanisms while leaving
`ai_mechanism_logic` empty (27% of substantive codes) and flagging
'mechanism missing' in specification_problem anyway (52%), so the coded
mechanism-missing rate (5%) understated the empty-logic-corrected rate (31%)
six-fold. Because black-box claims are the study's core diagnosis, spec-v3
adds coding-discipline rule 7 (a substantive mechanism code REQUIRES a
non-empty causal ai_mechanism_logic; otherwise 'mechanism missing'; the code
and the problem flag must agree) and rule 8 (needs_full_text is flagged only
when the abstract is genuinely insufficient, not as a routine caveat - under
spec-v2, nano flagged >90% of papers, destroying the signal). Schema and all
decoding parameters are unchanged from spec-v2. The `spec-v1*` and `spec-v2`
caches are retained as pilots and are never combined with spec-v3 data.

Decision rule (agreed 2026-07-11): a response that exceeds the uniform
4,096-token ceiling is roughly 4x the observed median and is therefore not
plausible as honest length; it indicates degenerate repetition on that paper.
Such responses raise on `finish_reason=length`, are appended to
`failures.jsonl`, are never cached as partials, and are documented in the
final analysis as rater non-response for that paper-model pair. The ceiling is
not raised again in response to such failures.

## Required record for every production or validation run

Before starting or resuming a full experiment, save the following in the
research log:

1. Experiment label and exact command.
2. Git commit and clean/dirty worktree status.
3. Master-corpus SHA-256 and paper count.
4. Provider name and base URL, excluding credentials.
5. Exact model identifier and release/version date when the provider exposes it.
6. OpenAI Python SDK version and Python environment export.
7. Explicit request parameters and the provider defaults listed above.
8. Start/end timestamps, cache directory, final coded count, failures, and runtime.
9. Worker count, hardware, and backend details.

For Ollama runs, also capture:

```bash
ollama --version
ollama list
ollama show llama3.2 --modelfile
nvidia-smi
conda env export --from-history > environment-history.yml
```

Record the model ID/digest printed by `ollama list`, quantization, context
length, Ollama version, GPU model, driver version, and CUDA version. Repeat the
commands with the Gemma identifier for its full experiment. Preserve Qwen's
artifact record only with the excluded pilot archive.

For remote providers, retain the provider's exact dated model/version identifier
where available, API/SDK version, service tier or region if relevant, and the
request report. Never commit API keys or raw credentials.

## Cache and interruption behavior

Each standardized response is flattened and written immediately to:

```text
data/interim/spec_cache/<protocol>/<sanitized-model-id>/<paper-id>.<sha1-prefix>.json
```

Pre-standardization root-level caches such as
`data/interim/spec_cache/llama3.2/`, and the superseded `spec-v1*` protocol
directories, are retained as pilot data and are never read by a `spec-v3` run.
No deletion or manual movement is required.

The cache records `paper_id` and `coding_model`, and reruns skip existing files.
Interrupted in-flight requests are retried because no cache file exists until a
response has parsed successfully. Cache directories are separated by protocol
and model. Each cache contains `protocol_manifest.json`, and each paper record
contains the protocol ID, fingerprint, and decoding parameters. Therefore:

- do not alter the prompt, schema, model artifact, or request settings midway
  through a model's full run;
- do not reuse a model cache after changing those inputs;
- archive the old cache under an experiment-specific name before intentionally
  starting a revised coding instrument.

The final outputs are:

```text
data/processed/specification/paper_specifications_<model>_<protocol>.csv
data/processed/specification/specification_report_<model>_<protocol>.json
```

The report records timestamp, protocol ID and fingerprint, parameters, timeout,
retry limit, model, provider mode, base URL, SDK and Python versions, corpus
checksum, counts, failures, and runtime. The external research log must still
retain the model artifact digest, Git commit and status, and hardware details.

## Protocol rationale

Temperature zero suits a classification instrument because variation is
undesirable. `top_p=1` leaves nucleus sampling unrestricted rather than adding
a second creativity constraint. Seed 42 supplies a conventional fixed seed
where supported. Zero penalties avoid modifying terminology probabilities. The
default 1,200-token ceiling is above the approximately 700-token expected
profile while preventing unbounded responses. Gemma uses 4,096 because observed
valid structured profiles exceeded both lower ceilings. Responses stopped by
the applicable ceiling are rejected rather than cached. The five-minute timeout
accommodates local inference while preventing indefinite stalled requests.

Ollama documents the OpenAI-compatible request fields used here, including
seed, temperature, top-p, penalties, `max_tokens`, and response formatting:
<https://docs.ollama.com/api/openai-compatibility>. OpenAI recommends pinned
model versions for consistent behavior, so the remote experiment uses the
dated GPT-4.1 nano snapshot rather than the moving alias.
