# ETV workbench access and Gemma runtime

This machine reaches the `Limmy` workbench through an SSH-forwarded ngrok TCP
endpoint. The workbench supplies two RTX 4090 24 GiB GPUs for the Gemma 4 31B
specification rater. Qwen 3.5 27B is no longer an active rater: its structured-
output pilot produced 4 successful records and 51 failures. Preserve that cache
for audit, but do not resume the Qwen runner.

## Local configuration

Keep these values in the gitignored project `.env`:

```text
WORKBENCH_SSH_HOST=<current ngrok host>
WORKBENCH_SSH_USER=<remote username>
WORKBENCH_SSH_PORT=<current ngrok port>
WORKBENCH_SSH_KEY=/home/suvh/.ssh/id_ed25519
WORKBENCH_OLLAMA_MODELS=/home/<remote-user>/.ollama/models
```

The ngrok host and port can change. The SSH key and all credentials remain
outside Git. Do not disable SSH host-key verification.

## Preferred helper

Run from the repository root. The helper reads `.env` itself and manages only
the user-owned Ollama listener on remote port 11437; it never requires sudo and
does not modify the system Ollama service.

```bash
bash scripts/workbench.sh start    # start Gemma if the port is free
bash scripts/workbench.sh restart  # apply the pinned runtime configuration
bash scripts/workbench.sh status   # endpoint, context, processor and both GPUs
bash scripts/workbench.sh tunnel   # foreground local 11435 -> remote 11437
bash scripts/workbench.sh logs     # recent Gemma Ollama log
bash scripts/workbench.sh stop     # stop only the :11437 user-owned listener
bash scripts/workbench.sh pull     # pull Gemma into the configured model store
```

The fixed server configuration is:

```text
Model: gemma4:31b
Remote endpoint: 127.0.0.1:11437
Local tunnel endpoint: 127.0.0.1:11435
OLLAMA_CONTEXT_LENGTH=16384
OLLAMA_NUM_PARALLEL=1
OLLAMA_MAX_LOADED_MODELS=1
CUDA_VISIBLE_DEVICES=0,1
```

Both GPUs are visible so Ollama can distribute the 31B model and 16k KV cache
if they do not fit on one card. `OLLAMA_NUM_PARALLEL=1` prevents the context
allocation from multiplying.

## Why 16,384 context is required

The strict JSON schema is part of the request and actual usage is approximately
4,866 prompt tokens, not the earlier 1,400-token cost estimate. With a uniform
4,096-token output ceiling, the worst case needs roughly 8,962 tokens. The
observed 8,192-context Gemma instance produced truncated and malformed JSON.
The corrected 16,384 context supplies headroom while retaining the frozen
`spec-v3` prompt, schema and decoding parameters.

Before the corrected full run, preserve the earlier Gemma cache as a pilot:

```bash
mkdir -p data/interim/spec_cache/spec-v3-gemma8192-pilot
mv data/interim/spec_cache/spec-v3/gemma4_31b \
  data/interim/spec_cache/spec-v3-gemma8192-pilot/
```

Do this once, only after stopping the Gemma Python runner. Never move Llama or
nano caches.

## Start and verify Gemma

```bash
bash scripts/workbench.sh restart
bash scripts/workbench.sh status
```

Keep the tunnel open in its own local tmux window:

```bash
bash scripts/workbench.sh tunnel
```

From the Gemma tmux window:

```bash
cd ~/projects/ETV_V2
conda activate graphrag
python scripts/run_specification.py \
  --local \
  --model gemma4:31b \
  --base-url http://127.0.0.1:11435/v1 \
  --workers 1
```

After the first request starts:

```bash
bash scripts/workbench.sh status
```

Required state:

```text
CONTEXT: 16384
PROCESSOR: 100% GPU
```

One or both GPUs may be used. Any CPU percentage means the model is offloading
and the full run should be paused for diagnosis.

## Interruption and recovery

The Python runner writes each successful paper immediately. A stopped runner,
SSH tunnel or user-owned Ollama server can be restarted without deleting the
correct cache.

If the ngrok endpoint changes:

1. Stop the Gemma runner and tunnel.
2. Update `WORKBENCH_SSH_HOST` and `WORKBENCH_SSH_PORT` in `.env`.
3. Verify the new host fingerprint with the workbench owner.
4. Run `bash scripts/workbench.sh status`.
5. Recreate the tunnel and resume the identical Gemma command.

If startup fails, inspect:

```bash
bash scripts/workbench.sh logs
```

The system model store may be unreadable to the SSH user. Use the user-local
`WORKBENCH_OLLAMA_MODELS` directory and run `bash scripts/workbench.sh pull`.
The recorded Gemma artifact is Q4_K_M with digest prefix `6316f0629137`.

## Verified hardware

```text
Host: Limmy
GPUs: 2 × NVIDIA RTX 4090, 24 GiB each
Driver: 550.163.01
CUDA: 12.4
Ollama: 0.20.7
Gemma: gemma4:31b, 31.3B, Q4_K_M
Artifact digest: 6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7
```

Record `bash scripts/workbench.sh status`, the model digest, Git commit, corpus
checksum, and protocol manifest with the completed experiment.
