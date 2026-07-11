#!/usr/bin/env bash
# Gemma-only workbench helper for the specification experiment.
#
# Inputs: WORKBENCH_* values from the project .env and a reachable SSH host.
# Outputs: a user-owned Ollama server on remote port 11437, an optional local
# SSH tunnel on port 11435, and status/log output. Qwen is a retired pilot.
set -euo pipefail
cd "$(dirname "$0")/.."

set -a
# shellcheck disable=SC1090
source <(grep -E '^WORKBENCH_[A-Z0-9_]+=' .env)
set +a
: "${WORKBENCH_SSH_HOST:?WORKBENCH_SSH_HOST missing from .env}"
: "${WORKBENCH_SSH_PORT:?WORKBENCH_SSH_PORT missing from .env}"
: "${WORKBENCH_SSH_USER:?WORKBENCH_SSH_USER missing from .env}"
: "${WORKBENCH_SSH_KEY:?WORKBENCH_SSH_KEY missing from .env}"

SSH=(ssh -p "$WORKBENCH_SSH_PORT" -i "$WORKBENCH_SSH_KEY"
     "$WORKBENCH_SSH_USER@$WORKBENCH_SSH_HOST")
MODELS_DIR="${WORKBENCH_OLLAMA_MODELS:-/usr/share/ollama/.ollama/models}"
REMOTE_PORT=11437
LOCAL_PORT=11435
CONTEXT_LENGTH=16384
LOG_PATH=~/ollama-logs/gemma.log

case "${1:-}" in
start)
  "${SSH[@]}" "
    set -eu
    mkdir -p ~/ollama-logs
    existing=\$(lsof -t -iTCP:$REMOTE_PORT -sTCP:LISTEN 2>/dev/null | head -n 1 || true)
    if [ -n \"\$existing\" ]; then
      echo \"Gemma instance already listening on :$REMOTE_PORT (PID \$existing).\"
      echo \"Run 'bash scripts/workbench.sh restart' to apply configuration.\"
      exit 0
    fi
    CUDA_VISIBLE_DEVICES=0,1 \
      OLLAMA_HOST=127.0.0.1:$REMOTE_PORT \
      OLLAMA_CONTEXT_LENGTH=$CONTEXT_LENGTH \
      OLLAMA_NUM_PARALLEL=1 \
      OLLAMA_MAX_LOADED_MODELS=1 \
      OLLAMA_MODELS=$MODELS_DIR \
      setsid nohup ollama serve > ~/ollama-logs/gemma.log 2>&1 < /dev/null &
    sleep 5
    tags=\$(curl -fsS http://127.0.0.1:$REMOTE_PORT/api/tags 2>/dev/null) || {
      echo 'Gemma instance failed to start:'
      tail -n 20 ~/ollama-logs/gemma.log
      exit 1
    }
    if echo \"\$tags\" | grep -q 'gemma4:31b'; then
      echo 'Gemma instance ready on :$REMOTE_PORT (context $CONTEXT_LENGTH; GPUs 0,1 visible).'
    else
      echo 'Instance is up but gemma4:31b is not in the configured model store.'
      echo "Run: bash scripts/workbench.sh pull"
      exit 1
    fi"
  ;;
stop)
  "${SSH[@]}" "
    pid=\$(lsof -t -iTCP:$REMOTE_PORT -sTCP:LISTEN 2>/dev/null | head -n 1 || true)
    if [ -n \"\$pid\" ]; then
      kill -TERM \"\$pid\"
      echo \"Stopped Gemma Ollama PID \$pid on :$REMOTE_PORT.\"
    else
      echo 'Gemma instance is already stopped.'
    fi"
  ;;
restart)
  "$0" stop
  sleep 3
  "$0" start
  ;;
status)
  "${SSH[@]}" "
    echo '--- Gemma instance (:$REMOTE_PORT)'
    OLLAMA_HOST=http://127.0.0.1:$REMOTE_PORT ollama ps 2>/dev/null || echo down
    echo '--- listener'
    lsof -nP -iTCP:$REMOTE_PORT -sTCP:LISTEN 2>/dev/null || true
    echo '--- GPU utilization and memory'
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv"
  ;;
tunnel)
  echo "Tunnel up: local $LOCAL_PORT -> Gemma workbench instance :$REMOTE_PORT."
  echo "Silence means healthy. Ctrl+C stops only the tunnel."
  exec ssh -N \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=10 \
    -o TCPKeepAlive=yes \
    -o ExitOnForwardFailure=yes \
    -p "$WORKBENCH_SSH_PORT" \
    -i "$WORKBENCH_SSH_KEY" \
    -L "$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" \
    "$WORKBENCH_SSH_USER@$WORKBENCH_SSH_HOST"
  ;;
logs)
  "${SSH[@]}" 'tail -n 50 ~/ollama-logs/gemma.log'
  ;;
pull)
  "${SSH[@]}" "
    OLLAMA_HOST=http://127.0.0.1:$REMOTE_PORT ollama pull gemma4:31b
    OLLAMA_HOST=http://127.0.0.1:$REMOTE_PORT ollama list | grep 'gemma4:31b'"
  ;;
*)
  echo "Usage: bash scripts/workbench.sh {start|stop|restart|status|tunnel|logs|pull}" >&2
  exit 1
  ;;
esac
