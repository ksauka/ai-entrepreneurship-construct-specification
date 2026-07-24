#!/usr/bin/env bash
# Build a checksummed desktop-failover bundle without placing secrets in Git.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/backup_review_host.sh DESTINATION [--no-secrets] [--include-project-env]

The destination may be a mounted flash drive or a private local directory.
Runtime data are stored in a portable tar archive so Windows-mounted drives do
not need to support Unix ownership or permission metadata. Host credentials and
the tunnel token are encrypted into one GPG/AES256 archive unless --no-secrets
is used.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

DESTINATION="$1"
shift
INCLUDE_SECRETS=1
INCLUDE_PROJECT_ENV=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-secrets) INCLUDE_SECRETS=0 ;;
    --include-project-env) INCLUDE_PROJECT_ENV=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ETV_PYTHON_BIN:-$HOME/miniconda3/envs/graphrag/bin/python}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$DESTINATION"
DESTINATION="$(cd "$DESTINATION" && pwd)"
BUNDLE_ROOT="${DESTINATION%/}/etv-review-host-${STAMP}"
AUTH_FILE="$HOME/.config/etv-dashboard/auth.env"
TOKEN_FILE="$HOME/.config/etv-dashboard/tunnel.token"

for command in git rsync sha256sum tar; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Required command is unavailable: $command" >&2
    exit 1
  fi
done
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is unavailable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ -n "$(git -C "$PROJECT_ROOT" status --porcelain)" ]]; then
  echo "Commit the repository before creating a failover bundle." >&2
  exit 1
fi
if [[ -e "$BUNDLE_ROOT" ]]; then
  echo "Bundle destination already exists: $BUNDLE_ROOT" >&2
  exit 1
fi

RUNTIME_PATHS=(
  "data/processed"
  "data/interim/theory_elaboration"
  "data/interim/human_validation"
  "data/interim/topic_review_figures"
  "reports/analysis/tables/stage4"
  "reports/analysis/figures/stage4"
)
for relative_path in "${RUNTIME_PATHS[@]}"; do
  if [[ ! -e "$PROJECT_ROOT/$relative_path" ]]; then
    echo "Required runtime path is missing: $relative_path" >&2
    exit 1
  fi
done

mkdir -p "$BUNDLE_ROOT"
git -C "$PROJECT_ROOT" bundle create "$BUNDLE_ROOT/repository.bundle" --all
git -C "$PROJECT_ROOT" rev-parse HEAD > "$BUNDLE_ROOT/CODE_COMMIT"

RUNTIME_STAGE="$(mktemp -d)"
SECRET_STAGE=""
cleanup() {
  rm -rf "$RUNTIME_STAGE"
  if [[ -n "$SECRET_STAGE" ]]; then
    rm -rf "$SECRET_STAGE"
  fi
}
trap cleanup EXIT
mkdir -p "$RUNTIME_STAGE/runtime"

(
  cd "$PROJECT_ROOT"
  rsync -aR --info=progress2 "${RUNTIME_PATHS[@]}" "$RUNTIME_STAGE/runtime/"
)

# Replace copied SQLite files with transactionally consistent online backups.
backup_sqlite() {
  local source_path="$1"
  local target_path="$2"
  [[ -f "$source_path" ]] || return 0
  mkdir -p "$(dirname "$target_path")"
  "$PYTHON_BIN" -c '
import sqlite3
import sys
source, target = sys.argv[1], sys.argv[2]
with sqlite3.connect(source) as source_db, sqlite3.connect(target) as target_db:
    source_db.backup(target_db)
' "$source_path" "$target_path"
}
backup_sqlite \
  "$PROJECT_ROOT/data/interim/human_validation/human_annotations.sqlite3" \
  "$RUNTIME_STAGE/runtime/data/interim/human_validation/human_annotations.sqlite3"
backup_sqlite \
  "$PROJECT_ROOT/data/interim/theory_elaboration/targeted_reading.sqlite3" \
  "$RUNTIME_STAGE/runtime/data/interim/theory_elaboration/targeted_reading.sqlite3"

tar -C "$RUNTIME_STAGE/runtime" -cf "$BUNDLE_ROOT/runtime.tar" .

SECRETS_INCLUDED="no"
if [[ "$INCLUDE_SECRETS" -eq 1 ]]; then
  if ! command -v gpg >/dev/null 2>&1; then
    echo "gpg is required to encrypt host credentials." >&2
    exit 1
  fi
  if [[ ! -s "$AUTH_FILE" || ! -s "$TOKEN_FILE" ]]; then
    echo "Host credential or tunnel-token file is missing." >&2
    exit 1
  fi
  SECRET_STAGE="$(mktemp -d)"
  mkdir -p "$SECRET_STAGE/etv-dashboard"
  install -m 0600 "$AUTH_FILE" "$SECRET_STAGE/etv-dashboard/auth.env"
  install -m 0600 "$TOKEN_FILE" "$SECRET_STAGE/etv-dashboard/tunnel.token"
  if [[ "$INCLUDE_PROJECT_ENV" -eq 1 && -s "$PROJECT_ROOT/.env" ]]; then
    install -m 0600 "$PROJECT_ROOT/.env" "$SECRET_STAGE/project.env"
  fi
  SECRET_ITEMS=(etv-dashboard)
  if [[ -f "$SECRET_STAGE/project.env" ]]; then
    SECRET_ITEMS+=(project.env)
  fi
  tar -C "$SECRET_STAGE" -czf "$SECRET_STAGE/host-secrets.tar.gz" \
    "${SECRET_ITEMS[@]}"
  gpg --symmetric --cipher-algo AES256 \
    --output "$BUNDLE_ROOT/host-secrets.tar.gz.gpg" \
    "$SECRET_STAGE/host-secrets.tar.gz"
  # Windows-mounted removable drives may not implement chmod. The archive is
  # encrypted; apply restrictive permissions where the destination supports it.
  chmod 600 "$BUNDLE_ROOT/host-secrets.tar.gz.gpg" 2>/dev/null || true
  SECRETS_INCLUDED="encrypted"
fi

{
  printf 'created_utc=%s\n' "$STAMP"
  printf 'source_host=%s\n' "$(hostname)"
  printf 'git_branch=%s\n' "$(git -C "$PROJECT_ROOT" branch --show-current)"
  printf 'git_commit=%s\n' "$(cat "$BUNDLE_ROOT/CODE_COMMIT")"
  printf 'secrets=%s\n' "$SECRETS_INCLUDED"
  printf 'runtime_paths=%s\n' "${RUNTIME_PATHS[*]}"
} > "$BUNDLE_ROOT/BUNDLE_INFO"

(
  cd "$BUNDLE_ROOT"
  find . -type f ! -name SHA256SUMS -print0 |
    sort -z |
    xargs -0 sha256sum > SHA256SUMS
)

echo
echo "Desktop failover bundle created:"
echo "  $BUNDLE_ROOT"
du -sh "$BUNDLE_ROOT"
echo "Keep the GPG passphrase separately from the bundle."
