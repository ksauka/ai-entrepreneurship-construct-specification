#!/usr/bin/env bash
# Restore a checksummed review-host bundle on a Linux desktop.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/restore_review_host.sh BUNDLE [--standby|--activate] [--skip-secrets]

Default: restore files only.
--standby: install and start the local dashboard, but keep the public tunnel off.
--activate: install and start both dashboard and permanent named tunnel.
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

BUNDLE_ROOT="$(cd "$1" && pwd)"
shift
MODE="restore-only"
RESTORE_SECRETS=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --standby) MODE="standby" ;;
    --activate) MODE="active" ;;
    --skip-secrets) RESTORE_SECRETS=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for required in SHA256SUMS CODE_COMMIT repository.bundle; do
  if [[ ! -e "$BUNDLE_ROOT/$required" ]]; then
    echo "Bundle item is missing: $required" >&2
    exit 1
  fi
done
if [[ ! -f "$BUNDLE_ROOT/runtime.tar" && ! -d "$BUNDLE_ROOT/runtime" ]]; then
  echo "Bundle item is missing: runtime.tar (or legacy runtime directory)" >&2
  exit 1
fi

(
  cd "$BUNDLE_ROOT"
  sha256sum -c SHA256SUMS
)

EXPECTED_COMMIT="$(tr -d '[:space:]' < "$BUNDLE_ROOT/CODE_COMMIT")"
CURRENT_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
if [[ "$CURRENT_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  cat >&2 <<EOF
This checkout does not match the bundle.
Expected: $EXPECTED_COMMIT
Current:  $CURRENT_COMMIT

Fetch or clone the recorded commit before restoring runtime data. The offline
Git bundle is available at:
  $BUNDLE_ROOT/repository.bundle
EOF
  exit 1
fi
if [[ -n "$(git -C "$PROJECT_ROOT" status --porcelain)" ]]; then
  echo "The desktop checkout has uncommitted changes; refusing to overwrite data." >&2
  exit 1
fi

if [[ -f "$BUNDLE_ROOT/runtime.tar" ]]; then
  tar -xf "$BUNDLE_ROOT/runtime.tar" -C "$PROJECT_ROOT"
else
  # Compatibility with directory-style bundles created before runtime archives
  # were introduced. Do not copy removable-drive ownership metadata.
  rsync -rlt --no-perms --no-owner --no-group \
    "$BUNDLE_ROOT/runtime/" "$PROJECT_ROOT/"
fi

if [[ "$RESTORE_SECRETS" -eq 1 && -f "$BUNDLE_ROOT/host-secrets.tar.gz.gpg" ]]; then
  SECRET_STAGE="$(mktemp -d)"
  trap 'rm -rf "$SECRET_STAGE"' EXIT
  gpg --decrypt "$BUNDLE_ROOT/host-secrets.tar.gz.gpg" |
    tar -xzf - -C "$SECRET_STAGE"
  mkdir -p "$HOME/.config/etv-dashboard"
  chmod 700 "$HOME/.config/etv-dashboard"
  install -m 0600 \
    "$SECRET_STAGE/etv-dashboard/auth.env" \
    "$HOME/.config/etv-dashboard/auth.env"
  install -m 0600 \
    "$SECRET_STAGE/etv-dashboard/tunnel.token" \
    "$HOME/.config/etv-dashboard/tunnel.token"
  if [[ -f "$SECRET_STAGE/project.env" ]]; then
    install -m 0600 "$SECRET_STAGE/project.env" "$PROJECT_ROOT/.env"
  fi
fi

case "$MODE" in
  standby)
    bash "$PROJECT_ROOT/scripts/install_review_host.sh" --standby
    ;;
  active)
    bash "$PROJECT_ROOT/scripts/install_review_host.sh"
    ;;
esac

echo "Review-host files restored at commit $EXPECTED_COMMIT."
if [[ "$MODE" == "restore-only" ]]; then
  echo "Services were not changed. Use --standby for local testing before cutover."
fi
