#!/usr/bin/env bash
# Report deployment readiness without printing credentials or tunnel tokens.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="$HOME/.config/etv-dashboard"
AUTH_FILE="$CONFIG_DIR/auth.env"
TOKEN_FILE="$CONFIG_DIR/tunnel.token"
failures=0

check() {
  local label="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    printf 'OK       %s\n' "$label"
  else
    printf 'PENDING  %s\n' "$label"
    failures=$((failures + 1))
  fi
}

check "Project root" test -d "$PROJECT_ROOT/src/aecsp"
check "Python environment" test -x "$HOME/miniconda3/envs/graphrag/bin/python"
check "Cloudflare connector" command -v cloudflared
check "Administrator/reviewer credential file" test -s "$AUTH_FILE"
check "Named-tunnel token" test -s "$TOKEN_FILE"
check "Dashboard user service installed" test -f "$HOME/.config/systemd/user/etv-dashboard.service"
check "Named-tunnel user service installed" test -f "$HOME/.config/systemd/user/etv-dashboard-tunnel-named.service"
check "Dashboard service active" systemctl --user is-active --quiet etv-dashboard.service
check "Named tunnel active" systemctl --user is-active --quiet etv-dashboard-tunnel-named.service

if [[ -s "$AUTH_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$AUTH_FILE"
  set +a
  if [[ -n "${ETV_DASHBOARD_USERNAME:-}" && -n "${ETV_DASHBOARD_PASSWORD:-}" ]]; then
    check "Authenticated local health endpoint" \
      curl -fsS \
      -u "$ETV_DASHBOARD_USERNAME:$ETV_DASHBOARD_PASSWORD" \
      http://127.0.0.1:8321/api/health
  fi
  if [[ -n "${ETV_DASHBOARD_REVIEW_USERNAME:-}" && -n "${ETV_DASHBOARD_REVIEW_PASSWORD:-}" ]]; then
    check "Reviewer read-only role" \
      bash -c \
      'curl -fsS -u "$1:$2" http://127.0.0.1:8321/api/access-mode | grep -q "\"read_only\":true"' \
      _ "$ETV_DASHBOARD_REVIEW_USERNAME" "$ETV_DASHBOARD_REVIEW_PASSWORD"
  else
    printf 'PENDING  Reviewer credentials in %s\n' "$AUTH_FILE"
    failures=$((failures + 1))
  fi
fi

exit "$failures"
