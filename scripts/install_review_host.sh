#!/usr/bin/env bash
# Install the authenticated dashboard and, unless in standby mode, its tunnel.

set -euo pipefail

MODE="${1:-active}"
if [[ "$MODE" != "active" && "$MODE" != "--standby" ]]; then
  echo "Usage: bash scripts/install_review_host.sh [--standby]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="$HOME/.config/etv-dashboard"
USER_UNIT_DIR="$HOME/.config/systemd/user"
AUTH_FILE="$CONFIG_DIR/auth.env"
TUNNEL_TOKEN_FILE="$CONFIG_DIR/tunnel.token"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}

require_file "$AUTH_FILE"
require_file "$PROJECT_ROOT/deploy/etv-dashboard.service"
require_file "$PROJECT_ROOT/deploy/etv-dashboard-tunnel-named.service"

if [[ ! -x "$HOME/miniconda3/envs/graphrag/bin/python" ]]; then
  echo "The graphrag Python environment is not available." >&2
  exit 1
fi
if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is not installed." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$AUTH_FILE"
set +a

for variable in \
  ETV_DASHBOARD_USERNAME \
  ETV_DASHBOARD_PASSWORD \
  ETV_DASHBOARD_REVIEW_USERNAME \
  ETV_DASHBOARD_REVIEW_PASSWORD
do
  if [[ -z "${!variable:-}" ]]; then
    echo "$variable is missing from $AUTH_FILE" >&2
    exit 1
  fi
done
if [[ "$ETV_DASHBOARD_USERNAME" == "$ETV_DASHBOARD_REVIEW_USERNAME" ]]; then
  echo "Administrator and reviewer usernames must be different." >&2
  exit 1
fi

chmod 700 "$CONFIG_DIR"
chmod 600 "$AUTH_FILE"
mkdir -p "$USER_UNIT_DIR"
install -m 0644 \
  "$PROJECT_ROOT/deploy/etv-dashboard.service" \
  "$USER_UNIT_DIR/etv-dashboard.service"
install -m 0644 \
  "$PROJECT_ROOT/deploy/etv-dashboard-tunnel-named.service" \
  "$USER_UNIT_DIR/etv-dashboard-tunnel-named.service"

systemctl --user daemon-reload
systemctl --user enable etv-dashboard.service
# Restart even when the service is already active so changed credentials and
# access-mode settings are loaded into the dashboard process.
systemctl --user restart etv-dashboard.service

dashboard_ready=0
for _attempt in $(seq 1 90); do
  if curl -fsS \
    -u "$ETV_DASHBOARD_USERNAME:$ETV_DASHBOARD_PASSWORD" \
    http://127.0.0.1:8321/api/health \
    >/dev/null 2>&1
  then
    dashboard_ready=1
    break
  fi
  sleep 2
done
if [[ "$dashboard_ready" -ne 1 ]]; then
  echo "Dashboard did not become healthy within 180 seconds." >&2
  echo "Inspect it with:" >&2
  echo "  journalctl --user -u etv-dashboard.service -n 100 --no-pager" >&2
  exit 1
fi
echo "Authenticated dashboard is healthy."

if [[ -s "$TUNNEL_TOKEN_FILE" && "$MODE" == "active" ]]; then
  chmod 600 "$TUNNEL_TOKEN_FILE"
  systemctl --user enable etv-dashboard-tunnel-named.service
  systemctl --user restart etv-dashboard-tunnel-named.service
  systemctl --user disable --now etv-dashboard-tunnel-quick.service 2>/dev/null || true
  echo "Permanent named tunnel enabled; Quick Tunnel disabled."
elif [[ -s "$TUNNEL_TOKEN_FILE" ]]; then
  chmod 600 "$TUNNEL_TOKEN_FILE"
  systemctl --user disable --now etv-dashboard-tunnel-named.service 2>/dev/null || true
  echo "Standby host prepared. Dashboard is local; named tunnel is installed but inactive."
else
  echo "Dashboard installed. Named tunnel remains pending until this file exists:"
  echo "  $TUNNEL_TOKEN_FILE"
  echo "The existing Quick Tunnel, if enabled, was left unchanged."
fi
