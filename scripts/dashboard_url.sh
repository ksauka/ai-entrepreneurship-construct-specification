#!/usr/bin/env bash
set -euo pipefail

# Print the Cloudflare Quick Tunnel URL created during the current WSL boot.
# The optional first argument is the number of seconds to wait for startup.

service="etv-dashboard-tunnel-quick.service"
wait_seconds="${1:-30}"

if ! [[ "$wait_seconds" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash scripts/dashboard_url.sh [wait-seconds]" >&2
  exit 2
fi

deadline=$((SECONDS + wait_seconds))
while (( SECONDS <= deadline )); do
  url="$({
    journalctl --user -u "$service" -b --no-pager -o cat 2>/dev/null || true
  } | grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -n 1 || true)"

  if [[ -n "$url" ]]; then
    printf '%s\n' "$url"
    exit 0
  fi

  sleep 1
done

echo "No Quick Tunnel URL was found for the current WSL boot." >&2
echo "Check: systemctl --user status $service" >&2
exit 1
