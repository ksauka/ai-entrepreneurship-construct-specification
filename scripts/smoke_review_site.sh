#!/usr/bin/env bash
# Verify the permanent reviewer site without printing credentials.

set -euo pipefail

BASE_URL="${1:-https://aitheoryelaboration.org}"
AUTH_FILE="${ETV_DASHBOARD_AUTH_FILE:-$HOME/.config/etv-dashboard/auth.env}"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT

if [[ ! -s "$AUTH_FILE" ]]; then
  echo "Authentication file is missing: $AUTH_FILE" >&2
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

status_code="$(
  curl -sS -o /dev/null -w '%{http_code}' --max-time 60 "$BASE_URL/"
)"
if [[ "$status_code" != "401" ]]; then
  echo "Expected unauthenticated HTTP 401, received $status_code." >&2
  exit 1
fi
echo "OK       Unauthenticated requests are challenged"

reviewer_auth="$ETV_DASHBOARD_REVIEW_USERNAME:$ETV_DASHBOARD_REVIEW_PASSWORD"
administrator_auth="$ETV_DASHBOARD_USERNAME:$ETV_DASHBOARD_PASSWORD"

reviewer_mode="$TEMP_DIR/reviewer-mode.json"
curl -fsS --max-time 120 -u "$reviewer_auth" \
  "$BASE_URL/api/access-mode" >"$reviewer_mode"
grep -q '"role":"reviewer"' "$reviewer_mode"
grep -q '"read_only":true' "$reviewer_mode"
grep -q '"writes_allowed":false' "$reviewer_mode"
echo "OK       Reviewer account is read-only"

administrator_mode="$TEMP_DIR/administrator-mode.json"
curl -fsS --max-time 120 -u "$administrator_auth" \
  "$BASE_URL/api/access-mode" >"$administrator_mode"
grep -q '"role":"administrator"' "$administrator_mode"
grep -q '"writes_allowed":true' "$administrator_mode"
echo "OK       Administrator account retains write capability"

routes=(
  "/"
  "/composition"
  "/contrasting"
  "/topic-review"
  "/human-annotation"
  "/knowledge-graph"
  "/assistant"
  "/api/health"
  "/api/scopes"
)
for route in "${routes[@]}"; do
  curl -fsS --max-time 180 -u "$reviewer_auth" \
    "$BASE_URL$route" >/dev/null
  printf 'OK       Reviewer route %s\n' "$route"
done

write_status="$(
  curl -sS -o "$TEMP_DIR/write-response.json" -w '%{http_code}' \
    --max-time 60 \
    -u "$reviewer_auth" \
    -H "Content-Type: application/json" \
    -d '{"reviewer_id":"public-smoke-test","paper_id":"nonexistent","review":{}}' \
    "$BASE_URL/api/targeted-reading/save"
)"
if [[ "$write_status" != "403" ]]; then
  echo "Expected reviewer write HTTP 403, received $write_status." >&2
  exit 1
fi
grep -q 'reviewer read-only access' "$TEMP_DIR/write-response.json"
echo "OK       Reviewer write attempt is rejected server-side"

echo "PASS     Permanent review site smoke test"
