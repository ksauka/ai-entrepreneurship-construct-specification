#!/usr/bin/env bash
# Add a separate generated read-only reviewer account to the private auth file.

set -euo pipefail

CONFIG_DIR="$HOME/.config/etv-dashboard"
AUTH_FILE="$CONFIG_DIR/auth.env"
RECEIPT_FILE="$CONFIG_DIR/reviewer-credentials.txt"
REVIEW_USERNAME="${1:-reviewer}"

if [[ ! "$REVIEW_USERNAME" =~ ^[A-Za-z0-9._-]{3,40}$ ]]; then
  echo "Reviewer username must use 3-40 letters, numbers, dots, underscores, or hyphens." >&2
  exit 1
fi
if [[ ! -s "$AUTH_FILE" ]]; then
  echo "Administrator credential file is missing: $AUTH_FILE" >&2
  exit 1
fi
if ! command -v openssl >/dev/null 2>&1; then
  echo "openssl is required to generate the reviewer password." >&2
  exit 1
fi

administrator_username="$(
  sed -n 's/^ETV_DASHBOARD_USERNAME=//p' "$AUTH_FILE" | tail -n 1
)"
if [[ -z "$administrator_username" ]]; then
  echo "ETV_DASHBOARD_USERNAME is missing from $AUTH_FILE" >&2
  exit 1
fi
if [[ "$administrator_username" == "$REVIEW_USERNAME" ]]; then
  echo "Reviewer and administrator usernames must be different." >&2
  exit 1
fi

review_password="$(openssl rand -hex 24)"
temporary_file="$(mktemp "$CONFIG_DIR/auth.env.XXXXXX")"
trap 'rm -f "$temporary_file"' EXIT

grep -v '^ETV_DASHBOARD_REVIEW_' "$AUTH_FILE" >"$temporary_file"
printf '\nETV_DASHBOARD_REVIEW_USERNAME=%s\n' "$REVIEW_USERNAME" >>"$temporary_file"
printf 'ETV_DASHBOARD_REVIEW_PASSWORD=%s\n' "$review_password" >>"$temporary_file"
install -m 0600 "$temporary_file" "$AUTH_FILE"

{
  printf 'ETV dashboard reviewer credentials\n'
  printf 'Username: %s\n' "$REVIEW_USERNAME"
  printf 'Password: %s\n' "$review_password"
  printf 'Access: read-only\n'
} >"$RECEIPT_FILE"
chmod 600 "$RECEIPT_FILE"

echo "Reviewer credentials configured without changing the administrator account."
echo "The private credential receipt is stored at:"
echo "  $RECEIPT_FILE"
echo "Do not commit, upload, or include the administrator credential in review files."
