"""Provide optional HTTP Basic authentication for the dashboard service.

Inputs: an HTTP Authorization header and expected administrator/reviewer
credentials.
Outputs: a constant-time credential-match decision or access role.
"""

from __future__ import annotations

import base64
import binascii
import hmac

ADMIN_ROLE = "administrator"
REVIEWER_ROLE = "reviewer"


def verify_basic_credentials(
    authorization: str | None,
    expected_username: str,
    expected_password: str,
) -> bool:
    """Validate an HTTP Basic header without leaking comparison timing."""

    if not authorization or not authorization.lower().startswith("basic "):
        return False
    try:
        encoded = authorization.split(" ", 1)[1].strip()
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
        username, password = decoded.split(":", 1)
    except (ValueError, UnicodeDecodeError, binascii.Error):
        return False
    return hmac.compare_digest(username, expected_username) and hmac.compare_digest(
        password, expected_password
    )


def resolve_basic_access_role(
    authorization: str | None,
    administrator_username: str,
    administrator_password: str,
    reviewer_username: str = "",
    reviewer_password: str = "",
) -> str | None:
    """Return the authenticated dashboard role without merging credentials."""

    if not authorization or not authorization.lower().startswith("basic "):
        return None
    try:
        encoded = authorization.split(" ", 1)[1].strip()
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
        username, password = decoded.split(":", 1)
    except (ValueError, UnicodeDecodeError, binascii.Error):
        return None
    return resolve_access_role(
        username,
        password,
        administrator_username,
        administrator_password,
        reviewer_username,
        reviewer_password,
    )


def resolve_access_role(
    username: str,
    password: str,
    administrator_username: str,
    administrator_password: str,
    reviewer_username: str = "",
    reviewer_password: str = "",
) -> str | None:
    """Resolve credentials supplied by the application login form."""

    administrator_match = hmac.compare_digest(
        username,
        administrator_username,
    ) and hmac.compare_digest(password, administrator_password)
    if administrator_match:
        return ADMIN_ROLE
    reviewer_match = (
        bool(reviewer_username)
        and bool(reviewer_password)
        and hmac.compare_digest(username, reviewer_username)
        and hmac.compare_digest(password, reviewer_password)
    )
    if reviewer_match:
        return REVIEWER_ROLE
    return None
