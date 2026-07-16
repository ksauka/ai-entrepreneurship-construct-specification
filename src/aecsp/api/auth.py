"""Provide optional HTTP Basic authentication for the dashboard service.

Inputs: an HTTP Authorization header and expected environment credentials.
Outputs: a constant-time credential-match decision.
"""

from __future__ import annotations

import base64
import binascii
import hmac


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
