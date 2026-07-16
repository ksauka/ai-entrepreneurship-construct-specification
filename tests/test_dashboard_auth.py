"""Tests for dashboard HTTP Basic credential validation."""

import base64

from aecsp.api.auth import verify_basic_credentials


def basic_header(username: str, password: str) -> str:
    encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {encoded}"


def test_correct_credentials_are_accepted():
    assert verify_basic_credentials(
        basic_header("supervisor", "long-random-password"),
        "supervisor",
        "long-random-password",
    )


def test_wrong_or_missing_credentials_are_rejected():
    assert not verify_basic_credentials(None, "supervisor", "password")
    assert not verify_basic_credentials("Bearer token", "supervisor", "password")
    assert not verify_basic_credentials(
        basic_header("supervisor", "wrong"), "supervisor", "password"
    )
    assert not verify_basic_credentials(
        basic_header("wrong", "password"), "supervisor", "password"
    )


def test_malformed_basic_header_is_rejected():
    assert not verify_basic_credentials("Basic !!!", "supervisor", "password")
    encoded_without_colon = base64.b64encode(b"supervisor").decode()
    assert not verify_basic_credentials(
        f"Basic {encoded_without_colon}", "supervisor", "password"
    )
