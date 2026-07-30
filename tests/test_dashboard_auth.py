"""Tests for dashboard HTTP Basic credential validation."""

import base64

import pytest
from fastapi import HTTPException, Request

from aecsp.api.auth import (
    ADMIN_ROLE,
    REVIEWER_ROLE,
    resolve_access_role,
    resolve_basic_access_role,
    verify_basic_credentials,
)
from aecsp.api.main import (
    DASHBOARD_SESSION_COOKIE,
    HTTP_SECURITY_HEADERS,
    LOGIN_CLIENT_FAILURE_LIMIT,
    LOGIN_FAILURE_WINDOW_SECONDS,
    LOGIN_PAIR_FAILURE_LIMIT,
    _clear_login_failures,
    _create_dashboard_session,
    _dashboard_session_role,
    _is_browser_navigation,
    _login_rate_limit_keys,
    _login_retry_after,
    _record_login_failure,
    _require_dashboard_write,
    _safe_login_destination,
    dashboard_login_failures,
    dashboard_sessions,
    logout_dashboard,
)


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


def test_administrator_and_reviewer_roles_are_resolved_independently():
    assert (
        resolve_basic_access_role(
            basic_header("supervisor", "admin-secret"),
            "supervisor",
            "admin-secret",
            "reviewer",
            "review-secret",
        )
        == ADMIN_ROLE
    )
    assert (
        resolve_basic_access_role(
            basic_header("reviewer", "review-secret"),
            "supervisor",
            "admin-secret",
            "reviewer",
            "review-secret",
        )
        == REVIEWER_ROLE
    )


def test_form_credentials_resolve_the_same_separate_roles():
    assert (
        resolve_access_role(
            "supervisor",
            "admin-secret",
            "supervisor",
            "admin-secret",
            "reviewer",
            "review-secret",
        )
        == ADMIN_ROLE
    )
    assert (
        resolve_access_role(
            "reviewer",
            "review-secret",
            "supervisor",
            "admin-secret",
            "reviewer",
            "review-secret",
        )
        == REVIEWER_ROLE
    )
    assert (
        resolve_access_role(
            "reviewer",
            "wrong",
            "supervisor",
            "admin-secret",
            "reviewer",
            "review-secret",
        )
        is None
    )


def test_access_role_rejects_unknown_or_partial_reviewer_credentials():
    assert (
        resolve_basic_access_role(
            basic_header("unknown", "secret"),
            "supervisor",
            "admin-secret",
            "reviewer",
            "review-secret",
        )
        is None
    )


def _request_with_role(role: str) -> Request:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [],
        }
    )
    request.state.dashboard_access_role = role
    return request


def test_reviewer_role_cannot_pass_the_server_side_write_gate(monkeypatch):
    monkeypatch.setenv("ETV_DASHBOARD_REQUIRE_AUTH", "1")
    with pytest.raises(HTTPException, match="reviewer read-only access") as error:
        _require_dashboard_write(
            _request_with_role(REVIEWER_ROLE),
            "Human-annotation writing",
        )
    assert error.value.status_code == 403


def test_administrator_role_can_pass_the_server_side_write_gate(monkeypatch):
    monkeypatch.setenv("ETV_DASHBOARD_REQUIRE_AUTH", "1")
    _require_dashboard_write(
        _request_with_role(ADMIN_ROLE),
        "Human-annotation writing",
    )
    assert (
        resolve_basic_access_role(
            basic_header("reviewer", "review-secret"),
            "supervisor",
            "admin-secret",
            "reviewer",
            "",
        )
        is None
    )


def test_logout_clears_browser_site_data():
    dashboard_sessions.clear()
    token = _create_dashboard_session(REVIEWER_ROLE)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/logout",
            "headers": [
                (
                    b"cookie",
                    f"{DASHBOARD_SESSION_COOKIE}={token}".encode("ascii"),
                )
            ],
        }
    )
    response = logout_dashboard(request)

    assert response.status_code == 303
    assert response.headers["location"] == "/login?signed_out=1"
    assert response.headers["clear-site-data"] == '"cache", "cookies", "storage"'
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert _dashboard_session_role(token) is None
    assert DASHBOARD_SESSION_COOKIE in response.headers["set-cookie"]


def test_browser_navigation_and_login_destination_are_bounded():
    browser_request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"accept", b"text/html,application/xhtml+xml")],
        }
    )

    assert _is_browser_navigation(browser_request)
    assert _safe_login_destination("/composition?scope=query_3") == (
        "/composition?scope=query_3"
    )
    assert _safe_login_destination("https://example.com") == "/"
    assert _safe_login_destination("//example.com") == "/"
    assert _safe_login_destination("/login?next=/") == "/"


def _login_request(address: str = "203.0.113.7") -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/login",
            "headers": [(b"cf-connecting-ip", address.encode("ascii"))],
            "client": ("127.0.0.1", 12345),
        }
    )


def test_repeated_login_failures_are_rate_limited_without_storing_identity():
    dashboard_login_failures.clear()
    request = _login_request()
    keys = _login_rate_limit_keys(request, "reviewer")

    assert len(keys) == 2
    assert all("reviewer" not in key for key in keys)
    assert _login_retry_after(keys, now=100.0) == 0

    for _ in range(LOGIN_PAIR_FAILURE_LIMIT):
        _record_login_failure(keys, now=100.0)

    assert _login_retry_after(keys, now=100.0) == LOGIN_FAILURE_WINDOW_SECONDS
    assert _login_retry_after(
        _login_rate_limit_keys(request, "different-user"),
        now=100.0,
    ) == 0

    _clear_login_failures(keys)
    assert _login_retry_after(keys, now=100.0) == 0


def test_client_limit_and_window_expiry_are_enforced():
    dashboard_login_failures.clear()
    keys = _login_rate_limit_keys(_login_request())
    assert len(keys) == 1

    for _ in range(LOGIN_CLIENT_FAILURE_LIMIT):
        _record_login_failure(keys, now=200.0)

    assert _login_retry_after(keys, now=200.0) == LOGIN_FAILURE_WINDOW_SECONDS
    assert (
        _login_retry_after(
            keys,
            now=200.0 + LOGIN_FAILURE_WINDOW_SECONDS + 1,
        )
        == 0
    )


def test_security_headers_prevent_sniffing_framing_and_referrer_leakage():
    assert HTTP_SECURITY_HEADERS["X-Content-Type-Options"] == "nosniff"
    assert HTTP_SECURITY_HEADERS["X-Frame-Options"] == "DENY"
    assert HTTP_SECURITY_HEADERS["Referrer-Policy"] == "no-referrer"
    assert "geolocation=()" in HTTP_SECURITY_HEADERS["Permissions-Policy"]
