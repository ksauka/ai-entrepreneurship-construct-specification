#!/usr/bin/env python3
"""Capture authenticated, reproducible screenshots of the local ETV platform.

The script talks directly to a locally cached Chromium instance through the
Chrome DevTools Protocol.  It does not log dashboard credentials or store them
in the browser profile after the temporary capture session is removed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import websocket


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHROME = Path.home() / ".cache/ms-playwright/chromium-1223/chrome-linux64/chrome"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/analysis/figures/platform/raw"


def read_env_value(path: Path, key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}=(.*)$")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(raw_line.strip())
        if not match:
            continue
        value = match.group(1).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value
    raise RuntimeError(f"Missing {key} in {path}")


class CDP:
    def __init__(self, websocket_url: str) -> None:
        self.ws = websocket.create_connection(
            websocket_url,
            timeout=60,
            origin="http://127.0.0.1",
        )
        self.counter = 0

    def close(self) -> None:
        self.ws.close()

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.counter += 1
        call_id = self.counter
        self.ws.send(json.dumps({"id": call_id, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.ws.recv())
            if message.get("id") != call_id:
                continue
            if "error" in message:
                raise RuntimeError(f"CDP {method} failed: {message['error']}")
            return message.get("result", {})

    def evaluate(self, expression: str) -> Any:
        result = self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": True,
                "returnByValue": True,
            },
        )
        return result.get("result", {}).get("value")

    def wait_for(self, expression: str, timeout: float = 120.0) -> None:
        deadline = time.monotonic() + timeout
        last_error: str | None = None
        while time.monotonic() < deadline:
            try:
                if self.evaluate(expression):
                    return
            except Exception as exc:  # page can be between navigations
                last_error = str(exc)
            time.sleep(0.5)
        suffix = f" Last error: {last_error}" if last_error else ""
        raise TimeoutError(f"Timed out waiting for: {expression}.{suffix}")

    def screenshot(self, output: Path) -> None:
        data = self.call(
            "Page.captureScreenshot",
            {
                "format": "png",
                "fromSurface": True,
                "captureBeyondViewport": False,
            },
        )["data"]
        output.write_bytes(base64.b64decode(data))


def wait_for_debug_port(port_file: Path, process: subprocess.Popen[str]) -> int:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Chromium exited before exposing its debugging port")
        if port_file.exists():
            lines = port_file.read_text(encoding="utf-8").splitlines()
            if lines:
                return int(lines[0])
        time.sleep(0.1)
    raise TimeoutError("Chromium did not expose a debugging port")


def new_page(port: int) -> str:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/json/new?about:blank",
        method="PUT",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)["webSocketDebuggerUrl"]


def navigate(cdp: CDP, url: str, ready: str, settle: float = 2.0) -> None:
    cdp.call("Page.navigate", {"url": url})
    cdp.wait_for("document.readyState === 'complete'", timeout=60)
    cdp.wait_for(ready, timeout=180)
    time.sleep(settle)


def sign_in(cdp: CDP, base_url: str, username: str, password: str) -> None:
    """Create the same role-aware session used by an interactive browser."""

    navigate(
        cdp,
        f"{base_url}/login",
        "document.querySelector('form[action=\"/login\"]') !== null",
        settle=0.2,
    )
    submitted = cdp.evaluate(
        f"""
        (() => {{
          const username = document.querySelector('#username');
          const password = document.querySelector('#password');
          const form = document.querySelector('form[action="/login"]');
          if (!username || !password || !form) return false;
          username.value = {json.dumps(username)};
          password.value = {json.dumps(password)};
          form.requestSubmit();
          return true;
        }})()
        """
    )
    if not submitted:
        raise RuntimeError("Could not submit the dashboard login form")
    cdp.wait_for(
        "location.pathname !== '/login' && "
        "document.querySelector('.app-header') !== null",
        timeout=60,
    )


def capture(
    cdp: CDP,
    output_dir: Path,
    base_url: str,
    username: str,
    password: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sign_in(cdp, base_url, username, password)

    navigate(
        cdp,
        f"{base_url}/",
        "document.querySelector('#scope')?.options.length > 2 && "
        "document.querySelector('#perfCards')?.children.length > 0",
        settle=4,
    )
    cdp.evaluate("window.scrollTo(0, 0)")
    cdp.screenshot(output_dir / "analytics_dashboard.png")

    navigate(
        cdp,
        f"{base_url}/composition",
        "document.querySelector('#scope')?.options.length > 2 && "
        "document.querySelector('#codingModel')?.options.length > 1 && "
        "document.querySelectorAll('.composition-panel').length >= 8",
        settle=4,
    )
    cdp.evaluate("window.scrollTo(0, 0)")
    cdp.screenshot(output_dir / "construct_specification.png")

    # Open the first available evidence-bearing bar/cell using the page's own
    # click handler, then retain the current analytical state in the panel.
    opened = cdp.evaluate(
        """
        (() => {
          const selectors = [
            '.composition-category',
            '[data-evidence-key]',
            '.clickable-bar',
            '.matrix-cell.clickable',
            'button[data-category]'
          ];
          for (const selector of selectors) {
            const node = document.querySelector(selector);
            if (node) { node.click(); return selector; }
          }
          return null;
        })()
        """
    )
    if opened:
        try:
            cdp.wait_for(
                "document.querySelector('#panel')?.classList.contains('open') || "
                "document.querySelector('.side-panel')?.classList.contains('open')",
                timeout=60,
            )
            cdp.wait_for(
                "document.querySelectorAll('#panelBody .paper-inspection-card').length > 0 && "
                "!document.querySelector('#panelBody').innerText.includes('Loading evidence papers')",
                timeout=120,
            )
            time.sleep(1)
            cdp.screenshot(output_dir / "construct_evidence_panel.png")
        except TimeoutError:
            pass

    navigate(
        cdp,
        f"{base_url}/contrasting",
        "document.querySelectorAll('select option').length > 8 && "
        "document.body.innerText.includes('Horizontal contrasting')",
        settle=4,
    )
    cdp.evaluate("window.scrollTo(0, 0)")
    cdp.screenshot(output_dir / "construct_contrasting.png")

    navigate(
        cdp,
        f"{base_url}/topic-review",
        "document.querySelectorAll('select option').length > 2 && "
        "document.body.innerText.includes('Topic')",
        settle=4,
    )
    cdp.evaluate("window.scrollTo(0, 0)")
    cdp.screenshot(output_dir / "topic_review.png")

    navigate(
        cdp,
        f"{base_url}/human-annotation",
        "document.body.innerText.includes('Coding instructions') && "
        "document.querySelector('input') !== null",
        settle=3,
    )
    cdp.evaluate("window.scrollTo(0, 0)")
    cdp.screenshot(output_dir / "human_annotation.png")

    navigate(
        cdp,
        f"{base_url}/assistant",
        "document.querySelector('#scope')?.options.length > 2 && "
        "document.querySelectorAll('.q-item').length >= 5",
        settle=3,
    )
    cdp.evaluate("window.scrollTo(0, 0)")
    cdp.screenshot(output_dir / "assistant.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8321")
    parser.add_argument("--chrome", type=Path, default=DEFAULT_CHROME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument(
        "--debug-port",
        type=int,
        help="Connect to an existing temporary Chromium debugging port instead of launching one.",
    )
    args = parser.parse_args()

    username = read_env_value(PROJECT_ROOT / ".env", "USERNAME")
    password = read_env_value(PROJECT_ROOT / ".env", "PASSWORD")
    authorization = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()

    profile = Path(tempfile.mkdtemp(prefix="etv-platform-capture-"))
    process: subprocess.Popen[str] | None = None
    cdp: CDP | None = None
    try:
        if args.debug_port is not None:
            port = args.debug_port
        else:
            process = subprocess.Popen(
                [
                    str(args.chrome),
                    "--headless=new",
                    "--no-sandbox",
                    "--disable-gpu",
                    "--hide-scrollbars",
                    "--remote-debugging-port=0",
                    "--remote-allow-origins=*",
                    f"--user-data-dir={profile}",
                    f"--window-size={args.width},{args.height}",
                    "about:blank",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            port = wait_for_debug_port(profile / "DevToolsActivePort", process)
        cdp = CDP(new_page(port))
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call("Network.enable")
        cdp.call("Network.setExtraHTTPHeaders", {"headers": {"Authorization": authorization}})
        cdp.call(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": args.width,
                "height": args.height,
                "deviceScaleFactor": 1,
                "mobile": False,
            },
        )
        capture(
            cdp,
            args.output_dir,
            args.base_url.rstrip("/"),
            username,
            password,
        )
        print(f"Captured platform screenshots in {args.output_dir}")
    finally:
        if cdp is not None:
            cdp.close()
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
