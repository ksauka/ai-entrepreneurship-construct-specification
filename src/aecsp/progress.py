"""Dependency-free, flushed progress reporting for long project workflows."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


@dataclass
class ProgressReporter:
    """Print durable progress lines suitable for terminals and redirected logs."""

    label: str
    total: int
    every: int = 1
    width: int = 30
    started: float = field(default_factory=time.time)
    in_place: bool = False
    _last_line_width: int = field(default=0, init=False, repr=False)

    def line(
        self,
        completed: int,
        *,
        detail: str = "",
        failures: int = 0,
    ) -> str:
        fraction = completed / self.total if self.total else 1.0
        fraction = min(1.0, max(0.0, fraction))
        filled = min(self.width, int(self.width * fraction))
        if completed >= self.total or filled == self.width:
            bar = "#" * self.width
        elif completed > 0:
            # Keep early progress visible even when the corpus is much larger
            # than the bar's character resolution.
            bar = "#" * filled + ">" + "-" * (self.width - filled - 1)
        else:
            bar = "-" * self.width
        elapsed = time.time() - self.started
        rate = completed / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - completed) / rate if rate > 0 else None
        suffix = f" | {detail}" if detail else ""
        return (
            f"{self.label} [{bar}] {completed:,}/{self.total:,} ({fraction:6.2%}) | "
            f"elapsed {format_duration(elapsed)} | {rate:.3f}/s | "
            f"ETA {format_duration(remaining)} | failures {failures}{suffix}"
        )

    def update(
        self,
        completed: int,
        *,
        detail: str = "",
        failures: int = 0,
        force: bool = False,
    ) -> None:
        if force or completed == self.total or completed % max(1, self.every) == 0:
            line = self.line(completed, detail=detail, failures=failures)
            if self.in_place and sys.stdout.isatty():
                padded = line.ljust(self._last_line_width)
                self._last_line_width = len(line)
                print(
                    f"\r{padded}",
                    end="\n" if completed == self.total else "",
                    flush=True,
                )
            else:
                print(line, flush=True)
