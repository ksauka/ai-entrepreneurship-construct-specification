"""Named platform views for the ETV_V2 interface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformView:
    """One navigable analytics or graph view."""

    id: str
    label: str
    purpose: str


PLATFORM_VIEWS: tuple[PlatformView, ...] = (
    PlatformView("overview", "Overview", "Corpus-level specification summary."),
    PlatformView("paper", "Paper", "Paper-centred diagnostic graph."),
    PlatformView("topic", "Topic", "Topic-level convergence and divergence."),
    PlatformView("journal", "Journal", "Journal-level AI specification patterns."),
    PlatformView("author", "Author", "Author-level specification consistency."),
    PlatformView("dimension", "Specification Dimension", "Navigate role, type, mechanism, level, process, scope, and clarity."),
    PlatformView("contrast", "Construct Contrast", "Compare papers that share one dimension but differ on another."),
)
