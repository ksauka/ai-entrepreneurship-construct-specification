"""VOSviewer assignment records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VOSClusterAssignment:
    """VOSviewer cluster membership for one publication."""

    publication_id: str
    cluster_id: str
    map_name: str = "full_corpus"
