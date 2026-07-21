"""Persist qualitative interpretation of the 136 previously read papers.

The targeted-reading layer is separate from blind human reliability coding.
It supports evidence-led interpretation of strong, contrasting, and boundary
cells without altering model codes or fitted analytical populations.
"""

from __future__ import annotations

import json
import hashlib
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd


REVIEWER_PATTERN = re.compile(r"^[A-Za-z0-9._-]{2,40}$")
VALID_STATUSES = {"pending", "reviewed", "revisit"}
VALID_RELATIONS = {"", "supports", "contrasts", "boundary", "context", "not_relevant"}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class TargetedReadingStore:
    """Store one independent qualitative interpretation per reviewer and paper."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.sample_path = (
            self.project_root
            / "data/interim/theory_elaboration/theory_elaboration_matched_papers.csv"
        )
        self.overlap_path = (
            self.project_root
            / "data/interim/theory_elaboration/theory_elaboration_probability_overlap_23.csv"
        )
        self.database_path = (
            self.project_root
            / "data/interim/theory_elaboration/targeted_reading.sqlite3"
        )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS targeted_reviews (
                    reviewer_id TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    context_id TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    evidence_note TEXT NOT NULL,
                    interpretation TEXT NOT NULL,
                    theoretical_implication TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (reviewer_id, paper_id, context_id)
                );
                CREATE TABLE IF NOT EXISTS targeted_review_audit (
                    revision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reviewer_id TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    context_id TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    saved_at TEXT NOT NULL
                );
                """
            )
            review_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(targeted_reviews)"
                ).fetchall()
            }
            if "context_id" not in review_columns:
                connection.executescript(
                    """
                    ALTER TABLE targeted_reviews RENAME TO targeted_reviews_legacy;
                    CREATE TABLE targeted_reviews (
                        reviewer_id TEXT NOT NULL,
                        paper_id TEXT NOT NULL,
                        context_id TEXT NOT NULL,
                        context_json TEXT NOT NULL,
                        status TEXT NOT NULL,
                        relation TEXT NOT NULL,
                        evidence_note TEXT NOT NULL,
                        interpretation TEXT NOT NULL,
                        theoretical_implication TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (reviewer_id, paper_id, context_id)
                    );
                    INSERT INTO targeted_reviews
                        (reviewer_id, paper_id, context_id, context_json, status,
                         relation, evidence_note, interpretation,
                         theoretical_implication, created_at, updated_at)
                    SELECT reviewer_id, paper_id, 'legacy-unconditioned', '{}',
                           status, relation, evidence_note, interpretation,
                           theoretical_implication, created_at, updated_at
                    FROM targeted_reviews_legacy;
                    DROP TABLE targeted_reviews_legacy;
                    """
                )
            audit_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(targeted_review_audit)"
                ).fetchall()
            }
            if "context_id" not in audit_columns:
                connection.execute(
                    "ALTER TABLE targeted_review_audit ADD COLUMN context_id TEXT NOT NULL DEFAULT 'legacy-unconditioned'"
                )
            if "context_json" not in audit_columns:
                connection.execute(
                    "ALTER TABLE targeted_review_audit ADD COLUMN context_json TEXT NOT NULL DEFAULT '{}'"
                )

    def sample(self) -> pd.DataFrame:
        if not self.sample_path.exists() or not self.overlap_path.exists():
            raise FileNotFoundError("The audited targeted-reading files are unavailable")
        frame = pd.read_csv(self.sample_path, dtype=str, keep_default_na=False).fillna("")
        overlap = pd.read_csv(self.overlap_path, dtype=str, keep_default_na=False).fillna("")
        if "paper_id" not in frame or "paper_id" not in overlap:
            raise ValueError("Targeted-reading files must contain paper_id")
        if frame["paper_id"].duplicated().any():
            raise ValueError("The 136-paper targeted-reading set contains duplicate IDs")
        overlap_ids = set(overlap["paper_id"])
        frame["human_validation_overlap"] = frame["paper_id"].isin(overlap_ids)
        return frame

    @staticmethod
    def validate_reviewer_id(value: str) -> str:
        reviewer = str(value).strip()
        if not REVIEWER_PATTERN.fullmatch(reviewer):
            raise ValueError(
                "Reviewer ID must contain 2-40 letters, numbers, dots, underscores, or hyphens"
            )
        return reviewer

    def metadata(self) -> dict[str, Any]:
        frame = self.sample()
        overlap_n = int(frame["human_validation_overlap"].sum())
        return {
            "target_papers": len(frame),
            "human_validation_overlap": overlap_n,
            "remaining_targeted_reading": len(frame) - overlap_n,
            "sets": [
                {"id": "remaining", "label": "Targeted reading outside human IRR", "papers": len(frame) - overlap_n},
                {"id": "human_overlap", "label": "Blind human-validation overlap", "papers": overlap_n},
                {"id": "all", "label": "All previously read papers", "papers": len(frame)},
            ],
            "methodological_boundary": (
                "The 23 human-validation papers remain a distinct blind reliability set. "
                "Do not inspect their model codes before completing independent annotation."
            ),
        }

    @staticmethod
    def canonical_context(context: dict[str, Any]) -> tuple[str, str]:
        """Return a stable ID and serialized record for one analytical pattern."""

        if not isinstance(context, dict):
            raise ValueError("Targeted reading requires an analytical context")
        patterns = context.get("patterns")
        if not isinstance(patterns, list) or not patterns:
            raise ValueError(
                "Select at least one specification pattern before saving an interpretation"
            )
        serialized = json.dumps(context, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20], serialized

    def review_map(
        self, reviewer_id: str, context: dict[str, Any]
    ) -> dict[str, dict]:
        reviewer = self.validate_reviewer_id(reviewer_id)
        context_id, _ = self.canonical_context(context)
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM targeted_reviews WHERE reviewer_id = ? AND context_id = ?",
                (reviewer, context_id),
            ).fetchall()
        return {str(row["paper_id"]): dict(row) for row in rows}

    def save(self, reviewer_id: str, paper_id: str, payload: dict[str, Any]) -> dict:
        reviewer = self.validate_reviewer_id(reviewer_id)
        paper = str(paper_id).strip()
        if paper not in set(self.sample()["paper_id"]):
            raise ValueError("Paper is not in the audited 136-paper set")
        status = str(payload.get("status", "pending")).strip()
        relation = str(payload.get("relation", "")).strip()
        if status not in VALID_STATUSES:
            raise ValueError(f"Unknown targeted-reading status: {status}")
        if relation not in VALID_RELATIONS:
            raise ValueError(f"Unknown theoretical relation: {relation}")
        context_id, context_json = self.canonical_context(payload.get("context", {}))
        values = {
            "status": status,
            "relation": relation,
            "evidence_note": str(payload.get("evidence_note", "")).strip()[:5000],
            "interpretation": str(payload.get("interpretation", "")).strip()[:5000],
            "theoretical_implication": str(payload.get("theoretical_implication", "")).strip()[:5000],
        }
        timestamp = _now()
        with self._lock, sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO targeted_reviews
                    (reviewer_id, paper_id, context_id, context_json, status, relation, evidence_note,
                     interpretation, theoretical_implication, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(reviewer_id, paper_id, context_id) DO UPDATE SET
                    context_json=excluded.context_json,
                    status=excluded.status, relation=excluded.relation,
                    evidence_note=excluded.evidence_note,
                    interpretation=excluded.interpretation,
                    theoretical_implication=excluded.theoretical_implication,
                    updated_at=excluded.updated_at
                """,
                (
                    reviewer,
                    paper,
                    context_id,
                    context_json,
                    values["status"],
                    values["relation"],
                    values["evidence_note"],
                    values["interpretation"],
                    values["theoretical_implication"],
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                "INSERT INTO targeted_review_audit (reviewer_id, paper_id, context_id, context_json, payload_json, saved_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    reviewer,
                    paper,
                    context_id,
                    context_json,
                    json.dumps(values, sort_keys=True),
                    timestamp,
                ),
            )
        return {
            "reviewer_id": reviewer,
            "paper_id": paper,
            "context_id": context_id,
            "context": json.loads(context_json),
            **values,
            "updated_at": timestamp,
        }

    def export(self) -> pd.DataFrame:
        with sqlite3.connect(self.database_path) as connection:
            reviews = pd.read_sql_query("SELECT * FROM targeted_reviews", connection)
        sample = self.sample()
        if reviews.empty:
            return reviews
        return reviews.merge(
            sample[["paper_id", "Title", "Source title", "Year", "human_validation_overlap"]],
            on="paper_id",
            how="left",
            validate="many_to_one",
        )
