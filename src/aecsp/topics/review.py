"""Manage traceable researcher review of data-specific topic labels.

Inputs: the 130-row scope-topic review CSV and the Stage 4 manifest.
Outputs: filtered review records, progress summaries, atomic CSV updates, and
an append-only JSONL audit history for every researcher decision.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from threading import Lock

import pandas as pd

ALLOWED_SCOPES = ("full_corpus", "query_1", "query_2", "query_3", "query_4")
ALLOWED_STATUSES = ("pending", "approved", "revise")
EXPECTED_BY_SCOPE = {
    "full_corpus": 53,
    "query_1": 50,
    "query_2": 13,
    "query_3": 6,
    "query_4": 8,
}
FIGURE_NAMES = (
    "topic_prevalence.png",
)
PREVIEW_FIGURE_VERSION = "v5-all-topic-prevalence"
REQUIRED_COLUMNS = {
    "scope",
    "topic_id",
    "automatic_label",
    "top_terms",
    "fitted_papers",
    "final_assigned_papers",
    "approved_label",
    "review_status",
    "reviewer_notes",
}
TOPIC_EVIDENCE_COLUMNS = {
    "full_corpus": {
        "topic_id": "bertopic_topic",
        "probability": "bertopic_topic_prob",
        "was_outlier": "bertopic_was_outlier",
    },
    **{
        f"query_{query}": {
            "topic_id": f"query_{query}_topic_id",
            "probability": f"query_{query}_topic_prob",
            "was_outlier": f"query_{query}_was_outlier",
        }
        for query in range(1, 5)
    },
}


def file_sha256(path: Path) -> str:
    """Return one file's SHA-256 checksum."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TopicReviewStore:
    """Read and update topic-label decisions without changing model outputs."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.review_path = (
            self.project_root
            / "data/processed/analysis/stage4/topic_label_review.csv"
        )
        self.audit_path = (
            self.project_root
            / "data/processed/analysis/stage4/topic_label_review_audit.jsonl"
        )
        self.manifest_path = (
            self.project_root / "data/processed/analysis/stage4/stage4_manifest.json"
        )
        self.figure_root = self.project_root / "reports/analysis/figures/stage4"
        self.preview_figure_root = (
            self.project_root / "data/interim/topic_review_figures"
        )
        self._lock = Lock()
        self._figure_lock = Lock()

    def _load(self) -> pd.DataFrame:
        if not self.review_path.exists():
            raise FileNotFoundError(
                "Topic review has not been prepared. Run "
                "scripts/prepare_topic_label_review.py first."
            )
        review = pd.read_csv(self.review_path, dtype=str, keep_default_na=False)
        missing = REQUIRED_COLUMNS - set(review.columns)
        if missing:
            raise ValueError(f"Topic review is missing columns: {sorted(missing)}")
        if review.duplicated(["scope", "topic_id"]).any():
            raise ValueError("Topic review contains duplicate scope-topic keys")
        invalid_statuses = sorted(
            set(review["review_status"].astype(str)) - set(ALLOWED_STATUSES)
        )
        if invalid_statuses:
            raise ValueError(
                f"Topic review contains invalid statuses: {invalid_statuses}"
            )
        for column in ("last_updated_at", "last_reviewer"):
            if column not in review.columns:
                review[column] = ""
        self._validate_coverage(review)
        return review

    @staticmethod
    def _validate_coverage(review: pd.DataFrame) -> None:
        observed_scopes = set(review["scope"])
        if observed_scopes != set(ALLOWED_SCOPES):
            raise ValueError(
                f"Topic review scopes are incomplete: {sorted(observed_scopes)}"
            )
        for scope, expected in EXPECTED_BY_SCOPE.items():
            scoped = review[review["scope"].eq(scope)]
            ids = set(pd.to_numeric(scoped["topic_id"], errors="raise").astype(int))
            if len(scoped) != expected or ids != set(range(expected)):
                raise ValueError(
                    f"Topic review coverage is invalid for {scope}: "
                    f"expected topics 0-{expected - 1}"
                )

    def summary(self) -> dict[str, object]:
        """Return review progress and whether generated outputs match the review."""

        review = self._load()
        status_counts = review["review_status"].value_counts().to_dict()
        scopes = []
        for scope in ALLOWED_SCOPES:
            scoped = review[review["scope"].eq(scope)]
            scopes.append(
                {
                    "scope": scope,
                    "topics": len(scoped),
                    "approved": int(scoped["review_status"].eq("approved").sum()),
                    "pending": int(scoped["review_status"].eq("pending").sum()),
                    "revise": int(scoped["review_status"].eq("revise").sum()),
                }
            )
        approved = int(status_counts.get("approved", 0))
        complete = approved == len(review) and review["approved_label"].str.strip().ne("").all()
        current_hash = file_sha256(self.review_path)
        manifest_review = {}
        generated_at = ""
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            manifest_review = manifest.get("topic_label_review", {})
            generated_at = str(manifest.get("generated_at", ""))
        applied_hash = str(manifest_review.get("sha256", ""))
        outputs_current = bool(
            complete
            and manifest_review.get("status") == "approved"
            and applied_hash
            and applied_hash == current_hash
        )
        return {
            "total_topics": len(review),
            "approved": approved,
            "pending": int(status_counts.get("pending", 0)),
            "revise": int(status_counts.get("revise", 0)),
            "complete": bool(complete),
            "outputs_current": outputs_current,
            "generated_at": generated_at,
            "review_sha256": current_hash,
            "applied_review_sha256": applied_hash,
            "release_id": (
                f"topics-{current_hash[:12]}"
                if outputs_current
                else f"draft-{current_hash[:12]}"
            ),
            "scopes": scopes,
            "figure_names": list(FIGURE_NAMES),
        }

    def records(
        self,
        scope: str = "full_corpus",
        status: str = "all",
        query: str = "",
    ) -> list[dict[str, object]]:
        """Return filtered evidence rows for the researcher interface."""

        if scope not in ALLOWED_SCOPES:
            raise ValueError(f"Unknown topic scope: {scope}")
        if status not in ("all", *ALLOWED_STATUSES):
            raise ValueError(f"Unknown review status: {status}")
        review = self._load()
        review = review[review["scope"].eq(scope)]
        if status != "all":
            review = review[review["review_status"].eq(status)]
        needle = query.strip().casefold()
        if needle:
            searchable = (
                review["topic_id"].astype(str)
                + " "
                + review["automatic_label"]
                + " "
                + review["approved_label"]
                + " "
                + review["top_terms"]
            ).str.casefold()
            review = review[searchable.str.contains(needle, regex=False)]
        review = review.assign(
            topic_id=pd.to_numeric(review["topic_id"], errors="raise").astype(int),
            fitted_papers=pd.to_numeric(
                review["fitted_papers"], errors="coerce"
            ).fillna(0).astype(int),
            final_assigned_papers=pd.to_numeric(
                review["final_assigned_papers"], errors="coerce"
            ).fillna(0).astype(int),
        )
        review["topic_uid"] = (
            review["scope"].astype(str) + ":" + review["topic_id"].astype(str)
        )
        review["display_label"] = review["approved_label"].str.strip().where(
            review["approved_label"].str.strip().ne(""),
            review["automatic_label"],
        )
        review = review.sort_values("topic_id")
        return review.to_dict("records")

    def fitted_papers(
        self,
        scope: str,
        topic_id: int,
        *,
        limit: int = 100,
    ) -> dict[str, object]:
        """Return papers originally fitted to one stable scope-topic ID."""

        if scope not in ALLOWED_SCOPES:
            raise ValueError(f"Unknown topic scope: {scope}")
        if topic_id not in range(EXPECTED_BY_SCOPE[scope]):
            raise ValueError(f"Unknown topic ID for {scope}: {topic_id}")
        if not 1 <= int(limit) <= 50_000:
            raise ValueError("Fitted-paper limit must be between 1 and 50,000")

        enriched_path = (
            self.project_root
            / "data/processed/analysis/primary_analysis_dataset_with_topics.csv"
        )
        if not enriched_path.exists():
            raise FileNotFoundError(
                "The topic-enriched dataset is unavailable. Run "
                "scripts/build_stage4_analysis.py first."
            )
        config = TOPIC_EVIDENCE_COLUMNS[scope]
        identity_columns = [
            "paper_id",
            "Title",
            "Authors",
            "Year",
            "Source title",
            "Cited by",
            "DOI",
            "Link",
        ]
        model_columns = [
            str(config["topic_id"]),
            str(config["probability"]),
            str(config["was_outlier"]),
        ]
        frame = pd.read_csv(
            enriched_path,
            dtype=str,
            keep_default_na=False,
            usecols=identity_columns + model_columns,
        )
        assigned_topic = pd.to_numeric(
            frame[str(config["topic_id"])], errors="coerce"
        )
        fitted = frame[str(config["was_outlier"])].astype(str).str.casefold().isin(
            {"false", "0", "no", "n"}
        )
        papers = frame[assigned_topic.eq(topic_id) & fitted].copy()
        papers["topic_probability"] = pd.to_numeric(
            papers[str(config["probability"])], errors="coerce"
        )
        papers["_citations"] = pd.to_numeric(
            papers["Cited by"], errors="coerce"
        ).fillna(0)
        papers = papers.sort_values(
            ["topic_probability", "_citations", "Title"],
            ascending=[False, False, True],
            na_position="last",
        )
        total = len(papers)
        output_columns = identity_columns + ["topic_probability"]
        records = papers.head(limit)[output_columns].fillna("").to_dict("records")
        for record in records:
            probability = record.get("topic_probability")
            record["topic_probability"] = (
                float(probability) if probability not in (None, "") else None
            )
        return {
            "scope": scope,
            "topic_id": topic_id,
            "total": total,
            "returned": len(records),
            "limit": int(limit),
            "papers": records,
        }

    def graph_preview(self, scope: str) -> dict[str, object]:
        """Return a topic-centred preview using the latest saved draft labels."""

        topics = self.records(scope=scope)
        nodes: list[dict[str, object]] = []
        edges: list[dict[str, object]] = []
        paper_ids: set[str] = set()
        for topic in topics:
            topic_uid = str(topic["topic_uid"])
            topic_node_id = f"Topic::{topic_uid}"
            nodes.append(
                {
                    "id": topic_node_id,
                    "caption": str(topic["display_label"]),
                    "nodeType": "Topic",
                    "degree": int(topic["final_assigned_papers"]),
                    "properties": {
                        "uid": topic_uid,
                        "scope": scope,
                        "topic_id": int(topic["topic_id"]),
                        "automatic_label": str(topic["automatic_label"]),
                        "approved_label": str(topic["approved_label"]),
                        "display_label": str(topic["display_label"]),
                        "review_status": str(topic["review_status"]),
                        "papers": int(topic["final_assigned_papers"]),
                    },
                }
            )
            for rank in (1, 2, 3):
                paper_id = str(topic.get(f"representative_{rank}_paper_id", "")).strip()
                title = str(topic.get(f"representative_{rank}_title", "")).strip()
                if not paper_id:
                    continue
                paper_node_id = f"Publication::{paper_id}"
                if paper_id not in paper_ids:
                    paper_ids.add(paper_id)
                    nodes.append(
                        {
                            "id": paper_node_id,
                            "caption": title or paper_id,
                            "nodeType": "Publication",
                            "degree": 1,
                            "properties": {
                                "id": paper_id,
                                "Title": title,
                                "representative_scope": scope,
                            },
                        }
                    )
                edges.append(
                    {
                        "id": f"{paper_node_id}::HAS_TOPIC::{topic_node_id}",
                        "from": paper_node_id,
                        "to": topic_node_id,
                        "type": "HAS_TOPIC",
                        "properties": {
                            "scope": scope,
                            "representative_rank": rank,
                        },
                    }
                )
        return {
            "scope": scope,
            "status": "draft",
            "nodes": nodes,
            "edges": edges,
            "topic_count": len(topics),
            "representative_paper_count": len(paper_ids),
        }

    def update(
        self,
        scope: str,
        topic_id: int,
        *,
        approved_label: str,
        review_status: str,
        reviewer_notes: str,
        reviewer: str,
    ) -> dict[str, object]:
        """Atomically save one decision and append its before/after audit record."""

        if scope not in ALLOWED_SCOPES:
            raise ValueError(f"Unknown topic scope: {scope}")
        if review_status not in ALLOWED_STATUSES:
            raise ValueError(f"Unknown review status: {review_status}")
        label = approved_label.strip()
        notes = reviewer_notes.strip()
        reviewer_name = reviewer.strip()
        if review_status == "approved" and not label:
            raise ValueError("An approved topic must have a non-empty label")
        if not reviewer_name:
            raise ValueError("Reviewer name is required for the audit trail")

        with self._lock:
            review = self._load()
            numeric_ids = pd.to_numeric(review["topic_id"], errors="raise").astype(int)
            mask = review["scope"].eq(scope) & numeric_ids.eq(int(topic_id))
            if int(mask.sum()) != 1:
                raise KeyError(f"Unknown scope-topic key: {scope}, {topic_id}")
            index = review.index[mask][0]
            before = {
                "approved_label": review.at[index, "approved_label"],
                "review_status": review.at[index, "review_status"],
                "reviewer_notes": review.at[index, "reviewer_notes"],
                "last_updated_at": review.at[index, "last_updated_at"],
                "last_reviewer": review.at[index, "last_reviewer"],
            }
            updated_at = datetime.now().astimezone().isoformat(timespec="seconds")
            review.at[index, "approved_label"] = label
            review.at[index, "review_status"] = review_status
            review.at[index, "reviewer_notes"] = notes
            review.at[index, "last_updated_at"] = updated_at
            review.at[index, "last_reviewer"] = reviewer_name
            self._atomic_write(review)
            after = {
                "approved_label": label,
                "review_status": review_status,
                "reviewer_notes": notes,
                "last_updated_at": updated_at,
                "last_reviewer": reviewer_name,
            }
            self._append_audit(
                {
                    "timestamp": updated_at,
                    "scope": scope,
                    "topic_id": int(topic_id),
                    "reviewer": reviewer_name,
                    "before": before,
                    "after": after,
                }
            )
            row = review.loc[index].to_dict()
            row["topic_id"] = int(topic_id)
            row["fitted_papers"] = int(float(row.get("fitted_papers") or 0))
            row["final_assigned_papers"] = int(
                float(row.get("final_assigned_papers") or 0)
            )
            return row

    def _atomic_write(self, review: pd.DataFrame) -> None:
        self.review_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.review_path.with_suffix(".csv.tmp")
        review.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, self.review_path)

    def _append_audit(self, record: dict[str, object]) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def figure_path(self, scope: str, figure_name: str) -> Path:
        """Return a whitelisted Stage 4 figure for one topic model."""

        if scope not in ALLOWED_SCOPES:
            raise ValueError(f"Unknown topic scope: {scope}")
        if figure_name not in FIGURE_NAMES:
            raise ValueError(f"Unknown topic figure: {figure_name}")
        path = self.figure_root / scope / figure_name
        if not path.exists():
            raise FileNotFoundError(f"Topic figure is unavailable: {scope}/{figure_name}")
        return path

    def preview_figure_path(self, scope: str, figure_name: str) -> Path:
        """Return a cached scope figure using the latest saved draft labels."""

        if scope not in ALLOWED_SCOPES:
            raise ValueError(f"Unknown topic scope: {scope}")
        if figure_name not in FIGURE_NAMES:
            raise ValueError(f"Unknown topic figure: {figure_name}")
        review_hash = file_sha256(self.review_path)
        output_dir = (
            self.preview_figure_root
            / PREVIEW_FIGURE_VERSION
            / review_hash
            / scope
        )
        path = output_dir / figure_name
        if path.exists():
            return path
        with self._figure_lock:
            if not path.exists():
                self._render_preview_figures(scope, output_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"Draft topic figure is unavailable: {scope}/{figure_name}"
            )
        return path

    def _render_preview_figures(self, scope: str, output_dir: Path) -> None:
        """Render all four figures for one scope without changing published files."""

        from scripts.build_stage4_analysis import (
            SCOPE_CONFIG,
            _scope_title,
            plot_topic_prevalence,
            scope_frame,
            topic_prevalence,
        )

        enriched_path = (
            self.project_root
            / "data/processed/analysis/primary_analysis_dataset_with_topics.csv"
        )
        if not enriched_path.exists():
            raise FileNotFoundError(
                "The topic-enriched dataset is unavailable. Run "
                "scripts/build_stage4_analysis.py first."
        )
        frame = pd.read_csv(enriched_path, dtype=str, keep_default_na=False)
        config = SCOPE_CONFIG[scope]
        topic_id_column = str(config["topic_id"])
        topic_label_column = str(config["topic_label"])
        review = pd.DataFrame(self.records(scope=scope))
        labels = review.set_index("topic_id")["display_label"].to_dict()
        topic_ids = pd.to_numeric(frame[topic_id_column], errors="coerce")
        draft_labels = topic_ids.map(labels)
        frame[topic_label_column] = draft_labels.fillna(frame[topic_label_column])

        scoped = scope_frame(frame, scope)
        prevalence = topic_prevalence(scoped)
        title = _scope_title(scoped)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_topic_prevalence(
            prevalence,
            output_dir / "topic_prevalence.png",
            title,
        )
