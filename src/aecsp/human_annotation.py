"""Manage blind, multi-human annotation and human-anchored reliability.

Inputs are the audited workbook/probability overlap, the frozen specification
schema, human annotation decisions, and available model specification files.
Outputs are resumable annotation records, progress summaries, and reliability
statistics calculated on exact common paper-ID intersections.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime
from itertools import combinations
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd

from aecsp.analytics.agreement import (
    krippendorff_alpha_nominal,
    pairwise_percent_agreement,
)
from aecsp.specification.analysis_columns import enrich_for_analysis
from aecsp.specification.llm_coder import (
    PROTOCOL_ID,
    SYSTEM_PROMPT,
    build_user_prompt,
    cache_key,
    protocol_fingerprint,
    response_json_schema,
)
from aecsp.specification.schema import (
    AI_STUDY_STATUS_FIELD,
    EVIDENCE_TYPES,
    SPECIFICATION_DIMENSIONS,
)

ANNOTATOR_PATTERN = re.compile(r"^[A-Za-z0-9._-]{2,40}$")
CORE_COLUMNS = (
    "ai_method_or_phenomenon",
    "ai_type_form",
    "ai_role_function",
    "ai_mechanism_analysis",
    "level_of_analysis",
    "scope_conditions",
)
EXPLORATORY_COLUMNS = (
    "entrepreneurial_process_stage",
    "definition_construct_clarity",
)
DISPLAY_COLUMNS = (*CORE_COLUMNS, *EXPLORATORY_COLUMNS)
MODEL_LABELS = {
    "gpt-5.4-mini-2026-03-17": "GPT-5.4 Mini",
    "gpt-4.1-nano-2025-04-14": "GPT-4.1 Nano",
    "claude-sonnet-5": "Claude Sonnet 5",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
    "llama3.2": "Llama 3.2",
    "gemma4:31b": "Gemma 4",
}
VALIDATION_CACHE_MODELS = {
    "llama3.2": "llama3.2",
}

DIMENSION_REFERENCES: dict[str, tuple[dict[str, str], ...]] = {
    "ai_type_form": (
        {
            "label": "Obschonka et al.",
            "citation": (
                "Obschonka et al. (2025). Artificial Intelligence and "
                "Entrepreneurship: A Call for Research to Prospect and "
                "Establish the Scholarly AI Frontiers."
            ),
            "url": "https://doi.org/10.1177/10422587241304676",
        },
    ),
    "ai_mechanism": (
        {
            "label": "Wiklund",
            "citation": (
                "Wiklund (2026). ETP at 50: The Past, Present, and Future "
                "of Entrepreneurship Research."
            ),
            "url": "https://doi.org/10.1177/10422587261441596",
        },
        {
            "label": "Maula et al.",
            "citation": (
                "Maula et al. (2026). Investing in Data Quality for "
                "High-Impact Entrepreneurship Research."
            ),
            "url": "https://doi.org/10.1177/10422587261435916",
        },
    ),
    "level_of_analysis": (
        {
            "label": "Fisher & Aguinis",
            "citation": (
                "Fisher and Aguinis (2017). Using Theory Elaboration to "
                "Make Theoretical Advancements."
            ),
            "url": "https://doi.org/10.1177/1094428116689707",
        },
    ),
    "entrepreneurial_process_stage": (
        {
            "label": "Burnell et al.",
            "citation": (
                "Burnell et al. (2026). Entrepreneurial Experimentation: "
                "Conceptual Foundations, Integrative Theoretical Framework, "
                "and Research Agenda."
            ),
            "url": "https://doi.org/10.1177/10422587251347046",
        },
        {
            "label": "Shepherd & Suddaby",
            "citation": (
                "Shepherd and Suddaby (2017). Theory Building: A Review "
                "and Integration."
            ),
            "url": "https://doi.org/10.1177/0149206316647102",
        },
    ),
    "scope_conditions": (
        {
            "label": "Suddaby",
            "citation": (
                "Suddaby (2010). Editor's Comments: Construct Clarity in "
                "Theories of Management and Organization."
            ),
            "url": "https://doi.org/10.5465/amr.35.3.zok346",
        },
    ),
    "definition_construct_clarity": (
        {
            "label": "Suddaby",
            "citation": (
                "Suddaby (2010). Editor's Comments: Construct Clarity in "
                "Theories of Management and Organization."
            ),
            "url": "https://doi.org/10.5465/amr.35.3.zok346",
        },
    ),
}


def _dimension_contract() -> tuple[dict[str, Any], ...]:
    """Return the eight human fields in the displayed IRR order."""

    by_column = {
        dimension.column: dimension for dimension in SPECIFICATION_DIMENSIONS
    }
    ordered = (
        AI_STUDY_STATUS_FIELD,
        by_column["ai_type_form"],
        by_column["ai_role_function"],
        by_column["ai_mechanism"],
        by_column["level_of_analysis"],
        by_column["scope_conditions"],
        by_column["entrepreneurial_process_stage"],
        by_column["definition_construct_clarity"],
    )
    core = {
        "ai_method_or_phenomenon",
        "ai_type_form",
        "ai_role_function",
        "ai_mechanism",
        "level_of_analysis",
        "scope_conditions",
    }
    return tuple(
        {
            "id": dimension.id,
            "label": dimension.label,
            "column": dimension.column,
            "analysis_column": (
                "ai_mechanism_analysis"
                if dimension.column == "ai_mechanism"
                else dimension.column
            ),
            "question": dimension.question,
            "diagnosis": dimension.diagnosis,
            "references": list(
                DIMENSION_REFERENCES.get(dimension.column, ())
            ),
            "allowed_values": list(dimension.allowed_values),
            "classification": (
                "Core" if dimension.column in core else "Exploratory"
            ),
        }
        for dimension in ordered
    )


HUMAN_DIMENSIONS = _dimension_contract()
DIMENSION_BY_COLUMN = {item["column"]: item for item in HUMAN_DIMENSIONS}
LABEL_BY_ANALYSIS_COLUMN = {
    item["analysis_column"]: item["label"] for item in HUMAN_DIMENSIONS
}


def _utc_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HumanAnnotationStore:
    """Persist independent human ratings and calculate balanced comparisons."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.sample_path = (
            self.project_root
            / "data/interim/theory_elaboration/"
            "theory_elaboration_probability_overlap_23.csv"
        )
        self.database_path = (
            self.project_root
            / "data/interim/human_validation/human_annotations.sqlite3"
        )
        self.specification_dir = (
            self.project_root / "data/processed/specification"
        )
        self._lock = Lock()
        self._model_cache: dict[str, tuple[int, pd.DataFrame]] = {}
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS annotations (
                    annotator_id TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    is_complete INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (annotator_id, paper_id)
                );
                CREATE TABLE IF NOT EXISTS annotation_audit (
                    revision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    annotator_id TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    is_complete INTEGER NOT NULL,
                    saved_at TEXT NOT NULL
                );
                """
            )

    def _sample(self) -> pd.DataFrame:
        if not self.sample_path.exists():
            raise FileNotFoundError(
                "The audited workbook/probability overlap is unavailable."
            )
        sample = pd.read_csv(
            self.sample_path, dtype=str, keep_default_na=False
        ).fillna("")
        required = {
            "paper_id",
            "Title",
            "Abstract",
            "Author Keywords",
            "Source title",
            "Year",
        }
        missing = required - set(sample.columns)
        if missing:
            raise ValueError(
                f"Human-annotation sample is missing columns: {sorted(missing)}"
            )
        if sample["paper_id"].duplicated().any():
            raise ValueError("Human-annotation sample contains duplicate paper IDs")
        return sample.reset_index(drop=True)

    @staticmethod
    def validate_annotator_id(annotator_id: str) -> str:
        value = str(annotator_id).strip()
        if not ANNOTATOR_PATTERN.fullmatch(value):
            raise ValueError(
                "Annotator ID must contain 2-40 letters, numbers, dots, "
                "underscores, or hyphens."
            )
        return value

    def instrument(self) -> dict[str, Any]:
        sample = self._sample()
        return {
            "instrument": "spec-v3-human",
            "model_protocol_id": PROTOCOL_ID,
            "model_protocol_label": "System prompt V3",
            "model_protocol_fingerprint": protocol_fingerprint(),
            "sample_id": "workbook_probability_overlap",
            "sample_label": "23-paper probability-sample overlap",
            "sample_papers": len(sample),
            "sample_sha256": _sha256(self.sample_path),
            "evidence_boundary": "Title, abstract, and author keywords only",
            "dimensions": list(HUMAN_DIMENSIONS),
            "evidence_types": list(EVIDENCE_TYPES),
            "instructions": [
                "Code independently without consulting any model output.",
                "Use only the displayed title, abstract, and author keywords.",
                "Record evidence before choosing each code.",
                "Mark evidence as stated, inferred, or absent.",
                "Use an unspecified or missing code when the text is silent.",
                (
                    "Choose a substantive mechanism only when the paper's "
                    "causal logic can be stated in its own terms."
                ),
                (
                    "Flag needs full text only when the displayed evidence is "
                    "genuinely insufficient."
                ),
            ],
            "completion_requirements": [
                (
                    "Complete all eight dimensions using only one permitted "
                    "code per dimension."
                ),
                (
                    "For stated or inferred evidence, enter a short quote or "
                    "close paraphrase before selecting the code."
                ),
                (
                    "When the text is silent, choose the relevant unspecified "
                    "or missing code, select absent, and leave evidence empty "
                    "or briefly state that it is not reported."
                ),
                (
                    "Use confidence of 0.90 or above for explicit support, "
                    "0.60-0.80 for a strong inference, and below 0.60 for a "
                    "weak inference requiring review."
                ),
                (
                    "Before completing the paper, challenge the three most "
                    "important codes and revise any that rely only on keyword "
                    "pattern matching."
                ),
            ],
            "full_model_prompt": SYSTEM_PROMPT,
            "full_model_user_prompt": build_user_prompt(
                "{title}",
                "{abstract}",
                "{author_keywords}",
                "{source_journal}",
                "{publication_year}",
            ),
            "full_model_response_schema": json.dumps(
                response_json_schema(), indent=2, ensure_ascii=False
            ),
            "full_prompt_note": (
                "This section reproduces the complete model coding request: "
                "the verbatim system instruction, the per-paper user-message "
                "template, and the required structured response schema. The "
                "system instruction defines seven evidence-bearing construct "
                "dimensions. The response schema separately requires Study "
                "status as the eighth comparability field. Model-only "
                "auxiliary outputs are not part of human IRR."
            ),
            "study_status_schema_note": (
                "AI positioning is the required response field that asks whether "
                "AI is the phenomenon being studied, a research method used by "
                "the authors, both, or unclear. Unlike the seven "
                "evidence-bearing dimensions, the model response does not attach "
                "a separate quotation, evidence-type label, or confidence value "
                "to AI positioning."
            ),
        }

    def _order(self, sample: pd.DataFrame) -> list[str]:
        """Return one reproducible blinded order shared by all annotators.

        A common order maximises the exact overlap when annotators stop before
        completing the full sample. The order is still independent of model
        output and is fixed by the audited sample fingerprint.
        """

        sample_hash = _sha256(self.sample_path)
        return sorted(
            sample["paper_id"].tolist(),
            key=lambda paper_id: hashlib.sha256(
                f"{sample_hash}|human-annotation-order|{paper_id}".encode()
            ).hexdigest(),
        )

    def _states(self, annotator_id: str) -> dict[str, dict[str, Any]]:
        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT paper_id, payload_json, is_complete, created_at, updated_at
                FROM annotations WHERE annotator_id = ?
                """,
                (annotator_id,),
            ).fetchall()
        return {
            row[0]: {
                "payload": json.loads(row[1]),
                "is_complete": bool(row[2]),
                "created_at": row[3],
                "updated_at": row[4],
            }
            for row in rows
        }

    def paper(
        self, annotator_id: str, paper_id: str | None = None
    ) -> dict[str, Any]:
        """Return one blinded paper and resumable annotator state."""

        annotator_id = self.validate_annotator_id(annotator_id)
        sample = self._sample()
        order = self._order(sample)
        states = self._states(annotator_id)
        if paper_id is None:
            paper_id = next(
                (
                    candidate
                    for candidate in order
                    if not states.get(candidate, {}).get("is_complete")
                ),
                order[0],
            )
        if paper_id not in set(order):
            raise KeyError(f"Unknown human-annotation paper: {paper_id}")
        row = sample.set_index("paper_id").loc[paper_id]
        navigation = [
            {
                "paper_id": candidate,
                "position": index + 1,
                "complete": bool(
                    states.get(candidate, {}).get("is_complete", False)
                ),
            }
            for index, candidate in enumerate(order)
        ]
        state = states.get(paper_id, {})
        return {
            "annotator_id": annotator_id,
            "paper": {
                "paper_id": paper_id,
                "Title": row["Title"],
                "Abstract": row["Abstract"],
                "Author Keywords": row["Author Keywords"],
                "Source title": row["Source title"],
                "Year": row["Year"],
            },
            "position": order.index(paper_id) + 1,
            "total": len(order),
            "annotation": state.get("payload", {}),
            "is_complete": bool(state.get("is_complete", False)),
            "navigation": navigation,
            "progress": self.progress(annotator_id),
        }

    def _validate_payload(
        self, payload: dict[str, Any], *, require_complete: bool
    ) -> tuple[dict[str, Any], bool]:
        dimensions = payload.get("dimensions", {})
        if not isinstance(dimensions, dict):
            raise ValueError("Annotation dimensions must be an object")
        cleaned_dimensions: dict[str, dict[str, Any]] = {}
        complete = True
        for contract in HUMAN_DIMENSIONS:
            column = contract["column"]
            entry = dimensions.get(column, {})
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid annotation entry for {column}")
            code = str(entry.get("code", "")).strip()
            evidence = str(entry.get("evidence", "")).strip()
            evidence_type = str(entry.get("evidence_type", "")).strip()
            confidence_value = entry.get("confidence")
            confidence = None
            if confidence_value not in (None, ""):
                try:
                    confidence = float(confidence_value)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Confidence for {column} must be numeric"
                    ) from error
                if not 0 <= confidence <= 1:
                    raise ValueError(
                        f"Confidence for {column} must be between 0 and 1"
                    )
            if code and code not in contract["allowed_values"]:
                raise ValueError(f"Unknown code for {column}: {code}")
            if evidence_type and evidence_type not in EVIDENCE_TYPES:
                raise ValueError(
                    f"Unknown evidence type for {column}: {evidence_type}"
                )
            evidence_complete = bool(evidence) or evidence_type == "absent"
            entry_complete = bool(
                code
                and evidence_type
                and evidence_complete
                and confidence is not None
            )
            complete = complete and entry_complete
            cleaned_dimensions[column] = {
                "code": code,
                "evidence": evidence,
                "evidence_type": evidence_type,
                "confidence": confidence,
            }
        mechanism_logic = str(payload.get("ai_mechanism_logic", "")).strip()
        mechanism_code = cleaned_dimensions["ai_mechanism"]["code"]
        if mechanism_code and mechanism_code != "mechanism missing":
            complete = complete and bool(mechanism_logic)
        needs_full_text = payload.get("needs_full_text", [])
        if not isinstance(needs_full_text, list):
            raise ValueError("needs_full_text must be a list")
        invalid_full_text = sorted(
            set(map(str, needs_full_text)) - set(DIMENSION_BY_COLUMN)
        )
        if invalid_full_text:
            raise ValueError(
                f"Unknown needs-full-text dimensions: {invalid_full_text}"
            )
        cleaned = {
            "dimensions": cleaned_dimensions,
            "ai_mechanism_logic": mechanism_logic,
            "needs_full_text": sorted(set(map(str, needs_full_text))),
            "annotator_notes": str(payload.get("annotator_notes", "")).strip(),
        }
        if require_complete and not complete:
            raise ValueError(
                "All eight dimensions, evidence types, and confidence values "
                "must be completed; substantive mechanisms also require logic."
            )
        return cleaned, complete

    def save(
        self,
        annotator_id: str,
        paper_id: str,
        payload: dict[str, Any],
        *,
        submit: bool,
    ) -> dict[str, Any]:
        annotator_id = self.validate_annotator_id(annotator_id)
        sample_ids = set(self._sample()["paper_id"])
        if paper_id not in sample_ids:
            raise KeyError(f"Unknown human-annotation paper: {paper_id}")
        cleaned, complete = self._validate_payload(
            payload, require_complete=submit
        )
        saved_complete = bool(submit and complete)
        encoded = json.dumps(cleaned, sort_keys=True)
        timestamp = _utc_now()
        with self._lock, sqlite3.connect(self.database_path) as connection:
            existing = connection.execute(
                """
                SELECT created_at FROM annotations
                WHERE annotator_id = ? AND paper_id = ?
                """,
                (annotator_id, paper_id),
            ).fetchone()
            created_at = existing[0] if existing else timestamp
            connection.execute(
                """
                INSERT INTO annotations
                    (annotator_id, paper_id, payload_json, is_complete,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(annotator_id, paper_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    is_complete = excluded.is_complete,
                    updated_at = excluded.updated_at
                """,
                (
                    annotator_id,
                    paper_id,
                    encoded,
                    int(saved_complete),
                    created_at,
                    timestamp,
                ),
            )
            connection.execute(
                """
                INSERT INTO annotation_audit
                    (annotator_id, paper_id, payload_json, is_complete, saved_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    annotator_id,
                    paper_id,
                    encoded,
                    int(saved_complete),
                    timestamp,
                ),
            )
        return {
            "annotator_id": annotator_id,
            "paper_id": paper_id,
            "is_complete": saved_complete,
            "updated_at": timestamp,
            "progress": self.progress(annotator_id),
        }

    def progress(self, annotator_id: str | None = None) -> dict[str, Any]:
        sample_n = len(self._sample())
        with sqlite3.connect(self.database_path) as connection:
            parameters: tuple[str, ...] = ()
            where = ""
            if annotator_id is not None:
                annotator_id = self.validate_annotator_id(annotator_id)
                where = "WHERE annotator_id = ?"
                parameters = (annotator_id,)
            rows = connection.execute(
                f"""
                SELECT annotator_id, COUNT(*), SUM(is_complete), MAX(updated_at)
                FROM annotations {where}
                GROUP BY annotator_id ORDER BY annotator_id
                """,
                parameters,
            ).fetchall()
        annotators = [
            {
                "annotator_id": row[0],
                "started_papers": int(row[1] or 0),
                "completed_papers": int(row[2] or 0),
                "target_papers": sample_n,
                "completion_share": (
                    round(int(row[2] or 0) / sample_n, 6) if sample_n else 0.0
                ),
                "updated_at": row[3] or "",
            }
            for row in rows
        ]
        if annotator_id is not None and not annotators:
            annotators = [
                {
                    "annotator_id": annotator_id,
                    "started_papers": 0,
                    "completed_papers": 0,
                    "target_papers": sample_n,
                    "completion_share": 0.0,
                    "updated_at": "",
                }
            ]
        return {
            "sample_papers": sample_n,
            "annotators": annotators,
            "annotator_count": len(annotators),
        }

    def _human_frame(self, annotator_id: str) -> pd.DataFrame:
        states = self._states(self.validate_annotator_id(annotator_id))
        rows = []
        for paper_id, state in states.items():
            if not state["is_complete"]:
                continue
            dimensions = state["payload"]["dimensions"]
            row: dict[str, Any] = {"paper_id": paper_id}
            for contract in HUMAN_DIMENSIONS:
                row[contract["analysis_column"]] = dimensions[
                    contract["column"]
                ]["code"]
            rows.append(row)
        return pd.DataFrame(rows, columns=["paper_id", *DISPLAY_COLUMNS])

    def _model_frames(self) -> dict[str, pd.DataFrame]:
        """Load model ratings currently materialised as processed CSV files."""

        sample_ids = set(self._sample()["paper_id"])
        frames: dict[str, pd.DataFrame] = {}
        if not self.specification_dir.exists():
            return frames
        for path in sorted(
            self.specification_dir.glob("paper_specifications_*_spec-v3.csv")
        ):
            mtime = path.stat().st_mtime_ns
            file_cache_key = str(path)
            cached = self._model_cache.get(file_cache_key)
            if cached and cached[0] == mtime:
                frame = cached[1]
            else:
                header = pd.read_csv(path, nrows=0)
                if "paper_id" not in header.columns:
                    continue
                frame = pd.read_csv(
                    path, dtype=str, keep_default_na=False
                ).fillna("")
                frame = enrich_for_analysis(frame)
                self._model_cache[file_cache_key] = (mtime, frame)
            model_values = (
                frame["coding_model"].astype(str).str.strip()
                if "coding_model" in frame.columns
                else pd.Series(dtype=str)
            )
            model = next(
                (value for value in model_values if value),
                path.stem.removeprefix("paper_specifications_").removesuffix(
                    "_spec-v3"
                ),
            )
            for column in DISPLAY_COLUMNS:
                if column not in frame.columns:
                    frame[column] = ""
            scoped = frame[
                frame["paper_id"].isin(sample_ids)
            ][["paper_id", *DISPLAY_COLUMNS]].copy()
            if not scoped["paper_id"].duplicated().any():
                frames[model] = scoped

        cache_root = (
            self.project_root / "data/interim/spec_cache" / PROTOCOL_ID
        )
        for expected_model, directory_name in VALIDATION_CACHE_MODELS.items():
            directory = cache_root / directory_name
            if not directory.exists():
                continue
            rows = []
            for paper_id in sorted(sample_ids):
                path = directory / cache_key(paper_id)
                if not path.exists():
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if (
                    str(payload.get("paper_id", "")).strip() != paper_id
                    or str(payload.get("coding_protocol", "")).strip()
                    != PROTOCOL_ID
                ):
                    continue
                rows.append(payload)
            if not rows:
                continue
            cache_frame = enrich_for_analysis(
                pd.DataFrame(rows).fillna("")
            )
            for column in DISPLAY_COLUMNS:
                if column not in cache_frame.columns:
                    cache_frame[column] = ""
            model_values = cache_frame.get(
                "coding_model", pd.Series(dtype=str)
            ).astype(str)
            model = next(
                (
                    value.strip()
                    for value in model_values
                    if value.strip()
                ),
                expected_model,
            )
            scoped_cache = cache_frame[
                cache_frame["paper_id"].isin(sample_ids)
            ][["paper_id", *DISPLAY_COLUMNS]].copy()
            if model in frames:
                scoped_cache = pd.concat(
                    [frames[model], scoped_cache], ignore_index=True
                ).drop_duplicates("paper_id", keep="first")
            if not scoped_cache["paper_id"].duplicated().any():
                frames[model] = scoped_cache
        return frames

    def rater_catalog(self) -> list[dict[str, Any]]:
        sample_n = len(self._sample())
        catalog = []
        for row in self.progress()["annotators"]:
            catalog.append(
                {
                    "id": f"human::{row['annotator_id']}",
                    "label": f"Human: {row['annotator_id']}",
                    "type": "human",
                    "available_papers": row["completed_papers"],
                    "target_papers": sample_n,
                    "default_selected": row["completed_papers"] > 0,
                }
            )
        for model, frame in self._model_frames().items():
            catalog.append(
                {
                    "id": f"model::{model}",
                    "label": MODEL_LABELS.get(model, model),
                    "type": "model",
                    "available_papers": len(frame),
                    "target_papers": sample_n,
                    "default_selected": len(frame) == sample_n,
                }
            )
        return catalog

    def reliability(
        self,
        annotator_ids: set[str] | None = None,
        model_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        """Calculate all selected raters on one exact balanced intersection."""

        humans = {
            row["annotator_id"]
            for row in self.progress()["annotators"]
            if row["completed_papers"] > 0
        }
        models = self._model_frames()
        selected_humans = humans if annotator_ids is None else annotator_ids
        selected_models = (
            {
                model
                for model, frame in models.items()
                if len(frame) == len(self._sample())
            }
            if model_ids is None
            else model_ids
        )
        unknown_humans = selected_humans - humans
        unknown_models = selected_models - set(models)
        if unknown_humans:
            raise ValueError(f"Unknown completed annotators: {sorted(unknown_humans)}")
        if unknown_models:
            raise ValueError(f"Unknown available models: {sorted(unknown_models)}")

        rater_frames: dict[str, pd.DataFrame] = {}
        rater_labels: dict[str, str] = {}
        rater_types: dict[str, str] = {}
        for annotator_id in sorted(selected_humans):
            key = f"human::{annotator_id}"
            rater_frames[key] = self._human_frame(annotator_id)
            rater_labels[key] = f"Human: {annotator_id}"
            rater_types[key] = "human"
        for model in sorted(selected_models):
            key = f"model::{model}"
            rater_frames[key] = models[model]
            rater_labels[key] = MODEL_LABELS.get(model, model)
            rater_types[key] = "model"

        id_sets = [
            set(frame["paper_id"]) for frame in rater_frames.values()
        ]
        balanced_ids = set.intersection(*id_sets) if id_sets else set()
        balanced_frames = {
            key: frame[frame["paper_id"].isin(balanced_ids)]
            .sort_values("paper_id")
            .set_index("paper_id")
            for key, frame in rater_frames.items()
        }
        pairs = []
        for left, right in combinations(sorted(balanced_frames), 2):
            dimensions = []
            for column in DISPLAY_COLUMNS:
                left_values = balanced_frames[left][column].tolist()
                right_values = balanced_frames[right][column].tolist()
                exact = pairwise_percent_agreement(left_values, right_values)
                alpha = krippendorff_alpha_nominal(
                    [list(pair) for pair in zip(left_values, right_values)]
                )
                dimensions.append(
                    {
                        "column": column,
                        "label": LABEL_BY_ANALYSIS_COLUMN[column],
                        "classification": (
                            "Core" if column in CORE_COLUMNS else "Exploratory"
                        ),
                        "comparable_papers": exact.comparable,
                        "agreements": exact.agreements,
                        "disagreements": exact.comparable - exact.agreements,
                        "percent_agreement": exact.percent_agreement,
                        "krippendorff_alpha": alpha,
                    }
                )
            core = [
                row for row in dimensions if row["classification"] == "Core"
            ]
            agreements = [
                row["percent_agreement"]
                for row in core
                if row["percent_agreement"] is not None
            ]
            alphas = [
                row["krippendorff_alpha"]
                for row in core
                if row["krippendorff_alpha"] is not None
            ]
            pairs.append(
                {
                    "left_model": left,
                    "left_label": rater_labels[left],
                    "right_model": right,
                    "right_label": rater_labels[right],
                    "intersection_papers": len(balanced_ids),
                    "mean_percent_agreement": (
                        sum(agreements) / len(agreements)
                        if agreements
                        else None
                    ),
                    "mean_krippendorff_alpha": (
                        sum(alphas) / len(alphas) if alphas else None
                    ),
                    "dimensions": dimensions,
                }
            )
        human_progress = self.progress()["annotators"]
        return {
            "sample_id": "workbook_probability_overlap",
            "sample_label": "23-paper probability-sample overlap",
            "sample_papers": len(self._sample()),
            "balanced_common_papers": len(balanced_ids),
            "provisional": any(
                row["completed_papers"] < row["target_papers"]
                for row in human_progress
                if row["annotator_id"] in selected_humans
            ),
            "raters": [
                {
                    "id": key,
                    "label": rater_labels[key],
                    "type": rater_types[key],
                    "available_papers": len(rater_frames[key]),
                }
                for key in sorted(rater_frames)
            ],
            "available_raters": self.rater_catalog(),
            "pairs": pairs,
            "dimension_count": len(DISPLAY_COLUMNS),
            "core_dimension_count": len(CORE_COLUMNS),
            "summary_method": (
                "arithmetic mean across six core dimensions on the exact "
                "balanced common-paper intersection"
            ),
        }

    def export(self, annotator_id: str | None = None) -> pd.DataFrame:
        """Return completed human codes in a traceable long table."""

        selected = (
            [self.validate_annotator_id(annotator_id)]
            if annotator_id
            else [
                row["annotator_id"]
                for row in self.progress()["annotators"]
            ]
        )
        rows = []
        for current in selected:
            states = self._states(current)
            for paper_id, state in states.items():
                payload = state["payload"]
                for contract in HUMAN_DIMENSIONS:
                    entry = payload["dimensions"][contract["column"]]
                    rows.append(
                        {
                            "annotator_id": current,
                            "paper_id": paper_id,
                            "is_complete": state["is_complete"],
                            "dimension": contract["column"],
                            "classification": contract["classification"],
                            **entry,
                            "ai_mechanism_logic": payload[
                                "ai_mechanism_logic"
                            ],
                            "needs_full_text": ";".join(
                                payload["needs_full_text"]
                            ),
                            "annotator_notes": payload["annotator_notes"],
                            "updated_at": state["updated_at"],
                        }
                    )
        return pd.DataFrame(rows)
