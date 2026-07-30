"""Provide graph and analytical data to the API.

Inputs: processed paper-level CSV files and an optional Neo4j driver.
Outputs: scope metrics, evidence records, and graph traversal results.
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

import pandas as pd

from aecsp.analytics.convergence import (
    DIMENSION_COLUMNS,
    construct_contrast,
    convergence_by,
    dimension_profile,
    group_convergence,
)
from aecsp.analytics.coder_robustness import build_coder_robustness
from aecsp.analytics.agreement import (
    krippendorff_alpha_nominal,
    pairwise_percent_agreement,
)
from aecsp.analytics.keyword_trends import (
    SEARCH_CUTOFF_DATE,
    SEARCH_CUTOFF_LABEL,
    SEARCH_CUTOFF_YEAR,
    analyze_keyword_evolution,
    keyword_evidence_mask,
    keyword_year_summary,
    search_keyword_series,
)
from aecsp.analytics.observed_composition import (
    OBSERVED_COMPOSITION_PANELS,
    STUDY_STATUS_FILTERS,
    analyze_observed_composition,
    observed_composition_evidence_mask,
)
from aecsp.analytics.publication_growth import (
    DEFAULT_GROWTH_PERIODS,
    cumulative_trace,
    growth_records,
)
from aecsp.analytics.theory_contrasting import (
    DIMENSIONS as THEORY_DIMENSIONS,
    STRUCTURING_PAIRS,
    VERTICAL_ROW_DIMENSIONS,
    dimension_column,
    distribution as theory_distribution,
    filter_study_status,
    observed_mask as theory_observed_mask,
    recurring_configurations,
    relationship_matrix,
)
from aecsp.corpus.business_domains import REGISTERED_QUERY_DOMAIN_RULES
from aecsp.corpus.scopes import (
    DATASET_SCOPES,
    SCOPE_BY_ID,
    ScopeError,
    scope_frame,
)
from aecsp.knowledge_graph.neo4j_reader import (
    GraphQueryError,
    Neo4jGraphReader,
    graph_node_id,
)
from aecsp.specification.schema import (
    AI_STUDY_STATUS_FIELD,
    AI_STUDY_STATUS_COLUMN,
    SPECIFICATION_COLUMNS,
    SPECIFICATION_DIMENSIONS,
    SPECIFICATION_PROBLEM_COLUMN,
)
from aecsp.specification.analysis_columns import enrich_for_analysis
from aecsp.specification.paths import (
    load_experiment_register,
    resolve_primary_model,
    specification_csv_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_COMPARISON_CONFIG = PROJECT_ROOT / "configs" / "model_comparison.json"
CLOSE_READING_LEDGER = (
    PROJECT_ROOT
    / "data/interim/theory_elaboration/theory_elaboration_matched_papers.csv"
)

MODEL_DISPLAY_NAMES = {
    "gpt-5.4-mini-2026-03-17": "GPT-5.4 Mini",
    "gpt-4.1-nano-2025-04-14": "GPT-4.1 Nano",
    "claude-sonnet-5": "Claude Sonnet 5",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
    "llama3.2": "Llama 3.2",
    "gemma4:31b": "Gemma 4 31B",
}

CORE_IRR_DIMENSIONS = (
    (AI_STUDY_STATUS_COLUMN, "AI as method or phenomenon"),
    ("ai_type_form", "Technical AI type/form"),
    ("ai_role_function", "AI role/function"),
    ("ai_mechanism_analysis", "AI mechanism"),
    ("level_of_analysis", "Level of analysis"),
    ("scope_conditions", "Scope conditions"),
)

EXPLORATORY_IRR_DIMENSIONS = (
    ("entrepreneurial_process_stage", "Entrepreneurial process stage"),
    ("definition_construct_clarity", "Definition clarity"),
)

DISPLAY_IRR_DIMENSIONS = tuple(
    (column, label, "Core") for column, label in CORE_IRR_DIMENSIONS
) + tuple(
    (column, label, "Exploratory")
    for column, label in EXPLORATORY_IRR_DIMENSIONS
)

IRR_UNOBSERVED_VALUES = {
    panel["column"]: frozenset(str(value).strip() for value in panel["excluded"])
    for panel in OBSERVED_COMPOSITION_PANELS
}

# Evidence columns surfaced whenever we return a paper list. DOI and Link let
# the UI build an in-text citation that links out to the article.
EVIDENCE_COLUMNS = [
    "paper_id",
    "Title",
    "Abstract",
    "Author Keywords",
    "Index Keywords",
    "Authors",
    "Source title",
    "Year",
    "Cited by",
    "DOI",
    "Link",
    "query_sources",
    "bertopic_topic_label",
    AI_STUDY_STATUS_COLUMN,
    *DIMENSION_COLUMNS,
    "ai_mechanism_analysis",
    "ai_mechanism_logic",
    "theories_mentioned",
    "needs_full_text",
    SPECIFICATION_PROBLEM_COLUMN,
]

INSPECTION_DIMENSIONS = (AI_STUDY_STATUS_FIELD, *SPECIFICATION_DIMENSIONS)
DIMENSION_EVIDENCE_COLUMNS = tuple(
    field
    for dimension in SPECIFICATION_DIMENSIONS
    for field in (
        f"{dimension.column}_evidence",
        f"{dimension.column}_evidence_type",
        f"{dimension.column}_confidence",
    )
)
SCOPE_LABEL_BY_ID = {scope.id: scope.label for scope in DATASET_SCOPES}

THEORY_POPULATIONS = (
    ("core", "Leading entrepreneurship journals"),
    ("other", "Additional entrepreneurship journals"),
    ("combined", "Combined entrepreneurship"),
    ("close_reading", "Systematic close-reading set"),
)

PENDING_ASJC_DOMAIN_LABELS = {
    "innovation": "Management of Technology and Innovation",
    "strategy": "Strategy and Management",
    "marketing": "Marketing",
    "information_systems": "Information systems",
    "finance": "Finance",
    "operations": "Management Science and Operations Research",
    "organization_studies": "Organization studies",
    "environmental_and_sustainability": "Environmental and sustainability",
    "tourism": "Tourism, Leisure and Hospitality Management",
}


def _short_title(title: object, width: int = 42) -> str:
    text = str(title or "").strip()
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


class GraphService:
    """Serves scope-aware analytics and evidence to the API."""

    def __init__(
        self,
        processed_dir: Path = PROCESSED_DIR,
        neo4j_driver=None,
        model: str | None = None,
        neo4j_database: str = "neo4j",
    ) -> None:
        self.processed_dir = processed_dir
        self.driver = neo4j_driver
        candidate = (
            Neo4jGraphReader(neo4j_driver, database=neo4j_database)
            if neo4j_driver is not None
            else None
        )
        self.neo4j_security = candidate.security if candidate is not None else None
        self.neo4j = (
            candidate
            if candidate is not None
            and candidate.security.get("read_only_verified")
            else None
        )
        self.model = model
        self.papers = self._load_papers()
        self._composition_frames: dict[str, pd.DataFrame] = {}
        self._composition_frame_signatures: dict[str, tuple] = {}
        self._comparison_config = self._load_comparison_config()
        self._reference_ids: frozenset[str] | None = None
        self._reference_signature: tuple | None = None
        self._close_reading_ids_cache: frozenset[str] | None = None

    # ---- loading --------------------------------------------------------
    def _load_papers(self) -> pd.DataFrame:
        analysis_dir = self.processed_dir / "analysis"
        enriched = analysis_dir / "primary_analysis_dataset_with_topics.csv"
        primary = analysis_dir / "primary_analysis_dataset.csv"
        topics = self.processed_dir / "master_corpus_topics.csv"
        base = self.processed_dir / "master_corpus.csv"
        candidates = (enriched, primary, topics, base)
        path = next((candidate for candidate in candidates if candidate.exists()), base)
        if not path.exists():
            return pd.DataFrame()
        papers = pd.read_csv(path, dtype=str, keep_default_na=False)

        # Keep the frozen primary table as the analytical base. Once the topic
        # runner has produced assignments, append only genuinely new columns by
        # paper_id until the versioned enriched dataset is materialised.
        if path == primary and topics.exists():
            topic_frame = pd.read_csv(topics, dtype=str, keep_default_na=False)
            new_columns = [
                column
                for column in topic_frame.columns
                if column == "paper_id" or column not in papers.columns
            ]
            if len(new_columns) > 1:
                papers = papers.merge(
                    topic_frame[new_columns], on="paper_id", how="left", validate="one_to_one"
                )

        spec_path = specification_csv_path(self.processed_dir, model=self.model)
        already_has_specs = any(column in papers.columns for column in SPECIFICATION_COLUMNS)
        if spec_path.exists() and not already_has_specs:
            specs = pd.read_csv(spec_path, dtype=str, keep_default_na=False)
            keep = ["paper_id"] + [c for c in SPECIFICATION_COLUMNS if c in specs.columns]
            papers = papers.merge(specs[keep], on="paper_id", how="left", suffixes=("", "_spec"))
        papers = self._apply_source_title_display_aliases(papers.fillna(""))
        return self._apply_topic_review_display_labels(papers)

    def _apply_source_title_display_aliases(
        self, papers: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge reviewed punctuation variants without altering raw source files."""

        alias_path = PROJECT_ROOT / "configs/business_domain_journal_aliases.csv"
        if "Source title" not in papers.columns or not alias_path.exists():
            return papers
        aliases = pd.read_csv(alias_path, dtype=str, keep_default_na=False)
        required = {"registered_title", "corpus_title", "review_status"}
        if not required.issubset(aliases.columns):
            return papers
        approved = aliases[
            aliases["review_status"].str.strip().str.lower().eq("approved")
        ]
        mapping = {
            str(row["corpus_title"]).strip(): str(row["registered_title"]).strip()
            for _, row in approved.iterrows()
            if str(row["corpus_title"]).strip()
            and str(row["registered_title"]).strip()
        }
        if not mapping:
            return papers
        result = papers.copy()
        result["Source title"] = result["Source title"].replace(mapping)
        return result

    def _apply_topic_review_display_labels(self, papers: pd.DataFrame) -> pd.DataFrame:
        """Apply saved UI labels in memory without changing processed datasets."""

        review_path = (
            self.processed_dir
            / "analysis/stage4/topic_label_review.csv"
        )
        if not review_path.exists() or "paper_id" not in papers.columns:
            return papers
        review = pd.read_csv(review_path, dtype=str, keep_default_na=False)
        required = {"scope", "topic_id", "automatic_label", "approved_label"}
        if not required.issubset(review.columns):
            return papers
        if "review_status" not in review.columns:
            review["review_status"] = "pending"
        result = papers.copy()
        for scope in ("full_corpus", "query_1", "query_2", "query_3", "query_4"):
            topic_id_column = (
                "bertopic_topic" if scope == "full_corpus" else f"{scope}_topic_id"
            )
            label_column = (
                "bertopic_topic_label"
                if scope == "full_corpus"
                else f"{scope}_topic_label"
            )
            if topic_id_column not in result.columns or label_column not in result.columns:
                continue
            scoped = review[review["scope"].eq(scope)].copy()
            if scoped.empty:
                continue
            scoped["topic_id"] = pd.to_numeric(scoped["topic_id"], errors="raise").astype(int)
            scoped["display_label"] = scoped["approved_label"].str.strip().where(
                scoped["approved_label"].str.strip().ne(""),
                scoped["automatic_label"],
            )
            mapping = scoped.set_index("topic_id")["display_label"].to_dict()
            status_mapping = scoped.set_index("topic_id")["review_status"].to_dict()
            automatic_column = f"{label_column}_automatic"
            if automatic_column not in result.columns:
                result[automatic_column] = result[label_column]
            topic_ids = pd.to_numeric(result[topic_id_column], errors="coerce")
            display = topic_ids.map(mapping)
            result[label_column] = display.fillna(result[label_column])
            result[f"{label_column}_review_status"] = topic_ids.map(
                status_mapping
            ).fillna("automatic")
        return result

    def reload_data(self) -> int:
        """Reload the preferred processed dataset after a controlled rebuild."""

        self.papers = self._load_papers()
        self._composition_frames.clear()
        self._composition_frame_signatures.clear()
        self._reference_ids = None
        self._reference_signature = None
        self._close_reading_ids_cache = None
        return len(self.papers)

    @property
    def has_specifications(self) -> bool:
        return bool(self.papers is not None) and any(
            c in self.papers.columns for c in DIMENSION_COLUMNS
        )

    def _scope(self, scope_id: str) -> pd.DataFrame:
        try:
            return scope_frame(self.papers, scope_id)
        except ScopeError as error:
            for population_id, _label, _definition, frame, _exclusive in (
                self._publication_growth_population_frames()
            ):
                if population_id == scope_id:
                    return frame.reset_index(drop=True)
            raise error

    def _close_reading_ids(self) -> frozenset[str]:
        """Return the fixed, audited 136-paper systematic close-reading set."""

        if self._close_reading_ids_cache is not None:
            return self._close_reading_ids_cache
        if not CLOSE_READING_LEDGER.exists():
            self._close_reading_ids_cache = frozenset()
            return self._close_reading_ids_cache
        ledger = pd.read_csv(
            CLOSE_READING_LEDGER,
            usecols=["paper_id"],
            dtype=str,
            keep_default_na=False,
        )
        self._close_reading_ids_cache = frozenset(
            paper_id
            for paper_id in ledger["paper_id"].astype(str).str.strip()
            if paper_id
        )
        return self._close_reading_ids_cache

    def _registered_composition_models(self) -> list[tuple[str, str]]:
        """Return every model eligible for an analytical model selector.

        The comparison configuration defines the prespecified IRR cohort.
        Supplementary validation models can still become selectable when a
        usable analytical export exists, without altering that cohort.
        """

        register = load_experiment_register()
        primary = str(register["primary_model"])
        registered = [(primary, "primary")]
        registered.extend(
            (str(model), "baseline")
            for model in register.get("baseline_models", [])
            if str(model) != primary
        )
        existing = {model for model, _role in registered}
        for model in self._comparison_config.get("models", []):
            model = str(model)
            if model and model not in existing:
                registered.append((model, "independent"))
                existing.add(model)
        for model in register.get("validation_models", []):
            model = str(model)
            if model and model not in existing:
                registered.append((model, "supplementary"))
                existing.add(model)
        return registered

    def _irr_composition_models(self) -> list[dict]:
        """Return fixed and coverage-qualified supplementary IRR models."""

        available = {item["id"]: item for item in self.composition_models()}
        selected = [
            available[str(model)]
            for model in self._comparison_config.get("models", [])
            if str(model) in available
        ]
        selected_ids = {item["id"] for item in selected}
        for model, minimum in self._supplementary_irr_thresholds().items():
            candidate = available.get(model)
            if (
                candidate is not None
                and model not in selected_ids
                and int(candidate["coded_papers"]) >= minimum
            ):
                selected.append(candidate)
                selected_ids.add(model)
        return selected

    def _cross_model_agreement_models(
        self,
        reference_model: str | None = None,
    ) -> list[dict]:
        """Return available models admitted to paper-level agreement evidence."""

        available = {item["id"]: item for item in self.composition_models()}
        configured = self._comparison_config.get(
            "cross_model_agreement_models",
            self._comparison_config.get("models", []),
        )
        if not isinstance(configured, list):
            raise ValueError(
                "Cross-model agreement configuration must contain a model list"
            )
        selected = [
            available[str(model)]
            for model in configured
            if str(model) in available
        ]
        selected_ids = {item["id"] for item in selected}
        if reference_model and reference_model in available and reference_model not in selected_ids:
            selected.append(available[reference_model])
        return selected

    def _supplementary_irr_thresholds(self) -> dict[str, int]:
        """Return validated model-specific minimum coverage counts for IRR."""

        thresholds: dict[str, int] = {}
        rules = self._comparison_config.get("supplementary_irr_models", [])
        if not isinstance(rules, list):
            raise ValueError(
                "Supplementary IRR model configuration must be a list"
            )
        for rule in rules:
            if not isinstance(rule, dict):
                raise ValueError(
                    "Every supplementary IRR model rule must be an object"
                )
            model = str(rule.get("model", "")).strip()
            minimum = int(rule.get("minimum_coded_papers", 0))
            if not model or minimum < 1:
                raise ValueError(
                    "Supplementary IRR rules require a model and positive "
                    "minimum_coded_papers"
                )
            thresholds[model] = minimum
        return thresholds

    @staticmethod
    def _load_comparison_config() -> dict:
        """Load the transparent model-comparison cohort contract."""

        if not MODEL_COMPARISON_CONFIG.exists():
            return {}
        payload = json.loads(MODEL_COMPARISON_CONFIG.read_text(encoding="utf-8"))
        if not isinstance(payload.get("models", []), list):
            raise ValueError("Model-comparison configuration must contain a model list")
        return payload

    @staticmethod
    def _file_signature(path: Path) -> tuple:
        if not path.exists():
            return (False, 0, 0)
        stat = path.stat()
        return (True, stat.st_mtime_ns, stat.st_size)

    def _comparison_reference_model(self) -> str | None:
        model = str(self._comparison_config.get("reference_model", "")).strip()
        if not model:
            return None
        path = specification_csv_path(self.processed_dir, model=model)
        return model if path.exists() else None

    def _comparison_reference_ids(self) -> frozenset[str]:
        """Return the successful reference-model IDs that bound model IRR."""

        reference_model = self._comparison_reference_model()
        if reference_model is None:
            return frozenset(self.papers.get("paper_id", pd.Series(dtype=str)).astype(str))
        path = specification_csv_path(self.processed_dir, model=reference_model)
        signature = (reference_model, *self._file_signature(path))
        if self._reference_ids is not None and signature == self._reference_signature:
            return self._reference_ids
        frame = pd.read_csv(
            path,
            usecols=["paper_id"],
            dtype=str,
            keep_default_na=False,
        )
        paper_ids = frame["paper_id"].astype(str).str.strip()
        if paper_ids.eq("").any():
            raise ValueError("Reference-model results contain blank paper IDs")
        if paper_ids.duplicated().any():
            raise ValueError("Reference-model results contain duplicate paper IDs")
        corpus_ids = set(self.papers["paper_id"].astype(str))
        self._reference_ids = frozenset(paper_ids[paper_ids.isin(corpus_ids)])
        self._reference_signature = signature
        return self._reference_ids

    def _comparison_reference_frame(self) -> pd.DataFrame:
        reference_ids = self._comparison_reference_ids()
        return self.papers[
            self.papers["paper_id"].astype(str).isin(reference_ids)
        ].copy()

    def _composition_model_frame(self, model: str) -> pd.DataFrame:
        """Join every successful code from one model to corpus metadata."""

        registered = {item[0] for item in self._registered_composition_models()}
        if model not in registered:
            raise ValueError(f"Unknown specification model: {model}")
        spec_path = specification_csv_path(self.processed_dir, model=model)
        signature = self._file_signature(spec_path)
        if (
            model in self._composition_frames
            and self._composition_frame_signatures.get(model) == signature
        ):
            return self._composition_frames[model]

        primary_model = resolve_primary_model()
        if not spec_path.exists():
            if model == primary_model and self.has_specifications:
                frame = self.papers.copy()
                self._composition_frames[model] = frame
                self._composition_frame_signatures[model] = signature
                return frame
            raise ValueError(f"Specification results are unavailable for model: {model}")

        specifications = pd.read_csv(
            spec_path, dtype=str, keep_default_na=False
        ).fillna("")
        if "paper_id" not in specifications.columns:
            raise ValueError(f"Specification results have no paper_id column: {model}")
        if specifications["paper_id"].duplicated().any():
            raise ValueError(f"Specification results contain duplicate paper IDs: {model}")
        specifications = enrich_for_analysis(specifications)

        # The preferred analysis dataset already carries the primary model's
        # fields. Remove every column supplied by the selected model before the
        # one-to-one merge so a Nano view can never inherit Mini values.
        replace_columns = {
            *SPECIFICATION_COLUMNS,
            *specifications.columns,
            "ai_mechanism_raw",
            "ai_mechanism_analysis",
            "mechanism_corrected",
            "mechanism_black_box",
            "specification_problem_count",
        }
        replace_columns.discard("paper_id")
        corpus = self.papers.drop(
            columns=[column for column in replace_columns if column in self.papers.columns]
        )
        frame = corpus.merge(
            specifications,
            on="paper_id",
            how="inner",
            validate="one_to_one",
        ).fillna("")
        self._composition_frames[model] = frame
        self._composition_frame_signatures[model] = signature
        return frame

    def composition_models(self) -> list[dict]:
        """Describe model outputs currently safe for full-corpus comparison."""

        reference_model = self._comparison_reference_model()
        corpus_n = len(self.papers)
        fixed_irr_models = {
            str(model) for model in self._comparison_config.get("models", [])
        }
        supplementary_thresholds = self._supplementary_irr_thresholds()
        models = []
        for model, role in self._registered_composition_models():
            try:
                coded_n = len(self._composition_model_frame(model))
            except ValueError:
                continue
            irr_minimum = supplementary_thresholds.get(model)
            irr_eligible = model in fixed_irr_models or (
                irr_minimum is not None and coded_n >= irr_minimum
            )
            models.append(
                {
                    "id": model,
                    "label": MODEL_DISPLAY_NAMES.get(model, model),
                    "role": role,
                    "coded_papers": coded_n,
                    "corpus_papers": corpus_n,
                    "missing_papers": max(0, corpus_n - coded_n),
                    "coverage_share": round(coded_n / corpus_n, 6) if corpus_n else 0.0,
                    "irr_eligible": irr_eligible,
                    "irr_minimum_coded_papers": irr_minimum,
                    "irr_status": (
                        "included"
                        if irr_eligible
                        else "pending_threshold"
                        if irr_minimum is not None
                        else "not_configured"
                    ),
                    "comparison_cohort": self._comparison_config.get("comparison_id", ""),
                    "reference_model": reference_model,
                    "reference_label": MODEL_DISPLAY_NAMES.get(
                        reference_model, reference_model
                    ) if reference_model else "Full available corpus",
                }
            )
        return models

    def _composition_scope(self, scope_id: str, model: str) -> tuple[pd.DataFrame, int]:
        corpus_scope = self._scope(scope_id)
        corpus_ids = set(corpus_scope["paper_id"].astype(str))
        model_frame = self._composition_model_frame(model)
        selected = model_frame[
            model_frame["paper_id"].astype(str).isin(corpus_ids)
        ].copy()
        return selected.reset_index(drop=True), len(corpus_scope)

    def _paper_inspection_records(
        self,
        frame: pd.DataFrame,
        selected_columns: tuple[str, ...] = (),
        limit: int = 100,
    ) -> list[dict]:
        """Build evidence-first paper records for the two construct views."""

        selected = set(selected_columns)
        available_columns = list(
            dict.fromkeys(
                column
                for column in (*EVIDENCE_COLUMNS, *DIMENSION_EVIDENCE_COLUMNS)
                if column in frame.columns
            )
        )
        records = []
        for _, row in frame.head(limit).iterrows():
            record = {column: row.get(column, "") for column in available_columns}
            query_ids = [
                value.strip()
                for value in re.split(r"[;,]", str(row.get("query_sources", "")))
                if value.strip()
            ]
            dimensions = []
            for dimension in INSPECTION_DIMENSIONS:
                value_column = dimension.column
                if (
                    dimension.column == "ai_mechanism"
                    and "ai_mechanism_analysis" in frame.columns
                ):
                    value_column = "ai_mechanism_analysis"
                dimensions.append(
                    {
                        "column": value_column,
                        "source_column": dimension.column,
                        "label": dimension.label,
                        "question": dimension.question,
                        "diagnosis": dimension.diagnosis,
                        "value": row.get(value_column, row.get(dimension.column, "")),
                        "evidence": row.get(f"{dimension.column}_evidence", ""),
                        "evidence_type": row.get(
                            f"{dimension.column}_evidence_type", ""
                        ),
                        "confidence": row.get(
                            f"{dimension.column}_confidence", ""
                        ),
                        "selected": bool(
                            {value_column, dimension.column}.intersection(selected)
                        ),
                    }
                )
            record["_inspection"] = {
                "evidence_boundary": "Title, abstract, and author keywords",
                "dataset_views": [
                    SCOPE_LABEL_BY_ID.get(query_id, query_id)
                    for query_id in query_ids
                ],
                "dimensions": dimensions,
                "mechanism_logic": row.get("ai_mechanism_logic", ""),
                "theories_mentioned": row.get("theories_mentioned", ""),
                "needs_full_text": row.get("needs_full_text", ""),
                "specification_problem": row.get(
                    SPECIFICATION_PROBLEM_COLUMN, ""
                ),
            }
            records.append(record)
        return records

    @staticmethod
    def _agreement_dimension_label(column: str) -> str:
        """Return the instrument label for one exact-match agreement column."""

        for dimension in INSPECTION_DIMENSIONS:
            aliases = {dimension.column}
            if dimension.column == "ai_mechanism":
                aliases.add("ai_mechanism_analysis")
            if column in aliases:
                return dimension.label
        return column.replace("_", " ").strip().title()

    def _model_pattern_agreement(
        self,
        paper_ids: list[str],
        filters: dict[str, str],
        reference_model: str | None = None,
    ) -> dict[str, dict]:
        """Compare other assignments with the selected model's evidence pattern."""

        models = self._cross_model_agreement_models(reference_model)
        normalized_filters = {
            str(column): str(value).strip()
            for column, value in filters.items()
        }
        pattern = [
            {
                "column": column,
                "label": self._agreement_dimension_label(column),
                "value": value,
                "display_value": value or "Missing value",
            }
            for column, value in normalized_filters.items()
        ]
        paper_ids = [str(paper_id) for paper_id in paper_ids]
        assignments: dict[str, list[dict]] = {paper_id: [] for paper_id in paper_ids}
        for model in models:
            frame = self._composition_model_frame(model["id"])
            subset = frame[frame["paper_id"].astype(str).isin(paper_ids)].copy()
            subset["paper_id"] = subset["paper_id"].astype(str)
            indexed = subset.set_index("paper_id", drop=False)
            for paper_id in paper_ids:
                available = paper_id in indexed.index
                values = {}
                matches = available and bool(normalized_filters)
                if available:
                    row = indexed.loc[paper_id]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    for column, expected in normalized_filters.items():
                        assigned = str(row.get(column, "")).strip()
                        values[column] = assigned
                        if assigned != expected:
                            matches = False
                else:
                    values = {column: "" for column in normalized_filters}
                assignments[paper_id].append(
                    {
                        "model": model["id"],
                        "label": model["label"],
                        "is_reference": model["id"] == reference_model,
                        "available": available,
                        "matches": matches,
                        "values": values,
                    }
                )

        result = {}
        model_count = len(models)
        preferred_ids = {
            str(model)
            for model in self._comparison_config.get(
                "preferred_agreement_models", []
            )
        }
        for paper_id in paper_ids:
            rows = assignments[paper_id]
            agreeing = [row for row in rows if row["matches"]]
            available = [row for row in rows if row["available"]]
            preferred_rows = [row for row in rows if row["model"] in preferred_ids]
            preferred_agreeing = [row for row in preferred_rows if row["matches"]]
            reference_row = next(
                (row for row in rows if row["model"] == reference_model),
                None,
            )
            result[paper_id] = {
                "pattern": pattern,
                "reference_model": (
                    {
                        "id": reference_row["model"],
                        "label": reference_row["label"],
                    }
                    if reference_row
                    else None
                ),
                "reference_model_matches": (
                    bool(reference_row["matches"]) if reference_row else None
                ),
                "models_total": model_count,
                "models_available": len(available),
                "models_agreeing": len(agreeing),
                "agreement_models": [
                    {"id": row["model"], "label": row["label"]}
                    for row in agreeing
                ],
                "all_models_agree": bool(pattern)
                and model_count > 0
                and len(agreeing) == model_count,
                "preferred_agreement_label": self._comparison_config.get(
                    "preferred_agreement_label",
                    "Preferred-model sweet spot",
                ),
                "preferred_models_total": len(preferred_ids),
                "preferred_models_agreeing": len(preferred_agreeing),
                "preferred_agreement_models": [
                    {"id": row["model"], "label": row["label"]}
                    for row in preferred_agreeing
                ],
                "preferred_sweet_spot": bool(pattern)
                and bool(preferred_ids)
                and len(preferred_rows) == len(preferred_ids)
                and len(preferred_agreeing) == len(preferred_ids),
                "assignments": rows,
            }
        return result

    def _agreement_evidence_bundle(
        self,
        frame: pd.DataFrame,
        filters: dict[str, str],
        selected_columns: tuple[str, ...],
        limit: int,
        minimum_agreement: int = 1,
        preferred_only: bool = False,
        reference_model: str | None = None,
    ) -> dict:
        """Attach model agreement and optionally require a minimum model count."""

        agreement_models = self._cross_model_agreement_models(reference_model)
        model_count = len(agreement_models)
        if minimum_agreement < 1 or minimum_agreement > max(1, model_count):
            raise ValueError(
                f"Minimum model agreement must be between 1 and {max(1, model_count)}"
            )
        paper_ids = frame["paper_id"].astype(str).tolist()
        agreement = self._model_pattern_agreement(
            paper_ids,
            filters,
            reference_model=reference_model,
        )
        if reference_model and filters:
            invalid_reference_ids = [
                paper_id
                for paper_id, detail in agreement.items()
                if detail["reference_model_matches"] is not True
            ]
            if invalid_reference_ids:
                raise ValueError(
                    "Evidence selection contains papers that do not match the "
                    f"selected model {reference_model}"
                )
        agreement_ids = {
            paper_id
            for paper_id, detail in agreement.items()
            if detail["models_agreeing"] >= minimum_agreement
        }
        two_model_ids = {
            paper_id
            for paper_id, detail in agreement.items()
            if detail["models_agreeing"] >= 2
        }
        all_model_ids = {
            paper_id
            for paper_id, detail in agreement.items()
            if detail["all_models_agree"]
        }
        preferred_ids = {
            paper_id
            for paper_id, detail in agreement.items()
            if detail["preferred_sweet_spot"]
        }
        agreement_threshold_counts = [
            {
                "minimum_models": threshold,
                "papers": sum(
                    detail["models_agreeing"] >= threshold
                    for detail in agreement.values()
                ),
                "additional_models_beyond_reference": (
                    threshold - 1 if reference_model else threshold
                ),
            }
            for threshold in range(1, model_count + 1)
        ]
        if not filters:
            selected_ids = set(paper_ids) if minimum_agreement == 1 and not preferred_only else set()
        else:
            selected_ids = preferred_ids if preferred_only else agreement_ids
        selected = frame[frame["paper_id"].astype(str).isin(selected_ids)]
        if not selected.empty:
            order = selected["paper_id"].astype(str).map(
                lambda paper_id: (
                    int(agreement[paper_id]["preferred_sweet_spot"]),
                    int(agreement[paper_id]["all_models_agree"]),
                    agreement[paper_id]["models_agreeing"],
                )
            )
            selected = selected.assign(
                _agreement_preferred=[item[0] for item in order],
                _agreement_all=[item[1] for item in order],
                _agreement_count=[item[2] for item in order],
            ).sort_values(
                ["_agreement_preferred", "_agreement_all", "_agreement_count"],
                ascending=[False, False, False],
                kind="stable",
            )
        records = self._paper_inspection_records(
            selected,
            selected_columns=selected_columns,
            limit=limit,
        )
        for record in records:
            record["_model_agreement"] = agreement.get(str(record.get("paper_id")), {})
        return {
            "minimum_agreement": minimum_agreement,
            "preferred_only": preferred_only,
            "total_papers": len(frame),
            "supporting_papers": len(frame),
            "agreement_papers": len(two_model_ids),
            "agreement_threshold_counts": agreement_threshold_counts,
            "all_model_agreement_papers": len(all_model_ids),
            "preferred_agreement_papers": len(preferred_ids),
            "preferred_agreement_label": self._comparison_config.get(
                "preferred_agreement_label",
                "Preferred-model sweet spot",
            ),
            "reference_model": (
                {
                    "id": reference_model,
                    "label": MODEL_DISPLAY_NAMES.get(reference_model, reference_model),
                }
                if reference_model
                else None
            ),
            "agreement_rule": (
                "The selected model defines the supporting-paper set and "
                "selected pattern; agreement counts other available models "
                "that assigned the same pattern to those papers."
            ),
            "filtered_agreement_papers": len(selected),
            "returned_papers": len(records),
            "models": [
                {"id": item["id"], "label": item["label"]}
                for item in agreement_models
            ],
            "pattern": next(iter(agreement.values()), {}).get("pattern", []),
            "papers": records,
        }

    def composition_export(
        self,
        scope_id: str,
        model: str,
        study_status: str = "all",
        filter_dimension: str | None = None,
        filter_value: str | None = None,
    ) -> pd.DataFrame:
        """Return the traceable paper table behind one composition view."""

        if study_status not in STUDY_STATUS_FILTERS:
            raise ValueError(f"Unknown AI positioning: {study_status}")
        frame, _ = self._composition_scope(scope_id, model)
        frame, _ = self._theory_controlled_frame(
            frame, filter_dimension, filter_value
        )
        if study_status != "all":
            frame = frame[
                frame[AI_STUDY_STATUS_COLUMN].astype(str).str.strip().eq(study_status)
            ]
        columns = [
            column
            for column in [
                "paper_id",
                "Title",
                "Authors",
                "Source title",
                "Year",
                "DOI",
                "Link",
                "query_sources",
                *(column for column, _ in CORE_IRR_DIMENSIONS),
                "entrepreneurial_process_stage",
                "definition_construct_clarity",
                "specification_problem",
                "coding_model",
                "coding_protocol",
                "coding_protocol_fingerprint",
            ]
            if column in frame.columns
        ]
        return frame[columns].reset_index(drop=True).copy()

    def composition_irr_units(
        self,
        scope_id: str,
        left_model: str,
        right_model: str,
        common_paper_ids: frozenset[str] | set[str] | None = None,
    ) -> pd.DataFrame:
        """Return paper-aligned ratings on one balanced comparison cohort."""

        if left_model == right_model:
            raise ValueError("IRR requires two different specification models")
        if common_paper_ids is None:
            common_paper_ids = self._balanced_composition_ids(scope_id)
        dimension_columns = [column for column, _, _ in DISPLAY_IRR_DIMENSIONS]
        left, _ = self._composition_scope(scope_id, left_model)
        right, _ = self._composition_scope(scope_id, right_model)
        left = left[
            left["paper_id"].astype(str).isin(common_paper_ids)
        ].copy()
        right = right[
            right["paper_id"].astype(str).isin(common_paper_ids)
        ].copy()
        for column in dimension_columns:
            if column not in left.columns:
                left[column] = ""
            if column not in right.columns:
                right[column] = ""
        left = left[["paper_id", *dimension_columns]].rename(
            columns={column: f"left__{column}" for column in dimension_columns}
        )
        right = right[["paper_id", *dimension_columns]].rename(
            columns={column: f"right__{column}" for column in dimension_columns}
        )
        return left.merge(
            right,
            on="paper_id",
            how="inner",
            validate="one_to_one",
        ).sort_values("paper_id").reset_index(drop=True)

    def _balanced_composition_ids(
        self,
        scope_id: str,
        model_ids: list[str] | tuple[str, ...] | None = None,
    ) -> frozenset[str]:
        """Return the exact paper-ID intersection shared by all comparison models."""

        if model_ids is None:
            model_ids = [
                item["id"] for item in self._irr_composition_models()
            ]
        scope_ids = set(self._scope(scope_id)["paper_id"].astype(str))
        reference_ids = set(
            self._comparison_reference_frame()["paper_id"].astype(str)
        )
        paper_sets = [scope_ids & reference_ids]
        for model_id in model_ids:
            frame, _ = self._composition_scope(scope_id, model_id)
            paper_sets.append(set(frame["paper_id"].astype(str)))
        if not paper_sets:
            return frozenset()
        return frozenset(set.intersection(*paper_sets))

    def composition_irr(
        self,
        scope_id: str,
        left_model: str,
        right_model: str,
        common_paper_ids: frozenset[str] | set[str] | None = None,
    ) -> dict:
        """Calculate exact agreement and nominal alpha on the balanced papers."""

        if common_paper_ids is None:
            common_paper_ids = self._balanced_composition_ids(scope_id)
        units = self.composition_irr_units(
            scope_id,
            left_model,
            right_model,
            common_paper_ids,
        )
        dimensions = []
        for column, label, classification in DISPLAY_IRR_DIMENSIONS:
            left_series = units[f"left__{column}"].fillna("").astype(str).str.strip()
            right_series = units[f"right__{column}"].fillna("").astype(str).str.strip()
            left_values = left_series.tolist()
            right_values = right_series.tolist()
            exact = pairwise_percent_agreement(left_values, right_values)
            alpha = krippendorff_alpha_nominal(
                [list(pair) for pair in zip(left_values, right_values)]
            )
            excluded = IRR_UNOBSERVED_VALUES.get(column, frozenset())
            left_observed = left_series.ne("") & ~left_series.isin(excluded)
            right_observed = right_series.ne("") & ~right_series.isin(excluded)

            observability_left = left_observed.map(
                {True: "observed", False: "unobserved"}
            ).tolist()
            observability_right = right_observed.map(
                {True: "observed", False: "unobserved"}
            ).tolist()
            observability_exact = pairwise_percent_agreement(
                observability_left,
                observability_right,
            )
            observability_alpha = krippendorff_alpha_nominal(
                [
                    list(pair)
                    for pair in zip(observability_left, observability_right)
                ]
            )

            jointly_observed = left_observed & right_observed
            observed_left_values = left_series.loc[jointly_observed].tolist()
            observed_right_values = right_series.loc[jointly_observed].tolist()
            observed_category_exact = pairwise_percent_agreement(
                observed_left_values,
                observed_right_values,
            )
            observed_category_alpha = krippendorff_alpha_nominal(
                [
                    list(pair)
                    for pair in zip(observed_left_values, observed_right_values)
                ]
            )
            dimensions.append(
                {
                    "column": column,
                    "label": label,
                    "classification": classification,
                    "comparable_papers": exact.comparable,
                    "agreements": exact.agreements,
                    "disagreements": exact.comparable - exact.agreements,
                    "percent_agreement": exact.percent_agreement,
                    "krippendorff_alpha": alpha,
                    "observability_comparable_papers": observability_exact.comparable,
                    "observability_agreements": observability_exact.agreements,
                    "observability_percent_agreement": observability_exact.percent_agreement,
                    "observability_krippendorff_alpha": observability_alpha,
                    "jointly_observed_papers": observed_category_exact.comparable,
                    "observed_category_agreements": observed_category_exact.agreements,
                    "observed_category_percent_agreement": observed_category_exact.percent_agreement,
                    "observed_category_krippendorff_alpha": observed_category_alpha,
                }
            )
        return {
            "scope": scope_id,
            "left_model": left_model,
            "left_label": MODEL_DISPLAY_NAMES.get(left_model, left_model),
            "right_model": right_model,
            "right_label": MODEL_DISPLAY_NAMES.get(right_model, right_model),
            "intersection_papers": len(units),
            "balanced_common_papers": len(common_paper_ids),
            "dimensions": dimensions,
            "study_status_filter_applied": False,
        }

    def composition_irr_matrix(self, scope_id: str) -> dict:
        """Calculate every available model pair for one dataset scope.

        The two matrix summaries are arithmetic means across the six locked
        dimensions. Dimension-level estimates remain the inferential record;
        the means are compact orientation measures for the matrix display.
        """

        models = self._irr_composition_models()
        model_ids = [item["id"] for item in models]
        balanced_ids = self._balanced_composition_ids(scope_id, model_ids)
        pairs = []
        for left, right in combinations(models, 2):
            result = self.composition_irr(
                scope_id,
                left["id"],
                right["id"],
                balanced_ids,
            )
            core_dimensions = [
                row for row in result["dimensions"] if row["classification"] == "Core"
            ]
            agreements = [
                row["percent_agreement"]
                for row in core_dimensions
                if row["percent_agreement"] is not None
            ]
            alphas = [
                row["krippendorff_alpha"]
                for row in core_dimensions
                if row["krippendorff_alpha"] is not None
            ]
            result["mean_percent_agreement"] = (
                sum(agreements) / len(agreements) if agreements else None
            )
            result["mean_krippendorff_alpha"] = (
                sum(alphas) / len(alphas) if alphas else None
            )
            pairs.append(result)
        return {
            "scope": scope_id,
            "models": models,
            "dimensions": [
                {"column": column, "label": label, "classification": classification}
                for column, label, classification in DISPLAY_IRR_DIMENSIONS
            ],
            "pairs": pairs,
            "dimension_count": len(DISPLAY_IRR_DIMENSIONS),
            "core_dimension_count": len(CORE_IRR_DIMENSIONS),
            "study_status_filter_applied": False,
            "summary_method": "arithmetic mean across the six estimable core dimensions",
            "balanced_common_papers": len(balanced_ids),
            "reference_model": self._comparison_reference_model(),
            "reference_label": MODEL_DISPLAY_NAMES.get(
                self._comparison_reference_model(),
                self._comparison_reference_model(),
            ),
            "reference_cohort_papers": len(self._comparison_reference_ids()),
            "comparison_cohort": self._comparison_config.get("comparison_id", ""),
            "cohort_rule": self._comparison_config.get("cohort_rule", ""),
            "irr_rule": self._comparison_config.get("irr_rule", ""),
        }

    def primary_coder_robustness(
        self,
        reference_model: str | None = None,
        comparison_model: str | None = None,
        min_support: int = 20,
    ) -> dict:
        """Re-estimate the five registered findings with any two available coders.

        The default remains the manuscript's Mini-to-Gemini check. Model
        selection changes only the source of the paper-level classifications;
        populations, observed-category rules, selected contrasts, and the
        support threshold remain constant.
        """

        registered_primary = resolve_primary_model()
        model_rows = self.composition_models()
        available = {row["id"] for row in model_rows}
        reference_model = reference_model or registered_primary
        if reference_model not in available:
            raise ValueError(f"Unknown robustness reference model: {reference_model}")
        if comparison_model is None:
            preferred = "gemini-3.1-pro-preview"
            comparison_model = (
                preferred
                if preferred in available and preferred != reference_model
                else next(
                    (row["id"] for row in model_rows if row["id"] != reference_model),
                    "",
                )
            )
        if comparison_model not in available:
            raise ValueError(f"Unknown robustness comparison model: {comparison_model}")
        if comparison_model == reference_model:
            raise ValueError("Coder robustness requires two different models")
        if min_support < 1:
            raise ValueError("Coder robustness support threshold must be positive")

        result = build_coder_robustness(
            self._composition_model_frame(reference_model),
            self._composition_model_frame(comparison_model),
            primary_model=reference_model,
            primary_label=MODEL_DISPLAY_NAMES.get(reference_model, reference_model),
            alternative_model=comparison_model,
            alternative_label=MODEL_DISPLAY_NAMES.get(
                comparison_model, comparison_model
            ),
            min_support=min_support,
        )
        result.update(
            {
                "reference_model": result["primary_model"],
                "comparison_model": result["alternative_model"],
                "registered_primary_model": {
                    "id": registered_primary,
                    "label": MODEL_DISPLAY_NAMES.get(
                        registered_primary, registered_primary
                    ),
                },
                "reference_is_registered_primary": reference_model
                == registered_primary,
                "available_models": model_rows,
            }
        )
        return result

    @staticmethod
    def _coder_robustness_population(
        frame: pd.DataFrame,
        population: str,
    ) -> pd.DataFrame:
        """Restrict a model frame to one entrepreneurship population."""

        core = frame["in_query_3"].astype(str).str.strip().str.lower().isin(
            {"1", "true", "yes", "y", "x"}
        )
        additional = frame["in_query_4"].astype(str).str.strip().str.lower().isin(
            {"1", "true", "yes", "y", "x"}
        )
        masks = {
            "core": core,
            "additional": additional,
            "combined": core | additional,
        }
        if population not in masks:
            raise ValueError(f"Unknown robustness population: {population}")
        return frame[masks[population]].copy()

    def _selected_model_pattern_agreement(
        self,
        paper_ids: list[str],
        filters: dict[str, str],
        model_ids: tuple[str, str],
    ) -> dict[str, dict]:
        """Return exact pattern assignments for one selected model pair."""

        all_agreement = self._model_pattern_agreement(paper_ids, filters)
        selected = set(model_ids)
        result: dict[str, dict] = {}
        for paper_id, detail in all_agreement.items():
            assignments = [
                row for row in detail["assignments"] if row["model"] in selected
            ]
            assignments.sort(key=lambda row: model_ids.index(row["model"]))
            agreeing = [row for row in assignments if row["matches"]]
            available = [row for row in assignments if row["available"]]
            result[paper_id] = {
                "pattern": detail["pattern"],
                "models_total": len(model_ids),
                "models_available": len(available),
                "models_agreeing": len(agreeing),
                "agreement_models": [
                    {"id": row["model"], "label": row["label"]}
                    for row in agreeing
                ],
                "all_models_agree": len(agreeing) == len(model_ids),
                "preferred_agreement_label": "Selected-model pair",
                "preferred_models_total": 0,
                "preferred_models_agreeing": 0,
                "preferred_agreement_models": [],
                "preferred_sweet_spot": False,
                "assignments": assignments,
            }
        return result

    def coder_robustness_evidence(
        self,
        reference_model: str,
        comparison_model: str,
        population: str,
        column: str,
        value: str,
        limit: int = 100,
        secondary_column: str | None = None,
        secondary_value: str | None = None,
        match_mode: str = "either",
    ) -> dict:
        """Return paper evidence behind one dynamic coder-robustness result."""

        if reference_model == comparison_model:
            raise ValueError("Coder robustness evidence requires two different models")
        allowed_columns = {
            AI_STUDY_STATUS_COLUMN,
            *(panel["column"] for panel in OBSERVED_COMPOSITION_PANELS),
        }
        if column not in allowed_columns:
            raise ValueError(f"Unknown robustness evidence column: {column}")
        if bool(secondary_column) != (secondary_value is not None):
            raise ValueError(
                "Secondary evidence column and value must be supplied together"
            )
        if secondary_column and secondary_column not in allowed_columns:
            raise ValueError(
                f"Unknown secondary robustness evidence column: {secondary_column}"
            )
        if match_mode not in {"either", "both", "reference", "comparison", "different"}:
            raise ValueError(f"Unknown robustness evidence match mode: {match_mode}")

        model_frames = {
            "reference": self._coder_robustness_population(
                self._composition_model_frame(reference_model), population
            ),
            "comparison": self._coder_robustness_population(
                self._composition_model_frame(comparison_model), population
            ),
        }
        filters = {column: str(value).strip()}
        if secondary_column:
            filters[secondary_column] = str(secondary_value).strip()

        supporting_ids: dict[str, set[str]] = {}
        for role, frame in model_frames.items():
            mask = pd.Series(True, index=frame.index)
            for filter_column, expected in filters.items():
                mask &= frame[filter_column].astype(str).str.strip().eq(expected)
            supporting_ids[role] = set(frame.loc[mask, "paper_id"].astype(str))

        reference_ids = supporting_ids["reference"]
        comparison_ids = supporting_ids["comparison"]
        both_ids = reference_ids & comparison_ids
        either_ids = reference_ids | comparison_ids
        selected_ids = {
            "either": either_ids,
            "both": both_ids,
            "reference": reference_ids,
            "comparison": comparison_ids,
            "different": either_ids - both_ids,
        }[match_mode]

        reference_frame = model_frames["reference"]
        comparison_frame = model_frames["comparison"]
        base = reference_frame[
            reference_frame["paper_id"].astype(str).isin(selected_ids)
        ].copy()
        present = set(base["paper_id"].astype(str))
        if present != selected_ids:
            base = pd.concat(
                [
                    base,
                    comparison_frame[
                        comparison_frame["paper_id"].astype(str).isin(
                            selected_ids - present
                        )
                    ],
                ],
                ignore_index=True,
            )
        base["_robustness_rank"] = base["paper_id"].astype(str).map(
            lambda paper_id: (
                0 if paper_id in both_ids else 1,
                0 if paper_id in reference_ids else 1,
            )
        )
        base = base.sort_values("_robustness_rank", kind="stable")
        paper_ids = base["paper_id"].astype(str).tolist()
        agreement = self._selected_model_pattern_agreement(
            paper_ids,
            filters,
            (reference_model, comparison_model),
        )
        records = self._paper_inspection_records(
            base,
            selected_columns=tuple(filters),
            limit=limit,
        )
        for record in records:
            record["_model_agreement"] = agreement.get(
                str(record.get("paper_id")), {}
            )
        population_labels = {
            "core": "Leading entrepreneurship journals",
            "additional": "Additional entrepreneurship journals",
            "combined": "Combined entrepreneurship",
        }
        return {
            "reference_model": {
                "id": reference_model,
                "label": MODEL_DISPLAY_NAMES.get(reference_model, reference_model),
            },
            "comparison_model": {
                "id": comparison_model,
                "label": MODEL_DISPLAY_NAMES.get(comparison_model, comparison_model),
            },
            "population": population,
            "population_label": population_labels[population],
            "match_mode": match_mode,
            "pattern": next(iter(agreement.values()), {}).get("pattern", []),
            "reference_supporting_papers": len(reference_ids),
            "comparison_supporting_papers": len(comparison_ids),
            "both_supporting_papers": len(both_ids),
            "either_supporting_papers": len(either_ids),
            "different_supporting_papers": len(either_ids - both_ids),
            "selected_papers": len(selected_ids),
            "returned_papers": len(records),
            "papers": records,
        }

    def export_scope(
        self, scope_id: str, filters: dict[str, str] | None = None
    ) -> pd.DataFrame:
        """Return a copy of one scope, optionally restricted by exact filters."""

        frame = self._scope(scope_id)
        for column, value in (filters or {}).items():
            if column not in frame.columns:
                raise ValueError(f"Unknown export filter column: {column}")
            frame = frame[frame[column].astype(str) == str(value)]
        return frame.reset_index(drop=True).copy()

    # ---- overview -------------------------------------------------------
    def scopes(self) -> list[dict]:
        """Return one shared dataset-scope registry for every public page."""

        scopes = list(DATASET_SCOPES)
        retrieval_definitions = {
            "full_corpus": (
                "All papers in the retained analytical corpus. This baseline "
                "includes papers outside the selected business-domain rows."
            ),
            "query_1": (
                "Papers retrieved through the broad business and management "
                "journal query."
            ),
            "query_2": (
                "Papers from source titles in the prespecified FT50 journal set."
            ),
            "query_3": (
                "Papers from the 15 represented Leading entrepreneurship source titles."
            ),
            "query_4": (
                "Papers from the 13 represented Additional entrepreneurship journal source titles."
            ),
        }
        rows = [
            {
                "id": scope.id,
                "label": scope.label,
                "papers": len(scope_frame(self.papers, scope.id)),
                "definition": retrieval_definitions.get(
                    scope.id,
                    "Papers inherited from this prespecified retrieval scope.",
                ),
                "scope_type": "retrieval_scope",
            }
            for scope in scopes
        ]
        existing = {row["id"] for row in rows}
        aliases_already_present = {
            "broad_business_management",
            "ft50",
            "core_entrepreneurship",
            "additional_entrepreneurship",
        }
        business_domain_ids = set(self._business_domain_manifest().get("domains", {}))
        for population_id, label, definition, frame, exclusive in (
            self._publication_growth_population_frames()
        ):
            if population_id in existing or population_id in aliases_already_present:
                continue
            if population_id in business_domain_ids:
                scope_type = "business_domain"
            elif population_id == "outside_selected_business_domains":
                scope_type = "analytical_residual"
            elif exclusive:
                scope_type = "exclusive_complement"
            else:
                scope_type = "analytical_population"
            rows.append(
                {
                    "id": population_id,
                    "label": label,
                    "papers": len(frame),
                    "definition": definition,
                    "scope_type": scope_type,
                }
            )
            existing.add(population_id)
        return rows

    def analytics_scopes(self) -> list[dict]:
        """Compatibility alias for the shared public dataset-scope registry."""

        return self.scopes()

    def scope_label(self, scope_id: str) -> str:
        """Resolve a public label for canonical and Analytics-only scopes."""

        canonical = next(
            (row["label"] for row in self.scopes() if row["id"] == scope_id),
            None,
        )
        if canonical is not None:
            return canonical
        for population_id, label, _definition, _frame, _exclusive in (
            self._publication_growth_population_frames()
        ):
            if population_id == scope_id:
                return label
        raise ScopeError(f"Unknown scope {scope_id!r}")

    def scope_overview(self, scope_id: str) -> dict:
        frame = self._scope(scope_id)
        conv = group_convergence(frame, group_label=scope_id)
        return {
            "scope": scope_id,
            "paper_count": len(frame),
            "has_specifications": self.has_specifications,
            "overall_specification_clarity_score": conv.overall_specification_clarity_score,
            "fragmentation_score": conv.fragmentation_score,
            # Clear public names; legacy keys above remain for saved clients.
            "mean_concentration_score": conv.overall_specification_clarity_score,
            "mean_dispersion_score": conv.fragmentation_score,
            "dimension_convergence": conv.dimension_scores,
            "dominant_values": conv.dominant_values,
            "dimension_profiles": {
                column: dimension_profile(frame, column)
                for column in DIMENSION_COLUMNS
                if column in frame.columns
            },
            "concentration_definition": (
                "Code concentration = 1 minus normalized Shannon entropy, displayed "
                "as a percentage (C = 1 - H/log2(k), where H = -sum[p log2(p)] "
                "and k is the number of categories present). 100% means every coded "
                "paper uses one category; 0% means papers are evenly distributed "
                "across the categories present. This describes a distribution, not "
                "clarity, validity, or quality."
            ),
        }

    def dimension_profile(self, scope_id: str, column: str) -> dict:
        """Return the complete distribution behind one concentration score."""

        if column not in DIMENSION_COLUMNS:
            raise ValueError(f"Unknown specification dimension: {column}")
        result = dimension_profile(self._scope(scope_id), column)
        result["scope"] = scope_id
        return result

    def dimension_distribution(self, scope_id: str, column: str) -> dict:
        frame = self._scope(scope_id)
        if column not in frame.columns:
            return {"scope": scope_id, "column": column, "values": []}
        counts = (
            frame[frame[column].astype(str).str.strip() != ""]
            .groupby(column)["paper_id"]
            .agg(list)
        )
        values = [
            {"value": value, "count": len(ids), "paper_ids": ids}
            for value, ids in counts.sort_values(key=lambda s: s.map(len), ascending=False).items()
        ]
        return {"scope": scope_id, "column": column, "values": values}

    # ---- aggregate views (journal / author / topic) ---------------------
    def group_convergence_table(
        self,
        scope_id: str,
        group_column: str,
        min_papers: int = 2,
    ) -> list[dict]:
        frame = self._scope(scope_id)
        if group_column not in frame.columns or not self.has_specifications:
            return []
        return convergence_by(frame, group_column, min_papers=min_papers).to_dict("records")

    # ---- performance analysis (productivity + impact) ------------------
    def performance(self, scope_id: str, top_n: int = 15) -> dict:
        """Bibliometric performance metrics for one scope.

        Complements the science-mapping (knowledge graph) side with the
        productivity and impact side: annual output, most productive journals
        and authors, citation impact, and the most cited papers.
        """

        frame = self._scope(scope_id)
        citations = self._numeric(frame, "Cited by")
        years = self._numeric(frame, "Year")
        total_cites = int(citations.sum())

        summary = {
            "papers": len(frame),
            "total_citations": total_cites,
            "mean_citations": round(citations.mean(), 2) if len(frame) else 0.0,
            "median_citations": float(citations.median()) if len(frame) else 0.0,
            "cited_share": round(float((citations > 0).mean()), 4) if len(frame) else 0.0,
            "year_min": int(years[years > 0].min()) if (years > 0).any() else None,
            "year_max": int(years[years > 0].max()) if (years > 0).any() else None,
        }

        annual = []
        valid_trend_year = years > 0
        if valid_trend_year.any():
            work = frame.assign(_y=years, _c=citations)
            grouped = {
                int(year): sub
                for year, sub in work[valid_trend_year].groupby("_y")
            }
            cumulative_papers = 0
            first_year = min(grouped)
            last_year = max(grouped)
            for year in range(first_year, last_year + 1):
                sub = grouped.get(year)
                papers = len(sub) if sub is not None else 0
                cohort_citations = int(sub["_c"].sum()) if sub is not None else 0
                cumulative_papers += papers
                annual.append({
                    "year": year,
                    "papers": papers,
                    "cumulative_papers": cumulative_papers,
                    "citations": cohort_citations,
                    "publication_year_after_retrieval_year": (
                        year > SEARCH_CUTOFF_YEAR
                    ),
                })

        publication_growth = [
            {
                **record,
                "end_is_retrieval_year": (
                    record["end_year"] == SEARCH_CUTOFF_YEAR
                ),
            }
            for record in growth_records(frame, DEFAULT_GROWTH_PERIODS)
        ]

        top_journals = self._top_counts(frame, "Source title", citations, top_n)
        top_authors = self._top_authors(frame, top_n)
        most_cited = self._most_cited(frame, citations, top_n)

        return {
            "scope": scope_id,
            "citation_definition": (
                "Current cumulative citations grouped by publication year; "
                "this is publication-cohort impact, not citations received during each year."
            ),
            "publication_trend_definition": (
                "Annual papers are publication counts for each year. "
                "Cumulative papers are the running total through that year."
            ),
            "growth_definition": (
                "Cumulative growth equals (end cumulative papers minus start "
                "cumulative papers) divided by start cumulative papers. It is "
                "not annual publication-output growth."
            ),
            "search_cutoff": {
                "date": SEARCH_CUTOFF_DATE.isoformat(),
                "label": SEARCH_CUTOFF_LABEL,
                "year": SEARCH_CUTOFF_YEAR,
            },
            "trend_reconciliation": {
                "scope_papers": len(frame),
                "valid_year_papers": int(valid_trend_year.sum()),
                "invalid_year": int((years.isna() | (years <= 0)).sum()),
                "records_dated_after_retrieval_year": int(
                    (years > SEARCH_CUTOFF_YEAR).sum()
                ),
                "final_cumulative_papers": (
                    annual[-1]["cumulative_papers"] if annual else 0
                ),
                "matches_scope_papers": (
                    bool(annual)
                    and annual[-1]["cumulative_papers"] == len(frame)
                ),
            },
            "summary": summary,
            "annual_production": annual,
            "publication_growth": publication_growth,
            "top_journals": top_journals,
            "top_authors": top_authors,
            "most_cited": most_cited,
        }

    def performance_papers(
        self,
        scope_id: str,
        year: int,
        mode: str = "annual",
        limit: int = 100,
    ) -> dict:
        """Return the papers behind one annual or cumulative chart point."""

        if mode not in {"annual", "cumulative"}:
            raise ValueError("Publication-paper mode must be annual or cumulative")
        frame = self._scope(scope_id).copy()
        years = self._numeric(frame, "Year")
        if mode == "annual":
            subset = frame[years == year].copy()
        else:
            subset = frame[(years > 0) & (years <= year)].copy()
        subset["_year_sort"] = self._numeric(subset, "Year")
        subset["_citation_sort"] = self._numeric(subset, "Cited by")
        subset["_title_sort"] = (
            subset["Title"].astype(str)
            if "Title" in subset.columns
            else pd.Series("", index=subset.index)
        )
        subset = subset.sort_values(
            ["_year_sort", "_citation_sort", "_title_sort"],
            ascending=[False, False, True],
            kind="stable",
        )
        papers = self._paper_inspection_records(subset, limit=limit)
        return {
            "scope": scope_id,
            "year": year,
            "mode": mode,
            "total_papers": len(subset),
            "returned_papers": min(limit, len(subset)),
            "papers": papers,
            "search_cutoff": {
                "date": SEARCH_CUTOFF_DATE.isoformat(),
                "label": SEARCH_CUTOFF_LABEL,
                "year": SEARCH_CUTOFF_YEAR,
            },
        }

    def publication_growth_comparison(self) -> dict:
        """Compare cumulative publication growth across scopes and populations."""

        rows = []
        for scope in (
            item for item in self.scopes()
            if item["scope_type"] == "retrieval_scope"
        ):
            performance = self.performance(scope["id"])
            rows.append(
                {
                    "scope": scope["id"],
                    "label": scope["label"],
                    "papers": scope["papers"],
                    "growth": performance["publication_growth"],
                }
            )
        population_frames = self._publication_growth_population_frames()
        populations = []
        for (
            population_id,
            label,
            definition,
            population,
            exclusive,
        ) in population_frames:
            populations.append(
                {
                    "id": population_id,
                    "label": label,
                    "definition": definition,
                    "papers": len(population),
                    "exclusive_complement": exclusive,
                    "growth": growth_records(population),
                    "trace": cumulative_trace(
                        population,
                        min(start for start, _ in DEFAULT_GROWTH_PERIODS),
                        max(end for _, end in DEFAULT_GROWTH_PERIODS),
                    ),
                }
            )

        return {
            "search_cutoff": {
                "date": SEARCH_CUTOFF_DATE.isoformat(),
                "label": SEARCH_CUTOFF_LABEL,
                "year": SEARCH_CUTOFF_YEAR,
            },
            "growth_definition": (
                "Cumulative growth equals (end cumulative papers minus start "
                "cumulative papers) divided by start cumulative papers."
            ),
            "scopes_overlap": True,
            "views": rows,
            "population_overlap_note": (
                "Registered business domains may overlap. Full corpus excluding "
                "Combined entrepreneurship is the exclusive entrepreneurship "
                "complement; it is not the outside-domain residual."
            ),
            "populations": populations,
        }

    def _publication_growth_population_frames(
        self,
    ) -> list[tuple[str, str, str, pd.DataFrame, bool]]:
        """Return the selectable populations behind the publication chart."""

        frame = self.papers.copy()
        def flag(column: str) -> pd.Series:
            return pd.to_numeric(
                frame.get(column, pd.Series(0, index=frame.index)),
                errors="coerce",
            ).fillna(0).eq(1)

        query_1 = flag("in_query_1")
        query_2 = flag("in_query_2")
        query_3 = flag("in_query_3")
        query_4 = flag("in_query_4")
        combined_mask = query_3 | query_4
        population_frames: list[tuple[str, str, str, pd.DataFrame, bool]] = [
            (
                "full_corpus",
                "Full corpus",
                "All papers in the frozen analytical corpus.",
                frame.copy(),
                False,
            ),
            (
                "broad_business_management",
                "Broad business and management journals",
                "Papers retrieved through the registered broad business and management journal query.",
                frame.loc[query_1].copy(),
                False,
            ),
            (
                "ft50",
                "FT50 journals",
                "Papers from source titles in the registered FT50 journal set.",
                frame.loc[query_2].copy(),
                False,
            ),
            (
                "core_entrepreneurship",
                "Leading entrepreneurship journals",
                "Papers from the prespecified leading entrepreneurship journal population.",
                frame.loc[query_3].copy(),
                False,
            ),
            (
                "additional_entrepreneurship",
                "Additional entrepreneurship journals",
                "Papers from the prespecified additional entrepreneurship journal population.",
                frame.loc[query_4].copy(),
                False,
            ),
            (
                "combined_entrepreneurship",
                "Combined entrepreneurship",
                "Union of the Leading and Additional entrepreneurship journal populations.",
                frame.loc[combined_mask].copy(),
                False,
            ),
            (
                "close_reading",
                "Systematic close-reading set",
                "The fixed 136-paper interpretive dataset: 124 papers from "
                "Leading and Additional entrepreneurship journals plus 12 "
                "cross-domain contrasts. It supports close interpretation and "
                "counterexample analysis; it does not estimate corpus prevalence.",
                frame.loc[
                    frame["paper_id"].astype(str).isin(self._close_reading_ids())
                ].copy(),
                False,
            ),
            (
                "remaining_full_corpus",
                "Full corpus excluding Combined entrepreneurship",
                f"The {len(frame) - int(combined_mask.sum()):,} papers remaining after "
                "the exact Combined entrepreneurship population is removed from the "
                f"{len(frame):,}-paper full corpus.",
                frame.loc[~combined_mask].copy(),
                True,
            ),
        ]
        assignments = self._theory_domain_assignments(frame)
        registered_query_domains = {
            "ft50",
            "core_entrepreneurship",
            "other_entrepreneurship",
        }
        domain_manifest = self._business_domain_manifest().get("domains", {})
        for domain_id, assigned in assignments.groupby("domain_id", sort=False):
            domain_id = str(domain_id)
            if domain_id in registered_query_domains:
                continue
            domain_frame = frame[
                frame["paper_id"].isin(set(assigned["paper_id"]))
            ].copy()
            assignment_mode = str(
                domain_manifest.get(domain_id, {}).get("mapping_mode", "")
            )
            definition = (
                "Papers inherit this domain through the disclosed reviewed "
                "source-title overlay."
                if assignment_mode == "reviewed_source_overlay"
                else "Papers inherit this domain through the explicit official "
                "Scopus ASJC-code aggregation."
            )
            population_frames.append(
                (
                    domain_id,
                    str(assigned["domain_label"].iloc[0]),
                    definition,
                    domain_frame,
                    False,
                )
            )

        business_domain_ids = set(domain_manifest)
        assigned_business_ids = set(
            assignments.loc[
                assignments["domain_id"].isin(business_domain_ids), "paper_id"
            ].astype(str)
        )
        outside_domains = frame[
            ~frame["paper_id"].astype(str).isin(assigned_business_ids)
        ].copy()
        if business_domain_ids:
            population_frames.append(
                (
                    "outside_selected_business_domains",
                    "Outside selected business-domain rows",
                    "Papers with official Scopus ASJC classifications that do not "
                    f"map to any of the {len(business_domain_ids)} selected business "
                    "domains. These papers "
                    "remain part of the full-corpus comparison baseline.",
                    outside_domains,
                    True,
                )
            )

        return population_frames

    def _publication_growth_population_frame(
        self, population_id: str
    ) -> tuple[str, pd.DataFrame]:
        """Resolve one publication-chart population by its stable ID."""

        for current_id, label, _definition, frame, _exclusive in (
            self._publication_growth_population_frames()
        ):
            if current_id == population_id:
                return label, frame
        raise ValueError(f"Unknown publication-growth population: {population_id}")

    def publication_growth_population_papers(
        self,
        population_id: str,
        year: int,
        mode: str = "annual",
        limit: int = 100,
    ) -> dict:
        """Return evidence papers behind a publication-population chart point."""

        if mode not in {"annual", "cumulative"}:
            raise ValueError("Publication-paper mode must be annual or cumulative")
        label, frame = self._publication_growth_population_frame(population_id)
        years = self._numeric(frame, "Year")
        subset = (
            frame[years == year].copy()
            if mode == "annual"
            else frame[(years > 0) & (years <= year)].copy()
        )
        subset["_year_sort"] = self._numeric(subset, "Year")
        subset["_citation_sort"] = self._numeric(subset, "Cited by")
        subset["_title_sort"] = (
            subset["Title"].astype(str)
            if "Title" in subset.columns
            else pd.Series("", index=subset.index)
        )
        subset = subset.sort_values(
            ["_year_sort", "_citation_sort", "_title_sort"],
            ascending=[False, False, True],
            kind="stable",
        )
        return {
            "population": population_id,
            "population_label": label,
            "year": year,
            "mode": mode,
            "total_papers": len(subset),
            "returned_papers": min(limit, len(subset)),
            "papers": self._paper_inspection_records(subset, limit=limit),
            "search_cutoff": {
                "date": SEARCH_CUTOFF_DATE.isoformat(),
                "label": SEARCH_CUTOFF_LABEL,
                "year": SEARCH_CUTOFF_YEAR,
            },
        }

    def publication_growth_population_keyword_year(
        self,
        population_id: str,
        source: str,
        year: int,
        limit: int = 20,
    ) -> dict:
        """Return top keywords for one publication-chart population and year."""

        label, frame = self._publication_growth_population_frame(population_id)
        result = keyword_year_summary(
            frame, source=source, year=year, limit=limit
        )
        result.update({"population": population_id, "population_label": label})
        return result

    def publication_growth_population_keyword_evidence(
        self,
        population_id: str,
        source: str,
        keyword: str,
        year: int,
        limit: int = 100,
    ) -> list[dict]:
        """Return papers behind one population-year keyword count."""

        _label, frame = self._publication_growth_population_frame(population_id)
        mask = keyword_evidence_mask(
            frame, source, keyword, None, year=year
        )
        return self._paper_inspection_records(frame[mask], limit=limit)

    # ---- keyword evolution --------------------------------------------
    def keyword_evolution(
        self,
        scope_id: str,
        source: str = "author",
        series_top_n: int = 10,
        period_top_n: int = 20,
    ) -> dict:
        """Return period prevalence and movement for one keyword source."""

        result = analyze_keyword_evolution(
            self._scope(scope_id),
            source=source,
            series_top_n=series_top_n,
            period_top_n=period_top_n,
        )
        result["scope"] = scope_id
        return result

    def keyword_evidence(
        self,
        scope_id: str,
        source: str,
        keyword: str,
        period_id: str | None = None,
        year: int | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return papers supporting one period- or year-keyword statistic."""

        frame = self._scope(scope_id)
        mask = keyword_evidence_mask(
            frame,
            source,
            keyword,
            period_id,
            year=year,
        )
        subset = frame[mask]
        return self._paper_inspection_records(subset, limit=limit)

    def keyword_search(
        self,
        scope_id: str,
        source: str,
        query: str,
        limit: int = 20,
    ) -> list[dict]:
        """Search all canonical keywords and include their period trajectories."""

        return search_keyword_series(
            self._scope(scope_id), source=source, query=query, limit=limit
        )

    def keyword_year_summary(
        self,
        scope_id: str,
        source: str,
        year: int,
        limit: int = 20,
    ) -> dict:
        """Return the leading canonical keywords for one publication year."""

        result = keyword_year_summary(
            self._scope(scope_id), source=source, year=year, limit=limit
        )
        result["scope"] = scope_id
        return result

    # ---- observed construct composition -------------------------------
    def observed_composition(
        self,
        scope_id: str,
        study_status: str = "all",
        model: str | None = None,
        filter_dimension: str | None = None,
        filter_value: str | None = None,
    ) -> dict:
        """Return full and observed composition under one optional dimension filter."""

        selected_model = model or resolve_primary_model()
        frame, corpus_scope_n = self._composition_scope(scope_id, selected_model)
        model_scope_n = len(frame)
        filter_options = self._composition_filter_options(frame)
        frame, control = self._theory_controlled_frame(
            frame, filter_dimension, filter_value
        )
        result = analyze_observed_composition(
            frame, study_status=study_status
        )
        result["scope"] = scope_id
        result["model"] = selected_model
        result["model_label"] = MODEL_DISPLAY_NAMES.get(selected_model, selected_model)
        result["corpus_scope_papers"] = corpus_scope_n
        result["model_scope_papers"] = model_scope_n
        result["model_missing_papers"] = max(0, corpus_scope_n - model_scope_n)
        result["model_coverage_share"] = (
            round(model_scope_n / corpus_scope_n, 6)
            if corpus_scope_n
            else 0.0
        )
        reference_model = self._comparison_reference_model()
        result["comparison_reference_model"] = reference_model
        result["comparison_reference_label"] = MODEL_DISPLAY_NAMES.get(
            reference_model,
            reference_model,
        ) if reference_model else "Full available corpus"
        result["comparison_cohort"] = self._comparison_config.get(
            "comparison_id", ""
        )
        result["control"] = control
        result["filter_options"] = filter_options
        return result

    def _composition_filter_options(self, frame: pd.DataFrame) -> list[dict]:
        """Return the shared eight-dimension filter contract and live values."""

        options = []
        for definition in THEORY_DIMENSIONS:
            column = dimension_column(frame, definition["id"])
            values = []
            if column in frame.columns:
                counts = (
                    frame[column].astype(str).str.strip().value_counts(dropna=False)
                )
                values = [
                    {
                        "value": str(value),
                        "label": "Missing value" if str(value) == "" else str(value),
                        "papers": int(count),
                    }
                    for value, count in counts.items()
                ]
            options.append(
                {
                    "id": definition["id"],
                    "label": definition["label"],
                    "column": column,
                    "values": values,
                }
            )
        return options

    def composition_relationship_matrix(
        self,
        scope_id: str,
        model: str | None,
        row_dimension: str,
        column_dimension: str,
        distribution_view: str = "observed",
        study_status: str = "all",
        filter_dimension: str | None = None,
        filter_value: str | None = None,
    ) -> dict:
        """Cross any two specification dimensions inside the active filters."""

        if row_dimension == column_dimension:
            raise ValueError("Matrix axes must use different dimensions")
        selected_model = model or resolve_primary_model()
        frame, corpus_scope_n = self._composition_scope(scope_id, selected_model)
        frame, control = self._theory_controlled_frame(
            frame, filter_dimension, filter_value
        )
        result = relationship_matrix(
            frame,
            row_dimension,
            column_dimension,
            distribution_view,
            study_status,
        )
        result.update(
            {
                "scope": scope_id,
                "model": selected_model,
                "model_label": MODEL_DISPLAY_NAMES.get(selected_model, selected_model),
                "corpus_scope_papers": corpus_scope_n,
                "control": control,
            }
        )
        return result

    def observed_composition_evidence(
        self,
        scope_id: str,
        study_status: str,
        column: str,
        value: str,
        limit: int = 100,
        model: str | None = None,
        filter_dimension: str | None = None,
        filter_value: str | None = None,
        secondary_column: str | None = None,
        secondary_value: str | None = None,
    ) -> list[dict]:
        """Return papers supporting one observed-composition bar."""

        return self.observed_composition_evidence_bundle(
            scope_id,
            study_status,
            column,
            value,
            limit,
            model,
            filter_dimension,
            filter_value,
            secondary_column,
            secondary_value,
        )["papers"]

    def observed_composition_evidence_bundle(
        self,
        scope_id: str,
        study_status: str,
        column: str,
        value: str,
        limit: int = 100,
        model: str | None = None,
        filter_dimension: str | None = None,
        filter_value: str | None = None,
        secondary_column: str | None = None,
        secondary_value: str | None = None,
        minimum_agreement: int = 1,
        preferred_only: bool = False,
    ) -> dict:
        """Return traceable evidence plus exact cross-model pattern agreement."""

        selected_model = model or resolve_primary_model()
        frame, _ = self._composition_scope(scope_id, selected_model)
        frame, control = self._theory_controlled_frame(
            frame, filter_dimension, filter_value
        )
        mask = observed_composition_evidence_mask(
            frame,
            study_status=study_status,
            column=column,
            value=value,
        )
        if bool(secondary_column) != (secondary_value is not None):
            raise ValueError(
                "Secondary evidence column and value must be supplied together"
            )
        if secondary_column:
            if secondary_column not in frame.columns:
                raise ValueError(f"Unknown evidence column: {secondary_column}")
            mask &= frame[secondary_column].astype(str).str.strip().eq(
                str(secondary_value).strip()
            )
        subset = frame[mask]
        agreement_filters: dict[str, str] = {}
        if study_status != "all":
            agreement_filters[AI_STUDY_STATUS_COLUMN] = study_status
        if control:
            agreement_filters[str(control["column"])] = str(control["value"])
        agreement_filters[column] = str(value)
        if secondary_column:
            agreement_filters[secondary_column] = str(secondary_value)
        return self._agreement_evidence_bundle(
            subset,
            agreement_filters,
            selected_columns=tuple(agreement_filters),
            limit=limit,
            minimum_agreement=minimum_agreement,
            preferred_only=preferred_only,
            reference_model=selected_model,
        )

    # ---- theory elaboration and construct contrasting -----------------
    def _theory_population_frame(
        self,
        frame: pd.DataFrame,
        population: str,
    ) -> pd.DataFrame:
        """Return one registered entrepreneurship population."""

        if population not in {item[0] for item in THEORY_POPULATIONS}:
            raise ValueError(f"Unknown entrepreneurship population: {population}")
        for column in ("in_query_3", "in_query_4"):
            if column not in frame.columns:
                return frame.iloc[0:0].copy()
        if population == "close_reading":
            return frame.loc[
                frame["paper_id"].astype(str).isin(self._close_reading_ids())
            ].copy()
        core = pd.to_numeric(frame["in_query_3"], errors="coerce").fillna(0).eq(1)
        other = pd.to_numeric(frame["in_query_4"], errors="coerce").fillna(0).eq(1)
        mask = core if population == "core" else other
        if population == "combined":
            mask = core | other
        return frame.loc[mask].copy()

    def _theory_journal_scope_frame(
        self,
        frame: pd.DataFrame,
        journal_scope: str,
    ) -> pd.DataFrame:
        """Apply the FT50 robustness restriction when requested."""

        if journal_scope not in {"all", "ft50"}:
            raise ValueError(f"Unknown journal scope: {journal_scope}")
        if journal_scope == "all":
            return frame.copy()
        if "in_query_2" not in frame.columns:
            return frame.iloc[0:0].copy()
        mask = pd.to_numeric(frame["in_query_2"], errors="coerce").fillna(0).eq(1)
        return frame.loc[mask].copy()

    def _theory_domain_assignments(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Load registered domains plus any reviewed ASJC-domain aggregation.

        The optional ASJC-derived table deliberately has the same long schema
        as the registered query-domain assignments. Until it exists, the API
        reports those domains as pending instead of inventing memberships.
        """

        rows = []
        for rule in REGISTERED_QUERY_DOMAIN_RULES:
            flag = rule["flag_column"]
            if flag not in frame.columns:
                continue
            mask = pd.to_numeric(frame[flag], errors="coerce").fillna(0).eq(1)
            selected = frame.loc[mask, ["paper_id"]].copy()
            selected["domain_id"] = rule["domain_id"]
            selected["domain_label"] = rule["domain_label"]
            selected["assignment_basis"] = rule["query_id"]
            rows.append(selected)

        derived_path = (
            self.processed_dir
            / "analysis/theory_elaboration/domains/business_domain_assignments.csv"
        )
        if derived_path.exists():
            derived = pd.read_csv(
                derived_path, dtype=str, keep_default_na=False
            ).fillna("")
            required = {"paper_id", "domain_id", "domain_label", "assignment_basis"}
            if not required.issubset(derived.columns):
                raise ValueError(
                    "Reviewed business-domain assignments are missing required columns"
                )
            derived = derived[
                ["paper_id", "domain_id", "domain_label", "assignment_basis"]
            ].copy()
            derived = derived[derived["paper_id"].isin(set(frame["paper_id"]))]
            rows.append(derived)

        if not rows:
            return pd.DataFrame(
                columns=["paper_id", "domain_id", "domain_label", "assignment_basis"]
            )
        assignments = pd.concat(rows, ignore_index=True)
        return assignments.drop_duplicates(["paper_id", "domain_id"]).reset_index(
            drop=True
        )

    def _business_domain_manifest(self) -> dict:
        """Return the frozen ASJC aggregation manifest when it is available."""

        path = (
            self.processed_dir
            / "analysis/theory_elaboration/domains/business_domain_manifest.json"
        )
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _theory_domain_coverage(
        self,
        frame: pd.DataFrame,
        assignments: pd.DataFrame,
    ) -> dict:
        """Audit coverage without treating the multi-label rows as additive."""

        manifest = self._business_domain_manifest()
        business_domain_ids = set(manifest.get("domains", {}))
        frame_ids = set(frame["paper_id"].astype(str))
        business_ids = set(
            assignments.loc[
                assignments["domain_id"].isin(business_domain_ids), "paper_id"
            ].astype(str)
        )
        all_registered_ids = set(assignments["paper_id"].astype(str))
        total = len(frame_ids)

        def share(value: int) -> float:
            return round(value / total * 100, 4) if total else 0.0

        return {
            "baseline_papers": total,
            "business_domain_count": len(business_domain_ids),
            "inside_selected_business_domains": len(business_ids),
            "inside_selected_business_domains_percent": share(len(business_ids)),
            "outside_selected_business_domains": len(frame_ids - business_ids),
            "outside_selected_business_domains_percent": share(
                len(frame_ids - business_ids)
            ),
            "inside_all_registered_groups": len(all_registered_ids),
            "inside_all_registered_groups_percent": share(len(all_registered_ids)),
            "outside_all_registered_groups": len(frame_ids - all_registered_ids),
            "outside_all_registered_groups_percent": share(
                len(frame_ids - all_registered_ids)
            ),
            "all_papers_have_official_asjc": bool(
                manifest.get("validation", {}).get(
                    "all_corpus_papers_have_official_asjc", False
                )
            ),
            "baseline_rule": str(
                manifest.get("aggregation", {}).get("baseline_rule", "")
            ).strip(),
            "residual_label": str(
                manifest.get("aggregation", {}).get(
                    "residual_label", "Outside selected analytical domains"
                )
            ),
        }

    def theory_contrasting_metadata(self, model: str | None = None) -> dict:
        """Describe available populations, domains, dimensions and model coverage."""

        selected_model = model or resolve_primary_model()
        frame = self._composition_model_frame(selected_model)
        assignments = self._theory_domain_assignments(frame)
        manifest = self._business_domain_manifest()
        manifest_domains = manifest.get("domains", {})
        domain_coverage = self._theory_domain_coverage(frame, assignments)
        domains = []
        query_domain_definitions = {
            "ft50": (
                "Papers from source titles in the registered FT50 journal set."
            ),
            "core_entrepreneurship": (
                "Papers from the prespecified leading entrepreneurship journal set."
            ),
            "other_entrepreneurship": (
                "Papers from the prespecified additional entrepreneurship journal set."
            ),
        }
        for domain_id, group in assignments.groupby("domain_id", sort=False):
            domain_id = str(domain_id)
            domain_manifest = manifest_domains.get(domain_id, {})
            paper_ids = set(group["paper_id"])
            source_counts = (
                frame.loc[frame["paper_id"].isin(paper_ids), "Source title"]
                .astype(str)
                .str.strip()
                .loc[lambda values: values.ne("")]
                .value_counts()
            )
            query_defined = domain_id in query_domain_definitions
            mapping_mode = str(domain_manifest.get("mapping_mode", ""))
            asjc_codes = dict(domain_manifest.get("asjc_codes", {}))
            registry_field = str(domain_manifest.get("registry_field", ""))
            if query_defined:
                assignment_basis = str(group["assignment_basis"].iloc[0])
                assignment_type = "Registered journal population"
                definition = query_domain_definitions[domain_id]
            elif mapping_mode == "official_asjc":
                assignment_basis = "official_scopus_asjc:" + ",".join(asjc_codes)
                assignment_type = "Official Scopus ASJC aggregation"
                definition = (
                    "Papers inherit this domain when their source carries at least "
                    "one of the explicitly registered official Scopus ASJC codes."
                )
            else:
                assignment_basis = f"reviewed_source_overlay:{registry_field}"
                assignment_type = "Reviewed source-title overlay"
                definition = (
                    "Papers inherit this domain through the disclosed reviewed "
                    "source-title overlay because no direct Scopus business-category "
                    "code represents this construct."
                )
            domains.append(
                {
                    "id": domain_id,
                    "label": str(group["domain_label"].iloc[0]),
                    "papers": int(group["paper_id"].nunique()),
                    "assignment_basis": assignment_basis,
                    "assignment_type": assignment_type,
                    "definition": definition,
                    "mapping_mode": mapping_mode,
                    "asjc_codes": asjc_codes,
                    "rationale": str(domain_manifest.get("rationale", "")),
                    "registry_field": registry_field,
                    "source_title_count": int(len(source_counts)),
                    "source_titles": [
                        {"title": str(title), "papers": int(count)}
                        for title, count in source_counts.items()
                    ],
                    "available": True,
                }
            )
        available_ids = {item["id"] for item in domains}
        pending = [
            {"id": domain_id, "label": label, "available": False}
            for domain_id, label in PENDING_ASJC_DOMAIN_LABELS.items()
            if domain_id not in available_ids
        ]
        populations = []
        for population_id, label in THEORY_POPULATIONS:
            population_frame = self._theory_population_frame(frame, population_id)
            populations.append(
                {
                    "id": population_id,
                    "label": label,
                    "papers": len(population_frame),
                }
            )
        pair_options = [
            {"id": f"{left}__{right}", "label": label}
            for left, right, label in STRUCTURING_PAIRS
        ]
        horizontal_controls = []
        for definition in THEORY_DIMENSIONS:
            control_column = dimension_column(frame, definition["id"])
            values = []
            if control_column in frame.columns:
                values = sorted(
                    value
                    for value in frame[control_column]
                    .astype(str)
                    .str.strip()
                    .unique()
                    if value
                )
            horizontal_controls.append(
                {
                    "id": definition["id"],
                    "label": definition["label"],
                    "column": control_column,
                    "values": values,
                }
            )
        if domain_coverage["all_papers_have_official_asjc"]:
            residual_explanation = (
                f"All {domain_coverage['baseline_papers']:,} papers in this model "
                "view have official Scopus ASJC classifications. "
                f"{domain_coverage['outside_selected_business_domains']:,} "
                f"({domain_coverage['outside_selected_business_domains_percent']:.2f}%) "
                f"do not map to one of the {domain_coverage['business_domain_count']} "
                "selected business-domain rows. "
                "This is an analytical residual, not missing Scopus classification. "
                "Those papers remain in the full-corpus baseline."
            )
        else:
            residual_explanation = (
                "The official ASJC aggregation manifest is not available in this "
                "runtime. Papers outside the available registered rows remain in "
                "the full-corpus baseline."
            )
        return {
            "model": selected_model,
            "model_label": MODEL_DISPLAY_NAMES.get(selected_model, selected_model),
            "model_coded_papers": len(frame),
            "corpus_papers": len(self.papers),
            "populations": populations,
            "domains": domains,
            "pending_domains": pending,
            "dimensions": [
                {key: definition[key] for key in ("id", "label", "column")}
                for definition in THEORY_DIMENSIONS
            ],
            "vertical_row_dimensions": list(VERTICAL_ROW_DIMENSIONS),
            "horizontal_controls": horizontal_controls,
            "structuring_pairs": pair_options,
            "domain_coverage": domain_coverage,
            "journal_scopes": [
                {"id": "all", "label": "All journals"},
                {"id": "ft50", "label": "FT50 robustness subset"},
            ],
            "domain_methodology": {
                "unit": "Paper inherited from the official classifications of its source journal",
                "construction": (
                    "Domains classify papers already present in the 22,345-paper "
                    "corpus. They do not retrieve or add papers."
                ),
                "classification": (
                    "FT50, Leading entrepreneurship journals, and Additional "
                    "entrepreneurship journals use prespecified journal populations. "
                    f"{domain_coverage['business_domain_count']} business domains are "
                    "aggregated from explicitly registered official Scopus ASJC "
                    "codes. No reviewed source-title overlay is included in the "
                    "primary horizontal comparison."
                ),
                "overlap": (
                    "Domain membership is multi-label. A journal, and therefore a "
                    "paper, may belong to more than one domain; rows must not be summed."
                ),
                "unclassified": (
                    residual_explanation
                ),
                "baseline": domain_coverage["baseline_rule"],
            },
            "domain_assignment_complete": not pending,
        }

    def theory_construct_specification(
        self,
        model: str,
        population: str,
        journal_scope: str = "all",
        study_status: str = "all",
    ) -> dict:
        """Return existing specification results for an entrepreneurship population."""

        frame = self._composition_model_frame(model)
        frame = self._theory_population_frame(frame, population)
        frame = self._theory_journal_scope_frame(frame, journal_scope)
        result = analyze_observed_composition(frame, study_status=study_status)
        population_label = dict(THEORY_POPULATIONS)[population]
        result.update(
            {
                "model": model,
                "model_label": MODEL_DISPLAY_NAMES.get(model, model),
                "population": population,
                "population_label": population_label,
                "journal_scope": journal_scope,
            }
        )
        return result

    def theory_horizontal_contrast(
        self,
        model: str,
        dimension_id: str,
        distribution_view: str = "observed",
        journal_scope: str = "all",
        study_status: str = "all",
        control_dimension: str | None = None,
        control_value: str | None = None,
    ) -> dict:
        """Compare one specification dimension across available domains."""

        frame = self._composition_model_frame(model)
        frame = self._theory_journal_scope_frame(frame, journal_scope)
        frame, control = self._theory_controlled_frame(
            frame, control_dimension, control_value
        )
        assignments = self._theory_domain_assignments(frame)
        baseline = theory_distribution(
            frame, dimension_id, distribution_view, study_status
        )
        baseline_frame = filter_study_status(frame, study_status)
        if distribution_view == "observed":
            baseline_frame = baseline_frame.loc[
                theory_observed_mask(baseline_frame, dimension_id)
            ].copy()
        baseline_assignments = assignments[
            assignments["paper_id"].isin(set(baseline_frame["paper_id"]))
        ].copy()
        baseline_domain_coverage = self._theory_domain_coverage(
            baseline_frame, baseline_assignments
        )
        baseline_shares = {
            item["raw_value"]: item["share"] for item in baseline["categories"]
        }
        metadata = self.theory_contrasting_metadata(model)
        domain_metadata = {item["id"]: item for item in metadata["domains"]}
        groups = []
        for domain_id, assigned in assignments.groupby("domain_id", sort=False):
            # Under an FT50-only replication, the FT50 group is identical to
            # the comparison baseline and therefore contains no contrast.
            if journal_scope == "ft50" and str(domain_id) == "ft50":
                continue
            domain_frame = frame[frame["paper_id"].isin(set(assigned["paper_id"]))]
            result = theory_distribution(
                domain_frame, dimension_id, distribution_view, study_status
            )
            for category in result["categories"]:
                category["percentage_point_difference"] = round(
                    (category["share"] - baseline_shares.get(category["raw_value"], 0.0))
                    * 100,
                    4,
                )
            groups.append(
                {
                    "id": str(domain_id),
                    "label": str(assigned["domain_label"].iloc[0]),
                    "assignment_basis": str(
                        domain_metadata.get(str(domain_id), {}).get(
                            "assignment_basis", assigned["assignment_basis"].iloc[0]
                        )
                    ),
                    "eligible": True,
                    "eligibility_note": "",
                    **result,
                }
            )

        # Keep registered rows visible when a restricted comparison corpus has
        # no intersecting papers. Absence is analytically meaningful and must
        # not look like a missing domain definition.
        group_ids = {group["id"] for group in groups}
        for domain in metadata["domains"]:
            domain_id = str(domain["id"])
            if journal_scope == "ft50" and domain_id == "ft50":
                continue
            if domain_id in group_ids:
                continue
            groups.append(
                {
                    "id": domain_id,
                    "label": str(domain["label"]),
                    "assignment_basis": str(domain.get("assignment_basis", "")),
                    "eligible": False,
                    "eligibility_note": (
                        "No papers from this group occur in the selected "
                        "comparison corpus."
                    ),
                    **theory_distribution(
                        frame.iloc[0:0],
                        dimension_id,
                        distribution_view,
                        study_status,
                    ),
                }
            )

        domain_order = {
            str(domain["id"]): index
            for index, domain in enumerate(metadata["domains"])
            if not (journal_scope == "ft50" and str(domain["id"]) == "ft50")
        }
        groups.sort(key=lambda group: domain_order.get(group["id"], len(domain_order)))

        combined_ids = set(
            assignments.loc[
                assignments["domain_id"].isin(
                    ["core_entrepreneurship", "other_entrepreneurship"]
                ),
                "paper_id",
            ]
        )
        if combined_ids:
            combined = theory_distribution(
                frame[frame["paper_id"].isin(combined_ids)],
                dimension_id,
                distribution_view,
                study_status,
            )
            for category in combined["categories"]:
                category["percentage_point_difference"] = round(
                    (category["share"] - baseline_shares.get(category["raw_value"], 0.0))
                    * 100,
                    4,
                )
            groups.append(
                {
                    "id": "combined_entrepreneurship",
                    "label": "Combined entrepreneurship",
                    "assignment_basis": "union of core and other entrepreneurship",
                    "eligible": True,
                    "eligibility_note": "",
                    **combined,
                }
            )
        return {
            "model": model,
            "model_label": MODEL_DISPLAY_NAMES.get(model, model),
            "dimension_id": dimension_id,
            "dimension_label": baseline["dimension_label"],
            "column": baseline["column"],
            "distribution": distribution_view,
            "journal_scope": journal_scope,
            "study_status": study_status,
            "control": control,
            "control_filter_label": (
                f"{control['dimension_label']} = {control['value']}"
                if control
                else "No additional dimension control"
            ),
            "baseline_label": (
                "FT50 corpus" if journal_scope == "ft50" else "Full corpus"
            ),
            "comparison_definition": (
                "Every domain row contains only papers that are also in the "
                "FT50 corpus and is compared with all eligible FT50 papers."
                if journal_scope == "ft50"
                else "Every domain row is compared with all eligible papers "
                "in the full corpus, including papers outside the selected "
                "domain rows."
            ),
            "baseline": baseline,
            "baseline_domain_coverage": baseline_domain_coverage,
            "groups": groups,
            "pending_domains": metadata["pending_domains"],
            "overlap_warning": (
                "Domain memberships can overlap; domain counts must not be summed."
            ),
            "entrepreneurship_comparison": self.theory_entrepreneurship_comparison(
                model,
                dimension_id,
                distribution_view,
                study_status,
                min_support=10,
                control_dimension=control_dimension,
                control_value=control_value,
            ),
        }

    def _theory_controlled_frame(
        self,
        frame: pd.DataFrame,
        control_dimension: str | None,
        control_value: str | None,
    ) -> tuple[pd.DataFrame, dict | None]:
        """Apply one optional registered specification control consistently."""

        dimension_id = str(control_dimension or "").strip()
        value = str(control_value or "").strip()
        if bool(dimension_id) != bool(value):
            raise ValueError(
                "Comparison control dimension and control value must be selected together"
            )
        if not dimension_id:
            return frame, None
        definition = next(
            (
                item
                for item in THEORY_DIMENSIONS
                if item["id"] == dimension_id
            ),
            None,
        )
        if definition is None:
            raise ValueError(f"Unknown comparison control dimension: {dimension_id}")
        column = dimension_column(frame, dimension_id)
        if column not in frame.columns:
            raise ValueError(
                f"{definition['label']} is unavailable for this model"
            )
        values = frame[column].astype(str).str.strip()
        if value not in set(values):
            raise ValueError(
                f"Unknown {definition['label'].lower()} control value: {value}"
            )
        return frame.loc[values.eq(value)].copy(), {
            "dimension_id": dimension_id,
            "dimension_label": definition["label"],
            "column": column,
            "value": value,
        }

    def theory_vertical_contrast(
        self,
        model: str,
        population: str,
        row_dimension: str = "ai_role",
        distribution_view: str = "observed",
        journal_scope: str = "all",
        study_status: str = "all",
        column_dimension: str = "level",
        control_dimension: str | None = None,
        control_value: str | None = None,
    ) -> dict:
        """Return a selectable specification-dimension-by-level matrix."""

        if row_dimension not in VERTICAL_ROW_DIMENSIONS:
            raise ValueError(f"Unknown vertical row dimension: {row_dimension}")
        if column_dimension not in VERTICAL_ROW_DIMENSIONS:
            raise ValueError(
                f"Unknown vertical column dimension: {column_dimension}"
            )
        if row_dimension == column_dimension:
            raise ValueError("Vertical matrix axes must use different dimensions")
        if "level" not in {row_dimension, column_dimension}:
            raise ValueError(
                "Vertical contrasting requires Level of analysis on one axis"
            )
        frame = self._composition_model_frame(model)
        frame, control = self._theory_controlled_frame(
            frame, control_dimension, control_value
        )
        frame = self._theory_population_frame(frame, population)
        frame = self._theory_journal_scope_frame(frame, journal_scope)
        result = relationship_matrix(
            frame,
            row_dimension,
            column_dimension,
            distribution_view,
            study_status,
        )
        result.update(
            {
                "model": model,
                "model_label": MODEL_DISPLAY_NAMES.get(model, model),
                "population": population,
                "population_label": dict(THEORY_POPULATIONS)[population],
                "journal_scope": journal_scope,
                "study_status": study_status,
                "control": control,
                "control_filter_label": (
                    f"{control['dimension_label']} = {control['value']}"
                    if control
                    else "No additional dimension filter"
                ),
            }
        )
        return result

    def theory_entrepreneurship_comparison(
        self,
        model: str,
        dimension_id: str = "ai_role",
        distribution_view: str = "observed",
        study_status: str = "all",
        min_support: int = 10,
        control_dimension: str | None = None,
        control_value: str | None = None,
    ) -> dict:
        """Compare specification and configurations across entrepreneurship sets."""

        frame = self._composition_model_frame(model)
        frame, control = self._theory_controlled_frame(
            frame, control_dimension, control_value
        )
        population_frames = {
            population_id: self._theory_population_frame(frame, population_id)
            for population_id, _ in THEORY_POPULATIONS
        }
        distributions = {
            population_id: theory_distribution(
                population_frame,
                dimension_id,
                distribution_view,
                study_status,
            )
            for population_id, population_frame in population_frames.items()
        }
        combined = distributions["combined"]
        combined_shares = {
            category["raw_value"]: category["share"]
            for category in combined["categories"]
        }
        groups = []
        for population_id, population_label in THEORY_POPULATIONS:
            result = distributions[population_id]
            for category in result["categories"]:
                category["percentage_point_difference_from_combined"] = round(
                    (
                        category["share"]
                        - combined_shares.get(category["raw_value"], 0.0)
                    )
                    * 100,
                    4,
                )
            groups.append(
                {
                    "id": population_id,
                    "label": population_label,
                    "comparison_role": (
                        "Union benchmark"
                        if population_id == "combined"
                        else "Interpretive set"
                        if population_id == "close_reading"
                        else "Journal set"
                    ),
                    **result,
                }
            )

        population_configurations = {
            population_id: recurring_configurations(
                population_frame,
                distribution_view,
                study_status,
                min_support=1,
            )
            for population_id, population_frame in population_frames.items()
        }

        def configuration_key(record: dict) -> tuple[str, ...]:
            return tuple(
                str(record[dimension_id])
                for dimension_id in (
                    "ai_role",
                    "mechanism",
                    "level",
                    "scope",
                    "process_stage",
                )
            )

        configuration_indexes = {
            population_id: {
                configuration_key(record): record
                for record in result["configurations"]
            }
            for population_id, result in population_configurations.items()
        }
        configurations = []
        for combined_record in population_configurations["combined"][
            "configurations"
        ]:
            if combined_record["papers"] < min_support:
                continue
            key = configuration_key(combined_record)
            population_values = []
            for population_id, population_label in THEORY_POPULATIONS:
                record = configuration_indexes[population_id].get(key)
                population_values.append(
                    {
                        "population": population_id,
                        "population_label": population_label,
                        "papers": int(record["papers"]) if record else 0,
                        "share": float(record["share"]) if record else 0.0,
                        "analyzed_n": population_configurations[population_id][
                            "analyzed_n"
                        ],
                    }
                )
            configurations.append(
                {
                    **{
                        dimension_id: combined_record[dimension_id]
                        for dimension_id in (
                            "ai_role",
                            "mechanism",
                            "level",
                            "scope",
                            "process_stage",
                        )
                    },
                    "filters": combined_record["filters"],
                    "population_values": population_values,
                }
            )

        return {
            "model": model,
            "model_label": MODEL_DISPLAY_NAMES.get(model, model),
            "dimension_id": dimension_id,
            "dimension_label": combined["dimension_label"],
            "column": combined["column"],
            "distribution": distribution_view,
            "study_status": study_status,
            "control": control,
            "control_filter_label": (
                f"{control['dimension_label']} = {control['value']}"
                if control
                else "No additional dimension control"
            ),
            "benchmark_label": "Combined entrepreneurship",
            "comparison_definition": (
                "Leading and Additional entrepreneurship journals are disjoint "
                "prespecified journal sets. Combined entrepreneurship is their "
                "union and is reported as a benchmark, not as an independent "
                "third tier. The systematic close-reading set is an overlapping "
                "interpretive dataset and its percentages describe only that set."
            ),
            "groups": groups,
            "configuration_dimensions": [
                "ai_role",
                "mechanism",
                "level",
                "scope",
                "process_stage",
            ],
            "configuration_denominators": {
                population_id: result["analyzed_n"]
                for population_id, result in population_configurations.items()
            },
            "configuration_min_support": min_support,
            "configurations": configurations,
        }

    def theory_structuring(
        self,
        model: str,
        population: str,
        pair_id: str = "ai_role__mechanism",
        distribution_view: str = "observed",
        journal_scope: str = "all",
        study_status: str = "all",
        min_support: int = 10,
        control_dimension: str | None = None,
        control_value: str | None = None,
    ) -> dict:
        """Return pairwise structure and recurring five-dimension configurations."""

        pairs = {
            f"{left}__{right}": (left, right, label)
            for left, right, label in STRUCTURING_PAIRS
        }
        if pair_id not in pairs:
            raise ValueError(f"Unknown structuring matrix: {pair_id}")
        frame = self._composition_model_frame(model)
        frame, control = self._theory_controlled_frame(
            frame, control_dimension, control_value
        )
        frame = self._theory_population_frame(frame, population)
        frame = self._theory_journal_scope_frame(frame, journal_scope)
        left, right, label = pairs[pair_id]
        matrix = relationship_matrix(
            frame, left, right, distribution_view, study_status
        )
        configurations = recurring_configurations(
            frame,
            distribution_view,
            study_status,
            min_support=min_support,
        )
        return {
            "model": model,
            "model_label": MODEL_DISPLAY_NAMES.get(model, model),
            "population": population,
            "population_label": dict(THEORY_POPULATIONS)[population],
            "journal_scope": journal_scope,
            "study_status": study_status,
            "pair_id": pair_id,
            "pair_label": label,
            "matrix": matrix,
            "configurations": configurations,
            "agency_operationalisation": (
                "AI role and observed mechanism jointly represent observable agency allocation; "
                "agency configuration is not an independently coded variable."
            ),
            "sequence_inference_permitted": False,
            "control": control,
            "control_filter_label": (
                f"{control['dimension_label']} = {control['value']}"
                if control
                else "No additional dimension filter"
            ),
        }

    def theory_contrasting_evidence(
        self,
        model: str,
        population: str | None = None,
        journal_scope: str = "all",
        study_status: str = "all",
        domain_id: str | None = None,
        filters: dict[str, str] | None = None,
        limit: int = 100,
        minimum_agreement: int = 1,
        preferred_only: bool = False,
    ) -> dict:
        """Return papers behind a construct-contrasting bar, cell or configuration."""

        frame = self._composition_model_frame(model)
        if population:
            frame = self._theory_population_frame(frame, population)
        frame = self._theory_journal_scope_frame(frame, journal_scope)
        frame = filter_study_status(frame, study_status)
        if domain_id:
            assignments = self._theory_domain_assignments(frame)
            paper_ids = set(
                assignments.loc[assignments["domain_id"].eq(domain_id), "paper_id"]
            )
            if domain_id == "combined_entrepreneurship":
                paper_ids = set(
                    assignments.loc[
                        assignments["domain_id"].isin(
                            ["core_entrepreneurship", "other_entrepreneurship"]
                        ),
                        "paper_id",
                    ]
                )
            frame = frame[frame["paper_id"].isin(paper_ids)]
        allowed_columns = {
            dimension_column(frame, definition["id"])
            for definition in THEORY_DIMENSIONS
        }
        exact_filters = {str(key): str(value) for key, value in (filters or {}).items()}
        for column, value in exact_filters.items():
            if column not in allowed_columns:
                raise ValueError(f"Unsupported evidence filter: {column}")
            if column not in frame.columns:
                frame = frame.iloc[0:0]
                break
            frame = frame[frame[column].astype(str).str.strip().eq(str(value))]
        agreement_filters = dict(exact_filters)
        if study_status != "all":
            agreement_filters[AI_STUDY_STATUS_COLUMN] = study_status
        return self._agreement_evidence_bundle(
            frame,
            agreement_filters,
            selected_columns=tuple(agreement_filters),
            limit=limit,
            minimum_agreement=minimum_agreement,
            preferred_only=preferred_only,
            reference_model=model,
        )

    def _numeric(self, frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series([0] * len(frame), index=frame.index, dtype="float")
        return pd.to_numeric(frame[column], errors="coerce").fillna(0)

    def _top_counts(self, frame, column, citations, top_n) -> list[dict]:
        if column not in frame.columns:
            return []
        work = frame.assign(_c=citations)
        rows = []
        for value, sub in work.groupby(column):
            if not str(value).strip():
                continue
            rows.append({column: value, "papers": len(sub), "citations": int(sub["_c"].sum())})
        rows.sort(key=lambda r: r["papers"], reverse=True)
        return rows[:top_n]

    def _top_authors(self, frame, top_n) -> list[dict]:
        if "Authors" not in frame.columns:
            return []
        counts: dict[str, int] = {}
        for value in frame["Authors"].fillna("").astype(str):
            for author in (a.strip() for a in value.split(";")):
                if author and author.lower() not in {"[no author name available]"}:
                    counts[author] = counts.get(author, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return [{"author": a, "papers": n} for a, n in ranked]

    def _most_cited(self, frame, citations, top_n) -> list[dict]:
        work = frame.assign(_c=citations).sort_values("_c", ascending=False).head(top_n)
        cols = [
            c for c in ["paper_id", "Title", "Authors", "Source title", "Year", "DOI", "Link"]
            if c in work.columns
        ]
        out = []
        for _, row in work.iterrows():
            record = {c: row[c] for c in cols}
            record["citations"] = int(row["_c"])
            out.append(record)
        return out

    # ---- construct contrast --------------------------------------------
    def contrast(self, scope_id: str, shared_column: str, contrast_column: str) -> list[dict]:
        frame = self._scope(scope_id)
        if not {shared_column, contrast_column} <= set(frame.columns):
            return []
        return construct_contrast(frame, shared_column, contrast_column).to_dict("records")

    def contrast_evidence(
        self,
        scope_id: str,
        shared_column: str,
        shared_value: str,
        contrast_column: str,
        limit: int = 100,
    ) -> dict:
        """Return the contrast distribution and its supporting papers."""

        frame = self._scope(scope_id)
        required = {shared_column, contrast_column}
        if not required <= set(frame.columns):
            return {
                "shared_column": shared_column,
                "shared_value": shared_value,
                "contrast_column": contrast_column,
                "total_papers": 0,
                "values": [],
                "papers": [],
            }
        subset = frame[frame[shared_column].astype(str) == str(shared_value)]
        counts = subset[contrast_column].astype(str).str.strip().value_counts()
        columns = [column for column in EVIDENCE_COLUMNS if column in subset.columns]
        return {
            "shared_column": shared_column,
            "shared_value": shared_value,
            "contrast_column": contrast_column,
            "total_papers": len(subset),
            "values": [
                {
                    "value": value or "not specified",
                    "filter_value": value,
                    "count": int(count),
                }
                for value, count in counts.items()
            ],
            "papers": subset[columns].head(limit).to_dict("records"),
        }

    # ---- paper + evidence ----------------------------------------------
    def paper(self, paper_id: str) -> dict | None:
        match = self.papers[self.papers["paper_id"] == paper_id]
        if match.empty:
            return None
        row = match.iloc[0].to_dict()
        record = self._paper_inspection_records(match, limit=1)[0]
        record["convergent_papers"] = self._nearest(row, same=True)
        record["contrasting_papers"] = self._nearest(row, same=False)
        return record

    def evidence(
        self,
        scope_id: str,
        column: str,
        value: str,
        limit: int = 100,
    ) -> list[dict]:
        """The paper list behind a statistic (brief section 10 traceability)."""

        frame = self._scope(scope_id)
        if column not in frame.columns:
            return []
        subset = frame[frame[column].astype(str) == str(value)]
        cols = [c for c in EVIDENCE_COLUMNS if c in subset.columns]
        return subset[cols].head(limit).to_dict("records")

    def query(self, scope_id: str, filters: dict[str, str], limit: int = 200) -> list[dict]:
        """Return evidence papers matching every column=value filter (Assistant)."""

        frame = self._scope(scope_id)
        for column, value in filters.items():
            if column in frame.columns:
                frame = frame[frame[column].astype(str) == str(value)]
        cols = [c for c in EVIDENCE_COLUMNS if c in frame.columns]
        return frame[cols].head(limit).to_dict("records")

    def distinct_values(self, scope_id: str, column: str) -> list[str]:
        """Distinct non-empty values of a column (for Assistant dropdowns)."""

        frame = self._scope(scope_id)
        if column not in frame.columns:
            return []
        vals = frame[column].astype(str).str.strip()
        return sorted(v for v in vals.unique() if v)

    def _nearest(self, row: dict, same: bool, limit: int = 10) -> list[dict]:
        """Papers agreeing on all dimensions (convergent) or differing (contrasting)."""

        if not self.has_specifications:
            return []
        dims = [c for c in DIMENSION_COLUMNS if c in self.papers.columns]
        others = self.papers[self.papers["paper_id"] != row.get("paper_id")]

        def score(other: pd.Series) -> int:
            return sum(1 for c in dims if str(other[c]) == str(row.get(c, "")))

        scored = others.assign(_match=others.apply(score, axis=1))
        if same:
            picked = scored[scored["_match"] == len(dims)]
        else:
            # share at least one dimension but differ on at least one.
            picked = scored[(scored["_match"] >= 1) & (scored["_match"] < len(dims))]
            picked = picked.sort_values("_match", ascending=False)
        cols = [c for c in ["paper_id", "Title", "Source title", "Year"] if c in picked.columns]
        return picked.head(limit)[cols].to_dict("records")

    # ---- graph data for vis-network ------------------------------------
    # On by default; Keyword and Reference are numerous, so off until toggled.
    _CORE_NODE_TYPES = (
        "Publication", "Author", "Journal", "Year", "SearchQuery", "Institution", "Topic",
        "AIRole", "AIType", "Mechanism", "LevelOfAnalysis", "ProcessStage",
        "ScopeCondition", "DefinitionClarity", "SpecificationProfile", "SpecificationProblem",
    )

    def scope_graph(
        self,
        scope_id: str,
        mode: str = "publications",
        limit: int = 40,
        include: set[str] | None = None,
        min_weight: int = 2,
    ) -> dict:
        """Per-scope graph for the Knowledge Graph view.

        mode='publications' (default): a publication-centred subgraph built with
        the same graph builder the platform uses, so every agreed node type that
        exists in the data (Publication, Author, Journal, Institution, Keyword,
        Reference, Topic, and the specification nodes) is rendered. ``include``
        selects which node types to show.
        This compatibility endpoint now serves only the locked graph schema.
        Analytical co-occurrence networks are not represented as graph edges.
        """

        if mode != "publications":
            raise ValueError("The knowledge graph supports publication mode only")
        frame = self._scope(scope_id)
        return self._publication_graph(frame, scope_id, limit, include)

    def _publication_graph(self, frame, scope_id, limit, include) -> dict:
        from collections import Counter

        from aecsp.knowledge_graph.builder import build_publication_graph

        if frame.empty:
            return {"mode": "publications", "scope": scope_id, "legend": {},
                    "nodes": [], "edges": [], "counts": {}, "available": False,
                    "backend": "csv", "message": "Neo4j is not connected."}

        include = set(include) if include else set(self._CORE_NODE_TYPES)
        include.add("Publication")

        citations = self._numeric(frame, "Cited by")
        focus = frame.assign(_c=citations).sort_values("_c", ascending=False).head(limit)
        meta_by_pid = self._publication_meta(focus)
        topic_scope = (
            scope_id
            if scope_id in {"full_corpus", "query_1", "query_2", "query_3", "query_4"}
            else "full_corpus"
        )
        graph = build_publication_graph(
            focus.drop(columns="_c").to_dict("records"),
            topic_scopes={topic_scope},
        )

        nodes: list[dict] = []
        kept: set[str] = set()
        for node in graph.nodes:
            label = node.ref.label
            if label not in include:
                continue
            value = node.ref.value
            nid = graph_node_id(label, value)
            kept.add(nid)
            is_pub = label == "Publication"
            meta = (
                meta_by_pid.get(value, {})
                if is_pub
                else {
                    "type": label,
                    node.ref.key: value,
                    **node.properties,
                }
            )
            display_value = (
                meta.get("Title")
                if is_pub
                else meta.get("display_label") or meta.get("label") or value
            )
            display = (
                _short_title(display_value)
                if is_pub
                else _short_title(display_value, 28)
            )
            citations_n = int(meta.get("Citations", 0)) if is_pub else 0
            nodes.append({
                "id": nid,
                "caption": display,
                "nodeType": label,
                "degree": 8 + min(citations_n, 60) if is_pub else 5,
                "properties": meta,
            })

        edges_by_id: dict[str, dict] = {}
        for rel in graph.relationships:
            source = graph_node_id(rel.start.label, rel.start.value)
            target = graph_node_id(rel.end.label, rel.end.value)
            if source in kept and target in kept:
                edge_id = f"{source}::{rel.relationship_type}::{target}"
                existing = edges_by_id.get(edge_id)
                if existing is not None:
                    existing["properties"]["weight"] = (
                        int(existing["properties"].get("weight", 1)) + 1
                    )
                    continue
                edge = {
                    "id": edge_id,
                    "from": source,
                    "to": target,
                    "type": rel.relationship_type,
                    "properties": {**rel.properties, "weight": 1},
                }
                edges_by_id[edge_id] = edge

        counts = dict(Counter(n["nodeType"] for n in nodes))
        return {
            "available": False,
            "backend": "csv",
            "message": "Neo4j is not connected. Showing a bounded dataframe seed.",
            "mode": "publications",
            "scope": scope_id,
            "counts": counts,
            "nodes": nodes,
            "edges": list(edges_by_id.values()),
        }

    def _publication_meta(self, focus) -> dict[str, dict]:
        """Rich per-publication metadata for the graph info panel."""

        spec_cols = [c for c in (*DIMENSION_COLUMNS, SPECIFICATION_PROBLEM_COLUMN) if c in focus.columns]
        meta: dict[str, dict] = {}
        for _, row in focus.iterrows():
            record = {
                "Title": str(row.get("Title", "")),
                "Authors": str(row.get("Authors", "")),
                "Journal": str(row.get("Source title", "")),
                "Year": str(row.get("Year", "")),
                "Citations": int(row["_c"]),
                "DOI": str(row.get("DOI", "")),
                "queries": str(row.get("query_sources", "")),
            }
            for column in spec_cols:
                value = str(row.get(column, "")).strip()
                if value:
                    record[column] = value
            meta[str(row["paper_id"])] = record
        return meta

    def connected_papers(self, scope_id: str, label: str, value: str) -> list[dict]:
        """Evidence papers connected to a clicked non-publication graph node."""

        frame = self._scope(scope_id)
        dim_col = {d.graph_node_label: d.column for d in SPECIFICATION_DIMENSIONS}

        exact_col = None
        contains_cols: list[str] = []
        if label == "Journal":
            exact_col = "Source title"
        elif label == "Topic":
            scoped_topic_column = (
                f"{scope_id}_topic_label"
                if scope_id.startswith("query_")
                else "bertopic_topic_label"
            )
            exact_col = (
                scoped_topic_column
                if scoped_topic_column in frame.columns
                else "bertopic_topic"
            )
        elif label in dim_col:
            exact_col = dim_col[label]
        elif label in DIMENSION_COLUMNS:          # contrast-mode nodes use column names
            exact_col = label
        elif label == "SpecificationProblem":
            contains_cols = [SPECIFICATION_PROBLEM_COLUMN]
        elif label == "Author":
            contains_cols = ["Authors"]
        elif label == "Institution":
            contains_cols = ["Authors with affiliations"]
        elif label == "Reference":
            contains_cols = ["References"]
        elif label == "Keyword":
            contains_cols = [c for c in ("Author Keywords", "Index Keywords", "keyphrases", "keybert_phrases") if c in frame.columns]

        if exact_col and exact_col in frame.columns:
            subset = frame[frame[exact_col].astype(str) == str(value)]
        elif contains_cols:
            mask = pd.Series(False, index=frame.index)
            for column in contains_cols:
                if column in frame.columns:
                    mask = mask | frame[column].astype(str).str.contains(re.escape(str(value)), case=False, na=False)
            subset = frame[mask]
        else:
            return []

        cols = [c for c in EVIDENCE_COLUMNS if c in subset.columns]
        return subset[cols].head(200).to_dict("records")

    # ---- graph exploration ---------------------------------------------
    def neo4j_available(self) -> bool:
        return self.neo4j is not None

    def graph_status(self) -> dict:
        """Return the graph backend and read-principal security posture."""

        if self.neo4j is None:
            rejected = self.neo4j_security is not None
            return {
                "connected": False,
                "backend": "csv",
                "neo4j_reachable": rejected,
                "raw_cypher_enabled": False,
                "message": (
                    self.neo4j_security.get("message", "Neo4j reader verification failed.")
                    if rejected
                    else (
                        "Neo4j is not connected. A bounded dataframe seed remains available; "
                        "focus, expansion, and graph search require Neo4j."
                    )
                ),
            }
        security = self.neo4j.security
        return {
            "connected": True,
            "backend": "neo4j",
            "raw_cypher_enabled": self.neo4j.raw_cypher_enabled,
            **security,
        }

    def graph_seed(
        self,
        scope_id: str,
        *,
        limit: int = 30,
        node_types: set[str] | None = None,
        relationship_types: set[str] | None = None,
        specification_label: str | None = None,
        specification_value: str | None = None,
    ) -> dict:
        """Return the Neo4j seed or a bounded dataframe fallback."""

        neo4j_scope = scope_id in SCOPE_BY_ID
        if self.neo4j is not None and neo4j_scope:
            return self.neo4j.seed(
                scope_id,
                limit=limit,
                node_types=node_types,
                relationship_types=relationship_types,
                specification_label=specification_label,
                specification_value=specification_value,
            )
        if specification_label or specification_value:
            return {
                "available": False,
                "backend": "csv",
                "scope": scope_id,
                "nodes": [],
                "edges": [],
                "counts": {},
                "message": (
                    "Specification-guided graph filtering requires a Neo4j-backed "
                    "retrieval scope. Reset the specification filter for this "
                    "derived dataset scope."
                ),
            }
        return self._publication_graph(
            self._scope(scope_id), scope_id, min(limit, 100), node_types
        )

    def graph_neighborhood(
        self,
        scope_id: str,
        node_id: str,
        *,
        relationship_types: set[str] | None = None,
    ) -> dict:
        """Return one focused one-hop neighborhood."""

        if self.neo4j is None:
            return self._graph_unavailable(scope_id, "Focus requires Neo4j.")
        if scope_id not in SCOPE_BY_ID:
            return self._graph_unavailable(
                scope_id,
                "Focus is unavailable for derived dataset scopes; reset the bounded scope seed.",
            )
        return self.neo4j.neighborhood(
            scope_id, node_id, relationship_types=relationship_types
        )

    def graph_expand(
        self,
        scope_id: str,
        node_id: str,
        *,
        relationship_types: set[str] | None = None,
    ) -> dict:
        """Return a one-hop expansion for client-side merging."""

        if self.neo4j is None:
            return self._graph_unavailable(scope_id, "Expansion requires Neo4j.")
        if scope_id not in SCOPE_BY_ID:
            return self._graph_unavailable(
                scope_id,
                "Expansion is unavailable for derived dataset scopes; reset the bounded scope seed.",
            )
        return self.neo4j.expand(
            scope_id, node_id, relationship_types=relationship_types
        )

    def graph_search(
        self,
        scope_id: str,
        text: str,
        *,
        node_types: set[str] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Search graph nodes within a scope."""

        if self.neo4j is None or scope_id not in SCOPE_BY_ID:
            return []
        return self.neo4j.search(
            scope_id, text, node_types=node_types, limit=limit
        )

    def graph_cypher(
        self,
        query: str,
        parameters: dict | None = None,
        *,
        limit: int = 500,
    ) -> dict:
        """Run guarded raw Cypher through a verified Neo4j reader role."""

        if self.neo4j is None:
            raise GraphQueryError("Raw Cypher requires Neo4j")
        return self.neo4j.raw_query(query, parameters, limit=limit)

    @staticmethod
    def _graph_unavailable(scope_id: str, message: str) -> dict:
        return {
            "available": False,
            "backend": "csv",
            "scope": scope_id,
            "nodes": [],
            "edges": [],
            "counts": {},
            "message": message,
        }

    def paper_neighbourhood(self, paper_id: str) -> dict:
        """Compatibility wrapper for a paper-centered full-corpus focus."""

        return self.graph_neighborhood(
            "full_corpus", graph_node_id("Publication", paper_id)
        )
