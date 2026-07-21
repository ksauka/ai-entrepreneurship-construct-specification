"""Provide graph and analytical data to the API.

Inputs: processed paper-level CSV files and an optional Neo4j driver.
Outputs: scope metrics, evidence records, and graph traversal results.
"""

from __future__ import annotations

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
    search_keyword_series,
)
from aecsp.analytics.observed_composition import (
    STUDY_STATUS_FILTERS,
    analyze_observed_composition,
    observed_composition_evidence_mask,
)
from aecsp.analytics.theory_contrasting import (
    DIMENSIONS as THEORY_DIMENSIONS,
    STRUCTURING_PAIRS,
    VERTICAL_ROW_DIMENSIONS,
    dimension_column,
    distribution as theory_distribution,
    filter_study_status,
    recurring_configurations,
    relationship_matrix,
)
from aecsp.corpus.business_domains import REGISTERED_QUERY_DOMAIN_RULES
from aecsp.corpus.scopes import DATASET_SCOPES, STRICT_AI_ENT_SCOPE, scope_frame
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

PROCESSED_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"

MODEL_DISPLAY_NAMES = {
    "gpt-5.4-mini-2026-03-17": "GPT-5.4 Mini",
    "gpt-4.1-nano-2025-04-14": "GPT-4.1 Nano",
    "claude-sonnet-5": "Claude Sonnet 5",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
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
    ("core", "Core entrepreneurship"),
    ("other", "Additional entrepreneurship"),
    ("combined", "Combined entrepreneurship"),
)

PENDING_ASJC_DOMAIN_LABELS = {
    "innovation": "Innovation",
    "strategy": "Strategy",
    "marketing": "Marketing",
    "information_systems": "Information systems",
    "finance": "Finance",
    "operations": "Operations",
    "organization_studies": "Organization studies",
    "environmental_and_sustainability": "Environmental and sustainability",
    "ethics_and_corporate_social_responsibility": "Ethics and CSR",
    "tourism": "Tourism",
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
        return self._apply_topic_review_display_labels(papers.fillna(""))

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
        return len(self.papers)

    @property
    def has_specifications(self) -> bool:
        return bool(self.papers is not None) and any(
            c in self.papers.columns for c in DIMENSION_COLUMNS
        )

    def _scope(self, scope_id: str) -> pd.DataFrame:
        return scope_frame(self.papers, scope_id)

    def _registered_composition_models(self) -> list[tuple[str, str]]:
        """Return the primary and full-corpus baseline models in stable order."""

        register = load_experiment_register()
        primary = str(register["primary_model"])
        registered = [(primary, "primary")]
        registered.extend(
            (str(model), "baseline")
            for model in register.get("baseline_models", [])
            if str(model) != primary
        )
        return registered

    def _composition_model_frame(self, model: str) -> pd.DataFrame:
        """Join one model's successful codes to the common corpus metadata."""

        registered = {item[0] for item in self._registered_composition_models()}
        if model not in registered:
            raise ValueError(f"Unknown specification model: {model}")
        if model in self._composition_frames:
            return self._composition_frames[model]

        spec_path = specification_csv_path(self.processed_dir, model=model)
        primary_model = resolve_primary_model()
        if not spec_path.exists():
            if model == primary_model and self.has_specifications:
                frame = self.papers.copy()
                self._composition_frames[model] = frame
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
        return frame

    def composition_models(self) -> list[dict]:
        """Describe model outputs currently safe for full-corpus comparison."""

        corpus_n = len(self.papers)
        models = []
        for model, role in self._registered_composition_models():
            try:
                coded_n = len(self._composition_model_frame(model))
            except ValueError:
                continue
            models.append(
                {
                    "id": model,
                    "label": MODEL_DISPLAY_NAMES.get(model, model),
                    "role": role,
                    "coded_papers": coded_n,
                    "corpus_papers": corpus_n,
                    "missing_papers": max(0, corpus_n - coded_n),
                    "coverage_share": round(coded_n / corpus_n, 6) if corpus_n else 0.0,
                }
            )
        return models

    def _composition_scope(self, scope_id: str, model: str) -> tuple[pd.DataFrame, int]:
        corpus_scope_n = len(self._scope(scope_id))
        return scope_frame(self._composition_model_frame(model), scope_id), corpus_scope_n

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

    def composition_export(
        self,
        scope_id: str,
        model: str,
        study_status: str = "all",
    ) -> pd.DataFrame:
        """Return the traceable paper table behind one composition view."""

        if study_status not in STUDY_STATUS_FILTERS:
            raise ValueError(f"Unknown study status: {study_status}")
        frame, _ = self._composition_scope(scope_id, model)
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
    ) -> pd.DataFrame:
        """Return paper-aligned ratings for all eight displayed dimensions."""

        if left_model == right_model:
            raise ValueError("IRR requires two different specification models")
        dimension_columns = [column for column, _, _ in DISPLAY_IRR_DIMENSIONS]
        left, _ = self._composition_scope(scope_id, left_model)
        right, _ = self._composition_scope(scope_id, right_model)
        left = left.copy()
        right = right.copy()
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

    def composition_irr(
        self,
        scope_id: str,
        left_model: str,
        right_model: str,
    ) -> dict:
        """Calculate exact agreement and nominal alpha on the common papers."""

        units = self.composition_irr_units(scope_id, left_model, right_model)
        dimensions = []
        for column, label, classification in DISPLAY_IRR_DIMENSIONS:
            left_values = units[f"left__{column}"].tolist()
            right_values = units[f"right__{column}"].tolist()
            exact = pairwise_percent_agreement(left_values, right_values)
            alpha = krippendorff_alpha_nominal(
                [list(pair) for pair in zip(left_values, right_values)]
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
                }
            )
        return {
            "scope": scope_id,
            "left_model": left_model,
            "left_label": MODEL_DISPLAY_NAMES.get(left_model, left_model),
            "right_model": right_model,
            "right_label": MODEL_DISPLAY_NAMES.get(right_model, right_model),
            "intersection_papers": len(units),
            "dimensions": dimensions,
            "study_status_filter_applied": False,
        }

    def composition_irr_matrix(self, scope_id: str) -> dict:
        """Calculate every available model pair for one dataset scope.

        The two matrix summaries are arithmetic means across the six locked
        dimensions. Dimension-level estimates remain the inferential record;
        the means are compact orientation measures for the matrix display.
        """

        models = self.composition_models()
        pairs = []
        for left, right in combinations(models, 2):
            result = self.composition_irr(scope_id, left["id"], right["id"])
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
        scopes = list(DATASET_SCOPES)
        if STRICT_AI_ENT_SCOPE.filter_column in self.papers.columns:
            scopes.append(STRICT_AI_ENT_SCOPE)
        return [
            {"id": s.id, "label": s.label, "papers": len(self._scope(s.id))}
            for s in scopes
        ]

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

        cumulative_by_year = {
            row["year"]: row["cumulative_papers"] for row in annual
        }

        def cumulative_at(year: int) -> int:
            eligible = [
                value
                for current_year, value in cumulative_by_year.items()
                if current_year <= year
            ]
            return max(eligible, default=0)

        growth_periods = (
            (2000, SEARCH_CUTOFF_YEAR),
            (2010, 2020),
            (2020, 2023),
            (2023, SEARCH_CUTOFF_YEAR),
        )
        publication_growth = []
        for start_year, end_year in growth_periods:
            start_count = cumulative_at(start_year)
            end_count = cumulative_at(end_year)
            added = end_count - start_count
            publication_growth.append(
                {
                    "start_year": start_year,
                    "end_year": end_year,
                    "start_cumulative_papers": start_count,
                    "end_cumulative_papers": end_count,
                    "added_papers": added,
                    "percent_growth": (
                        round(added / start_count, 6)
                        if start_count > 0
                        else None
                    ),
                    "end_is_retrieval_year": (
                        end_year == SEARCH_CUTOFF_YEAR
                    ),
                }
            )

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
        columns = list(
            dict.fromkeys(
                column
                for column in [*EVIDENCE_COLUMNS, "Cited by"]
                if column in subset.columns
            )
        )
        return {
            "scope": scope_id,
            "year": year,
            "mode": mode,
            "total_papers": len(subset),
            "returned_papers": min(limit, len(subset)),
            "papers": subset[columns].head(limit).to_dict("records"),
            "search_cutoff": {
                "date": SEARCH_CUTOFF_DATE.isoformat(),
                "label": SEARCH_CUTOFF_LABEL,
                "year": SEARCH_CUTOFF_YEAR,
            },
        }

    def publication_growth_comparison(self) -> dict:
        """Compare cumulative publication growth across every dataset view."""

        rows = []
        for scope in self.scopes():
            performance = self.performance(scope["id"])
            rows.append(
                {
                    "scope": scope["id"],
                    "label": scope["label"],
                    "papers": scope["papers"],
                    "growth": performance["publication_growth"],
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
        }

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
        columns = list(
            dict.fromkeys(
                column
                for column in [
                    *EVIDENCE_COLUMNS,
                    "Author Keywords",
                    "Index Keywords",
                ]
                if column in subset.columns
            )
        )
        return subset[columns].head(limit).to_dict("records")

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

    # ---- observed construct composition -------------------------------
    def observed_composition(
        self,
        scope_id: str,
        study_status: str = "all",
        model: str | None = None,
    ) -> dict:
        """Return observed-only dimension composition after status filtering."""

        selected_model = model or resolve_primary_model()
        frame, corpus_scope_n = self._composition_scope(scope_id, selected_model)
        result = analyze_observed_composition(
            frame, study_status=study_status
        )
        result["scope"] = scope_id
        result["model"] = selected_model
        result["model_label"] = MODEL_DISPLAY_NAMES.get(selected_model, selected_model)
        result["corpus_scope_papers"] = corpus_scope_n
        result["model_missing_papers"] = max(0, corpus_scope_n - len(frame))
        result["model_coverage_share"] = (
            round(len(frame) / corpus_scope_n, 6) if corpus_scope_n else 0.0
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
    ) -> list[dict]:
        """Return papers supporting one observed-composition bar."""

        selected_model = model or resolve_primary_model()
        frame, _ = self._composition_scope(scope_id, selected_model)
        mask = observed_composition_evidence_mask(
            frame,
            study_status=study_status,
            column=column,
            value=value,
        )
        subset = frame[mask]
        return self._paper_inspection_records(
            subset,
            selected_columns=(column,),
            limit=limit,
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

    def theory_contrasting_metadata(self, model: str | None = None) -> dict:
        """Describe available populations, domains, dimensions and model coverage."""

        selected_model = model or resolve_primary_model()
        frame = self._composition_model_frame(selected_model)
        assignments = self._theory_domain_assignments(frame)
        domains = []
        query_domain_definitions = {
            "ft50": (
                "Papers from source titles in the registered FT50 journal set."
            ),
            "core_entrepreneurship": (
                "Papers from the registered leading entrepreneurship journal set."
            ),
            "other_entrepreneurship": (
                "Papers from the registered additional entrepreneurship journal set."
            ),
        }
        for domain_id, group in assignments.groupby("domain_id", sort=False):
            assignment_basis = str(group["assignment_basis"].iloc[0])
            paper_ids = set(group["paper_id"])
            source_counts = (
                frame.loc[frame["paper_id"].isin(paper_ids), "Source title"]
                .astype(str)
                .str.strip()
                .loc[lambda values: values.ne("")]
                .value_counts()
            )
            query_defined = str(domain_id) in query_domain_definitions
            registry_field = (
                assignment_basis.removeprefix("journal_registry:")
                if assignment_basis.startswith("journal_registry:")
                else ""
            )
            domains.append(
                {
                    "id": str(domain_id),
                    "label": str(group["domain_label"].iloc[0]),
                    "papers": int(group["paper_id"].nunique()),
                    "assignment_basis": assignment_basis,
                    "assignment_type": (
                        "Registered journal population"
                        if query_defined
                        else "Reviewed journal-domain registry"
                    ),
                    "definition": query_domain_definitions.get(
                        str(domain_id),
                        (
                            "Papers inherit this domain from their Scopus source "
                            "title through the reviewed journal-domain registry."
                        ),
                    ),
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
            "journal_scopes": [
                {"id": "all", "label": "All journals"},
                {"id": "ft50", "label": "FT50 robustness subset"},
            ],
            "domain_methodology": {
                "unit": "Paper inherited from its source journal",
                "construction": (
                    "Domains classify papers already present in the 22,345-paper "
                    "corpus. They do not retrieve or add papers."
                ),
                "classification": (
                    "FT50, Core entrepreneurship, and Additional entrepreneurship "
                    "use registered journal populations. The remaining business "
                    "domains use reviewed journal-domain registries informed by "
                    "Scopus ASJC source classifications."
                ),
                "overlap": (
                    "Domain membership is multi-label. A journal, and therefore a "
                    "paper, may belong to more than one domain; rows must not be summed."
                ),
                "unclassified": (
                    "Papers whose source journal is not represented in a registered "
                    "domain remain outside these domain rows but remain in the baseline."
                ),
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
        baseline_shares = {
            item["raw_value"]: item["share"] for item in baseline["categories"]
        }
        metadata = self.theory_contrasting_metadata(model)
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
                    "assignment_basis": str(assigned["assignment_basis"].iloc[0]),
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
                "in the full corpus."
            ),
            "baseline": baseline,
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
                        "Union benchmark" if population_id == "combined" else "Journal set"
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
                "Core and Additional entrepreneurship are disjoint registered "
                "journal sets. Combined entrepreneurship is their union and is "
                "reported as a benchmark, not as an independent third tier."
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
        for column, value in (filters or {}).items():
            if column not in allowed_columns:
                raise ValueError(f"Unsupported evidence filter: {column}")
            if column not in frame.columns:
                frame = frame.iloc[0:0]
                break
            frame = frame[frame[column].astype(str).str.strip().eq(str(value))]
        total = len(frame)
        return {
            "total_papers": total,
            "returned_papers": min(total, limit),
            "papers": self._paper_inspection_records(
                frame,
                selected_columns=tuple((filters or {}).keys()),
                limit=limit,
            ),
        }

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
        record = {k: row.get(k, "") for k in EVIDENCE_COLUMNS if k in row}
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

        if self.neo4j is not None:
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
                "message": "Specification-guided graph filtering requires Neo4j.",
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

        if self.neo4j is None:
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
