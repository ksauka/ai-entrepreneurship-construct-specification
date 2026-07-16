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
    analyze_keyword_evolution,
    keyword_evidence_mask,
    search_keyword_series,
)
from aecsp.analytics.observed_composition import (
    STUDY_STATUS_FILTERS,
    analyze_observed_composition,
    observed_composition_evidence_mask,
)
from aecsp.corpus.scopes import DATASET_SCOPES, STRICT_AI_ENT_SCOPE, scope_frame
from aecsp.knowledge_graph.neo4j_reader import (
    GraphQueryError,
    Neo4jGraphReader,
    graph_node_id,
)
from aecsp.specification.schema import (
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
    "Authors",
    "Source title",
    "Year",
    "DOI",
    "Link",
    "query_sources",
    "bertopic_topic_label",
    AI_STUDY_STATUS_COLUMN,
    *DIMENSION_COLUMNS,
    SPECIFICATION_PROBLEM_COLUMN,
]


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
        if (years > 0).any():
            work = frame.assign(_y=years, _c=citations)
            for year, sub in work[work["_y"] > 0].groupby("_y"):
                annual.append({
                    "year": int(year),
                    "papers": len(sub),
                    "citations": int(sub["_c"].sum()),
                })
            annual.sort(key=lambda r: r["year"])

        top_journals = self._top_counts(frame, "Source title", citations, top_n)
        top_authors = self._top_authors(frame, top_n)
        most_cited = self._most_cited(frame, citations, top_n)

        return {
            "scope": scope_id,
            "citation_definition": (
                "Current cumulative citations grouped by publication year; "
                "this is publication-cohort impact, not citations received during each year."
            ),
            "summary": summary,
            "annual_production": annual,
            "top_journals": top_journals,
            "top_authors": top_authors,
            "most_cited": most_cited,
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
        columns = [
            column
            for column in [*EVIDENCE_COLUMNS, "Author Keywords", "Index Keywords"]
            if column in subset.columns
        ]
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
        columns = [column for column in EVIDENCE_COLUMNS if column in subset.columns]
        return subset[columns].head(limit).to_dict("records")

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
