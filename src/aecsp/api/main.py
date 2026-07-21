"""Define the FastAPI application for construct-specification analytics.

Inputs: processed datasets and optional Neo4j connection settings.
Outputs: HTTP analytics, evidence, graph, report, and static-interface routes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Lock
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from aecsp.api.auth import verify_basic_credentials
from aecsp.api.graph_service import GraphService
from aecsp.api.report import (
    build_composition_report,
    build_scope_report,
    build_theory_contrasting_report,
)
from aecsp.corpus.scopes import SCOPE_BY_ID
from aecsp.human_annotation import HumanAnnotationStore
from aecsp.knowledge_graph.neo4j_reader import GraphQueryError
from aecsp.specification.llm_coder import load_env
from aecsp.topics.review import TopicReviewStore

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = Path(__file__).resolve().parent / "static"
HTML_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Pragma": "no-cache",
}

state: dict = {}
topic_finalize_lock = Lock()

TOPIC_TABLE_DIR = PROJECT_ROOT / "reports/analysis/tables/stage4"
TOPIC_ENRICHED_DATASET = (
    PROJECT_ROOT / "data/processed/analysis/primary_analysis_dataset_with_topics.csv"
)
GRAPH_EXPORT_DIR = PROJECT_ROOT / "data/processed/graph"
TOPIC_TABLE_FILES = (
    "scope_topic_prevalence.csv",
    "scope_topic_by_era.csv",
    "scope_topic_by_journal.csv",
    "scope_topic_dimension_distribution.csv",
    "scope_topic_paper_index.csv",
)


class NoCacheStaticFiles(StaticFiles):
    """Serve dashboard assets without retaining stale interface behavior."""

    async def get_response(self, path: str, scope: dict) -> Response:
        response = await super().get_response(path, scope)
        response.headers.update(HTML_NO_CACHE_HEADERS)
        return response


class CypherQueryRequest(BaseModel):
    """One bounded read-only Cypher request."""

    query: str = Field(min_length=1, max_length=10_000)
    parameters: dict[str, object] = Field(default_factory=dict)
    limit: int = Field(default=500, ge=1, le=500)


class TopicReviewUpdateRequest(BaseModel):
    """One traceable researcher decision about a scope-specific topic label."""

    approved_label: str = Field(default="", max_length=180)
    review_status: str = Field(pattern="^(pending|approved|revise)$")
    reviewer_notes: str = Field(default="", max_length=2_000)
    reviewer: str = Field(min_length=1, max_length=160)


class TopicFinalizeRequest(BaseModel):
    """Explicit confirmation before regenerating approved derivative outputs."""

    confirmation: str = Field(max_length=80)


class HumanAnnotationSaveRequest(BaseModel):
    """One resumable, independently identified human paper annotation."""

    annotator_id: str = Field(min_length=2, max_length=40)
    paper_id: str = Field(min_length=1, max_length=180)
    annotation: dict[str, object] = Field(default_factory=dict)
    submit: bool = False


def _csv_set(value: str) -> set[str] | None:
    items = {part.strip() for part in value.split(",") if part.strip()}
    return items or None


def _graph_error(error: Exception) -> HTTPException:
    if isinstance(error, (GraphQueryError, ValueError)):
        return HTTPException(status_code=400, detail=str(error))
    return HTTPException(
        status_code=503,
        detail="The Neo4j graph request could not be completed.",
    )


def _connect_neo4j():
    env = load_env(PROJECT_ROOT / ".env")
    uri = env.get("NEO4J_URI")
    app_user = env.get("NEO4J_APP_USER")
    app_password = env.get("NEO4J_APP_PASSWORD")
    database = env.get("NEO4J_DATABASE", "neo4j")
    if not uri or not app_user or not app_password:
        return None, database
    try:
        from aecsp.knowledge_graph.neo4j_loader import connect

        driver = connect(uri, app_user, app_password)
        driver.verify_connectivity()
        return driver, database
    except Exception:
        return None, database  # bounded dataframe fallback


@asynccontextmanager
async def lifespan(app: FastAPI):
    driver, database = _connect_neo4j()
    state["service"] = GraphService(
        neo4j_driver=driver, neo4j_database=database
    )
    state["topic_review"] = TopicReviewStore(PROJECT_ROOT)
    state["human_annotation"] = HumanAnnotationStore(PROJECT_ROOT)
    yield
    if driver is not None:
        driver.close()


app = FastAPI(title="ETV_V2 Construct Specification Platform", lifespan=lifespan)


@app.middleware("http")
async def require_dashboard_authentication(request: Request, call_next):
    """Protect every page, static asset, API route, and generated report."""

    required = _dashboard_authentication_enabled()
    if not required:
        return await call_next(request)

    username = os.getenv("ETV_DASHBOARD_USERNAME", "")
    password = os.getenv("ETV_DASHBOARD_PASSWORD", "")
    if not username or not password:
        return PlainTextResponse(
            "Dashboard authentication is required but not configured.",
            status_code=503,
            headers={"Cache-Control": "no-store"},
        )
    if not verify_basic_credentials(
        request.headers.get("Authorization"), username, password
    ):
        return PlainTextResponse(
            "Authentication required.",
            status_code=401,
            headers={
                "WWW-Authenticate": 'Basic realm="ETV Dashboard", charset="UTF-8"',
                "Cache-Control": "no-store",
            },
        )
    return await call_next(request)


def service() -> GraphService:
    return state["service"]


def topic_review_store() -> TopicReviewStore:
    store = state.get("topic_review")
    if store is None:
        store = TopicReviewStore(PROJECT_ROOT)
        state["topic_review"] = store
    return store


def human_annotation_store() -> HumanAnnotationStore:
    store = state.get("human_annotation")
    if store is None:
        store = HumanAnnotationStore(PROJECT_ROOT)
        state["human_annotation"] = store
    return store


def _dashboard_authentication_enabled() -> bool:
    return os.getenv("ETV_DASHBOARD_REQUIRE_AUTH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _require_authenticated_topic_write() -> None:
    if not _dashboard_authentication_enabled():
        raise HTTPException(
            status_code=403,
            detail=(
                "Topic-review writes are disabled unless dashboard "
                "authentication is enabled."
            ),
        )


def _require_authenticated_annotation_write() -> None:
    if not _dashboard_authentication_enabled():
        raise HTTPException(
            status_code=403,
            detail=(
                "Human-annotation writes are disabled unless dashboard "
                "authentication is enabled."
            ),
        )


def _topic_review_frame(scope: str) -> "pd.DataFrame":
    """Return one scope's review decisions with current draft display labels."""

    import pandas as pd

    records = topic_review_store().records(scope=scope)
    return pd.DataFrame(records)


def _apply_draft_topic_labels(frame, review, scope: str):
    """Overlay saved draft labels on one scope-keyed exported dataframe."""

    import pandas as pd

    result = frame.copy()
    if "scope" in result.columns:
        result = result[result["scope"].astype(str).eq(scope)].copy()
    if "topic_id" not in result.columns:
        return result
    result["topic_id"] = result["topic_id"].astype(str)
    labels = review[
        [
            "topic_id",
            "topic_uid",
            "automatic_label",
            "approved_label",
            "display_label",
            "review_status",
            "last_reviewer",
            "last_updated_at",
        ]
    ].copy()
    labels["topic_id"] = labels["topic_id"].astype(str)
    if "topic_label" in result.columns:
        result = result.rename(columns={"topic_label": "topic_label_at_last_build"})
    result = result.merge(labels, on="topic_id", how="left", validate="many_to_one")
    result.insert(
        min(3, len(result.columns)),
        "topic_label",
        result["display_label"].fillna(result.get("topic_label_at_last_build", "")),
    )
    return result


def _topic_release_response(scope: str, bundle: str) -> Response:
    """Build a checksummed, scope-aware topic artifact archive in memory."""

    import pandas as pd

    if scope not in ("full_corpus", "query_1", "query_2", "query_3", "query_4"):
        raise HTTPException(status_code=400, detail=f"Unknown topic scope: {scope}")
    if bundle not in {"topics", "figures", "graph", "release"}:
        raise HTTPException(status_code=400, detail=f"Unknown topic bundle: {bundle}")

    store = topic_review_store()
    summary = store.summary()
    review = _topic_review_frame(scope)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    release_id = str(summary["release_id"])
    files: dict[str, bytes] = {}

    def add_bytes(name: str, content: bytes | str) -> None:
        files[name] = content.encode("utf-8") if isinstance(content, str) else content

    if bundle in {"topics", "release"}:
        add_bytes(
            f"topics/{scope}/topic_label_review.csv",
            review.to_csv(index=False),
        )
        for filename in TOPIC_TABLE_FILES:
            path = TOPIC_TABLE_DIR / filename
            if not path.exists():
                continue
            table = pd.read_csv(path, dtype=str, keep_default_na=False)
            table = _apply_draft_topic_labels(table, review, scope)
            add_bytes(f"topics/{scope}/{filename}", table.to_csv(index=False))

        if store.audit_path.exists():
            scoped_audit = []
            for line in store.audit_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("scope") == scope:
                    scoped_audit.append(json.dumps(record, ensure_ascii=False))
            add_bytes(
                f"topics/{scope}/topic_label_review_audit.jsonl",
                "\n".join(scoped_audit) + ("\n" if scoped_audit else ""),
            )

    if bundle == "release" and TOPIC_ENRICHED_DATASET.exists():
        enriched = pd.read_csv(
            TOPIC_ENRICHED_DATASET, dtype=str, keep_default_na=False
        )
        if scope != "full_corpus":
            flag = f"in_{scope}"
            if flag in enriched.columns:
                enriched = enriched[
                    enriched[flag].astype(str).str.lower().isin(("1", "true", "yes"))
                ].copy()
        topic_id_column = (
            "bertopic_topic" if scope == "full_corpus" else f"{scope}_topic_id"
        )
        label_column = (
            "bertopic_topic_label"
            if scope == "full_corpus"
            else f"{scope}_topic_label"
        )
        if topic_id_column in enriched.columns:
            mapping = review.set_index("topic_id")["display_label"].to_dict()
            numeric_ids = pd.to_numeric(enriched[topic_id_column], errors="coerce")
            display = numeric_ids.map(mapping)
            if label_column in enriched.columns:
                automatic_column = f"{label_column}_automatic"
                if automatic_column not in enriched.columns:
                    enriched[automatic_column] = enriched[label_column]
                enriched[label_column] = display.fillna(enriched[label_column])
            enriched[f"{label_column}_review_status"] = numeric_ids.map(
                review.set_index("topic_id")["review_status"].to_dict()
            ).fillna("")
        add_bytes(
            f"data/{scope}/topic_enriched_papers.csv",
            enriched.to_csv(index=False),
        )

    if bundle in {"figures", "release"}:
        for figure_name in summary["figure_names"]:
            try:
                figure = store.preview_figure_path(scope, str(figure_name))
            except FileNotFoundError:
                continue
            add_bytes(f"figures/{scope}/{figure_name}", figure.read_bytes())

    if bundle in {"graph", "release"}:
        preview = store.graph_preview(scope)
        add_bytes(
            f"graph/{scope}/draft_topic_nodes.csv",
            pd.DataFrame(preview["nodes"]).to_csv(index=False),
        )
        add_bytes(
            f"graph/{scope}/draft_topic_relationships.csv",
            pd.DataFrame(preview["edges"]).to_csv(index=False),
        )
        if summary["outputs_current"]:
            for filename in ("nodes.csv", "relationships.csv"):
                path = GRAPH_EXPORT_DIR / filename
                if path.exists():
                    add_bytes(f"graph/published/{filename}", path.read_bytes())

    if store.manifest_path.exists():
        add_bytes("provenance/stage4_manifest.json", store.manifest_path.read_bytes())
    manifest = {
        "release_id": release_id,
        "release_state": "published" if summary["outputs_current"] else "draft",
        "bundle": bundle,
        "scope": scope,
        "generated_at": generated_at,
        "review_sha256": summary["review_sha256"],
        "applied_review_sha256": summary["applied_review_sha256"],
        "approved_topics": summary["approved"],
        "required_topics": summary["total_topics"],
        "topic_assignments_changed": False,
        "files": [
            {
                "path": name,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for name, content in sorted(files.items())
        ],
    }
    add_bytes("manifest.json", json.dumps(manifest, indent=2))

    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            archive.writestr(name, content)
    archive_name = f"etv_{scope}_{bundle}_{release_id}.zip"
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{archive_name}"',
            "Cache-Control": "no-store",
            "X-ETV-Release-ID": release_id,
            "X-ETV-Release-State": manifest["release_state"],
        },
    )


def _composition_release_response(
    scope: str,
    bundle: str,
    model: str,
    study_status: str,
    distribution: str = "compare",
) -> Response:
    """Build a filter-aware composition and IRR archive in memory."""

    import pandas as pd

    if bundle not in {"composition", "irr", "release"}:
        raise HTTPException(status_code=400, detail=f"Unknown composition bundle: {bundle}")
    if distribution not in {"compare", "full", "observed"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown composition distribution: {distribution}",
        )
    svc = service()
    try:
        composition = svc.observed_composition(
            scope,
            study_status=study_status,
            model=model,
        )
        irr = svc.composition_irr_matrix(scope)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    files: dict[str, bytes] = {}

    def add_bytes(name: str, content: bytes | str) -> None:
        files[name] = content.encode("utf-8") if isinstance(content, str) else content

    if bundle in {"composition", "release"}:
        summary_rows = []
        for panel in composition["panels"]:
            for category in panel["comparison_categories"]:
                common = {
                    "scope": scope,
                    "model": model,
                    "study_status_filter": study_status,
                    "dimension": panel["label"],
                    "column": panel["column"],
                    "category": category["value"],
                }
                if distribution in {"compare", "full"}:
                    summary_rows.append(
                        {
                            **common,
                            "distribution": "full",
                            "denominator": panel["full_n"],
                            "papers": category["full_count"],
                            "share": category["full_share"],
                        }
                    )
                if distribution in {"compare", "observed"} and category["is_observed"]:
                    summary_rows.append(
                        {
                            **common,
                            "distribution": "observed",
                            "denominator": panel["observed_n"],
                            "papers": category["observed_count"],
                            "share": category["observed_share"],
                        }
                    )
        add_bytes(
            "composition/observed_composition_summary.csv",
            pd.DataFrame(summary_rows).to_csv(index=False),
        )
        add_bytes(
            "composition/observed_composition_papers.csv",
            svc.composition_export(scope, model, study_status).to_csv(index=False),
        )

    if bundle in {"irr", "release"}:
        matrix_rows = []
        dimension_rows = []
        for pair in irr["pairs"]:
            matrix_rows.append(
                {
                    "scope": scope,
                    "left_model": pair["left_model"],
                    "left_label": pair["left_label"],
                    "right_model": pair["right_model"],
                    "right_label": pair["right_label"],
                    "common_papers": pair["intersection_papers"],
                    "mean_percent_agreement": pair["mean_percent_agreement"],
                    "mean_krippendorff_alpha": pair["mean_krippendorff_alpha"],
                }
            )
            for row in pair["dimensions"]:
                dimension_rows.append(
                    {
                        "scope": scope,
                        "left_model": pair["left_model"],
                        "right_model": pair["right_model"],
                        "common_papers": pair["intersection_papers"],
                        **row,
                    }
                )
            units = svc.composition_irr_units(
                scope, pair["left_model"], pair["right_model"]
            )
            units = units.rename(
                columns={
                    column: column.replace(
                        "left__", f"{pair['left_model']}__"
                    ).replace("right__", f"{pair['right_model']}__")
                    for column in units.columns
                }
            )
            pair_name = re.sub(
                r"[^a-z0-9]+",
                "_",
                f"{pair['left_model']}_vs_{pair['right_model']}".lower(),
            ).strip("_")
            add_bytes(
                f"irr/common_paper_ratings/{pair_name}.csv",
                units.to_csv(index=False),
            )
        add_bytes(
            "irr/pairwise_irr_matrix.csv",
            pd.DataFrame(matrix_rows).to_csv(index=False),
        )
        add_bytes(
            "irr/pairwise_irr_by_dimension.csv",
            pd.DataFrame(dimension_rows).to_csv(index=False),
        )

    if bundle == "release":
        add_bytes(
            "report/observed_composition_report.html",
            build_composition_report(
                svc,
                scope,
                model,
                study_status,
                distribution,
            ),
        )

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    artifact_hash = hashlib.sha256(
        b"".join(content for _, content in sorted(files.items()))
    ).hexdigest()
    manifest = {
        "artifact_id": f"composition-{artifact_hash[:16]}",
        "bundle": bundle,
        "scope": scope,
        "model": model,
        "model_label": composition["model_label"],
        "study_status_filter": study_status,
        "distribution_view": distribution,
        "filtered_papers": composition["filtered_papers"],
        "model_coded_papers": composition["scope_papers"],
        "corpus_scope_papers": composition["corpus_scope_papers"],
        "model_coverage_share": composition["model_coverage_share"],
        "irr_models": [item["id"] for item in irr["models"]],
        "irr_pair_count": len(irr["pairs"]),
        "irr_dimension_count": irr["dimension_count"],
        "irr_core_dimension_count": irr["core_dimension_count"],
        "irr_summary_method": irr["summary_method"],
        "irr_pairs": [
            {
                "left_model": pair["left_model"],
                "right_model": pair["right_model"],
                "common_papers": pair["intersection_papers"],
            }
            for pair in irr["pairs"]
        ],
        "irr_study_status_filter_applied": False,
        "generated_at": generated_at,
        "raw_model_records_changed": False,
        "files": [
            {
                "path": name,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for name, content in sorted(files.items())
        ],
    }
    add_bytes("manifest.json", json.dumps(manifest, indent=2))

    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            archive.writestr(name, content)
    archive_name = (
        f"etv_composition_{re.sub(r'[^a-z0-9]+', '_', scope.lower()).strip('_')}_"
        f"{bundle}_{generated_at[:10]}.zip"
    )
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{archive_name}"',
            "Cache-Control": "no-store",
            "X-ETV-Artifact-ID": manifest["artifact_id"],
            "X-ETV-Paper-Count": str(composition["filtered_papers"]),
        },
    )


@app.get("/api/health")
def health() -> dict:
    svc = service()
    return {
        "status": "ok",
        "papers_loaded": len(svc.papers),
        "has_specifications": svc.has_specifications,
        "neo4j": svc.neo4j_available(),
        "graph": svc.graph_status(),
    }


@app.get("/api/scopes")
def scopes() -> list[dict]:
    return service().scopes()


@app.get("/api/specification/models")
def specification_models() -> list[dict]:
    return service().composition_models()


@app.get("/api/scope/{scope_id}/overview")
def overview(scope_id: str) -> dict:
    return service().scope_overview(scope_id)


@app.get("/api/scope/{scope_id}/distribution")
def distribution(scope_id: str, column: str = Query(...)) -> dict:
    return service().dimension_distribution(scope_id, column)


@app.get("/api/scope/{scope_id}/groups")
def groups(
    scope_id: str,
    by: str = Query("Source title"),
    min_papers: int = Query(2, ge=2, le=10000),
) -> list[dict]:
    return service().group_convergence_table(scope_id, by, min_papers=min_papers)


@app.get("/api/scope/{scope_id}/dimension-profile")
def dimension_profile(scope_id: str, column: str = Query(...)) -> dict:
    try:
        return service().dimension_profile(scope_id, column)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/scope/{scope_id}/performance")
def performance(scope_id: str) -> dict:
    return service().performance(scope_id)


@app.get("/api/scope/{scope_id}/performance/papers")
def performance_papers(
    scope_id: str,
    year: int = Query(..., ge=1800, le=2200),
    mode: str = Query("annual", pattern="^(annual|cumulative)$"),
    limit: int = Query(100, ge=1, le=50000),
) -> dict:
    try:
        return service().performance_papers(
            scope_id, year=year, mode=mode, limit=limit
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/performance/publication-growth")
def publication_growth_comparison() -> dict:
    return service().publication_growth_comparison()


@app.get("/api/scope/{scope_id}/keywords")
def keyword_evolution(
    scope_id: str,
    source: str = Query("author", pattern="^(author|index|combined)$"),
    series_top_n: int = Query(10, ge=1, le=20),
    period_top_n: int = Query(20, ge=1, le=50),
) -> dict:
    return service().keyword_evolution(
        scope_id,
        source=source,
        series_top_n=series_top_n,
        period_top_n=period_top_n,
    )


@app.get("/api/scope/{scope_id}/keywords/evidence")
def keyword_evidence(
    scope_id: str,
    source: str = Query("author", pattern="^(author|index|combined)$"),
    keyword: str = Query(...),
    period: str | None = Query(None),
    year: int | None = Query(None, ge=1900, le=2100),
    limit: int = Query(100, ge=1, le=200),
) -> list[dict]:
    try:
        return service().keyword_evidence(
            scope_id,
            source=source,
            keyword=keyword,
            period_id=period,
            year=year,
            limit=limit,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/scope/{scope_id}/keywords/search")
def keyword_search(
    scope_id: str,
    source: str = Query("author", pattern="^(author|index|combined)$"),
    q: str = Query(..., min_length=1, max_length=120),
    limit: int = Query(20, ge=1, le=50),
) -> list[dict]:
    return service().keyword_search(
        scope_id, source=source, query=q, limit=limit
    )


@app.get("/api/scope/{scope_id}/composition")
def observed_composition(
    scope_id: str,
    study_status: str = Query("all", pattern="^(all|phenomenon|method|both)$"),
    model: str | None = Query(None),
) -> dict:
    try:
        return service().observed_composition(
            scope_id, study_status=study_status, model=model
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/scope/{scope_id}/composition/evidence")
def observed_composition_evidence(
    scope_id: str,
    study_status: str = Query("all", pattern="^(all|phenomenon|method|both)$"),
    column: str = Query(...),
    value: str = Query(...),
    limit: int = Query(100, ge=1, le=50000),
    model: str | None = Query(None),
) -> list[dict]:
    try:
        return service().observed_composition_evidence(
            scope_id,
            study_status=study_status,
            column=column,
            value=value,
            limit=limit,
            model=model,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/scope/{scope_id}/composition/irr")
def observed_composition_irr(
    scope_id: str,
    left_model: str = Query("gpt-5.4-mini-2026-03-17"),
    right_model: str = Query("gpt-4.1-nano-2025-04-14"),
) -> dict:
    try:
        return service().composition_irr(scope_id, left_model, right_model)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/scope/{scope_id}/composition/irr/matrix")
def observed_composition_irr_matrix(scope_id: str) -> dict:
    try:
        return service().composition_irr_matrix(scope_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/scope/{scope_id}/composition/report", response_class=HTMLResponse)
def observed_composition_report(
    scope_id: str,
    model: str = Query("gpt-5.4-mini-2026-03-17"),
    study_status: str = Query("all", pattern="^(all|phenomenon|method|both)$"),
    distribution: str = Query("compare", pattern="^(compare|full|observed)$"),
) -> str:
    try:
        return build_composition_report(
            service(),
            scope_id,
            model,
            study_status,
            distribution,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/scope/{scope_id}/composition/download/{bundle}")
def observed_composition_download(
    scope_id: str,
    bundle: str,
    model: str = Query("gpt-5.4-mini-2026-03-17"),
    study_status: str = Query("all", pattern="^(all|phenomenon|method|both)$"),
    distribution: str = Query("compare", pattern="^(compare|full|observed)$"),
) -> Response:
    return _composition_release_response(
        scope_id,
        bundle,
        model,
        study_status,
        distribution,
    )


def _theory_result_rows(
    tactic: str,
    payload: dict,
    distribution: str,
) -> list[dict[str, object]]:
    """Flatten one theory-elaboration result without recalculating its metrics."""

    if tactic == "construct":
        rows = []
        for panel in payload["panels"]:
            for category in panel["comparison_categories"]:
                if distribution == "observed" and not category["is_observed"]:
                    continue
                rows.append(
                    {
                        "population": payload["population_label"],
                        "dimension": panel["label"],
                        "column": panel["column"],
                        "category": category["value"],
                        "distribution": distribution,
                        "denominator": panel[
                            "observed_n" if distribution == "observed" else "full_n"
                        ],
                        "papers": category[
                            "observed_count"
                            if distribution == "observed"
                            else "full_count"
                        ],
                        "share": category[
                            "observed_share"
                            if distribution == "observed"
                            else "full_share"
                        ],
                    }
                )
        return rows
    if tactic == "horizontal":
        domain_rows = [
            {
                "record_type": "domain_comparison",
                "domain_id": group["id"],
                "domain": group["label"],
                "assignment_basis": group["assignment_basis"],
                "dimension": payload["dimension_label"],
                "column": payload["column"],
                "distribution": payload["distribution"],
                "denominator": group["denominator"],
                "category": category["value"],
                "papers": category["papers"],
                "share": category["share"],
                "percentage_point_difference_from_corpus": category[
                    "percentage_point_difference"
                ],
            }
            for group in payload["groups"]
            for category in group["categories"]
        ]
        entrepreneurship_rows = _theory_result_rows(
            "entrepreneurship",
            payload["entrepreneurship_comparison"],
            distribution,
        )
        return [*domain_rows, *entrepreneurship_rows]
    if tactic == "vertical":
        return [
            {
                "population": payload["population_label"],
                "row_dimension": payload["row_label"],
                "row_value": cell["row_value"],
                "column_dimension": payload["column_label"],
                "column_value": cell["column_value"],
                **{
                    key: cell[key]
                    for key in (
                        "papers",
                        "share_of_analyzed",
                        "share_within_row",
                        "share_within_column",
                    )
                },
            }
            for cell in payload["cells"]
        ]
    if tactic == "entrepreneurship":
        specification_rows = [
            {
                "record_type": "specification_distribution",
                "population": group["label"],
                "comparison_role": group["comparison_role"],
                "dimension": payload["dimension_label"],
                "distribution": payload["distribution"],
                "denominator": group["denominator"],
                "category": category["value"],
                "papers": category["papers"],
                "share": category["share"],
                "percentage_point_difference_from_combined": category[
                    "percentage_point_difference_from_combined"
                ],
            }
            for group in payload["groups"]
            for category in group["categories"]
        ]
        configuration_rows = [
            {
                "record_type": "recurring_configuration",
                "population": value["population_label"],
                "comparison_role": (
                    "Union benchmark"
                    if value["population"] == "combined"
                    else "Journal set"
                ),
                "denominator": value["analyzed_n"],
                "ai_role": configuration["ai_role"],
                "mechanism": configuration["mechanism"],
                "level": configuration["level"],
                "scope": configuration["scope"],
                "process_stage": configuration["process_stage"],
                "papers": value["papers"],
                "share": value["share"],
            }
            for configuration in payload["configurations"]
            for value in configuration["population_values"]
        ]
        return [*specification_rows, *configuration_rows]
    if tactic == "structuring_matrix":
        matrix = payload["matrix"]
        return [
            {
                "population": payload["population_label"],
                "matrix": payload["pair_label"],
                "row_value": cell["row_value"],
                "column_value": cell["column_value"],
                **{
                    key: cell[key]
                    for key in (
                        "papers",
                        "share_of_analyzed",
                        "share_within_row",
                        "share_within_column",
                    )
                },
            }
            for cell in matrix["cells"]
        ]
    if tactic == "structuring":
        return [
            {
                "population": payload["population_label"],
                "distribution": payload["configurations"]["distribution"],
                "analyzed_papers": payload["configurations"]["analyzed_n"],
                "minimum_support": payload["configurations"]["min_support"],
                "ai_role": row["ai_role"],
                "mechanism": row["mechanism"],
                "level": row["level"],
                "scope": row["scope"],
                "process_stage": row["process_stage"],
                "papers": row["papers"],
                "share": row["share"],
            }
            for row in payload["configurations"]["configurations"]
        ]
    raise ValueError(f"Unknown contrasting tactic: {tactic}")


def _theory_payload(
    tactic: str,
    model: str,
    population: str,
    journal_scope: str,
    study_status: str,
    distribution: str,
    dimension: str,
    row_dimension: str,
    column_dimension: str,
    pair: str,
    min_support: int,
    control_dimension: str | None,
    control_value: str | None,
) -> dict:
    svc = service()
    if tactic == "construct":
        return svc.theory_construct_specification(
            model, population, journal_scope, study_status
        )
    if tactic == "horizontal":
        return svc.theory_horizontal_contrast(
            model,
            dimension,
            distribution,
            journal_scope,
            study_status,
            control_dimension,
            control_value,
        )
    if tactic == "vertical":
        return svc.theory_vertical_contrast(
            model,
            population,
            row_dimension,
            distribution,
            journal_scope,
            study_status,
            column_dimension,
            control_dimension,
            control_value,
        )
    if tactic == "entrepreneurship":
        return svc.theory_entrepreneurship_comparison(
            model,
            dimension,
            distribution,
            study_status,
            min_support,
            control_dimension,
            control_value,
        )
    if tactic == "structuring":
        return svc.theory_structuring(
            model,
            population,
            pair,
            distribution,
            journal_scope,
            study_status,
            min_support,
            control_dimension,
            control_value,
        )
    raise ValueError(f"Unknown contrasting tactic: {tactic}")


def _theory_context(
    tactic: str,
    model: str,
    population: str,
    journal_scope: str,
    study_status: str,
    distribution: str,
) -> dict[str, object]:
    return {
        "analysis_tactic": tactic,
        "coding_model": model,
        "entrepreneurship_population": population,
        "journal_scope": journal_scope,
        "study_status_filter": study_status,
        "evidence_view": distribution,
        "evidence_boundary": "title, abstract, and author keywords",
    }


def _theory_release_response(
    model: str,
    population: str,
    journal_scope: str,
    study_status: str,
    distribution: str,
    min_support: int,
) -> Response:
    """Build a complete, checksummed release for all available tactics."""

    import pandas as pd

    svc = service()
    metadata = svc.theory_contrasting_metadata(model)
    files: dict[str, bytes] = {}

    def add(name: str, content: str | bytes) -> None:
        files[name] = content.encode("utf-8") if isinstance(content, str) else content

    construct = svc.theory_construct_specification(
        model, population, "all", study_status
    )
    add(
        "construct_specification/entrepreneurship_specification.csv",
        pd.DataFrame(
            _theory_result_rows("construct", construct, distribution)
        ).to_csv(index=False),
    )
    for dimension in metadata["dimensions"]:
        horizontal = svc.theory_horizontal_contrast(
            model,
            dimension["id"],
            distribution,
            journal_scope,
            study_status,
            None,
            None,
        )
        add(
            f"horizontal/{journal_scope}/{dimension['id']}.csv",
            pd.DataFrame(
                row
                for row in _theory_result_rows(
                    "horizontal", horizontal, distribution
                )
                if row["record_type"] == "domain_comparison"
            ).to_csv(index=False),
        )
        entrepreneurship = svc.theory_entrepreneurship_comparison(
            model,
            dimension["id"],
            distribution,
            study_status,
            min_support,
            None,
            None,
        )
        entrepreneurship_rows = _theory_result_rows(
            "entrepreneurship", entrepreneurship, distribution
        )
        add(
            f"within_entrepreneurship/specification/{dimension['id']}.csv",
            pd.DataFrame(
                row
                for row in entrepreneurship_rows
                if row["record_type"] == "specification_distribution"
            ).to_csv(index=False),
        )
        if dimension["id"] == metadata["dimensions"][0]["id"]:
            add(
                "within_entrepreneurship/recurring_configurations.csv",
                pd.DataFrame(
                    row
                    for row in entrepreneurship_rows
                    if row["record_type"] == "recurring_configuration"
                ).to_csv(index=False),
            )
    for row_dimension in metadata["vertical_row_dimensions"]:
        if row_dimension == "level":
            continue
        vertical = svc.theory_vertical_contrast(
            model,
            population,
            row_dimension,
            distribution,
            "all",
            study_status,
            "level",
        )
        add(
            f"vertical/{row_dimension}_by_level.csv",
            pd.DataFrame(
                _theory_result_rows("vertical", vertical, distribution)
            ).to_csv(index=False),
        )
    structuring_payloads = []
    for pair_option in metadata["structuring_pairs"]:
        structuring = svc.theory_structuring(
            model,
            population,
            pair_option["id"],
            distribution,
            "all",
            study_status,
            min_support,
        )
        structuring_payloads.append(structuring)
        add(
            f"structuring/matrices/{pair_option['id']}.csv",
            pd.DataFrame(
                _theory_result_rows(
                    "structuring_matrix", structuring, distribution
                )
            ).to_csv(index=False),
        )
    if structuring_payloads:
        add(
            "structuring/recurring_configurations.csv",
            pd.DataFrame(
                _theory_result_rows(
                    "structuring", structuring_payloads[0], distribution
                )
            ).to_csv(index=False),
        )
    evidence = svc.theory_contrasting_evidence(
        model,
        population=population,
        journal_scope="all",
        study_status=study_status,
        limit=50000,
    )
    add(
        "evidence/filtered_entrepreneurship_papers.csv",
        pd.DataFrame(evidence["papers"]).to_csv(index=False),
    )
    context = _theory_context(
        "all", model, population, "tactic-specific", study_status, distribution
    )
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest = {
        **context,
        "generated_at": generated_at,
        "model_label": metadata["model_label"],
        "model_coded_papers": metadata["model_coded_papers"],
        "corpus_papers": metadata["corpus_papers"],
        "available_domains": metadata["domains"],
        "pending_domains": metadata["pending_domains"],
        "domain_assignment_complete": metadata["domain_assignment_complete"],
        "journal_scope_by_tactic": {
            "construct_specification": "all",
            "horizontal_contrasting": journal_scope,
            "vertical_contrasting": "all",
            "structuring": "all",
            "within_entrepreneurship": "core, additional, and combined journal sets",
        },
        "raw_model_records_changed": False,
        "files": [
            {
                "path": name,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for name, content in sorted(files.items())
        ],
    }
    add("manifest.json", json.dumps(manifest, indent=2))
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            archive.writestr(name, content)
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f'attachment; filename="etv_construct_contrasting_{generated_at[:10]}.zip"'
            ),
            "Cache-Control": "no-store",
        },
    )


@app.get("/api/contrasting/metadata")
def theory_contrasting_metadata(model: str | None = Query(None)) -> dict:
    try:
        return service().theory_contrasting_metadata(model)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/contrasting/construct")
def theory_construct_specification(
    model: str = Query("gpt-5.4-mini-2026-03-17"),
    population: str = Query("combined", pattern="^(core|other|combined)$"),
    journal_scope: str = Query("all", pattern="^(all|ft50)$"),
    study_status: str = Query("all", pattern="^(all|phenomenon|method|both)$"),
) -> dict:
    try:
        return service().theory_construct_specification(
            model, population, journal_scope, study_status
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/contrasting/horizontal")
def theory_horizontal_contrast(
    model: str = Query("gpt-5.4-mini-2026-03-17"),
    dimension: str = Query("ai_role"),
    distribution: str = Query("observed", pattern="^(full|observed)$"),
    journal_scope: str = Query("all", pattern="^(all|ft50)$"),
    study_status: str = Query("all", pattern="^(all|phenomenon|method|both)$"),
    control_dimension: str | None = Query(None),
    control_value: str | None = Query(None),
) -> dict:
    try:
        return service().theory_horizontal_contrast(
            model,
            dimension,
            distribution,
            journal_scope,
            study_status,
            control_dimension,
            control_value,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/contrasting/vertical")
def theory_vertical_contrast(
    model: str = Query("gpt-5.4-mini-2026-03-17"),
    population: str = Query("combined", pattern="^(core|other|combined)$"),
    row_dimension: str = Query("ai_role"),
    column_dimension: str = Query("level"),
    distribution: str = Query("observed", pattern="^(full|observed)$"),
    journal_scope: str = Query("all", pattern="^(all|ft50)$"),
    study_status: str = Query("all", pattern="^(all|phenomenon|method|both)$"),
    control_dimension: str | None = Query(None),
    control_value: str | None = Query(None),
) -> dict:
    try:
        return service().theory_vertical_contrast(
            model,
            population,
            row_dimension,
            distribution,
            journal_scope,
            study_status,
            column_dimension,
            control_dimension,
            control_value,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/contrasting/entrepreneurship")
def theory_entrepreneurship_comparison(
    model: str = Query("gpt-5.4-mini-2026-03-17"),
    dimension: str = Query("ai_role"),
    distribution: str = Query("observed", pattern="^(full|observed)$"),
    study_status: str = Query("all", pattern="^(all|phenomenon|method|both)$"),
    min_support: int = Query(10, ge=1, le=10000),
    control_dimension: str | None = Query(None),
    control_value: str | None = Query(None),
) -> dict:
    try:
        return service().theory_entrepreneurship_comparison(
            model,
            dimension,
            distribution,
            study_status,
            min_support,
            control_dimension,
            control_value,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/contrasting/structuring")
def theory_structuring(
    model: str = Query("gpt-5.4-mini-2026-03-17"),
    population: str = Query("combined", pattern="^(core|other|combined)$"),
    pair: str = Query("ai_role__mechanism"),
    distribution: str = Query("observed", pattern="^(full|observed)$"),
    journal_scope: str = Query("all", pattern="^(all|ft50)$"),
    study_status: str = Query("all", pattern="^(all|phenomenon|method|both)$"),
    min_support: int = Query(10, ge=1, le=10000),
    control_dimension: str | None = Query(None),
    control_value: str | None = Query(None),
) -> dict:
    try:
        return service().theory_structuring(
            model,
            population,
            pair,
            distribution,
            journal_scope,
            study_status,
            min_support,
            control_dimension,
            control_value,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/contrasting/evidence")
def theory_contrasting_evidence(
    model: str = Query("gpt-5.4-mini-2026-03-17"),
    population: str | None = Query(None, pattern="^(core|other|combined)$"),
    journal_scope: str = Query("all", pattern="^(all|ft50)$"),
    study_status: str = Query("all", pattern="^(all|phenomenon|method|both)$"),
    domain: str | None = Query(None),
    filters: str = Query("{}", max_length=4000),
    limit: int = Query(100, ge=1, le=50000),
) -> dict:
    try:
        parsed_filters = json.loads(filters)
        if not isinstance(parsed_filters, dict):
            raise ValueError("Evidence filters must be a JSON object")
        return service().theory_contrasting_evidence(
            model,
            population=population,
            journal_scope=journal_scope,
            study_status=study_status,
            domain_id=domain,
            filters={str(key): str(value) for key, value in parsed_filters.items()},
            limit=limit,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/contrasting/report", response_class=HTMLResponse)
def theory_contrasting_report(
    tactic: str = Query(
        "construct", pattern="^(construct|horizontal|vertical|structuring|entrepreneurship)$"
    ),
    model: str = Query("gpt-5.4-mini-2026-03-17"),
    population: str = Query("combined", pattern="^(core|other|combined)$"),
    journal_scope: str = Query("all", pattern="^(all|ft50)$"),
    study_status: str = Query("all", pattern="^(all|phenomenon|method|both)$"),
    distribution: str = Query("observed", pattern="^(full|observed)$"),
    dimension: str = Query("ai_role"),
    row_dimension: str = Query("ai_role"),
    column_dimension: str = Query("level"),
    pair: str = Query("ai_role__mechanism"),
    min_support: int = Query(10, ge=1, le=10000),
    control_dimension: str | None = Query(None),
    control_value: str | None = Query(None),
) -> str:
    try:
        payload = _theory_payload(
            tactic,
            model,
            population,
            journal_scope,
            study_status,
            distribution,
            dimension,
            row_dimension,
            column_dimension,
            pair,
            min_support,
            control_dimension,
            control_value,
        )
        rows = _theory_result_rows(tactic, payload, distribution)
        title = f"Construct contrasting: {tactic.replace('_', ' ')}"
        return build_theory_contrasting_report(
            title,
            _theory_context(
                tactic,
                model,
                population,
                journal_scope,
                study_status,
                distribution,
            ),
            rows,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/contrasting/download/{bundle}")
def theory_contrasting_download(
    bundle: str,
    tactic: str = Query(
        "construct", pattern="^(construct|horizontal|vertical|structuring|entrepreneurship)$"
    ),
    model: str = Query("gpt-5.4-mini-2026-03-17"),
    population: str = Query("combined", pattern="^(core|other|combined)$"),
    journal_scope: str = Query("all", pattern="^(all|ft50)$"),
    study_status: str = Query("all", pattern="^(all|phenomenon|method|both)$"),
    distribution: str = Query("observed", pattern="^(full|observed)$"),
    dimension: str = Query("ai_role"),
    row_dimension: str = Query("ai_role"),
    column_dimension: str = Query("level"),
    pair: str = Query("ai_role__mechanism"),
    min_support: int = Query(10, ge=1, le=10000),
    control_dimension: str | None = Query(None),
    control_value: str | None = Query(None),
) -> Response:
    import pandas as pd

    if bundle not in {"current", "release"}:
        raise HTTPException(status_code=400, detail=f"Unknown contrasting bundle: {bundle}")
    try:
        if bundle == "release":
            return _theory_release_response(
                model,
                population,
                journal_scope,
                study_status,
                distribution,
                min_support,
            )
        payload = _theory_payload(
            tactic,
            model,
            population,
            journal_scope,
            study_status,
            distribution,
            dimension,
            row_dimension,
            column_dimension,
            pair,
            min_support,
            control_dimension,
            control_value,
        )
        rows = _theory_result_rows(tactic, payload, distribution)
        content = pd.DataFrame(rows).to_csv(index=False)
        filename = re.sub(r"[^a-z0-9]+", "_", tactic.lower()).strip("_")
        return Response(
            content=content,
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="construct_contrasting_{filename}.csv"'
                ),
                "Cache-Control": "no-store",
            },
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/scope/{scope_id}/contrast")
def contrast(
    scope_id: str,
    shared: str = Query("ai_type_form"),
    differ: str = Query("ai_role_function"),
) -> list[dict]:
    return service().contrast(scope_id, shared, differ)


@app.get("/api/scope/{scope_id}/contrast/evidence")
def contrast_evidence(
    scope_id: str,
    shared: str = Query("ai_type_form"),
    value: str = Query(...),
    differ: str = Query("ai_role_function"),
    limit: int = Query(100, ge=1, le=50000),
) -> dict:
    return service().contrast_evidence(
        scope_id,
        shared_column=shared,
        shared_value=value,
        contrast_column=differ,
        limit=limit,
    )


@app.get("/api/scope/{scope_id}/evidence")
def evidence(
    scope_id: str,
    column: str = Query(...),
    value: str = Query(...),
    limit: int = Query(100, ge=1, le=50000),
) -> list[dict]:
    return service().evidence(scope_id, column, value, limit=limit)


@app.get("/api/paper/{paper_id:path}")
def paper(paper_id: str) -> dict:
    result = service().paper(paper_id)
    if result is None:
        raise HTTPException(status_code=404, detail="paper not found")
    return result


@app.get("/api/paper/{paper_id:path}/graph")
def paper_graph(paper_id: str) -> dict:
    return service().paper_neighbourhood(paper_id)


@app.get("/api/graph/status")
def graph_status() -> dict:
    return service().graph_status()


@app.get("/api/graph/seed")
def graph_seed(
    scope: str = Query("full_corpus"),
    limit: int = Query(30, ge=1, le=100),
    node_types: str = Query("", description="Comma-separated allowed node labels"),
    relationship_types: str = Query(
        "", description="Comma-separated allowed relationship types"
    ),
    specification_label: str | None = Query(None),
    specification_value: str | None = Query(None),
) -> dict:
    try:
        return service().graph_seed(
            scope,
            limit=limit,
            node_types=_csv_set(node_types),
            relationship_types=_csv_set(relationship_types),
            specification_label=specification_label,
            specification_value=specification_value,
        )
    except Exception as error:
        raise _graph_error(error) from error


@app.get("/api/graph/neighborhood")
def graph_neighborhood(
    scope: str = Query("full_corpus"),
    node_id: str = Query(..., min_length=3, max_length=2_000),
    relationship_types: str = Query(""),
) -> dict:
    try:
        return service().graph_neighborhood(
            scope,
            node_id,
            relationship_types=_csv_set(relationship_types),
        )
    except Exception as error:
        raise _graph_error(error) from error


@app.get("/api/graph/expand")
def graph_expand(
    scope: str = Query("full_corpus"),
    node_id: str = Query(..., min_length=3, max_length=2_000),
    relationship_types: str = Query(""),
) -> dict:
    try:
        return service().graph_expand(
            scope,
            node_id,
            relationship_types=_csv_set(relationship_types),
        )
    except Exception as error:
        raise _graph_error(error) from error


@app.get("/api/graph/search")
def graph_search(
    scope: str = Query("full_corpus"),
    text: str = Query(..., min_length=1, max_length=300),
    node_types: str = Query(""),
    limit: int = Query(20, ge=1, le=50),
) -> list[dict]:
    try:
        return service().graph_search(
            scope,
            text,
            node_types=_csv_set(node_types),
            limit=limit,
        )
    except Exception as error:
        raise _graph_error(error) from error


@app.post("/api/graph/cypher")
def graph_cypher(request: CypherQueryRequest) -> dict:
    try:
        return service().graph_cypher(
            request.query,
            request.parameters,
            limit=request.limit,
        )
    except Exception as error:
        raise _graph_error(error) from error


@app.get("/api/scope/{scope_id}/graph")
def scope_graph(
    scope_id: str,
    mode: str = "publications",
    limit: int = 40,
    edge_limit: int = 0,
    include: str = Query("", description="comma-separated node labels to show"),
    min_weight: int = 2,
) -> dict:
    labels = {part.strip() for part in include.split(",") if part.strip()} or None
    try:
        result = service().scope_graph(
            scope_id,
            mode=mode,
            limit=limit,
            include=labels,
            min_weight=min_weight,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if edge_limit and len(result.get("edges", [])) > edge_limit:
        result["edges"] = result["edges"][:edge_limit]
    return result


@app.get("/api/scope/{scope_id}/connected")
def connected(scope_id: str, label: str = Query(...), value: str = Query(...)) -> list[dict]:
    return service().connected_papers(scope_id, label, value)


@app.get("/api/scope/{scope_id}/query")
def scope_query(
    scope_id: str,
    filters: str = Query("", description="col:val,col:val"),
    limit: int = Query(100, ge=1, le=50000),
) -> list[dict]:
    parsed = dict(
        part.split(":", 1) for part in filters.split(",") if ":" in part
    )
    return service().query(scope_id, parsed, limit=limit)


@app.get("/api/scope/{scope_id}/values")
def scope_values(scope_id: str, column: str = Query(...)) -> list[str]:
    return service().distinct_values(scope_id, column)


@app.get("/api/scope/{scope_id}/report", response_class=HTMLResponse)
def scope_report(scope_id: str) -> str:
    return build_scope_report(service(), scope_id)


@app.get("/api/scope/{scope_id}/download")
def scope_download(
    scope_id: str,
    filters: str = Query("", description="Optional exact filters: col:val,col:val"),
) -> Response:
    """Download one analytical scope with a machine-readable provenance manifest."""

    parsed = dict(
        part.split(":", 1) for part in filters.split(",") if ":" in part
    )
    try:
        frame = service().export_scope(scope_id, parsed)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    scope = SCOPE_BY_ID.get(scope_id)
    scope_label = scope.label if scope else scope_id
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    slug = re.sub(r"[^a-z0-9]+", "_", scope_id.lower()).strip("_") or "scope"
    date_stamp = generated_at[:10]
    csv_name = f"etv_{slug}_{len(frame)}_papers.csv"
    archive_name = f"etv_{slug}_{len(frame)}_papers_{date_stamp}.zip"
    manifest = {
        "scope_id": scope_id,
        "scope_label": scope_label,
        "filters": parsed,
        "paper_count": len(frame),
        "column_count": len(frame.columns),
        "columns": frame.columns.tolist(),
        "generated_at": generated_at,
        "data_file": csv_name,
    }

    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(csv_name, frame.to_csv(index=False))
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))

    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{archive_name}"',
            "Cache-Control": "no-store",
            "X-ETV-Scope": scope_id,
            "X-ETV-Paper-Count": str(len(frame)),
            "X-ETV-Generated-At": generated_at,
        },
    )


@app.get("/api/human-annotation/instrument")
def human_annotation_instrument() -> dict:
    """Return the frozen, model-blind human coding instrument."""

    try:
        return human_annotation_store().instrument()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/api/human-annotation/progress")
def human_annotation_progress(
    annotator_id: str | None = Query(None, max_length=40),
) -> dict:
    """Return independent completion totals for every known annotator."""

    try:
        return human_annotation_store().progress(annotator_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/human-annotation/paper")
def human_annotation_paper(
    annotator_id: str = Query(..., min_length=2, max_length=40),
    paper_id: str | None = Query(None, max_length=180),
) -> dict:
    """Return one blinded paper and the annotator's resumable draft."""

    try:
        return human_annotation_store().paper(annotator_id, paper_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/human-annotation/save")
def save_human_annotation(request: HumanAnnotationSaveRequest) -> dict:
    """Save a draft or completed paper with an append-only audit revision."""

    _require_authenticated_annotation_write()
    try:
        return human_annotation_store().save(
            request.annotator_id,
            request.paper_id,
            request.annotation,
            submit=request.submit,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/human-annotation/reliability")
def human_annotation_reliability(
    annotators: str = Query("", max_length=1_000),
    models: str = Query("", max_length=2_000),
) -> dict:
    """Return multi-human/model IRR on one exact balanced paper intersection."""

    try:
        return human_annotation_store().reliability(
            annotator_ids=_csv_set(annotators),
            model_ids=_csv_set(models),
        )
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/human-annotation/export")
def export_human_annotations(
    annotator_id: str | None = Query(None, max_length=40),
) -> Response:
    """Download traceable human codes without exposing model ratings."""

    try:
        frame = human_annotation_store().export(annotator_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    suffix = f"_{annotator_id}" if annotator_id else "_all"
    return Response(
        content=frame.to_csv(index=False),
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                f'attachment; filename="human_annotations{suffix}.csv"'
            ),
            "Cache-Control": "no-store",
        },
    )


@app.get("/api/topic-review")
def topic_review_summary() -> dict:
    """Return review progress across all five independently fitted models."""

    try:
        return topic_review_store().summary()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/api/topic-review/topics")
def topic_review_topics(
    scope: str = Query("full_corpus"),
    status: str = Query("all", pattern="^(all|pending|approved|revise)$"),
    q: str = Query("", max_length=180),
) -> list[dict]:
    """Return searchable topic-label evidence for one data-specific model."""

    try:
        return topic_review_store().records(scope=scope, status=status, query=q)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/topic-review/fitted-papers")
def topic_review_fitted_papers(
    scope: str = Query("full_corpus"),
    topic_id: int = Query(..., ge=0),
    limit: int = Query(100, ge=1, le=50_000),
) -> dict:
    """Return inspectable papers originally fitted to one scope-topic."""

    try:
        return topic_review_store().fitted_papers(
            scope, topic_id, limit=limit
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.patch("/api/topic-review/{scope}/{topic_id}")
def update_topic_review(
    scope: str,
    topic_id: int,
    update: TopicReviewUpdateRequest,
) -> dict:
    """Save one audited label decision without changing topic assignments."""

    _require_authenticated_topic_write()
    try:
        row = topic_review_store().update(
            scope,
            topic_id,
            approved_label=update.approved_label,
            review_status=update.review_status,
            reviewer_notes=update.reviewer_notes,
            reviewer=update.reviewer,
        )
        graph_service = state.get("service")
        papers_loaded = graph_service.reload_data() if graph_service is not None else 0
        return {
            "topic": row,
            "summary": topic_review_store().summary(),
            "papers_loaded": papers_loaded,
        }
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/topic-review/figure/{scope}/{figure_name}")
def topic_review_figure(scope: str, figure_name: str) -> FileResponse:
    """Serve a scope figure whose axes use the latest saved draft labels."""

    try:
        path = topic_review_store().preview_figure_path(scope, figure_name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return FileResponse(path, media_type="image/png", headers=HTML_NO_CACHE_HEADERS)


@app.get("/api/topic-review/download/{bundle}")
def topic_review_download(
    bundle: str,
    scope: str = Query("full_corpus"),
) -> Response:
    """Download checksummed topic, figure, graph, or complete release artifacts."""

    try:
        return _topic_release_response(scope, bundle)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/topic-review/finalize")
def finalize_topic_review(request: TopicFinalizeRequest) -> dict:
    """Apply a complete review and rebuild Stage 4 plus graph CSV exports."""

    _require_authenticated_topic_write()
    if request.confirmation != "APPLY ALL 130 APPROVED LABELS":
        raise HTTPException(status_code=400, detail="Finalization was not confirmed")
    if not topic_finalize_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Topic finalization is already running")
    try:
        summary = topic_review_store().summary()
        if not summary["complete"]:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"All 130 labels must be approved first; "
                    f"currently {summary['approved']}/130 are approved."
                ),
            )
        commands = (
            [sys.executable, "scripts/build_stage4_analysis.py"],
            [sys.executable, "scripts/build_graph.py", "--export-csv"],
        )
        command_output = []
        for command in commands:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=900,
                check=False,
            )
            command_output.append(
                {
                    "command": " ".join(command),
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-4_000:],
                    "stderr": completed.stderr[-4_000:],
                }
            )
            if completed.returncode != 0:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "message": "Derived-output regeneration failed",
                        "commands": command_output,
                    },
                )
        papers_loaded = service().reload_data()
        return {
            "status": "completed",
            "papers_loaded": papers_loaded,
            "neo4j_note": (
                "Stage 4 and graph CSV exports were rebuilt. A configured Neo4j "
                "database is not wiped or reloaded automatically."
            ),
            "commands": command_output,
            "summary": topic_review_store().summary(),
        }
    finally:
        topic_finalize_lock.release()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=HTML_NO_CACHE_HEADERS)


@app.get("/graph")
def legacy_graph_page() -> RedirectResponse:
    return RedirectResponse(
        "/knowledge-graph",
        status_code=307,
        headers=HTML_NO_CACHE_HEADERS,
    )


@app.get("/knowledge-graph")
def graph_page() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "knowledge_graph.html", headers=HTML_NO_CACHE_HEADERS
    )


@app.get("/assistant")
def assistant_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "assistant.html", headers=HTML_NO_CACHE_HEADERS)


@app.get("/composition")
def composition_page() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "observed_composition.html", headers=HTML_NO_CACHE_HEADERS
    )


@app.get("/contrasting")
def contrasting_page() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "construct_contrasting.html", headers=HTML_NO_CACHE_HEADERS
    )


@app.get("/topic-review")
def topic_review_page() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "topic_review.html", headers=HTML_NO_CACHE_HEADERS
    )


@app.get("/human-annotation")
def human_annotation_page() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "human_annotation.html", headers=HTML_NO_CACHE_HEADERS
    )


if STATIC_DIR.exists():
    app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")
