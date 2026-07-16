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
from aecsp.api.report import build_composition_report, build_scope_report
from aecsp.corpus.scopes import SCOPE_BY_ID
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


@app.get("/topic-review")
def topic_review_page() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "topic_review.html", headers=HTML_NO_CACHE_HEADERS
    )


if STATIC_DIR.exists():
    app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")
