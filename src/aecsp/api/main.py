"""Stage 3 FastAPI app: scope-aware construct-specification analytics.

Run (project root, graphrag env):
    uvicorn aecsp.api.main:app --reload --app-dir src
Then open http://localhost:8000

Neo4j is used when reachable (creds from .env); otherwise the app runs from the
processed CSVs. Every analytics response includes the paper_ids behind it.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from aecsp.api.graph_service import GraphService
from aecsp.api.report import build_scope_report
from aecsp.specification.llm_coder import load_env

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = Path(__file__).resolve().parent / "static"

state: dict = {}


def _connect_neo4j():
    env = load_env(PROJECT_ROOT / ".env")
    uri = env.get("NEO4J_URI")
    if not uri:
        return None
    try:
        from aecsp.knowledge_graph.neo4j_loader import connect

        driver = connect(uri, env.get("NEO4J_USER", "neo4j"), env.get("NEO4J_PASSWORD", ""))
        driver.verify_connectivity()
        return driver
    except Exception:
        return None  # CSV fallback


@asynccontextmanager
async def lifespan(app: FastAPI):
    driver = _connect_neo4j()
    state["service"] = GraphService(neo4j_driver=driver)
    yield
    if driver is not None:
        driver.close()


app = FastAPI(title="ETV_V2 Construct Specification Platform", lifespan=lifespan)


def service() -> GraphService:
    return state["service"]


@app.get("/api/health")
def health() -> dict:
    svc = service()
    return {
        "status": "ok",
        "papers_loaded": len(svc.papers),
        "has_specifications": svc.has_specifications,
        "neo4j": svc.neo4j_available(),
    }


@app.get("/api/scopes")
def scopes() -> list[dict]:
    return service().scopes()


@app.get("/api/scope/{scope_id}/overview")
def overview(scope_id: str) -> dict:
    return service().scope_overview(scope_id)


@app.get("/api/scope/{scope_id}/distribution")
def distribution(scope_id: str, column: str = Query(...)) -> dict:
    return service().dimension_distribution(scope_id, column)


@app.get("/api/scope/{scope_id}/groups")
def groups(scope_id: str, by: str = Query("Source title")) -> list[dict]:
    return service().group_convergence_table(scope_id, by)


@app.get("/api/scope/{scope_id}/performance")
def performance(scope_id: str) -> dict:
    return service().performance(scope_id)


@app.get("/api/scope/{scope_id}/contrast")
def contrast(
    scope_id: str,
    shared: str = Query("ai_type_form"),
    differ: str = Query("ai_role_function"),
) -> list[dict]:
    return service().contrast(scope_id, shared, differ)


@app.get("/api/scope/{scope_id}/evidence")
def evidence(scope_id: str, column: str = Query(...), value: str = Query(...)) -> list[dict]:
    return service().evidence(scope_id, column, value)


@app.get("/api/paper/{paper_id:path}")
def paper(paper_id: str) -> dict:
    result = service().paper(paper_id)
    if result is None:
        raise HTTPException(status_code=404, detail="paper not found")
    return result


@app.get("/api/paper/{paper_id:path}/graph")
def paper_graph(paper_id: str) -> dict:
    return service().paper_neighbourhood(paper_id)


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
    result = service().scope_graph(scope_id, mode=mode, limit=limit, include=labels, min_weight=min_weight)
    if edge_limit and len(result.get("edges", [])) > edge_limit:
        result["edges"] = result["edges"][:edge_limit]
    return result


@app.get("/api/scope/{scope_id}/connected")
def connected(scope_id: str, label: str = Query(...), value: str = Query(...)) -> list[dict]:
    return service().connected_papers(scope_id, label, value)


@app.get("/api/scope/{scope_id}/query")
def scope_query(scope_id: str, filters: str = Query("", description="col:val,col:val")) -> list[dict]:
    parsed = dict(
        part.split(":", 1) for part in filters.split(",") if ":" in part
    )
    return service().query(scope_id, parsed)


@app.get("/api/scope/{scope_id}/values")
def scope_values(scope_id: str, column: str = Query(...)) -> list[str]:
    return service().distinct_values(scope_id, column)


@app.get("/api/scope/{scope_id}/report", response_class=HTMLResponse)
def scope_report(scope_id: str) -> str:
    return build_scope_report(service(), scope_id)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/graph")
def graph_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "knowledge_graph.html")


@app.get("/assistant")
def assistant_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "assistant.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
