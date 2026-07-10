"""Stage 3 data-access layer: Neo4j-backed with a CSV fallback.

Tabular analytics (scope overviews, distributions, convergence/contrast,
evidence lists) are computed from the processed paper-level data so the app
works even before Neo4j is running. When a Neo4j driver is available, the
graph-traversal endpoints (paper neighbourhood, contrast network) use it.

Every method that reports a statistic also exposes the underlying paper_ids so
the UI can show the evidence behind any number (brief section 10).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from aecsp.analytics.convergence import (
    DIMENSION_COLUMNS,
    construct_contrast,
    convergence_by,
    group_convergence,
)
from aecsp.corpus.scopes import DATASET_SCOPES, scope_frame
from aecsp.specification.schema import (
    SPECIFICATION_COLUMNS,
    SPECIFICATION_DIMENSIONS,
    SPECIFICATION_PROBLEM_COLUMN,
)

PROCESSED_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"

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
    *DIMENSION_COLUMNS,
    SPECIFICATION_PROBLEM_COLUMN,
]


def _short_title(title: object, width: int = 42) -> str:
    text = str(title or "").strip()
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


class GraphService:
    """Serves scope-aware analytics and evidence to the API."""

    def __init__(self, processed_dir: Path = PROCESSED_DIR, neo4j_driver=None) -> None:
        self.processed_dir = processed_dir
        self.driver = neo4j_driver
        self.papers = self._load_papers()

    # ---- loading --------------------------------------------------------
    def _load_papers(self) -> pd.DataFrame:
        topics = self.processed_dir / "master_corpus_topics.csv"
        base = self.processed_dir / "master_corpus.csv"
        path = topics if topics.exists() else base
        if not path.exists():
            return pd.DataFrame()
        papers = pd.read_csv(path, dtype=str, keep_default_na=False)

        spec_path = self.processed_dir / "specification" / "paper_specifications.csv"
        if spec_path.exists():
            specs = pd.read_csv(spec_path, dtype=str, keep_default_na=False)
            keep = ["paper_id"] + [c for c in SPECIFICATION_COLUMNS if c in specs.columns]
            papers = papers.merge(specs[keep], on="paper_id", how="left", suffixes=("", "_spec"))
        return papers.fillna("")

    @property
    def has_specifications(self) -> bool:
        return bool(self.papers is not None) and any(
            c in self.papers.columns for c in DIMENSION_COLUMNS
        )

    def _scope(self, scope_id: str) -> pd.DataFrame:
        return scope_frame(self.papers, scope_id)

    # ---- overview -------------------------------------------------------
    def scopes(self) -> list[dict]:
        return [
            {"id": s.id, "label": s.label, "papers": len(self._scope(s.id))}
            for s in DATASET_SCOPES
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
            "dimension_convergence": conv.dimension_scores,
            "dominant_values": conv.dominant_values,
        }

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

    # ---- group views (journal / author / topic) -------------------------
    def group_convergence_table(self, scope_id: str, group_column: str) -> list[dict]:
        frame = self._scope(scope_id)
        if group_column not in frame.columns or not self.has_specifications:
            return []
        return convergence_by(frame, group_column).to_dict("records")

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
            "summary": summary,
            "annual_production": annual,
            "top_journals": top_journals,
            "top_authors": top_authors,
            "most_cited": most_cited,
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

    def evidence(self, scope_id: str, column: str, value: str) -> list[dict]:
        """The paper list behind a statistic (brief section 10 traceability)."""

        frame = self._scope(scope_id)
        if column not in frame.columns:
            return []
        subset = frame[frame[column].astype(str) == str(value)]
        cols = [c for c in EVIDENCE_COLUMNS if c in subset.columns]
        return subset[cols].to_dict("records")

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
    # Dimension node colours (aligned with the esd palette family).
    _DIM_COLOURS = {
        "ai_role_function": "#e74c3c",
        "ai_type_form": "#3498db",
        "ai_mechanism": "#27ae60",
        "level_of_analysis": "#f39c12",
        "entrepreneurial_process_stage": "#9b59b6",
    }
    _CORE_DIMS = ("ai_type_form", "ai_role_function", "ai_mechanism")

    # Node colour per graph label (same builder labels the whole platform uses).
    _LABEL_COLOURS = {
        "Publication": "#3498db",
        "Author": "#16a085",
        "Journal": "#bdc3c7",
        "Institution": "#8e44ad",
        "Keyword": "#f39c12",
        "Reference": "#7f8c8d",
        "Topic": "#e67e22",
        "AIRole": "#e74c3c",
        "AIType": "#2980b9",
        "Mechanism": "#27ae60",
        "LevelOfAnalysis": "#d4ac0d",
        "ProcessStage": "#9b59b6",
        "ScopeCondition": "#1abc9c",
        "DefinitionClarity": "#34495e",
        "SpecificationProblem": "#c0392b",
        "SpecificationProfile": "#7f8c8d",
    }
    # On by default; Keyword and Reference are numerous, so off until toggled.
    _CORE_NODE_TYPES = (
        "Publication", "Author", "Journal", "Institution", "Topic",
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
        mode='contrast': the AI role/type/mechanism co-occurrence lens.
        """

        frame = self._scope(scope_id)
        if mode == "contrast" and self.has_specifications and all(d in frame.columns for d in self._CORE_DIMS):
            return self._specification_graph(frame, scope_id, min_weight, 120)
        return self._publication_graph(frame, scope_id, limit, include)

    def _publication_graph(self, frame, scope_id, limit, include) -> dict:
        from collections import Counter

        from aecsp.knowledge_graph.builder import build_publication_graph

        if frame.empty:
            return {"mode": "publications", "scope": scope_id, "legend": {},
                    "nodes": [], "edges": [], "counts": {}}

        include = set(include) if include else set(self._CORE_NODE_TYPES)
        include.add("Publication")

        citations = self._numeric(frame, "Cited by")
        focus = frame.assign(_c=citations).sort_values("_c", ascending=False).head(limit)
        meta_by_pid = self._publication_meta(focus)
        graph = build_publication_graph(focus.drop(columns="_c").to_dict("records"))

        nodes: list[dict] = []
        kept: set[str] = set()
        for node in graph.nodes:
            label = node.ref.label
            if label not in include:
                continue
            value = node.ref.value
            nid = f"{label}::{value}"
            kept.add(nid)
            is_pub = label == "Publication"
            meta = meta_by_pid.get(value, {}) if is_pub else {"type": label, "name": value}
            display = _short_title(meta.get("Title") or value) if is_pub else _short_title(value, 28)
            citations_n = int(meta.get("Citations", 0)) if is_pub else 0
            nodes.append({
                "id": nid,
                "label": display,
                "group": label,
                "value": 8 + min(citations_n, 60) if is_pub else 5,
                "color": self._LABEL_COLOURS.get(label, "#95a5a6"),
                "meta": meta,
            })

        edges: list[dict] = []
        for rel in graph.relationships:
            source = f"{rel.start.label}::{rel.start.value}"
            target = f"{rel.end.label}::{rel.end.value}"
            if source in kept and target in kept:
                edge = {"from": source, "to": target, "type": rel.relationship_type, "value": 1}
                if rel.relationship_type in ("CITES", "REFERENCES"):
                    edge["arrows"] = "to"
                edges.append(edge)

        counts = dict(Counter(n["group"] for n in nodes))
        legend = {g: self._LABEL_COLOURS[g] for g in self._LABEL_COLOURS if g in counts}
        return {"mode": "publications", "scope": scope_id, "legend": legend,
                "counts": counts, "nodes": nodes, "edges": edges}

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
            exact_col = "bertopic_topic_label" if "bertopic_topic_label" in frame.columns else "bertopic_topic"
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

    def _specification_graph(self, frame, scope_id, min_weight, max_nodes) -> dict:
        from itertools import combinations

        dims = [d for d in self._CORE_DIMS if d in frame.columns]
        node_count: dict[str, int] = {}
        edges: dict[tuple[str, str], int] = {}
        for _, row in frame.iterrows():
            present = [
                (d, str(row[d]).strip()) for d in dims if str(row[d]).strip()
            ]
            for dim, value in present:
                nid = f"{dim}::{value}"
                node_count[nid] = node_count.get(nid, 0) + 1
            for (d1, v1), (d2, v2) in combinations(present, 2):
                if d1 == d2:
                    continue
                key = tuple(sorted([f"{d1}::{v1}", f"{d2}::{v2}"]))
                edges[key] = edges.get(key, 0) + 1

        top = dict(sorted(node_count.items(), key=lambda kv: kv[1], reverse=True)[:max_nodes])
        nodes = [
            {
                "id": nid,
                "label": nid.split("::", 1)[1],
                "group": nid.split("::", 1)[0],
                "value": count,
                "color": self._DIM_COLOURS.get(nid.split("::", 1)[0], "#95a5a6"),
            }
            for nid, count in top.items()
        ]
        edge_list = [
            {"from": a, "to": b, "value": w}
            for (a, b), w in edges.items()
            if w >= min_weight and a in top and b in top
        ]
        return {
            "mode": "specification",
            "scope": scope_id,
            "legend": {d: self._DIM_COLOURS.get(d) for d in dims},
            "nodes": nodes,
            "edges": edge_list,
        }

    # ---- graph-native (Neo4j only) -------------------------------------
    def neo4j_available(self) -> bool:
        return self.driver is not None

    def paper_neighbourhood(self, paper_id: str) -> dict:
        """Node + immediate relationships from Neo4j for the network view."""

        if self.driver is None:
            return {"available": False}
        query = (
            "MATCH (p:Publication {id: $pid})-[r]-(n) "
            "RETURN type(r) AS rel, labels(n)[0] AS label, "
            "coalesce(n.name, n.label, n.id, n.value) AS name LIMIT 200"
        )
        with self.driver.session() as session:
            records = [dict(r) for r in session.run(query, pid=paper_id)]
        return {"available": True, "paper_id": paper_id, "edges": records}
