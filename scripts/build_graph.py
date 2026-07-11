"""Build and export or load the project knowledge graph.

Inputs: the processed corpus with optional topic and specification columns.
Outputs: graph node and relationship CSVs and, when requested, Neo4j records.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from aecsp.knowledge_graph.builder import build_publication_graph  # noqa: E402
from aecsp.specification.llm_coder import load_env  # noqa: E402
from aecsp.specification.paths import specification_csv_path  # noqa: E402
from aecsp.specification.schema import SPECIFICATION_COLUMNS  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_DIR = PROCESSED_DIR / "graph"


def load_corpus() -> pd.DataFrame:
    topics_path = PROCESSED_DIR / "master_corpus_topics.csv"
    base_path = PROCESSED_DIR / "master_corpus.csv"
    path = topics_path if topics_path.exists() else base_path
    print(f"Loading corpus: {path.name}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def attach_specifications(master: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
    spec_path = specification_csv_path(PROCESSED_DIR, model=model)
    if not spec_path.exists():
        print("  (no specification file yet — graph will omit specification nodes)")
        return master
    specs = pd.read_csv(spec_path, dtype=str, keep_default_na=False)
    keep = ["paper_id"] + [c for c in SPECIFICATION_COLUMNS if c in specs.columns]
    merged = master.merge(specs[keep], on="paper_id", how="left", suffixes=("", "_spec"))
    print(f"  merged specification codes for {specs['paper_id'].nunique():,} papers")
    return merged


def prepare_topic_columns(master: pd.DataFrame) -> pd.DataFrame:
    """Give the graph builder clean Topic nodes with extraction_method metadata."""

    master = master.copy()
    # BERTopic label becomes the Topic node text (drop the numeric id / -1 outliers).
    if "bertopic_topic_label" in master.columns:
        label = master["bertopic_topic_label"].fillna("")
        master["bertopic_topic"] = label.where(~label.str.startswith("-1"), "")
    # KeyBERT "phrase:score;..." -> "phrase;phrase" so nodes are the phrases only.
    if "keybert_phrases" in master.columns:
        master["keyphrases"] = master["keybert_phrases"].fillna("").map(_strip_scores)
    return master


def _strip_scores(value: str) -> str:
    parts = [p.split(":")[0].strip() for p in str(value).split(";") if p.strip()]
    return ";".join(p for p in parts if p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-csv", action="store_true", help="Write node/rel CSVs.")
    parser.add_argument("--load", action="store_true", help="Load into Neo4j.")
    parser.add_argument("--wipe", action="store_true", help="Clear Neo4j before loading.")
    parser.add_argument("--model", default=None, help="Specification model; defaults to the experiment register primary.")
    args = parser.parse_args()
    if not (args.export_csv or args.load):
        args.export_csv = True  # sensible default: no DB required

    master = load_corpus()
    master = attach_specifications(master, model=args.model)
    master = prepare_topic_columns(master)
    print(f"Building graph from {len(master):,} publications...")

    graph = build_publication_graph(master.to_dict("records"), show_progress=True)
    print(f"  nodes: {graph.node_count():,} | relationships: {graph.relationship_count():,}")
    for label in (
        "Publication", "Author", "Journal", "Year", "SearchQuery", "Institution",
        "Keyword", "Reference", "Topic", "SpecificationProfile",
    ):
        print(f"    {label}: {graph.node_count(label):,}")
    for rel in ("WROTE", "CO_AUTHORED_WITH", "AFFILIATED_WITH", "HAS_KEYWORD", "REFERENCES", "CITES"):
        print(f"    [{rel}]: {graph.relationship_count(rel):,}")

    if args.export_csv:
        GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(graph.to_node_rows()).to_csv(GRAPH_DIR / "nodes.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(graph.to_relationship_rows()).to_csv(
            GRAPH_DIR / "relationships.csv", index=False, encoding="utf-8-sig"
        )
        print(f"  exported CSVs to {GRAPH_DIR}")

    if args.load:
        from aecsp.knowledge_graph.neo4j_loader import connect, load_graph

        env = load_env(PROJECT_ROOT / ".env")
        uri = env.get("NEO4J_URI", "bolt://localhost:7687")
        user = env.get("NEO4J_USER", "neo4j")
        password = env.get("NEO4J_PASSWORD", "aecsp_password")
        print(f"Loading into Neo4j at {uri} (wipe={args.wipe})...")
        driver = connect(uri, user, password)
        try:
            counts = load_graph(driver, graph, wipe=args.wipe, show_progress=True)
        finally:
            driver.close()
        print(f"  loaded {counts['nodes']:,} nodes, {counts['relationships']:,} relationships")

    print("Done.")


if __name__ == "__main__":
    main()
