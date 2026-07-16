"""Build and export or load the project knowledge graph.

Inputs: the processed corpus with optional topic and specification columns.
Outputs: graph node and relationship CSVs and, when requested, Neo4j records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from aecsp.knowledge_graph.builder import build_publication_graph  # noqa: E402
from aecsp.specification.llm_coder import load_env  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_DIR = PROCESSED_DIR / "graph"
ANALYSIS_DIR = PROCESSED_DIR / "analysis"
PRIMARY_DATASET = ANALYSIS_DIR / "primary_analysis_dataset.csv"
DATASET_MANIFEST = ANALYSIS_DIR / "dataset_manifest.json"
STAGE4_MANIFEST = ANALYSIS_DIR / "stage4" / "stage4_manifest.json"

TOPIC_LABEL_COLUMNS = (
    "bertopic_topic_label",
    "query_1_topic_label",
    "query_2_topic_label",
    "query_3_topic_label",
    "query_4_topic_label",
)


def load_corpus() -> pd.DataFrame:
    """Load and verify the frozen primary table, with optional topic columns."""

    if not PRIMARY_DATASET.exists() or not DATASET_MANIFEST.exists():
        raise FileNotFoundError(
            "The canonical primary dataset or its manifest is missing. "
            "Expected data/processed/analysis/primary_analysis_dataset.csv "
            "and dataset_manifest.json."
        )
    manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["primary_dataset"]
    actual_sha = _sha256(PRIMARY_DATASET)
    if actual_sha != expected["sha256"]:
        raise RuntimeError(
            "Canonical dataset checksum mismatch. The graph load was refused. "
            f"Expected {expected['sha256']}; found {actual_sha}."
        )

    print(f"Loading canonical dataset: {_display_path(PRIMARY_DATASET)}")
    print(f"  SHA-256 verified: {actual_sha}")
    master = pd.read_csv(PRIMARY_DATASET, dtype=str, keep_default_na=False)
    if len(master) != int(expected["rows"]):
        raise RuntimeError(
            f"Canonical row count mismatch: expected {expected['rows']:,}; found {len(master):,}."
        )
    unique_ids = master["paper_id"].nunique()
    if unique_ids != int(expected["unique_paper_ids"]):
        raise RuntimeError(
            "Canonical paper_id count mismatch: "
            f"expected {expected['unique_paper_ids']:,}; found {unique_ids:,}."
        )

    topic_candidates = (
        ANALYSIS_DIR / "primary_analysis_dataset_with_topics.csv",
        PROCESSED_DIR / "master_corpus_topics.csv",
    )
    topic_path = next((path for path in topic_candidates if path.exists()), None)
    if topic_path is not None and topic_path != PRIMARY_DATASET:
        topic_frame = pd.read_csv(topic_path, dtype=str, keep_default_na=False)
        new_columns = [
            column
            for column in topic_frame.columns
            if column == "paper_id" or column not in master.columns
        ]
        if len(new_columns) > 1:
            master = master.merge(
                topic_frame[new_columns],
                on="paper_id",
                how="left",
                validate="one_to_one",
            )
            print(
                f"  merged {len(new_columns) - 1} topic-only columns from "
                f"{_display_path(topic_path)}"
            )
    return master


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> Path:
    """Return a concise project-relative path when the file is in the project."""

    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def prepare_topic_columns(master: pd.DataFrame) -> pd.DataFrame:
    """Attach topic review state without replacing stable numeric topic IDs."""

    master = master.copy()
    review_status = "automatic"
    if STAGE4_MANIFEST.exists():
        manifest = json.loads(STAGE4_MANIFEST.read_text(encoding="utf-8"))
        review_status = str(
            manifest.get("topic_label_review", {}).get("status", "automatic")
        )
    for label_column in TOPIC_LABEL_COLUMNS:
        if label_column in master.columns:
            master[f"{label_column}_review_status"] = review_status
    # KeyBERT "phrase:score;..." -> "phrase;phrase" so nodes are the phrases only.
    if "keybert_phrases" in master.columns:
        master["keyphrases"] = master["keybert_phrases"].fillna("").map(_strip_scores)
    return master


def _strip_scores(value: str) -> str:
    parts = [p.split(":")[0].strip() for p in str(value).split(";") if p.strip()]
    return ";".join(p for p in parts if p)


def _print_database_counts(summary: dict) -> None:
    print("Neo4j verification:")
    print(f"  nodes: {summary['nodes']:,}")
    for label, count in summary["node_labels"].items():
        print(f"    {label}: {count:,}")
    print(f"  relationships: {summary['relationships']:,}")
    for relationship, count in summary["relationship_types"].items():
        print(f"    [{relationship}]: {count:,}")
    unexpected_labels = summary.get("unexpected_node_labels", [])
    unexpected_relationships = summary.get("unexpected_relationship_types", [])
    if unexpected_labels or unexpected_relationships:
        raise RuntimeError(
            "Neo4j contains graph elements outside the locked contract: "
            f"labels={unexpected_labels}, relationships={unexpected_relationships}"
        )
    print("  locked node and relationship contract: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-csv", action="store_true", help="Write node/rel CSVs.")
    parser.add_argument("--load", action="store_true", help="Load into Neo4j.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Read and print Neo4j node/relationship counts without loading.",
    )
    parser.add_argument("--wipe", action="store_true", help="Clear Neo4j before loading.")
    args = parser.parse_args()
    if args.wipe and not args.load:
        parser.error("--wipe requires --load")
    if not (args.export_csv or args.load or args.verify):
        args.export_csv = True  # sensible default: no DB required

    env = load_env(PROJECT_ROOT / ".env")
    uri = env.get("NEO4J_URI", "bolt://localhost:7687")
    user = env.get("NEO4J_USER", "neo4j")
    password = env.get("NEO4J_PASSWORD", "aecsp_password")
    database = env.get("NEO4J_DATABASE", "neo4j")

    if args.verify and not (args.export_csv or args.load):
        from aecsp.knowledge_graph.neo4j_loader import connect, database_counts

        driver = connect(uri, user, password)
        try:
            _print_database_counts(database_counts(driver, database))
        finally:
            driver.close()
        return

    master = load_corpus()
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
        from aecsp.knowledge_graph.neo4j_loader import (
            connect,
            database_counts,
            load_graph,
        )

        print(f"Loading into Neo4j at {uri} (wipe={args.wipe})...")
        driver = connect(uri, user, password)
        try:
            counts = load_graph(
                driver,
                graph,
                wipe=args.wipe,
                show_progress=True,
                database=database,
            )
            database_summary = database_counts(driver, database)
        finally:
            driver.close()
        print(f"  loaded {counts['nodes']:,} nodes, {counts['relationships']:,} relationships")
        _print_database_counts(database_summary)

    print("Done.")


if __name__ == "__main__":
    main()
