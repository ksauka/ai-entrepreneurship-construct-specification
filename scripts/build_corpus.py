"""Run Stages 0 -> 1B: raw Query 1-4 exports to master corpus + VOS files.

Usage (from the ETV_V2 project root):

    python scripts/build_corpus.py [--skip-relevance-filter]

Outputs:
    data/interim/stage0_5_merged.csv            merged, deduplicated, provenance-aware
    data/interim/stage1_rejected_source_title.csv
    data/interim/stage1_5_rejected_relevance.csv
    data/processed/master_corpus.csv            Stage 1.6 master analytical dataset
    data/processed/query_1.csv ... query_4.csv  overlapping query views
    data/exports/vosviewer/vos_*.csv            five VOSviewer scope exports
    data/processed/pipeline_report.json         stage-by-stage statistics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from etv_v2.corpus.ingest import (  # noqa: E402
    PAPER_ID_COLUMN,
    load_query_frame,
    merge_query_frames,
    query_view,
)
from etv_v2.corpus.query_provenance import (  # noqa: E402
    QUERY_COUNT_COLUMN,
    SEARCH_QUERIES,
)
from etv_v2.corpus.relevance import (  # noqa: E402
    RELEVANT_COLUMN,
    STRICT_RELEVANT_COLUMN,
    load_relevance_config,
    score_relevance,
)
from etv_v2.corpus.source_titles import (  # noqa: E402
    SOURCE_TITLE_VALID_COLUMN,
    load_source_title_universe,
    validate_source_titles,
)
from etv_v2.vos.export import export_vos_scopes  # noqa: E402

CONFIG_DIR = PROJECT_ROOT / "configs"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VOS_DIR = PROJECT_ROOT / "data" / "exports" / "vosviewer"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-relevance-filter",
        action="store_true",
        help="Keep relevance columns but do not drop irrelevant records.",
    )
    parser.add_argument(
        "--keep-invalid-source-titles",
        action="store_true",
        help="Keep records whose source title is outside the Query 1-4 universe.",
    )
    args = parser.parse_args()

    report: dict = {"timestamp": datetime.now().isoformat(), "stages": {}}
    started = time.time()

    # ---- Stage 0: load raw exports -------------------------------------
    print("Stage 0: loading raw Query 1-4 exports...")
    with open(CONFIG_DIR / "data_sources.yaml", encoding="utf-8") as handle:
        sources = yaml.safe_load(handle)

    query_frames = {}
    stage0 = {}
    for query_id, rel_paths in sources["raw_exports"].items():
        paths = [PROJECT_ROOT / rel for rel in rel_paths]
        frame = load_query_frame(paths)
        query_frames[query_id] = frame
        expected = sources.get("expected_counts", {}).get(query_id)
        stage0[query_id] = {
            "files": len(paths),
            "records": len(frame),
            "expected": expected,
            "matches_expected": (expected is None or len(frame) == expected),
        }
        print(f"  {query_id}: {len(frame):,} records (expected {expected:,})")
    report["stages"]["stage_0"] = stage0

    # ---- Stage 0.5: merge + dedup + provenance --------------------------
    print("Stage 0.5: merging and deduplicating with query provenance...")
    master = merge_query_frames(query_frames)
    total_raw = sum(len(f) for f in query_frames.values())
    overlap = master[master[QUERY_COUNT_COLUMN] > 1]
    report["stages"]["stage_0_5"] = {
        "raw_records": total_raw,
        "unique_publications": len(master),
        "duplicates_collapsed": total_raw - len(master),
        "multi_query_publications": len(overlap),
        "query_count_distribution": {
            str(k): int(v)
            for k, v in master[QUERY_COUNT_COLUMN].value_counts().items()
        },
    }
    print(
        f"  {total_raw:,} raw -> {len(master):,} unique publications "
        f"({total_raw - len(master):,} duplicates collapsed, "
        f"{len(overlap):,} captured by multiple queries)"
    )
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    master.to_csv(INTERIM_DIR / "stage0_5_merged.csv", index=False, encoding="utf-8-sig")

    # ---- Stage 1: source-title validation --------------------------------
    print("Stage 1: validating source titles against the query universe...")
    universe = load_source_title_universe(CONFIG_DIR / "search_queries_july2026_q1_q4.yaml")
    master = validate_source_titles(master, universe["combined"])
    invalid = master[master[SOURCE_TITLE_VALID_COLUMN] == 0]
    report["stages"]["stage_1"] = {
        "universe_size": len(universe["combined"]),
        "valid_records": int(master[SOURCE_TITLE_VALID_COLUMN].sum()),
        "invalid_records": len(invalid),
    }
    print(
        f"  universe of {len(universe['combined']):,} source titles; "
        f"{len(invalid):,} records outside it"
    )
    if len(invalid):
        invalid.to_csv(
            INTERIM_DIR / "stage1_rejected_source_title.csv", index=False, encoding="utf-8-sig"
        )
    if not args.keep_invalid_source_titles:
        master = master[master[SOURCE_TITLE_VALID_COLUMN] == 1].reset_index(drop=True)

    # ---- Stage 1.5: AI x entrepreneurship relevance ----------------------
    print("Stage 1.5: scoring AI x entrepreneurship relevance...")
    relevance_config = load_relevance_config(CONFIG_DIR / "ai_keyword_config.yaml")
    master = score_relevance(master, relevance_config)
    rejected = master[master[RELEVANT_COLUMN] == 0]
    report["stages"]["stage_1_5"] = {
        "scored_records": len(master),
        "corpus_relevant_records": int(master[RELEVANT_COLUMN].sum()),
        "strict_ai_ent_relevant_records": int(master[STRICT_RELEVANT_COLUMN].sum()),
        "rejected_records": len(rejected),
        "filter_applied": not args.skip_relevance_filter,
    }
    print(
        f"  broad corpus_relevant: {int(master[RELEVANT_COLUMN].sum()):,} / {len(master):,} scored "
        f"({len(rejected):,} rejected)"
    )
    print(
        f"  strict ai_ent_relevant (kept as column): "
        f"{int(master[STRICT_RELEVANT_COLUMN].sum()):,}"
    )
    if len(rejected):
        rejected.to_csv(
            INTERIM_DIR / "stage1_5_rejected_relevance.csv", index=False, encoding="utf-8-sig"
        )
    if not args.skip_relevance_filter:
        master = master[master[RELEVANT_COLUMN] == 1].reset_index(drop=True)

    # ---- Stage 1.6: master corpus + query views --------------------------
    print("Stage 1.6: writing master corpus and query views...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    master.to_csv(PROCESSED_DIR / "master_corpus.csv", index=False, encoding="utf-8-sig")
    stage16 = {"master_records": len(master), "views": {}}
    for query in SEARCH_QUERIES:
        view = query_view(master, query.id)
        view.to_csv(PROCESSED_DIR / f"{query.id}.csv", index=False, encoding="utf-8-sig")
        stage16["views"][query.id] = len(view)
        print(f"  {query.id}: {len(view):,} records")
    report["stages"]["stage_1_6"] = stage16
    print(f"  master corpus: {len(master):,} records")

    # ---- Stage 1B: VOSviewer exports --------------------------------------
    print("Stage 1B: exporting VOSviewer files for the five scopes...")
    vos_stats = export_vos_scopes(master, VOS_DIR)
    report["stages"]["stage_1b"] = vos_stats
    for scope, info in vos_stats.items():
        print(f"  {scope}: {info['records']:,} records")

    report["paper_id_column"] = PAPER_ID_COLUMN
    report["runtime_seconds"] = round(time.time() - started, 1)
    with open(PROCESSED_DIR / "pipeline_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Done in {report['runtime_seconds']}s. Report: data/processed/pipeline_report.json")


if __name__ == "__main__":
    main()
