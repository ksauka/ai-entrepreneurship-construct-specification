"""Optimize and train global and per-query BERTopic models.

Inputs: the master corpus, domain seed configuration, and optional checkpoints
or approved optimization settings. Outputs: topic assignments, term tables,
diagnostics, model artifacts, an enriched master corpus, and a run report.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from aecsp.corpus.query_provenance import SEARCH_QUERIES  # noqa: E402
from aecsp.corpus.scopes import iter_scopes  # noqa: E402
from aecsp.progress import ProgressReporter  # noqa: E402
from aecsp.topics.pipeline import extraction, optimization, phrase_detection, training  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TOPICS_DIR = PROCESSED_DIR / "topics"
TOPIC_RECOMMENDATIONS_PATH = TOPICS_DIR / "optimization" / "recommendations.json"
TOPIC_APPROVAL_PATH = TOPICS_DIR / "optimization" / "topic_selection_review.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "ai_keyword_config.yaml"
TOPIC_INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "topics"
CHECKPOINT_VERSION = 1

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Column contract consumed by scripts/build_graph.py and the Stage 3 app.
TOPIC_ID_COLUMN = "bertopic_topic"
TOPIC_LABEL_COLUMN = "bertopic_topic_label"
TOPIC_PROB_COLUMN = "bertopic_topic_prob"
OUTLIER_COLUMN = "bertopic_was_outlier"
KEYBERT_COLUMN = "keybert_phrases"

TEXT_COLUMNS = ("Title", "Abstract", "Author Keywords", "Index Keywords")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("run_topics")


def scaled_min_topic_size(n_docs: int) -> int:
    """Smaller datasets need smaller topics: ~2% of corpus, clamped to [8, 30]."""

    return int(min(30, max(8, n_docs // 50)))


def build_unified_texts(
    master: pd.DataFrame, min_text_len: int
) -> tuple[list[str], list[int], pd.DataFrame]:
    """Unified text per paper plus the length gate the reference pipeline applies."""

    unified: list[str] = []
    for _, row in master.iterrows():
        parts = []
        for column in TEXT_COLUMNS:
            value = str(row.get(column, "")).strip()
            if column == "Abstract" and value == "[No abstract available]":
                value = ""
            if value:
                parts.append(value)
        unified.append(". ".join(parts).strip(". ").strip())

    df_work = master.copy()
    df_work["__Unified_Text__"] = unified

    documents: list[str] = []
    original_idx: list[int] = []
    for idx, text in enumerate(unified):
        if len(text) >= min_text_len:
            documents.append(text)
            original_idx.append(idx)

    logger.info("Valid documents for modeling: %s", f"{len(documents):,}")
    logger.info("Skipped (too short): %s", f"{len(master) - len(documents):,}")
    return documents, original_idx, df_work


def build_doc_index(df_work: pd.DataFrame, original_idx: list[int]) -> pd.DataFrame:
    """doc_local_index <-> original_index map with the ids extraction needs."""

    rows = []
    for local_i, orig_i in enumerate(original_idx):
        row = df_work.iloc[orig_i]
        rows.append(
            {
                "doc_local_index": local_i,
                "original_index": orig_i,
                "paper_id": row.get("paper_id", ""),
                "EID": row.get("EID", "") or None,
                "DOI": row.get("DOI", "") or None,
                "Title": row.get("Title", "") or None,
                "YEAR": row.get("Year", "") or None,
            }
        )
    return pd.DataFrame(rows)


def load_seed_terms() -> tuple[list[str], list[str]]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return (
        config.get("ai_seed_terms_expanded", []),
        config.get("entrepreneurship_seed_terms", []),
    )


def load_approved_topic_selections(
    recommendations_path: Path = TOPIC_RECOMMENDATIONS_PATH,
    approval_path: Path = TOPIC_APPROVAL_PATH,
) -> tuple[dict[str, dict], dict]:
    """Load human-approved grid choices and validate them against the grid.

    The optimization output is deliberately advisory. Final training must use
    the separately recorded approval artifact so a fresh optimization run
    cannot silently turn an automatic recommendation into a final parameter.
    """

    if not recommendations_path.exists():
        raise ValueError("a completed --optimize-only run is required")
    if not approval_path.exists():
        raise ValueError(
            "explicit topic approval is missing; review the grid graphs and "
            "candidate evidence, then create topic_selection_review.json"
        )

    recommendations_payload = json.loads(
        recommendations_path.read_text(encoding="utf-8")
    )
    approval_payload = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval_payload.get("approval_status") != "approved":
        raise ValueError("topic selections are not marked approved")

    recommendations = recommendations_payload.get("scopes", {})
    approvals = approval_payload.get("scopes", {})
    expected_scopes = {"full_corpus", *(query.id for query in SEARCH_QUERIES)}
    missing = sorted(expected_scopes - approvals.keys())
    if missing:
        raise ValueError(f"topic approval is missing scope(s): {', '.join(missing)}")

    selected: dict[str, dict] = {}
    for scope_id in sorted(expected_scopes):
        if scope_id not in recommendations:
            raise ValueError(f"optimization recommendations are missing {scope_id}")
        approval = approvals[scope_id]
        approved_size = int(approval["selected_min_topic_size"])
        candidates = {
            int(value)
            for value in recommendations[scope_id].get(
                "candidate_min_topic_sizes", []
            )
        }
        if approved_size not in candidates:
            raise ValueError(
                f"approved min_topic_size={approved_size} for {scope_id} "
                f"was not in the tested grid {sorted(candidates)}"
            )
        selected[scope_id] = {
            **recommendations[scope_id],
            "automatic_recommended_min_topic_size": recommendations[scope_id][
                "recommended_min_topic_size"
            ],
            "recommended_min_topic_size": approved_size,
            "approved_topic_count": int(approval["selected_topic_count"]),
            "selection_status": approval.get("decision", "human_approved"),
        }

    return selected, approval_payload


def run_topic_model(
    documents: list[str],
    doc_index_df: pd.DataFrame,
    sent_model,
    nr_topics,
    min_topic_size: int,
    min_reassign_prob: float,
    out_dir: Path,
    embeddings=None,
):
    """Train one BERTopic model and write its per-model artifacts."""

    model = training.train_bertopic(
        documents=documents,
        embedding_model=sent_model,
        nr_topics=nr_topics,
        min_topic_size=min_topic_size,
        calculate_probabilities=True,
        seed=42,
        embeddings=embeddings,
    )
    topics_df = extraction.extract_topics(model)
    doc_topics_df = extraction.assign_document_topics(
        model,
        documents,
        min_reassign_prob=min_reassign_prob,
        reassign_outliers=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    topics_df.to_csv(out_dir / "topics_summary.csv", index=False, encoding="utf-8-sig")
    extraction.export_bertopic_artifacts(model, doc_topics_df, doc_index_df, out_dir)
    return model, topics_df, doc_topics_df


def _checkpoint_signature(master_path: Path, min_text_len: int) -> dict:
    master_stat = master_path.stat()
    config_stat = CONFIG_PATH.stat()
    return {
        "version": CHECKPOINT_VERSION,
        "master_size": master_stat.st_size,
        "master_mtime_ns": master_stat.st_mtime_ns,
        "config_size": config_stat.st_size,
        "config_mtime_ns": config_stat.st_mtime_ns,
        "min_text_len": min_text_len,
        "embedding_model": EMBEDDING_MODEL,
    }


def _load_topic_checkpoint(signature: dict):
    meta_path = TOPIC_INTERIM_DIR / "checkpoint.json"
    docs_path = TOPIC_INTERIM_DIR / "phrase_documents.json.gz"
    embeddings_path = TOPIC_INTERIM_DIR / "embeddings.npy"
    stats_path = TOPIC_INTERIM_DIR / "keyphrase_stats.json"
    if not all(path.exists() for path in (meta_path, docs_path, embeddings_path, stats_path)):
        return None
    if json.loads(meta_path.read_text(encoding="utf-8")) != signature:
        return None
    import numpy as np

    with gzip.open(docs_path, "rt", encoding="utf-8") as handle:
        phrase_documents = json.load(handle)
    embeddings = np.load(embeddings_path)
    keyphrase_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if len(phrase_documents) != len(embeddings):
        return None
    return phrase_documents, embeddings, keyphrase_stats


def _save_topic_checkpoint(signature, phrase_documents, embeddings, keyphrase_stats) -> None:
    import numpy as np

    TOPIC_INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(TOPIC_INTERIM_DIR / "phrase_documents.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(phrase_documents, handle, ensure_ascii=False)
    np.save(TOPIC_INTERIM_DIR / "embeddings.npy", embeddings)
    (TOPIC_INTERIM_DIR / "keyphrase_stats.json").write_text(
        json.dumps(keyphrase_stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (TOPIC_INTERIM_DIR / "checkpoint.json").write_text(
        json.dumps(signature, indent=2), encoding="utf-8"
    )


def _parse_int_list(value: str) -> list[int]:
    values = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not values or any(item < 2 for item in values):
        raise argparse.ArgumentTypeError("provide comma-separated integers >= 2")
    return values


def _parse_topic_range(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    try:
        low, high = (int(part.strip()) for part in value.split(":", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("topic range must look like MIN:MAX") from exc
    if low < 2 or high < low:
        raise argparse.ArgumentTypeError("topic range must satisfy 2 <= MIN <= MAX")
    return low, high


def run_grid_searches(
    phrase_documents,
    embeddings,
    df_work,
    original_idx,
    doc_index_df,
    sent_model,
    min_topic_sizes,
    target_topic_range,
) -> dict:
    """Optimize the global corpus and four native query views independently."""

    root = TOPICS_DIR / "optimization"
    recommendations: dict[str, dict] = {}
    scope_progress = ProgressReporter("Topic optimization scopes", 5, every=1)
    scopes_done = 0

    def optimize_scope(scope_id, docs, scope_embeddings, scope_doc_index):
        nonlocal scopes_done

        eligible_sizes = [size for size in min_topic_sizes if size < len(docs)]
        if scope_id not in {"full_corpus", "query_1"}:
            eligible_sizes = optimization.native_grid_min_topic_sizes(
                len(docs), eligible_sizes
            )
        if not eligible_sizes:
            raise ValueError(f"no min_topic_size is smaller than {scope_id} ({len(docs)} papers)")
        logger.info(
            "%s candidate grid: %s; recommendation floor: %d topics",
            scope_id,
            eligible_sizes,
            optimization.MIN_TOPICS_FOR_RECOMMENDATION,
        )
        recommended, payload = optimization.optimize_topic_count_grid_search(
            documents=docs,
            embeddings=scope_embeddings,
            embedding_model=sent_model,
            min_topic_sizes=eligible_sizes,
            target_topic_range=target_topic_range,
            plot_metrics=True,
            out_dir=root / scope_id,
            document_metadata=scope_doc_index[["paper_id", "Title"]].to_dict("records"),
            min_topics_for_recommendation=optimization.MIN_TOPICS_FOR_RECOMMENDATION,
        )
        recommendations[scope_id] = {
            "papers": len(docs),
            "candidate_min_topic_sizes": eligible_sizes,
            "minimum_topics_for_recommendation": (
                optimization.MIN_TOPICS_FOR_RECOMMENDATION
            ),
            "recommended_min_topic_size": recommended,
            "recommended_topic_count": payload["recommended"]["n_topics"],
            "selection_status": payload["selection_status"],
        }
        scopes_done += 1
        scope_progress.update(scopes_done, detail=scope_id)

    optimize_scope("full_corpus", phrase_documents, embeddings, doc_index_df)
    for query in SEARCH_QUERIES:
        flags = pd.to_numeric(df_work[query.one_hot_column], errors="coerce").fillna(0).astype(int)
        local_indices = [i for i, orig in enumerate(original_idx) if flags.iloc[orig] == 1]
        optimize_scope(
            query.id,
            [phrase_documents[i] for i in local_indices],
            embeddings[local_indices],
            doc_index_df.iloc[local_indices].reset_index(drop=True),
        )

    root.mkdir(parents=True, exist_ok=True)
    with open(root / "recommendations.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "selection_status": "recommendations_require_human_approval",
                "scopes": recommendations,
            },
            handle,
            indent=2,
        )
    return recommendations


def assignment_frame(
    doc_topics_df: pd.DataFrame,
    topics_df: pd.DataFrame,
    doc_index_df: pd.DataFrame,
    id_col: str = TOPIC_ID_COLUMN,
    label_col: str = TOPIC_LABEL_COLUMN,
    prob_col: str = TOPIC_PROB_COLUMN,
    outlier_col: str = OUTLIER_COLUMN,
) -> pd.DataFrame:
    """paper_id + topic id/label/probability/outlier flag for one model."""

    labels = dict(zip(topics_df["topic_id"], topics_df["topic_label"]))
    merged = doc_topics_df.merge(
        doc_index_df[["doc_local_index", "paper_id"]], on="doc_local_index"
    )
    return pd.DataFrame(
        {
            "paper_id": merged["paper_id"],
            id_col: merged["topic_id"].astype("Int64"),
            label_col: [
                labels.get(int(t), "") if pd.notna(t) else ""
                for t in merged["topic_id"]
            ],
            prob_col: pd.to_numeric(merged["topic_prob"], errors="coerce").round(4),
            outlier_col: merged["was_outlier"],
        }
    )


def scope_topic_summary(view: pd.DataFrame) -> pd.DataFrame:
    """Global-topic distribution for one scope: paper_count and share."""

    labels = view[TOPIC_LABEL_COLUMN].fillna("")
    assigned = view[labels != ""]
    if assigned.empty:
        return pd.DataFrame(
            columns=[TOPIC_ID_COLUMN, TOPIC_LABEL_COLUMN, "paper_count", "share"]
        )
    counts = (
        assigned.groupby([TOPIC_ID_COLUMN, TOPIC_LABEL_COLUMN])
        .size()
        .rename("paper_count")
        .reset_index()
        .sort_values("paper_count", ascending=False)
    )
    counts["share"] = (counts["paper_count"] / len(view)).round(4)
    return counts.reset_index(drop=True)


def paper_term_columns(
    doc_index_df: pd.DataFrame,
    ai_by_paper: pd.DataFrame,
    ent_by_paper: pd.DataFrame,
) -> pd.DataFrame:
    """Per-paper AI/ent term columns plus the combined keybert_phrases column.

    keybert_phrases carries term:count pairs joined with ';' so the graph
    builder can split phrases off their scores, matching the old contract.
    """

    frame = doc_index_df[["doc_local_index", "paper_id"]].copy()
    frame = frame.merge(
        ai_by_paper[["doc_local_index", "ai_keywords", "ai_keyword_count"]],
        on="doc_local_index",
        how="left",
    )
    frame = frame.merge(
        ent_by_paper[["doc_local_index", "ent_keywords", "ent_keyword_count"]],
        on="doc_local_index",
        how="left",
    )

    def joined(keywords) -> str:
        if not isinstance(keywords, dict) or not keywords:
            return ""
        return "; ".join(term for term, _ in Counter(keywords).most_common())

    def combined(row) -> str:
        merged = Counter(row["ai_keywords"] if isinstance(row["ai_keywords"], dict) else {})
        merged += Counter(row["ent_keywords"] if isinstance(row["ent_keywords"], dict) else {})
        return ";".join(f"{term}:{count}" for term, count in merged.most_common())

    frame["ai_terms"] = frame["ai_keywords"].map(joined)
    frame["ai_term_count"] = frame["ai_keyword_count"].fillna(0).astype(int)
    frame["ent_terms"] = frame["ent_keywords"].map(joined)
    frame["ent_term_count"] = frame["ent_keyword_count"].fillna(0).astype(int)
    frame[KEYBERT_COLUMN] = frame.apply(combined, axis=1)
    return frame[
        ["paper_id", "ai_terms", "ai_term_count", "ent_terms", "ent_term_count", KEYBERT_COLUMN]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--global-min-topic-size",
        type=int,
        default=None,
        help="Approved final global value. Required unless --use-optimized is supplied.",
    )
    parser.add_argument(
        "--nr-topics", default="auto", help="'auto' or an integer topic count."
    )
    parser.add_argument("--min-text-len", type=int, default=50)
    parser.add_argument("--min-reassign-prob", type=float, default=0.05)
    parser.add_argument("--skip-native", action="store_true", help="Global model only.")
    parser.add_argument("--device", default="cuda", help="SentenceTransformer device.")
    parser.add_argument(
        "--optimize-only",
        action="store_true",
        help="Run five-scope grid searches, write diagnostics, and stop before final models.",
    )
    parser.add_argument(
        "--grid-min-topic-sizes",
        type=_parse_int_list,
        default=_parse_int_list("20,30,40,50,75,100"),
        help="Comma-separated HDBSCAN min_topic_size candidates.",
    )
    parser.add_argument(
        "--target-topic-range",
        type=_parse_topic_range,
        default=None,
        help="Optional review range MIN:MAX; never required or silently forced.",
    )
    parser.add_argument(
        "--use-optimized",
        action="store_true",
        help="Use the separately approved global and native grid-search values.",
    )
    parser.add_argument(
        "--refresh-checkpoint",
        action="store_true",
        help="Recompute phrase documents and embeddings even if a valid checkpoint exists.",
    )
    args = parser.parse_args()

    nr_topics = args.nr_topics if args.nr_topics == "auto" else int(args.nr_topics)

    recommendations = None
    if args.use_optimized:
        try:
            recommendations, approval_payload = load_approved_topic_selections()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            parser.error(f"--use-optimized: {error}")
        args.global_min_topic_size = recommendations["full_corpus"]["recommended_min_topic_size"]
    elif not args.optimize_only and args.global_min_topic_size is None:
        parser.error(
            "final modeling requires --global-min-topic-size N or --use-optimized; "
            "run --optimize-only first"
        )

    started = time.time()
    report: dict = {"timestamp": datetime.now().isoformat(), "parameters": vars(args)}
    if args.use_optimized:
        report["topic_selection_approval"] = approval_payload

    logger.info("Loading master corpus...")
    master_path = PROCESSED_DIR / "master_corpus.csv"
    master = pd.read_csv(master_path, dtype=str, keep_default_na=False)
    master = master.reset_index(drop=True)
    logger.info("%s papers", f"{len(master):,}")

    documents, original_idx, df_work = build_unified_texts(master, args.min_text_len)
    if not documents:
        raise SystemExit("No documents passed the minimum length gate.")
    doc_index_df = build_doc_index(df_work, original_idx)

    logger.info("Loading SentenceTransformer %s on %s...", EMBEDDING_MODEL, args.device)
    from sentence_transformers import SentenceTransformer

    sent_model = SentenceTransformer(EMBEDDING_MODEL, device=args.device)

    ai_seeds, ent_seeds = load_seed_terms()
    all_seeds = ai_seeds + ent_seeds
    logger.info(
        "Encoding %d seed terms (%d AI, %d entrepreneurship)...",
        len(all_seeds), len(ai_seeds), len(ent_seeds),
    )
    seed_embeddings = sent_model.encode(all_seeds, batch_size=32, show_progress_bar=False)

    # ---- Hybrid phrase detection + embeddings: checkpointed once -----------
    signature = _checkpoint_signature(master_path, args.min_text_len)
    checkpoint = None if args.refresh_checkpoint else _load_topic_checkpoint(signature)
    if checkpoint is None:
        _, _, phrase_documents, keyphrase_stats = phrase_detection.detect_phrases(
            documents,
            seed_model=sent_model,
            seed_embeddings=seed_embeddings,
            ai_seeds=ai_seeds,
            ent_seeds=ent_seeds,
            all_seeds=all_seeds,
        )
        logger.info("Encoding phrase-enhanced documents once for optimization and final models...")
        embeddings = sent_model.encode(phrase_documents, show_progress_bar=True)
        _save_topic_checkpoint(signature, phrase_documents, embeddings, keyphrase_stats)
        logger.info("Saved reusable topic checkpoint under data/interim/topics/")
    else:
        phrase_documents, embeddings, keyphrase_stats = checkpoint
        logger.info("Reused phrase documents and embeddings from data/interim/topics/")
    TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TOPICS_DIR / "keyphrases_detected.json", "w", encoding="utf-8") as handle:
        json.dump(keyphrase_stats, handle, indent=2, ensure_ascii=False)
    report["phrase_detection"] = keyphrase_stats["summary"]

    if args.optimize_only:
        recommendations = run_grid_searches(
            phrase_documents,
            embeddings,
            df_work,
            original_idx,
            doc_index_df,
            sent_model,
            args.grid_min_topic_sizes,
            args.target_topic_range,
        )
        logger.info(
            "Optimization complete. Review data/processed/topics/optimization/ before "
            "running with --use-optimized. No final model was selected automatically."
        )
        return

    # ---- Global model (comparability backbone, feeds the KG) --------------
    logger.info(
        "Training GLOBAL BERTopic on %s papers (min_topic_size=%d)...",
        f"{len(phrase_documents):,}", args.global_min_topic_size,
    )
    global_dir = TOPICS_DIR / "global"
    model, topics_df, doc_topics_df = run_topic_model(
        phrase_documents,
        doc_index_df,
        sent_model,
        nr_topics,
        args.global_min_topic_size,
        args.min_reassign_prob,
        global_dir,
        embeddings=embeddings,
    )
    report["global_model"] = {
        "topics": int(len(topics_df)),
        "outliers_unassigned": int(doc_topics_df["topic_id"].isna().sum()),
        "reassigned_outliers": int(
            (doc_topics_df["was_outlier"] & doc_topics_df["topic_id"].notna()).sum()
        ),
        "diagnostics": optimization.export_model_diagnostics(
            model, global_dir / "diagnostics"
        ),
    }
    logger.info("%d global topics", len(topics_df))

    # ---- AI / entrepreneurship term tables ---------------------------------
    ai_by_topic = extraction.extract_ai_terms_by_topic(
        topics_df, doc_topics_df, doc_index_df, df_work, keyphrase_stats
    )
    ai_by_topic.to_csv(global_dir / "ai_terms_by_topic.csv", index=False, encoding="utf-8-sig")
    ent_by_topic = extraction.extract_entrepreneurship_terms_by_topic(
        topics_df, doc_topics_df, doc_index_df, df_work, keyphrase_stats
    )
    ent_by_topic.to_csv(global_dir / "ent_terms_by_topic.csv", index=False, encoding="utf-8-sig")
    ai_by_paper = extraction.extract_ai_terms_by_paper(doc_index_df, df_work, keyphrase_stats)
    ai_by_paper.to_csv(global_dir / "ai_terms_by_paper.csv", index=False, encoding="utf-8-sig")
    ent_by_paper = extraction.extract_entrepreneurship_terms_by_paper(
        doc_index_df, df_work, keyphrase_stats
    )
    ent_by_paper.to_csv(global_dir / "ent_terms_by_paper.csv", index=False, encoding="utf-8-sig")

    # ---- Merge global topics + terms onto the master corpus ---------------
    master = master.merge(
        assignment_frame(doc_topics_df, topics_df, doc_index_df),
        on="paper_id",
        how="left",
    )
    master = master.merge(
        paper_term_columns(doc_index_df, ai_by_paper, ent_by_paper),
        on="paper_id",
        how="left",
    )
    for column in (TOPIC_LABEL_COLUMN, "ai_terms", "ent_terms", KEYBERT_COLUMN):
        master[column] = master[column].fillna("")

    report["global_scope_summaries"] = {}
    for scope_id, view in iter_scopes(master).items():
        summary = scope_topic_summary(view)
        summary.to_csv(
            global_dir / f"topic_summary_{scope_id}.csv", index=False, encoding="utf-8-sig"
        )
        report["global_scope_summaries"][scope_id] = {
            "papers": len(view),
            "top_topic": summary.iloc[0][TOPIC_LABEL_COLUMN] if not summary.empty else None,
        }

    # ---- Native per-query models (reuse the single phrase pass) -----------
    if not args.skip_native:
        report["native_models"] = {}
        native_progress = ProgressReporter("Native topic models", len(SEARCH_QUERIES), every=1)
        for query_number, query in enumerate(SEARCH_QUERIES, start=1):
            flags = (
                pd.to_numeric(df_work[query.one_hot_column], errors="coerce")
                .fillna(0)
                .astype(int)
            )
            local_indices = [
                i for i, orig in enumerate(original_idx) if flags.iloc[orig] == 1
            ]
            docs_q = [phrase_documents[i] for i in local_indices]
            index_q = doc_index_df.iloc[local_indices].reset_index(drop=True).copy()
            index_q["doc_local_index"] = range(len(index_q))

            size = (
                recommendations[query.id]["recommended_min_topic_size"]
                if recommendations is not None
                else scaled_min_topic_size(len(docs_q))
            )
            logger.info(
                "Training NATIVE model for %s (%s papers, min_topic_size=%d)...",
                query.id, f"{len(docs_q):,}", size,
            )
            native_dir = TOPICS_DIR / "native" / query.id
            model_q, topics_q, doc_topics_q = run_topic_model(
                docs_q,
                index_q,
                sent_model,
                "auto",
                size,
                args.min_reassign_prob,
                native_dir,
                embeddings=embeddings[local_indices],
            )
            assignments = assignment_frame(
                doc_topics_q,
                topics_q,
                index_q,
                id_col="native_topic_id",
                label_col="native_topic_label",
                prob_col="native_topic_prob",
                outlier_col="native_was_outlier",
            )
            assignments.to_csv(
                native_dir / "assignments.csv", index=False, encoding="utf-8-sig"
            )
            distribution = assignments.rename(
                columns={
                    "native_topic_id": TOPIC_ID_COLUMN,
                    "native_topic_label": TOPIC_LABEL_COLUMN,
                }
            )
            scope_topic_summary(distribution).to_csv(
                native_dir / "topic_summary.csv", index=False, encoding="utf-8-sig"
            )
            report["native_models"][query.id] = {
                "papers": len(docs_q),
                "topics": int(len(topics_q)),
                "min_topic_size": size,
                "diagnostics": optimization.export_model_diagnostics(
                    model_q, native_dir / "diagnostics"
                ),
            }
            logger.info("%d native topics for %s", len(topics_q), query.id)
            native_progress.update(query_number, detail=query.id)

    logger.info("Writing master_corpus_topics.csv...")
    master.to_csv(
        PROCESSED_DIR / "master_corpus_topics.csv", index=False, encoding="utf-8-sig"
    )

    report["runtime_seconds"] = round(time.time() - started, 1)
    with open(TOPICS_DIR / "topics_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info(
        "Done in %ss. Report: data/processed/topics/topics_report.json",
        report["runtime_seconds"],
    )


if __name__ == "__main__":
    main()
