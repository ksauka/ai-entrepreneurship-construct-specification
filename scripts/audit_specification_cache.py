"""Write non-mutating QC results for one model's specification cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from aecsp.specification.llm_coder import PROTOCOL_ID, cache_key, model_cache_dir  # noqa: E402
from aecsp.specification.paths import resolve_primary_model  # noqa: E402
from aecsp.specification.validators import validate_profile  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    model = args.model or resolve_primary_model()
    corpus = pd.read_csv(PROJECT_ROOT / "data/processed/master_corpus.csv", dtype=str, keep_default_na=False)
    cache = model_cache_dir(PROJECT_ROOT / "data/interim/spec_cache", model, PROTOCOL_ID)
    rows = []
    for _, paper in corpus.iterrows():
        path = cache / cache_key(paper["paper_id"])
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        qc = validate_profile(record, {
            "title": paper.get("Title", ""), "abstract": paper.get("Abstract", ""),
            "keywords": paper.get("Author Keywords", ""),
        })
        rows.append({
            "paper_id": paper["paper_id"],
            "validation_passed": qc["validation_passed"],
            "validation_error_count": qc["validation_error_count"],
            "validation_warning_count": qc["validation_warning_count"],
            "validation_issues_json": json.dumps(qc["validation_issues"], sort_keys=True),
            "grounding_scores_json": json.dumps(qc["grounding_scores"], sort_keys=True),
        })
    output = PROJECT_ROOT / "data/processed/specification" / f"qc_{model.replace(':', '_')}_{PROTOCOL_ID}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows):,} non-mutating QC records to {output}")


if __name__ == "__main__":
    main()
