"""Write pairwise query-scope overlap and Jaccard diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from aecsp.corpus.scope_overlap import overlap_table  # noqa: E402


def main() -> None:
    master = pd.read_csv(PROJECT_ROOT / "data/processed/master_corpus.csv", dtype=str, keep_default_na=False)
    result = overlap_table(master, ("query_1", "query_2", "query_3", "query_4"))
    output = PROJECT_ROOT / "data/processed/specification/scope_overlap.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    (output.with_suffix(".json")).write_text(json.dumps(result.to_dict("records"), indent=2), encoding="utf-8")
    print(f"Wrote {len(result)} pairwise scope comparisons to {output}")


if __name__ == "__main__":
    main()
