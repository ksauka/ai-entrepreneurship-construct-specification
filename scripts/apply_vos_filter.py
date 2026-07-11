"""Filter corpus scopes by VOSviewer citation connectivity.

Inputs: the master corpus and current per-scope VOSviewer maps.
Outputs: retained and dropped CSV files per scope plus a filter report.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from aecsp.vos.filter import filter_all_scopes  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VOS_DIR = PROJECT_ROOT / "data" / "vosdata"
OUTPUT_DIR = PROCESSED_DIR / "vos_filtered"


def main() -> None:
    master_path = PROCESSED_DIR / "master_corpus.csv"
    if not master_path.exists():
        sys.exit("master_corpus.csv not found. Run scripts/build_corpus.py first.")
    if not VOS_DIR.exists():
        sys.exit(f"{VOS_DIR} does not exist. Save VOSviewer maps there as <scope>_vos.csv.")

    print("Loading master corpus...")
    master = pd.read_csv(master_path, dtype=str, keep_default_na=False)

    print(f"Applying VOS filter from {VOS_DIR}...")
    stats = filter_all_scopes(
        master, VOS_DIR, master_path, OUTPUT_DIR, show_progress=True
    )

    for scope_id, info in stats.items():
        if info["status"] == "filtered":
            print(
                f"  {scope_id}: retained {info['retained']:,}, dropped {info['dropped']:,} "
                f"({round(info['retained_share'] * 100)}% connected)"
            )
        else:
            print(f"  {scope_id}: skipped ({info['status']})")

    report = {"timestamp": datetime.now().isoformat(), "vos_dir": str(VOS_DIR), "scopes": stats}
    with open(PROCESSED_DIR / "vos_filter_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    if any(s["status"] == "filtered" for s in stats.values()):
        print(f"Wrote retained/dropped CSVs to {OUTPUT_DIR}")
    else:
        print("No current VOS maps found. Nothing written.")


if __name__ == "__main__":
    main()
