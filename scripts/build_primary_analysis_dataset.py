"""Build canonical primary and model-validation analysis datasets.

Inputs: the master corpus, full Mini spec-v3 output and QC export, frozen
validation target, and model-specific spec-v3 exports. Outputs: an immutable
22,345-paper primary study table, a rater-prefixed validation table, and a
checksummed manifest. Source files are never modified.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

PROCESSED = PROJECT_ROOT / "data/processed"
OUTPUT = PROCESSED / "analysis"
MASTER = PROCESSED / "master_corpus.csv"
MINI = PROCESSED / "specification/paper_specifications_gpt-5.4-mini-2026-03-17_spec-v3.csv"
MINI_QC = PROCESSED / "specification/qc_gpt-5.4-mini-2026-03-17_spec-v3.csv"
TARGET = PROJECT_ROOT / "data/interim/proprietary_validation/proprietary_rater_target_2276_papers.csv"
PROBABILITY = PROJECT_ROOT / "data/interim/proprietary_validation/proprietary_probability_sample_2235.csv"
MODEL_FILES = {
    "mini": MINI,
    "nano": PROCESSED / "specification/paper_specifications_gpt-4.1-nano-2025-04-14_spec-v3.csv",
    "claude": PROCESSED / "specification/paper_specifications_claude-sonnet-5_spec-v3.csv",
    "gemini": PROCESSED / "specification/paper_specifications_gemini-3.1-pro-preview_spec-v3.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique(frame: pd.DataFrame, label: str) -> None:
    if "paper_id" not in frame:
        raise SystemExit(f"{label} has no paper_id column")
    duplicated = frame["paper_id"].duplicated()
    if duplicated.any():
        raise SystemExit(f"{label} has {int(duplicated.sum())} duplicate paper IDs")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(MASTER, dtype=str, keep_default_na=False)
    mini = pd.read_csv(MINI, dtype=str, keep_default_na=False)
    qc = pd.read_csv(MINI_QC, dtype=str, keep_default_na=False)
    for label, frame in (("master", master), ("Mini", mini), ("Mini QC", qc)):
        unique(frame, label)
    if set(master.paper_id) != set(mini.paper_id) or set(master.paper_id) != set(qc.paper_id):
        raise SystemExit("Master, Mini and Mini QC paper-ID sets are not identical")

    primary = master.merge(mini, on="paper_id", how="left", validate="one_to_one")
    qc = qc.rename(columns={column: f"qc_{column}" for column in qc if column != "paper_id"})
    primary = primary.merge(qc, on="paper_id", how="left", validate="one_to_one")
    if len(primary) != len(master) or primary.paper_id.nunique() != len(master):
        raise SystemExit("Primary analysis join failed its one-to-one invariant")
    primary_path = OUTPUT / "primary_analysis_dataset.csv"
    primary.to_csv(primary_path, index=False, encoding="utf-8-sig")

    target = pd.read_csv(TARGET, dtype=str, keep_default_na=False)
    probability = pd.read_csv(PROBABILITY, dtype=str, keep_default_na=False)
    unique(target, "validation target")
    validation = target.merge(
        probability[["paper_id", "selection_probability", "sampling_weight"]],
        on="paper_id", how="left", validate="one_to_one",
    )
    model_counts = {}
    for prefix, path in MODEL_FILES.items():
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        unique(frame, prefix)
        columns = {column: f"{prefix}_{column}" for column in frame if column != "paper_id"}
        frame = frame.rename(columns=columns)
        validation = validation.merge(frame, on="paper_id", how="left", validate="one_to_one")
        model_counts[prefix] = int(validation[f"{prefix}_coding_model"].fillna("").ne("").sum())
    validation_path = OUTPUT / "model_validation_dataset.csv"
    validation.to_csv(validation_path, index=False, encoding="utf-8-sig")

    inputs = [MASTER, MINI, MINI_QC, TARGET, PROBABILITY, *MODEL_FILES.values()]
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True,
        ).stdout.strip(),
        "primary_dataset": {
            "path": str(primary_path.relative_to(PROJECT_ROOT)),
            "rows": len(primary), "unique_paper_ids": primary.paper_id.nunique(),
            "columns": len(primary.columns), "sha256": sha256(primary_path),
            "primary_rater": "gpt-5.4-mini-2026-03-17", "protocol": "spec-v3",
        },
        "validation_dataset": {
            "path": str(validation_path.relative_to(PROJECT_ROOT)),
            "rows": len(validation), "unique_paper_ids": validation.paper_id.nunique(),
            "columns": len(validation.columns), "model_success_counts": model_counts,
            "sha256": sha256(validation_path),
        },
        "inputs": {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in dict.fromkeys(inputs)},
    }
    manifest_path = OUTPUT / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Primary study dataset: {len(primary):,} rows × {len(primary.columns):,} columns -> {primary_path}")
    print(f"Validation dataset: {len(validation):,} rows × {len(validation.columns):,} columns -> {validation_path}")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
