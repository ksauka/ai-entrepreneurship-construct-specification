"""Canonical-input gates for the Stage 2B graph builder."""

import hashlib
import json

import pandas as pd
import pytest

from scripts import build_graph


def _write_primary(tmp_path):
    path = tmp_path / "primary_analysis_dataset.csv"
    pd.DataFrame(
        [{"paper_id": "P1", "Title": "One paper", "in_query_1": "1"}]
    ).to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "primary_dataset": {
                    "rows": 1,
                    "unique_paper_ids": 1,
                    "sha256": digest,
                }
            }
        ),
        encoding="utf-8",
    )
    return path, manifest


def test_load_corpus_requires_matching_canonical_manifest(tmp_path, monkeypatch):
    primary, manifest = _write_primary(tmp_path)
    monkeypatch.setattr(build_graph, "PRIMARY_DATASET", primary)
    monkeypatch.setattr(build_graph, "DATASET_MANIFEST", manifest)
    monkeypatch.setattr(build_graph, "ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(build_graph, "PROCESSED_DIR", tmp_path)
    loaded = build_graph.load_corpus()
    assert loaded["paper_id"].tolist() == ["P1"]


def test_load_corpus_refuses_checksum_mismatch(tmp_path, monkeypatch):
    primary, manifest = _write_primary(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["primary_dataset"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(build_graph, "PRIMARY_DATASET", primary)
    monkeypatch.setattr(build_graph, "DATASET_MANIFEST", manifest)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        build_graph.load_corpus()
