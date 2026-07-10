"""Tests for the per-scope VOS citation-connectivity cleaning filter."""

import time
from pathlib import Path

import pandas as pd
import pytest

from aecsp.vos.filter import (
    filter_all_scopes,
    load_vos_dois,
    normalize_doi,
    split_scope,
    vos_status,
)

# A VOS map lists only the citation-connected papers (P1, P2); P3 is absent.
VOS_MAP = "\n".join([
    "id\tlabel\tcluster\tweight<Total link strength>\turl",
    "1\tp1\t1\t5\thttps://doi.org/10.1000/aaa",
    "2\tp2\t2\t3\thttps://doi.org/10.2000/bbb",
])


@pytest.fixture
def master() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"paper_id": "P1", "DOI": "10.1000/aaa", "in_query_1": "1", "in_query_2": "0", "in_query_3": "1", "in_query_4": "0"},
            {"paper_id": "P2", "DOI": "10.2000/bbb", "in_query_1": "1", "in_query_2": "0", "in_query_3": "0", "in_query_4": "0"},
            {"paper_id": "P3", "DOI": "10.3000/ccc", "in_query_1": "0", "in_query_2": "0", "in_query_3": "1", "in_query_4": "0"},
        ]
    )


def test_normalize_doi_strips_url_and_trailing_punct():
    assert normalize_doi("https://doi.org/10.1000/AAA;") == "10.1000/aaa"


def test_load_vos_dois_reads_connected_set(tmp_path: Path):
    path = tmp_path / "master_corpus_vos.csv"
    path.write_text(VOS_MAP, encoding="utf-8")
    assert load_vos_dois(path) == {"10.1000/aaa", "10.2000/bbb"}


def test_split_scope_separates_retained_and_dropped(master):
    dois = {"10.1000/aaa", "10.2000/bbb"}
    # Query 3 holds P1 (connected) and P3 (dropped).
    retained, dropped = split_scope(master, "query_3", dois)
    assert list(retained["paper_id"]) == ["P1"]
    assert list(dropped["paper_id"]) == ["P3"]
    # retained carries no VOS columns
    assert not any(c.startswith("vos_") for c in retained.columns)


def test_filter_all_scopes_writes_two_datasets(master, tmp_path):
    reference = tmp_path / "master_corpus.csv"
    master.to_csv(reference, index=False)
    vos_dir = tmp_path / "vosdata"
    vos_dir.mkdir()
    (vos_dir / "master_corpus_vos.csv").write_text(VOS_MAP, encoding="utf-8")
    out = tmp_path / "out"

    stats = filter_all_scopes(master, vos_dir, reference, out)

    assert stats["full_corpus"]["status"] == "filtered"
    assert stats["full_corpus"]["retained"] == 2
    assert stats["full_corpus"]["dropped"] == 1
    assert stats["query_2"]["status"] == "missing"
    assert (out / "full_corpus_retained.csv").exists()
    assert (out / "full_corpus_dropped.csv").exists()


def test_stale_map_is_skipped(master, tmp_path):
    vos_dir = tmp_path / "vosdata"
    vos_dir.mkdir()
    vos_path = vos_dir / "master_corpus_vos.csv"
    vos_path.write_text(VOS_MAP, encoding="utf-8")
    time.sleep(0.01)
    reference = tmp_path / "master_corpus.csv"       # newer than the map
    master.to_csv(reference, index=False)

    assert vos_status(vos_path, reference) == "stale"
    stats = filter_all_scopes(master, vos_dir, reference, tmp_path / "out")
    assert stats["full_corpus"]["status"] == "stale"
