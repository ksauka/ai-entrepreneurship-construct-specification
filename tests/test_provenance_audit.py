"""Tests for sample-to-source metadata provenance auditing."""

import pandas as pd

from aecsp.corpus.provenance_audit import audit_sample_provenance


def frames():
    row = {"paper_id": "eid:E1", "EID": "E1", "Title": "T", "Abstract": "A", "Author Keywords": "K", "Source title": "J", "Year": "2020"}
    sample = pd.DataFrame([{key: value for key, value in row.items() if key != "EID"}])
    master = pd.DataFrame([row])
    raw = pd.DataFrame([{key: value for key, value in row.items() if key != "paper_id"} | {"raw_source_file": "SQ1.csv", "raw_source_row": 2}])
    return sample, master, raw


def test_exact_provenance_passes():
    audit, discrepancies = audit_sample_provenance(*frames())
    assert audit.iloc[0]["sample_master_all_exact"]
    assert audit.iloc[0]["master_raw_all_exact"]
    assert discrepancies.empty


def test_changed_sample_value_is_reported():
    sample, master, raw = frames()
    sample.loc[0, "Abstract"] = "changed"
    audit, discrepancies = audit_sample_provenance(sample, master, raw)
    assert not audit.iloc[0]["sample_master_all_exact"]
    assert set(discrepancies["field"]) == {"Abstract"}


def test_trailing_source_whitespace_is_non_substantive():
    sample, master, raw = frames()
    raw.loc[0, "Source title"] = "J "
    audit, discrepancies = audit_sample_provenance(sample, master, raw)
    assert not audit.iloc[0]["master_raw_all_exact"]
    assert audit.iloc[0]["master_raw_all_normalized"]
    assert set(discrepancies["discrepancy_type"]) == {"whitespace_normalization"}
