"""Audit sampled paper metadata against master and original Scopus exports."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

AUDIT_FIELDS = ("Title", "Abstract", "Author Keywords", "Source title", "Year")


def normalized_source_value(value: object) -> str:
    """Apply the corpus builder's non-substantive edge-whitespace cleanup."""

    return str(value).strip()


def audit_sample_provenance(
    sample: pd.DataFrame,
    master: pd.DataFrame,
    raw: pd.DataFrame,
    fields: Iterable[str] = AUDIT_FIELDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return paper-level checks and field-level discrepancies.

    A master-to-raw field passes when at least one original Scopus row with the
    same EID contains the exact master value. The all-fields check requires one
    original row to match every audited field simultaneously.
    """

    fields = tuple(fields)
    required_sample = {"paper_id", *fields}
    required_master = {"paper_id", "EID", *fields}
    required_raw = {"EID", *fields, "raw_source_file", "raw_source_row"}
    for name, frame, required in (
        ("sample", sample, required_sample),
        ("master", master, required_master),
        ("raw", raw, required_raw),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{name} missing columns: {sorted(missing)}")
    if sample["paper_id"].duplicated().any():
        raise ValueError("sample contains duplicate paper_id values")
    if master["paper_id"].duplicated().any():
        raise ValueError("master contains duplicate paper_id values")

    master_index = master.set_index("paper_id", drop=False)
    raw_groups = {eid: group for eid, group in raw.groupby("EID", sort=False)}
    rows: list[dict] = []
    discrepancies: list[dict] = []
    for sample_row in sample.to_dict("records"):
        paper_id = str(sample_row["paper_id"])
        master_found = paper_id in master_index.index
        if not master_found:
            rows.append({"paper_id": paper_id, "master_found": False, "raw_found": False, "sample_master_all_exact": False, "master_raw_all_exact": False})
            discrepancies.append({"paper_id": paper_id, "comparison": "sample_to_master", "field": "paper_id", "expected": paper_id, "observed": "", "raw_source_file": "", "raw_source_row": ""})
            continue
        master_row = master_index.loc[paper_id].to_dict()
        eid = str(master_row["EID"])
        candidates = raw_groups.get(eid, pd.DataFrame())
        sample_matches = {field: str(sample_row.get(field, "")) == str(master_row.get(field, "")) for field in fields}
        raw_field_matches = {
            field: bool((candidates[field].astype(str) == str(master_row.get(field, ""))).any()) if not candidates.empty else False
            for field in fields
        }
        raw_normalized_matches = {
            field: bool(
                candidates[field].astype(str).map(normalized_source_value).eq(
                    normalized_source_value(master_row.get(field, ""))
                ).any()
            )
            if not candidates.empty
            else False
            for field in fields
        }
        if candidates.empty:
            raw_all_exact = False
            matching_sources = ""
        else:
            exact_mask = pd.Series(True, index=candidates.index)
            for field in fields:
                exact_mask &= candidates[field].astype(str).eq(str(master_row.get(field, "")))
            exact_rows = candidates[exact_mask]
            raw_all_exact = not exact_rows.empty
            matching_sources = ";".join(sorted(set(exact_rows["raw_source_file"].astype(str))))
        if candidates.empty:
            raw_all_normalized = False
        else:
            normalized_mask = pd.Series(True, index=candidates.index)
            for field in fields:
                normalized_mask &= candidates[field].astype(str).map(normalized_source_value).eq(
                    normalized_source_value(master_row.get(field, ""))
                )
            raw_all_normalized = bool(normalized_mask.any())
        for field, passed in sample_matches.items():
            if not passed:
                discrepancies.append({"paper_id": paper_id, "comparison": "sample_to_master", "field": field, "expected": str(master_row.get(field, "")), "observed": str(sample_row.get(field, "")), "raw_source_file": "", "raw_source_row": ""})
        for field, passed in raw_field_matches.items():
            if not passed:
                discrepancies.append({"paper_id": paper_id, "comparison": "master_to_raw", "field": field, "discrepancy_type": "whitespace_normalization" if raw_normalized_matches[field] else "substantive", "expected": str(master_row.get(field, "")), "observed": "<no byte-exact candidate value>", "raw_source_file": ";".join(sorted(set(candidates.get("raw_source_file", pd.Series(dtype=str)).astype(str)))), "raw_source_row": ""})
        rows.append(
            {
                "paper_id": paper_id,
                "eid": eid,
                "master_found": True,
                "raw_found": not candidates.empty,
                "raw_candidate_rows": len(candidates),
                "sample_master_all_exact": all(sample_matches.values()),
                "master_raw_all_exact": raw_all_exact,
                "master_raw_all_normalized": raw_all_normalized,
                "exact_raw_source_files": matching_sources,
                **{f"sample_master_{field}_exact": passed for field, passed in sample_matches.items()},
                **{f"master_raw_{field}_exact": passed for field, passed in raw_field_matches.items()},
                **{f"master_raw_{field}_normalized": passed for field, passed in raw_normalized_matches.items()},
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(discrepancies)
