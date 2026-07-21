import sqlite3

import pandas as pd

from aecsp.targeted_reading import TargetedReadingStore


def test_targeted_reading_keeps_blind_overlap_and_reviewer_records_separate(tmp_path):
    data_dir = tmp_path / "data/interim/theory_elaboration"
    data_dir.mkdir(parents=True)
    papers = pd.DataFrame(
        [
            {"paper_id": "P1", "Title": "One", "Source title": "Journal", "Year": "2024"},
            {"paper_id": "P2", "Title": "Two", "Source title": "Journal", "Year": "2025"},
            {"paper_id": "P3", "Title": "Three", "Source title": "Journal", "Year": "2026"},
        ]
    )
    papers.to_csv(data_dir / "theory_elaboration_matched_papers.csv", index=False)
    papers.iloc[[0]].to_csv(
        data_dir / "theory_elaboration_probability_overlap_23.csv", index=False
    )

    store = TargetedReadingStore(tmp_path)
    metadata = store.metadata()
    assert metadata["target_papers"] == 3
    assert metadata["human_validation_overlap"] == 1
    assert metadata["remaining_targeted_reading"] == 2
    context = {
        "dataset_scope": "full_corpus",
        "model": "test-model",
        "patterns": [
            {
                "dimension_id": "ai_role",
                "dimension_label": "AI role",
                "column": "ai_role_function",
                "value": "AI as tool",
                "value_label": "AI as tool",
            }
        ],
    }

    first = store.save(
        "reviewer-one",
        "P2",
        {
            "status": "reviewed",
            "relation": "supports",
            "evidence_note": "Evidence",
            "interpretation": "Interpretation",
            "theoretical_implication": "Implication",
            "context": context,
        },
    )
    assert first["status"] == "reviewed"
    store.save(
        "reviewer-two",
        "P2",
        {"status": "revisit", "relation": "contrasts", "context": context},
    )
    assert store.review_map("reviewer-one", context)["P2"]["relation"] == "supports"
    assert store.review_map("reviewer-two", context)["P2"]["relation"] == "contrasts"

    exported = store.export()
    assert len(exported) == 2
    assert set(exported["reviewer_id"]) == {"reviewer-one", "reviewer-two"}
    with sqlite3.connect(store.database_path) as connection:
        revisions = connection.execute(
            "SELECT COUNT(*) FROM targeted_review_audit"
        ).fetchone()[0]
    assert revisions == 2
