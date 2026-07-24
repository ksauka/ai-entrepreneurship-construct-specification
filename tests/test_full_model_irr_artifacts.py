import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports/analysis/tables/model_validation"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_model_irr_release_is_balanced_and_checksummed():
    coverage = pd.read_csv(TABLES / "full_corpus_model_coverage.csv")
    summary = pd.read_csv(TABLES / "full_corpus_pairwise_irr_core_summary.csv")
    dimensions = pd.read_csv(TABLES / "full_corpus_pairwise_irr_dimensions.csv")
    consensus = pd.read_csv(TABLES / "full_corpus_dimension_consensus.csv")
    manifest = json.loads(
        (TABLES / "full_corpus_model_irr_manifest.json").read_text(encoding="utf-8")
    )

    assert dict(
        zip(coverage["model_label"], coverage["successful_corpus_papers"])
    ) == {
        "GPT-5.4 Mini": 22345,
        "GPT-4.1 Nano": 22335,
        "Claude Sonnet 5": 21940,
        "Gemini 3.1 Pro Preview": 22345,
    }
    assert set(coverage["balanced_common_papers"]) == {21930}
    assert len(summary) == 6
    assert set(summary["balanced_common_papers"]) == {21930}
    assert len(dimensions) == 48
    assert set(dimensions["comparable_papers"]) == {21930}
    assert set(dimensions["classification"]) == {"Core", "Exploratory"}
    assert {
        "observability_exact_agreement",
        "observability_krippendorff_alpha",
        "jointly_observed_papers",
        "observed_category_exact_agreement",
        "observed_category_krippendorff_alpha",
    }.issubset(dimensions.columns)
    assert len(consensus) == 8
    assert {
        "preferred_trio_unobserved_agreement_papers",
        "preferred_trio_observed_agreement_papers",
        "all_four_unobserved_agreement_papers",
        "all_four_observed_agreement_papers",
    }.issubset(consensus.columns)
    assert (
        consensus["preferred_trio_agreement_papers"]
        >= consensus["all_four_agreement_papers"]
    ).all()
    assert (
        consensus["all_four_unobserved_agreement_papers"]
        + consensus["all_four_observed_agreement_papers"]
        == consensus["all_four_agreement_papers"]
    ).all()
    definition = consensus.loc[
        consensus["dimension"].eq("definition_construct_clarity")
    ].iloc[0]
    assert definition["all_four_unobserved_agreement_papers"] == 11448
    assert definition["all_four_observed_agreement_papers"] == 49
    assert manifest["balanced_common_papers"] == 21930
    assert manifest["core_dimension_count"] == 6
    assert manifest["dimension_count"] == 8

    for name, item in manifest["outputs"].items():
        path = TABLES / name
        assert path.exists()
        assert len(pd.read_csv(path)) == item["rows"]
        assert _sha256(path) == item["sha256"]
