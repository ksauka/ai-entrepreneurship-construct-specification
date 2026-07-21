"""Recalculate KS-MK Task 2 agreement on the current-corpus IRR subset.

Source: ``docs/archive/2026-07/research_sources/IIR workings FINAL.xlsx``.
Only Task 2 from the KS and MK tabs is used. The historical exercise contains
15 papers; S09 (Lada et al., 2023) is absent from the frozen current corpus and
is excluded from this aligned calculation.

This is a researcher-human-coder evaluation of allocation to three
interpretive insight families plus fragmentation. It is not reliability for
the eight construct-specification dimensions.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports/analysis/tables/contrasting"
RATINGS_OUTPUT = OUTPUT_DIR / "human_insight_irr_ks_mk_14_ratings.csv"
METRICS_OUTPUT = OUTPUT_DIR / "human_insight_irr_ks_mk_14_metrics.json"

# Task 2 values transcribed from the workbook's KS and MK tabs. S09 is
# deliberately omitted because it is absent from the current frozen corpus.
RATINGS = {
    "S01": ("P1S", "P1S"),
    "S02": ("P1S", "P1S"),
    "S03": ("P1S", "P1S"),
    "S04": ("P2S", "P2S"),
    "S05": ("P2S", "P2S"),
    "S06": ("P2S", "P2S"),
    "S07": ("P2C", "P2C"),
    "S08": ("P2C", "P2C"),
    "S10": ("P3S", "P3S"),
    "S11": ("P3S", "P3S"),
    "S12": ("P3S", "P3S"),
    "S13": ("FRAG", "FRAG"),
    "S14": ("FRAG", "FRAG"),
    "S15": ("FRAG", "FRAG"),
}


def calculate() -> dict[str, object]:
    n_papers = len(RATINGS)
    exact_agreements = sum(ks == mk for ks, mk in RATINGS.values())
    observed_agreement = exact_agreements / n_papers
    ks_counts = Counter(ks for ks, _ in RATINGS.values())
    mk_counts = Counter(mk for _, mk in RATINGS.values())
    categories = sorted(set(ks_counts) | set(mk_counts))
    expected_agreement = sum(
        (ks_counts[category] / n_papers) * (mk_counts[category] / n_papers)
        for category in categories
    )
    cohen_kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    return {
        "analysis": "KS-MK Task 2 human insight-allocation reliability",
        "source": "docs/archive/2026-07/research_sources/IIR workings FINAL.xlsx",
        "tabs": ["KS", "MK"],
        "field": "Task 2",
        "excluded_paper": "S09: Lada et al. (2023), absent from the frozen current corpus",
        "n_papers": n_papers,
        "exact_agreements": exact_agreements,
        "observed_agreement": observed_agreement,
        "expected_agreement": expected_agreement,
        "cohen_kappa": cohen_kappa,
        "ks_category_counts": dict(sorted(ks_counts.items())),
        "mk_category_counts": dict(sorted(mk_counts.items())),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RATINGS_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("paper_id", "KS_task_2", "MK_task_2", "exact_agreement"))
        for paper_id, (ks, mk) in RATINGS.items():
            writer.writerow((paper_id, ks, mk, ks == mk))
    METRICS_OUTPUT.write_text(
        json.dumps(calculate(), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {RATINGS_OUTPUT}")
    print(f"Wrote {METRICS_OUTPUT}")


if __name__ == "__main__":
    main()
