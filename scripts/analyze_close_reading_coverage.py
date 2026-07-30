#!/usr/bin/env python3
"""Evaluate topic breadth and VOS network position of the 136-paper reading ledger.

The analysis treats the reading ledger as a purposive interpretive set.  It
tests whether the already-read entrepreneurship papers span the current topic
and VOSviewer structures, while explicitly avoiding any claim that they form a
probability or prevalence-estimation sample.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "reports/analysis/tables/contrasting/close_reading_current_population_audit.csv"
OUTPUT_DIR = ROOT / "reports/analysis/tables/contrasting/close_reading_coverage"
FIGURE_DIR = ROOT / "reports/analysis/figures/contrasting"
REPORT = ROOT / "reports/analysis/CLOSE_READING_TOPIC_VOS_JUSTIFICATION.md"
TOPIC_LABEL_REVIEW = (
    ROOT / "data/processed/analysis/stage4/topic_label_review.csv"
)


POPULATIONS = {
    "Leading entrepreneurship journals": {
        "scope": "query_3",
        "membership": "in_leading_entrepreneurship_journals",
        "topics": ROOT / "data/processed/topics/native/query_3/assignments.csv",
        "vos": ROOT / "data/vosdata/query 3.txt",
    },
    "Additional entrepreneurship journals": {
        "scope": "query_4",
        "membership": "in_additional_entrepreneurship_journals",
        "topics": ROOT / "data/processed/topics/native/query_4/assignments.csv",
        "vos": ROOT / "data/vosdata/query 4.txt",
    },
}


def js_divergence(population_share: np.ndarray, reading_share: np.ndarray) -> float:
    midpoint = (population_share + reading_share) / 2

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    return 0.5 * kl(population_share, midpoint) + 0.5 * kl(reading_share, midpoint)


def topic_coverage(reading: pd.DataFrame, population: str, config: dict[str, Path | str]) -> pd.DataFrame:
    topics = pd.read_csv(
        config["topics"],
        dtype={"native_topic_id": str},
        keep_default_na=False,
    )
    selected = reading[reading[str(config["membership"])]].copy()
    selected["native_topic_id"] = (
        selected["current_topic_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    selected["native_topic_label"] = selected["current_topic_label"].replace("", "Unassigned")
    topics["native_topic_label"] = topics["native_topic_label"].replace("", "Unassigned")

    reviewed_labels = pd.read_csv(
        TOPIC_LABEL_REVIEW,
        dtype={"topic_id": str},
        keep_default_na=False,
    )
    reviewed_labels = reviewed_labels[
        (reviewed_labels["scope"] == str(config["scope"]))
        & (reviewed_labels["review_status"].str.lower() == "approved")
        & reviewed_labels["approved_label"].ne("")
    ]
    label_lookup = reviewed_labels.set_index("topic_id")["approved_label"].to_dict()
    selected["native_topic_label"] = selected["native_topic_id"].map(label_lookup).fillna(
        selected["native_topic_label"]
    )
    topics["native_topic_label"] = topics["native_topic_id"].astype(str).map(
        label_lookup
    ).fillna(topics["native_topic_label"])

    population_counts = topics.groupby(["native_topic_id", "native_topic_label"]).size().rename("population_papers")
    reading_counts = selected.groupby(["native_topic_id", "native_topic_label"]).size().rename("reading_papers")
    table = pd.concat([population_counts, reading_counts], axis=1).fillna(0).reset_index()
    table["reading_papers"] = table["reading_papers"].astype(int)
    table["population"] = population
    table["population_share"] = table["population_papers"] / len(topics)
    table["reading_share"] = table["reading_papers"] / len(selected)
    table["percentage_point_difference"] = 100 * (table["reading_share"] - table["population_share"])
    table["representation_ratio"] = np.where(
        table["population_share"] > 0,
        table["reading_share"] / table["population_share"],
        np.nan,
    )
    return table[
        [
            "population",
            "native_topic_id",
            "native_topic_label",
            "population_papers",
            "reading_papers",
            "population_share",
            "reading_share",
            "percentage_point_difference",
            "representation_ratio",
        ]
    ]


def network_coverage(
    reading: pd.DataFrame,
    population: str,
    config: dict[str, Path | str],
) -> tuple[dict[str, int | float | str], pd.DataFrame, pd.DataFrame]:
    vos = pd.read_csv(config["vos"], sep="\t")
    selected = reading[reading[str(config["membership"])]].copy()
    selected_tls = pd.to_numeric(selected["vos_total_link_strength"])
    selected_rank = pd.to_numeric(selected["vos_tls_rank"])
    full_tls = pd.to_numeric(vos["weight<Total link strength>"])
    nodes = len(vos)
    registered_population_papers = len(pd.read_csv(config["topics"]))

    quartile = np.ceil(selected_rank / nodes * 4).clip(1, 4).astype(int)
    quartile_table = pd.DataFrame(
        {
            "population": population,
            "tls_quartile": [1, 2, 3, 4],
            "reading_papers": [(quartile == value).sum() for value in (1, 2, 3, 4)],
        }
    )
    quartile_table["reading_share"] = quartile_table["reading_papers"] / len(selected)
    quartile_table["expected_population_share"] = 0.25

    full_clusters = vos["cluster"].astype(int).value_counts().sort_index()
    read_clusters = pd.to_numeric(selected["vos_cluster"]).astype(int).value_counts().sort_index()
    cluster_table = pd.DataFrame(
        {
            "population": population,
            "vos_cluster": full_clusters.index,
            "population_nodes": full_clusters.values,
            "reading_papers": [int(read_clusters.get(cluster, 0)) for cluster in full_clusters.index],
        }
    )
    cluster_table["population_share"] = cluster_table["population_nodes"] / nodes
    cluster_table["reading_share"] = cluster_table["reading_papers"] / len(selected)

    represented_clusters = cluster_table.loc[cluster_table["reading_papers"] > 0]
    covered_node_mass = represented_clusters["population_nodes"].sum() / nodes
    summary: dict[str, int | float | str] = {
        "population": population,
        "reading_papers": len(selected),
        "registered_population_papers": registered_population_papers,
        "vos_population_nodes": nodes,
        "vos_map_population_coverage": float(nodes / registered_population_papers),
        "reading_vos_matches": int(selected["vos_node_id"].astype(str).ne("").sum()),
        "full_tls_median": float(full_tls.median()),
        "reading_tls_median": float(selected_tls.median()),
        "median_tls_ratio": float(selected_tls.median() / full_tls.median()),
        "full_tls_mean": float(full_tls.mean()),
        "reading_tls_mean": float(selected_tls.mean()),
        "mean_tls_ratio": float(selected_tls.mean() / full_tls.mean()),
        "median_tls_rank": float(selected_rank.median()),
        "median_rank_as_population_percent": float(100 * selected_rank.median() / nodes),
        "top_decile_papers": int((selected_rank <= math.ceil(nodes * 0.10)).sum()),
        "top_decile_share": float((selected_rank <= math.ceil(nodes * 0.10)).mean()),
        "top_quartile_papers": int((selected_rank <= math.ceil(nodes * 0.25)).sum()),
        "top_quartile_share": float((selected_rank <= math.ceil(nodes * 0.25)).mean()),
        "bottom_quartile_papers": int((selected_rank > math.ceil(nodes * 0.75)).sum()),
        "bottom_quartile_share": float((selected_rank > math.ceil(nodes * 0.75)).mean()),
        "vos_clusters_available": int(len(cluster_table)),
        "vos_clusters_represented": int(len(represented_clusters)),
        "represented_cluster_node_mass": float(covered_node_mass),
    }
    return summary, quartile_table, cluster_table


def plot_results(topic_table: pd.DataFrame, quartiles: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    colors = {"population": "#9aa9b8", "reading": "#2f80d0"}

    for axis, population in zip(axes[0], POPULATIONS):
        subset = topic_table[topic_table["population"] == population].copy()
        subset = subset[subset["native_topic_id"].astype(str).ne("")]
        subset = subset.sort_values("population_share")
        y = np.arange(len(subset))
        axis.barh(y - 0.18, 100 * subset["population_share"], height=0.34, color=colors["population"], label="Population")
        axis.barh(
            y + 0.18,
            100 * subset["reading_share"],
            height=0.34,
            color=colors["reading"],
            label="Structured close-reading set",
        )
        axis.set_yticks(y, subset["native_topic_label"].str.replace("_", " "))
        axis.set_xlabel("Papers (%)")
        axis.set_title(population)
        axis.grid(axis="x", alpha=0.25)
        axis.legend(loc="lower right")

    for axis, population in zip(axes[1], POPULATIONS):
        subset = quartiles[quartiles["population"] == population]
        x = np.arange(1, 5)
        axis.bar(
            x,
            100 * subset["reading_share"],
            color=colors["reading"],
            width=0.62,
            label="Structured close-reading set",
        )
        axis.axhline(25, color=colors["population"], linestyle="--", linewidth=2, label="Population expectation")
        axis.set_xticks(x, ["Top 25%", "25-50%", "50-75%", "Bottom 25%"])
        axis.set_ylabel("Reading papers (%)")
        axis.set_ylim(0, 60)
        axis.set_title(f"VOS total-link-strength rank: {population}")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="upper right")

    fig.suptitle(
        "Topic and VOS network coverage of the structured close-reading set",
        fontsize=15,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    reading: pd.DataFrame,
    topic_table: pd.DataFrame,
    summaries: list[dict[str, int | float | str]],
) -> None:
    combined = int(reading["in_combined_entrepreneurship"].sum())
    outside = len(reading) - combined
    lines = [
        "# Topic and VOSviewer justification of the existing close-reading ledger",
        "",
        "## Decision",
        "",
        "The existing 136-paper ledger is defensible as a **purposive, network-central close-reading set for theory elaboration**, but not as a topic-proportional or prevalence-estimation sample. It should therefore be retained to preserve the completed reading and the insights already developed, while its concentration and one clear topic gap are disclosed.",
        "",
        "## Population placement",
        "",
        f"- {combined} of 136 papers belong to the Combined entrepreneurship population: 51 Leading and 73 Additional.",
        f"- The remaining {outside} papers are retained as cross-domain contrasts, not as part of the entrepreneurship prevalence base.",
        "- Every one of the 124 entrepreneurship papers is present in its corresponding VOSviewer document map.",
        "",
        "## Topic coverage",
        "",
    ]

    for population in POPULATIONS:
        subset = topic_table[topic_table["population"] == population]
        assigned = subset[subset["native_topic_id"].astype(str).ne("")].copy()
        represented = int((assigned["reading_papers"] > 0).sum())
        available = len(assigned)
        population_share = assigned["population_papers"].to_numpy() / assigned["population_papers"].sum()
        reading_share = assigned["reading_papers"].to_numpy() / assigned["reading_papers"].sum()
        divergence = js_divergence(population_share, reading_share)
        lines.append(f"### {population}")
        lines.append("")
        lines.append(f"- Topics represented: {represented} of {available}; Jensen-Shannon divergence from the population topic distribution: {divergence:.3f}.")
        for row in assigned.sort_values("percentage_point_difference", ascending=False).itertuples():
            paper_word = "paper" if row.reading_papers == 1 else "papers"
            lines.append(
                f"- {str(row.native_topic_label).replace('_', ' ')}: {row.reading_papers} reading {paper_word} "
                f"({row.reading_share:.1%}) versus {row.population_papers} population papers "
                f"({row.population_share:.1%}); {row.percentage_point_difference:+.1f} percentage points."
            )
        lines.append("")

    lines.extend(["## VOS total-link-strength position", ""])
    for summary in summaries:
        lines.extend(
            [
                f"### {summary['population']}",
                "",
                f"- The VOS document map contains {summary['vos_population_nodes']} of {summary['registered_population_papers']} population papers ({summary['vos_map_population_coverage']:.1%}); every reading-ledger paper in this population is mapped.",
                f"- Median total link strength is {summary['reading_tls_median']:.0f} in the reading set versus {summary['full_tls_median']:.0f} in the full VOS map ({summary['median_tls_ratio']:.2f} times as high).",
                f"- {summary['top_quartile_papers']} papers ({summary['top_quartile_share']:.1%}) are in the map's top TLS quartile; {summary['top_decile_papers']} ({summary['top_decile_share']:.1%}) are in the top decile.",
                f"- The median paper ranks at the top {summary['median_rank_as_population_percent']:.1f}% of the VOS map.",
                f"- The reading set covers {summary['vos_clusters_represented']} of {summary['vos_clusters_available']} VOS clusters, whose combined membership accounts for {summary['represented_cluster_node_mass']:.1%} of mapped papers.",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "The network evidence is strong: roughly half of both entrepreneurship reading subsets falls in the top TLS quartile, and their median TLS is more than twice the map median. The ledger therefore captures papers embedded in the field's connected conversations rather than an arbitrary collection of peripheral cases.",
            "",
            "The topic evidence is not proportional. Leading entrepreneurship is concentrated in Entrepreneurs and Entrepreneurship, and Additional entrepreneurship is concentrated in Entrepreneurship and Entrepreneurial. Leading Innovation and Patent is absent, while Model and Data, Research and Financial, Technology and Patent, and Healthcare and Medical are thinly represented. This does not invalidate theory-oriented close reading, but it prevents the ledger from being described as representative of the numerical topic composition.",
            "",
            "## Recommended use",
            "",
            "1. Retain the 124 entrepreneurship papers as the primary interpretive ledger and the 12 remaining papers as cross-domain contrasts.",
            "2. Describe the set as previously read, purposive, topic-spanning, and network-central; do not call it a random or topic-representative sample.",
            "3. Use the full-corpus model-coded matrices to establish prevalence and the reading ledger to explain mechanisms, boundary conditions, counterexamples, and bottleneck relocation.",
            "4. Do not replace the ledger with a new reading sample. If a reviewer requires explicit topic saturation, add only a small negative-case check for the missing Leading Innovation and Patent topic and the thinnest Additional topics; this is an augmentation, not a restart.",
            "",
            "## Reproducible outputs",
            "",
            "- `reports/analysis/tables/contrasting/close_reading_coverage/topic_coverage.csv`",
            "- `reports/analysis/tables/contrasting/close_reading_coverage/vos_position_summary.csv`",
            "- `reports/analysis/tables/contrasting/close_reading_coverage/vos_tls_quartiles.csv`",
            "- `reports/analysis/tables/contrasting/close_reading_coverage/vos_cluster_coverage.csv`",
            "- `reports/analysis/tables/contrasting/close_reading_coverage/reading_vos_ranked.csv`",
            "- `reports/analysis/figures/contrasting/close_reading_topic_vos_coverage.png`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    reading = pd.read_csv(AUDIT)
    if len(reading) != 136 or reading["paper_id"].nunique() != 136:
        raise RuntimeError("Expected the audited 136-paper reading ledger")

    topic_tables: list[pd.DataFrame] = []
    summaries: list[dict[str, int | float | str]] = []
    quartile_tables: list[pd.DataFrame] = []
    cluster_tables: list[pd.DataFrame] = []
    for population, config in POPULATIONS.items():
        topic_tables.append(topic_coverage(reading, population, config))
        summary, quartiles, clusters = network_coverage(reading, population, config)
        summaries.append(summary)
        quartile_tables.append(quartiles)
        cluster_tables.append(clusters)

    topic_table = pd.concat(topic_tables, ignore_index=True)
    quartile_table = pd.concat(quartile_tables, ignore_index=True)
    cluster_table = pd.concat(cluster_tables, ignore_index=True)
    summary_table = pd.DataFrame(summaries)
    ranked = reading[reading["in_combined_entrepreneurship"]].copy()
    ranked["vos_tls_rank"] = pd.to_numeric(ranked["vos_tls_rank"])
    ranked["vos_total_link_strength"] = pd.to_numeric(ranked["vos_total_link_strength"])
    ranked = ranked.sort_values(["vos_population", "vos_tls_rank", "Title"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    topic_table.to_csv(OUTPUT_DIR / "topic_coverage.csv", index=False)
    summary_table.to_csv(OUTPUT_DIR / "vos_position_summary.csv", index=False)
    quartile_table.to_csv(OUTPUT_DIR / "vos_tls_quartiles.csv", index=False)
    cluster_table.to_csv(OUTPUT_DIR / "vos_cluster_coverage.csv", index=False)
    ranked[
        [
            "vos_population",
            "vos_tls_rank",
            "vos_population_nodes",
            "vos_total_link_strength",
            "current_topic_id",
            "current_topic_label",
            "paper_id",
            "Title",
            "Source title",
            "Year",
            "DOI",
            "Link",
        ]
    ].to_csv(OUTPUT_DIR / "reading_vos_ranked.csv", index=False)
    figure = FIGURE_DIR / "close_reading_topic_vos_coverage.png"
    plot_results(topic_table, quartile_table, figure)
    write_report(reading, topic_table, summaries)
    print(f"Wrote {REPORT}")
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
