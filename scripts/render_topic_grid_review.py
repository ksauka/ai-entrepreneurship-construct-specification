"""Render a consolidated review of BERTopic grid-search decisions.

Inputs: per-scope grid-search CSV files and topic_selection_review.json.
Outputs: a combined PNG figure and a decision-level CSV audit table.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZATION_DIR = PROJECT_ROOT / "data" / "processed" / "topics" / "optimization"
SCOPES = ("full_corpus", "query_1", "query_2", "query_3", "query_4")


def main() -> None:
    approval_path = OPTIMIZATION_DIR / "topic_selection_review.json"
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    approval_status = approval.get("approval_status")
    if approval_status not in {"approved", "pending_researcher_confirmation"}:
        raise SystemExit("Topic selections are not in a reviewable state")
    selection_label = (
        "Human-approved selection"
        if approval_status == "approved"
        else "Proposed selection"
    )

    fig, axes = plt.subplots(len(SCOPES), 2, figsize=(14, 18))
    decision_rows: list[dict] = []

    for row_index, scope in enumerate(SCOPES):
        metrics = pd.read_csv(OPTIMIZATION_DIR / scope / "grid_search_results.csv")
        decision = approval["scopes"][scope]
        automatic = int(decision["automatic_min_topic_size"])
        selected = int(decision["selected_min_topic_size"])

        metrics["scope"] = scope
        metrics["automatic_recommendation"] = metrics["min_topic_size"].eq(automatic)
        metrics["selected_for_final_model"] = metrics["min_topic_size"].eq(selected)
        decision_rows.extend(metrics.to_dict("records"))

        score_ax, count_ax = axes[row_index]
        x = metrics["min_topic_size"]
        score_ax.plot(x, metrics["composite_score"], "o-", color="#315b7d")
        count_ax.plot(x, metrics["n_topics"], "o-", color="#315b7d")

        for axis, y_column in (
            (score_ax, "composite_score"),
            (count_ax, "n_topics"),
        ):
            automatic_row = metrics.loc[metrics["min_topic_size"].eq(automatic)].iloc[0]
            selected_row = metrics.loc[metrics["min_topic_size"].eq(selected)].iloc[0]
            axis.scatter(
                [automatic],
                [automatic_row[y_column]],
                marker="X",
                s=125,
                color="#d97904",
                label="Automatic recommendation" if row_index == 0 else None,
                zorder=4,
            )
            axis.scatter(
                [selected],
                [selected_row[y_column]],
                marker="*",
                s=190,
                color="#18864b",
                label=selection_label if row_index == 0 else None,
                zorder=5,
            )
            axis.grid(alpha=0.25)
            axis.set_xlabel("Minimum topic size")

        score_ax.set_ylabel("Composite score")
        count_ax.set_ylabel("Topics discovered")
        score_ax.set_title(f"{scope.replace('_', ' ').title()}: diagnostic score")
        count_ax.set_title(f"{scope.replace('_', ' ').title()}: topic count")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.93, bottom=0.04, hspace=0.48, wspace=0.22)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        f"BERTopic grid-search review: automatic and {selection_label.lower()}",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )
    fig.savefig(
        OPTIMIZATION_DIR / "grid_search_review_overview.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    pd.DataFrame(decision_rows).to_csv(
        OPTIMIZATION_DIR / "grid_search_review_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    full_metrics = pd.read_csv(
        OPTIMIZATION_DIR / "full_corpus" / "grid_search_results.csv"
    )
    query_1_metrics = pd.read_csv(
        OPTIMIZATION_DIR / "query_1" / "grid_search_results.csv"
    )
    paired = full_metrics.merge(
        query_1_metrics,
        on="min_topic_size",
        suffixes=("_full_corpus", "_query_1"),
        validate="one_to_one",
    )
    paired["mean_composite_score"] = paired[
        ["composite_score_full_corpus", "composite_score_query_1"]
    ].mean(axis=1)
    joint_size = int(
        approval["selection_rule"]["full_corpus_query_1"][
            "selected_common_min_topic_size"
        ]
    )
    paired["selected_by_joint_rule"] = paired["min_topic_size"].eq(joint_size)
    paired.to_csv(
        OPTIMIZATION_DIR / "full_corpus_query_1_joint_grid.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(OPTIMIZATION_DIR / "grid_search_review_overview.png")
    print(OPTIMIZATION_DIR / "grid_search_review_table.csv")
    print(OPTIMIZATION_DIR / "full_corpus_query_1_joint_grid.csv")


if __name__ == "__main__":
    main()
