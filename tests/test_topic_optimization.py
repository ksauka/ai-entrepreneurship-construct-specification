"""Tests for topic optimization diagnostics without fitting BERTopic."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from aecsp.topics.pipeline.optimization import (
    export_model_diagnostics,
    native_grid_min_topic_sizes,
    optimize_topic_count_grid_search,
)


class _InteractiveFigure:
    def write_html(self, path, include_plotlyjs):
        assert include_plotlyjs == "cdn"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("<html>diagnostic</html>")


class _FakeTopicModel:
    def get_topic_info(self):
        return pd.DataFrame(
            {"Topic": [-1, 0, 1], "Count": [2, 10, 5]}
        )

    def visualize_topics(self):
        return _InteractiveFigure()

    def visualize_hierarchy(self):
        return _InteractiveFigure()

    def visualize_barchart(self, top_n_topics):
        assert top_n_topics == 50
        return _InteractiveFigure()


def test_export_model_diagnostics_writes_static_and_interactive_graphs(tmp_path):
    status = export_model_diagnostics(_FakeTopicModel(), tmp_path)

    assert (tmp_path / "topic_size_distribution.html").exists()
    assert (tmp_path / "intertopic_distance.html").exists()
    assert (tmp_path / "topic_hierarchy.html").exists()
    assert (tmp_path / "topic_terms.html").exists()
    saved = json.loads((tmp_path / "diagnostics_status.json").read_text())
    assert saved == status
    assert not any(
        key.endswith("_error") and not key.startswith("topic_size_distribution_png")
        for key in status
    )


def test_grid_search_records_invalid_candidate_and_continues(monkeypatch, tmp_path):
    from sklearn import metrics

    class FakeUMAP:
        def __init__(self, **kwargs):
            self.parameters = kwargs

    class FakeHDBSCAN:
        def __init__(self, min_cluster_size, **kwargs):
            self.min_cluster_size = min_cluster_size

    class FakeBERTopic:
        def __init__(self, hdbscan_model, **kwargs):
            self.min_cluster_size = hdbscan_model.min_cluster_size

        def fit_transform(self, documents, embeddings):
            if self.min_cluster_size == 2:
                raise ValueError("max_df corresponds to < documents than min_df")
            return [0, 0, 0, 1, 1, -1], None

        def get_topic_info(self):
            return pd.DataFrame({"Topic": [-1, 0, 1], "Count": [1, 3, 2]})

        def get_topic(self, topic_id):
            return [(f"topic_{topic_id}", 1.0)]

    monkeypatch.setitem(sys.modules, "bertopic", SimpleNamespace(BERTopic=FakeBERTopic))
    monkeypatch.setitem(sys.modules, "hdbscan", SimpleNamespace(HDBSCAN=FakeHDBSCAN))
    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=FakeUMAP))
    monkeypatch.setattr(metrics, "silhouette_score", lambda *args, **kwargs: 0.25)

    selected, payload = optimize_topic_count_grid_search(
        documents=[f"document {index}" for index in range(6)],
        embeddings=np.zeros((6, 4)),
        embedding_model=None,
        min_topic_sizes=[2, 3],
        plot_metrics=False,
        out_dir=tmp_path,
        document_metadata=[
            {"paper_id": f"paper-{index}", "Title": f"Title {index}"}
            for index in range(6)
        ],
        min_topics_for_recommendation=2,
    )

    assert selected == 3
    assert [row["min_topic_size"] for row in payload["grid_search"]] == [3]
    assert payload["failed_configurations"] == [
        {
            "min_topic_size": 2,
            "error_type": "ValueError",
            "error": "max_df corresponds to < documents than min_df",
        }
    ]
    topics = pd.read_csv(tmp_path / "candidates/min_topic_size_3/topics.csv")
    representatives = pd.read_csv(
        tmp_path / "candidates/min_topic_size_3/representative_papers.csv"
    )
    assert topics[["topic_id", "paper_count"]].to_dict("records") == [
        {"topic_id": 0, "paper_count": 3},
        {"topic_id": 1, "paper_count": 2},
    ]
    assert set(representatives["paper_id"]) <= {
        "paper-0", "paper-1", "paper-2", "paper-3", "paper-4"
    }


def test_native_grid_adds_scope_scaled_candidates():
    assert native_grid_min_topic_sizes(438, [20, 30, 40]) == [8, 12, 20, 30, 40]
    assert native_grid_min_topic_sizes(646, [20, 30, 40]) == [12, 18, 20, 30, 40]
    assert native_grid_min_topic_sizes(986, [20, 30, 40]) == [19, 20, 28, 30, 40]


def test_grid_search_excludes_collapsed_candidate_from_recommendation(monkeypatch):
    from sklearn import metrics

    class FakeUMAP:
        def __init__(self, **kwargs):
            self.parameters = kwargs

    class FakeHDBSCAN:
        def __init__(self, min_cluster_size, **kwargs):
            self.min_cluster_size = min_cluster_size

    class FakeBERTopic:
        def __init__(self, hdbscan_model, **kwargs):
            self.min_cluster_size = hdbscan_model.min_cluster_size
            self.labels = []

        def fit_transform(self, documents, embeddings):
            if self.min_cluster_size == 2:
                self.labels = [0, 0, 0, 0, 1, 1, 1, 1, -1, -1]
            else:
                self.labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, -1]
            return self.labels, None

        def get_topic_info(self):
            counts = pd.Series(self.labels).value_counts().sort_index()
            return pd.DataFrame({"Topic": counts.index, "Count": counts.values})

        def get_topic(self, topic_id):
            return [(f"topic_{topic_id}", 1.0)]

    monkeypatch.setitem(sys.modules, "bertopic", SimpleNamespace(BERTopic=FakeBERTopic))
    monkeypatch.setitem(sys.modules, "hdbscan", SimpleNamespace(HDBSCAN=FakeHDBSCAN))
    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=FakeUMAP))
    monkeypatch.setattr(metrics, "silhouette_score", lambda *args, **kwargs: 0.25)

    selected, payload = optimize_topic_count_grid_search(
        documents=[f"document {index}" for index in range(10)],
        embeddings=np.eye(10),
        embedding_model=None,
        min_topic_sizes=[2, 3],
        min_topics_for_recommendation=5,
        target_topic_range=(2, 5),
        plot_metrics=False,
    )

    assert selected == 3
    assert payload["minimum_topics_for_recommendation"] == 5
    assert [row["n_topics"] for row in payload["excluded_from_recommendation"]] == [2]
    assert [row["n_topics"] for row in payload["eligible"]] == [5]
