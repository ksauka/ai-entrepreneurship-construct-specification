"""Tests for topic optimization diagnostics without fitting BERTopic."""

import json

import pandas as pd

from aecsp.topics.pipeline.optimization import export_model_diagnostics


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
