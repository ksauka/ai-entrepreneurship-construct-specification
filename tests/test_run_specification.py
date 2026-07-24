"""Recovery contracts for the resumable specification runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def load_script():
    path = Path(__file__).parents[1] / "scripts/run_specification.py"
    spec = importlib.util.spec_from_file_location("run_specification", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_atomic_csv_write_replaces_completed_artifact(tmp_path):
    module = load_script()
    path = tmp_path / "results.csv"
    path.write_text("old\n", encoding="utf-8")

    module.write_csv_atomically(pd.DataFrame([{"paper_id": "P1"}]), path)

    assert pd.read_csv(path)["paper_id"].tolist() == ["P1"]
    assert not path.with_suffix(".csv.tmp").exists()


def test_atomic_csv_write_preserves_previous_artifact_on_failure(
    tmp_path,
    monkeypatch,
):
    module = load_script()
    path = tmp_path / "results.csv"
    path.write_text("stable\n", encoding="utf-8")
    frame = pd.DataFrame([{"paper_id": "P1"}])

    def fail_after_partial_write(target, **_kwargs):
        Path(target).write_text("partial", encoding="utf-8")
        raise UnicodeEncodeError("utf-8", "\udcda", 0, 1, "surrogate")

    monkeypatch.setattr(frame, "to_csv", fail_after_partial_write)

    with pytest.raises(UnicodeEncodeError):
        module.write_csv_atomically(frame, path)

    assert path.read_text(encoding="utf-8") == "stable\n"
    assert not path.with_suffix(".csv.tmp").exists()
