"""Resolve the registered primary specification model and output paths."""

from __future__ import annotations

import json
import re
from pathlib import Path

REGISTER_PATH = Path(__file__).resolve().parents[3] / "configs" / "experiment_register.json"


def load_experiment_register(path: Path = REGISTER_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"experiment register not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def resolve_primary_model(register_path: Path = REGISTER_PATH) -> str:
    return str(load_experiment_register(register_path)["primary_model"])


def specification_csv_path(
    processed_dir: Path,
    model: str | None = None,
    protocol: str | None = None,
    register_path: Path = REGISTER_PATH,
) -> Path:
    register = load_experiment_register(register_path)
    model = model or str(register["primary_model"])
    protocol = protocol or str(register["protocol_id"])
    return processed_dir / "specification" / f"paper_specifications_{model_slug(model)}_{protocol}.csv"
