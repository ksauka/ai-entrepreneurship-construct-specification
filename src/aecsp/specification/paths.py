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


# Coding levels are the user-facing choice; protocol ids stay an implementation
# detail read from the register, so the two can never disagree. "abstract" is
# the frozen spec-v3 arm over the full corpus; "full_text" is spec-ft-v1 over
# the purposive 131-paper set. Analysis, the app and the RAG all select a level,
# and the ablation conditions are the same selection.
ABSTRACT_LEVEL = "abstract"
FULL_TEXT_LEVEL = "full_text"


def resolve_protocol(
    level: str | None = None,
    register_path: Path = REGISTER_PATH,
) -> str:
    """Map a coding level to its registered protocol id."""
    register = load_experiment_register(register_path)
    if level in (None, ABSTRACT_LEVEL):
        return str(register["protocol_id"])
    if level == FULL_TEXT_LEVEL:
        arm = register.get("fulltext_arm")
        if not arm:
            raise ValueError(
                "No 'fulltext_arm' block in the experiment register; the "
                "full-text protocol is not registered."
            )
        return str(arm["protocol_id"])
    raise ValueError(
        f"Unknown coding level '{level}'. "
        f"Known levels: {ABSTRACT_LEVEL}, {FULL_TEXT_LEVEL}."
    )


def specification_csv_path(
    processed_dir: Path,
    model: str | None = None,
    protocol: str | None = None,
    register_path: Path = REGISTER_PATH,
    level: str | None = None,
) -> Path:
    """Path to a coded dataset.

    Pass `level` ("abstract" or "full_text") for the readable selection, or
    `protocol` when an exact id is already known. Passing both is a caller
    error rather than a silent precedence rule.
    """
    if level is not None and protocol is not None:
        raise ValueError("Pass either 'level' or 'protocol', not both.")
    register = load_experiment_register(register_path)
    model = model or str(register["primary_model"])
    if protocol is None:
        protocol = resolve_protocol(level, register_path)
    return processed_dir / "specification" / f"paper_specifications_{model_slug(model)}_{protocol}.csv"


def coding_levels(
    processed_dir: Path,
    model: str | None = None,
    register_path: Path = REGISTER_PATH,
) -> dict[str, dict]:
    """Report which coding levels have a dataset on disk.

    The full-text level is registered before it has been run, so callers must be
    able to ask what exists rather than assume. A selector that silently falls
    back to the abstract level would make an ablation condition indistinguishable
    from its control.
    """
    out: dict[str, dict] = {}
    for level in (ABSTRACT_LEVEL, FULL_TEXT_LEVEL):
        try:
            protocol = resolve_protocol(level, register_path)
        except ValueError:
            continue
        path = specification_csv_path(
            processed_dir, model, protocol, register_path
        )
        out[level] = {"protocol": protocol, "path": path, "available": path.is_file()}
    return out
