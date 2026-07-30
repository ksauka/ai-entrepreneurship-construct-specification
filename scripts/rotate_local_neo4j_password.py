"""Rotate the ignored local Neo4j password without printing the new secret.

The command updates only an unused local environment. If an existing Neo4j
data store is present, rotate the database credential through Neo4j instead.
"""

from __future__ import annotations

import os
from pathlib import Path
import secrets
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
NEO4J_DATA = PROJECT_ROOT / "data/neo4j/data"


def _existing_store() -> bool:
    return NEO4J_DATA.is_dir() and any(NEO4J_DATA.iterdir())


def main() -> None:
    if not ENV_FILE.is_file():
        raise RuntimeError(f"Local environment file does not exist: {ENV_FILE}")
    if _existing_store():
        raise RuntimeError(
            "An existing Neo4j data store was detected. Rotate its database "
            "credential through Neo4j before updating .env."
        )

    password = secrets.token_urlsafe(48)
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    replacement = f"NEO4J_PASSWORD={password}"
    updated: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith("NEO4J_PASSWORD="):
            updated.append(replacement)
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.extend(("", replacement))

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".env.",
        dir=ENV_FILE.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write("\n".join(updated) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, ENV_FILE)
    finally:
        temporary.unlink(missing_ok=True)
    os.chmod(ENV_FILE, 0o600)
    print("Local Neo4j password rotated; .env permissions set to 0600.")


if __name__ == "__main__":
    main()
