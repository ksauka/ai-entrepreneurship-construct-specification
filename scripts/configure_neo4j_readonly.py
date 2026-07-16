"""Create and verify the app's Neo4j reader principal on Enterprise Edition.

Inputs: administrative and app Neo4j credentials from the project .env file.
Outputs: an idempotently created user assigned only PUBLIC and reader roles.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aecsp.knowledge_graph.neo4j_loader import connect  # noqa: E402
from aecsp.specification.llm_coder import load_env  # noqa: E402

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")


def _identifier(value: str, name: str) -> str:
    if not _SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"{name} must start with a letter and contain only letters, numbers, or underscores"
        )
    return f"`{value}`"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    env = load_env(PROJECT_ROOT / ".env")
    required = (
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "NEO4J_APP_USER",
        "NEO4J_APP_PASSWORD",
    )
    missing = [name for name in required if not env.get(name)]
    if missing:
        raise RuntimeError("Missing .env values: " + ", ".join(missing))

    admin_user = env["NEO4J_USER"]
    app_user = env["NEO4J_APP_USER"]
    if app_user == admin_user:
        raise RuntimeError("NEO4J_APP_USER must differ from the administrative loader user")
    app_identifier = _identifier(app_user, "NEO4J_APP_USER")
    database = env.get("NEO4J_DATABASE", "neo4j")

    driver = connect(env["NEO4J_URI"], admin_user, env["NEO4J_PASSWORD"])
    try:
        with driver.session(database=database) as session:
            component = session.run(
                "CALL dbms.components() YIELD name, versions, edition "
                "RETURN name, versions[0] AS version, edition"
            ).single()
        edition = str(component["edition"] if component else "unknown").lower()
        if "enterprise" not in edition:
            raise RuntimeError(
                "A secure reader role cannot be created on Neo4j Community Edition. "
                "Use a licensed Enterprise deployment or keep the app in dataframe fallback mode."
            )

        with driver.session(database="system") as session:
            session.run(
                f"CREATE USER {app_identifier} IF NOT EXISTS "
                "SET PLAINTEXT PASSWORD $password CHANGE NOT REQUIRED",
                password=env["NEO4J_APP_PASSWORD"],
            ).consume()
            for role in ("admin", "architect", "publisher", "editor"):
                session.run(
                    f"REVOKE ROLE {role} FROM {app_identifier}"
                ).consume()
            session.run(f"GRANT ROLE reader TO {app_identifier}").consume()
            record = session.run(
                f"SHOW USER {app_identifier} YIELD user, roles RETURN user, roles"
            ).single()

        roles = sorted(str(role) for role in (record["roles"] if record else []))
        if "reader" not in roles or any(
            role in roles for role in ("admin", "architect", "publisher", "editor")
        ):
            raise RuntimeError(f"Reader-role verification failed for {app_user}: {roles}")
        print(f"Neo4j app principal: {app_user}")
        print(f"Neo4j database: {database}")
        print("Neo4j roles: " + ", ".join(roles))
        print("Read-only role verification: PASS")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
