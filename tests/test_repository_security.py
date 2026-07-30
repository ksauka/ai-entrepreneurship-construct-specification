"""Regression checks for public repository security defaults."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_public_environment_example_uses_only_placeholders():
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    values = {}
    for line in example.splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value

    assert values["WORKBENCH_SSH_HOST"] == "your-workbench-host"
    assert values["WORKBENCH_SSH_USER"] == "your-workbench-user"
    assert "/home/" not in values["WORKBENCH_SSH_KEY"]
    assert "/home/" not in values["WORKBENCH_OLLAMA_MODELS"]
    assert "replace" in values["NEO4J_PASSWORD"]
    assert "replace" in values["NEO4J_APP_PASSWORD"]


def test_neo4j_compose_defaults_are_local_and_fail_closed():
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert '"127.0.0.1:7474:7474"' in compose
    assert '"127.0.0.1:7687:7687"' in compose
    assert "NEO4J_PASSWORD:?" in compose
    assert "NEO4J_PASSWORD:-" not in compose


def test_private_operations_are_ignored_from_public_repository():
    ignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "/deploy" in ignore
    assert "/scripts/update_ks_*.py" in ignore
    assert "/scripts/add_ks_platform_figure.py" in ignore
    assert "/scripts/analyze_theory_elaboration_integration.py" in ignore
    assert "/scripts/audit_manuscript_exhibits.py" in ignore
    assert "/scripts/build_full_theory_elaboration_manuscript.py" in ignore
    assert "/scripts/build_nested_specification_supplement.py" in ignore
    assert "/scripts/build_probability_sample_results_report.py" in ignore
    assert "/scripts/recalculate_human_insight_irr.py" in ignore
    assert "/scripts/prepare_gpt54mini_challenge.py" in ignore
    assert "/scripts/prepare_human_validation.py" in ignore
    assert "/scripts/workbench.sh" in ignore
    assert "/scripts/*review_host*" in ignore
    assert "/scripts/auto_failover_check.sh" in ignore
    assert "!docs/DESKTOP_FAILOVER_SETUP.md" not in ignore
    assert "!docs/PIPELINE.md" not in ignore
