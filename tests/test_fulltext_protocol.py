"""Contract tests for the full-text arm (spec-ft-v1).

The full-text protocol was added alongside the frozen spec-v3 instrument. These
tests exist to make the separation enforced rather than merely intended: they
fail if spec-v3's prompt, fingerprint or cache location shifts by a single
character, and if spec-ft-v1 ever writes into spec-v3's cache.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aecsp.specification.llm_coder import (
    FULLTEXT_MAX_OUTPUT_TOKENS,
    FULLTEXT_PROTOCOL_ID,
    MAX_OUTPUT_TOKENS,
    PROTOCOL_ID,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_FULLTEXT,
    build_user_prompt,
    build_user_prompt_for,
    build_user_prompt_fulltext,
    model_cache_dir,
    protocol_fingerprint,
    system_prompt_for,
)

SPEC_V3_FINGERPRINT = "04d00994822c239a35149a6ac4dcf46db9b803095f582a400bb133a5f1e4457c"

PAPER = {
    "paper_id": "eid:2-s2.0-000",
    "title": "T",
    "abstract": "A",
    "keywords": "K",
    "journal": "J",
    "year": "2024",
    "full_text": "# T\n\n## Abstract\n\nA\n\n## Introduction\n\nBody.",
}


def test_spec_v3_fingerprint_is_unchanged():
    """The frozen instrument must not move. Everything else rests on this."""
    assert protocol_fingerprint(PROTOCOL_ID, MAX_OUTPUT_TOKENS) == SPEC_V3_FINGERPRINT


def test_spec_v3_prompts_are_untouched_by_the_new_protocol():
    assert system_prompt_for(PROTOCOL_ID) is SYSTEM_PROMPT
    assert build_user_prompt_for(PROTOCOL_ID, PAPER) == build_user_prompt(
        "T", "A", "K", "J", "2024"
    )


def test_fulltext_protocol_is_distinct():
    assert FULLTEXT_PROTOCOL_ID != PROTOCOL_ID
    assert SYSTEM_PROMPT_FULLTEXT != SYSTEM_PROMPT
    assert protocol_fingerprint(
        FULLTEXT_PROTOCOL_ID, FULLTEXT_MAX_OUTPUT_TOKENS
    ) != SPEC_V3_FINGERPRINT


def test_fulltext_cache_can_never_land_in_spec_v3(tmp_path: Path):
    """The separation that keeps 22,345 spec-v3 records safe."""
    v3 = model_cache_dir(tmp_path, "gpt-5.4-mini-2026-03-17", PROTOCOL_ID)
    ft = model_cache_dir(tmp_path, "gpt-5.4-mini-2026-03-17", FULLTEXT_PROTOCOL_ID)
    assert v3 != ft
    assert PROTOCOL_ID in str(v3)
    assert FULLTEXT_PROTOCOL_ID in str(ft)
    assert not str(ft).startswith(str(v3))


def test_fulltext_prompt_carries_the_document_not_the_abstract():
    prompt = build_user_prompt_for(FULLTEXT_PROTOCOL_ID, PAPER)
    assert "FULL TEXT:" in prompt
    assert "## Introduction" in prompt
    assert "ABSTRACT: A" not in prompt


def test_fulltext_prompt_keeps_the_mechanism_rule_verbatim():
    """Rule 7 is what the pre-registered empty-logic correction depends on.

    If its wording drifts, mechanism codes stop being comparable between the
    abstract and full-text arms and the correction no longer applies equally.
    """
    rule7 = (
        "Mechanism requires causal logic. Code a substantive ai_mechanism ONLY if\n"
        "   you can also state the paper's causal logic in ai_mechanism_logic, in the\n"
        "   paper's own terms."
    )
    assert rule7 in SYSTEM_PROMPT
    assert rule7 in SYSTEM_PROMPT_FULLTEXT


def test_fulltext_prompt_drops_the_abstract_only_boundary():
    assert "title, abstract, and author keywords only" in SYSTEM_PROMPT
    assert "title, abstract, and author keywords only" not in SYSTEM_PROMPT_FULLTEXT
    assert "full text of the paper as supplied" in SYSTEM_PROMPT_FULLTEXT


def test_fulltext_retires_needs_full_text_guidance():
    """needs_full_text records metadata insufficiency; the full text is supplied."""
    assert "needs_full_text is a signal, not a caveat" in SYSTEM_PROMPT
    assert "needs_full_text is a signal, not a caveat" not in SYSTEM_PROMPT_FULLTEXT


def test_fulltext_ceiling_is_its_own_and_spec_v3_is_not_raised():
    assert MAX_OUTPUT_TOKENS == 4096
    assert FULLTEXT_MAX_OUTPUT_TOKENS == 8192


def test_spec_ft_v2_differs_from_spec_v3_by_exactly_two_lines():
    """The clean paired protocol: only the boundary and the evidence source.

    This is the method claim made testable. If anyone later edits the full-text
    prompt to add guidance, this fails, and the paired comparison stops being
    attributable to the evidentiary boundary alone.
    """
    import difflib

    from aecsp.specification.llm_coder import SYSTEM_PROMPT_FULLTEXT_V2

    a = SYSTEM_PROMPT.splitlines()
    b = SYSTEM_PROMPT_FULLTEXT_V2.splitlines()
    assert len(a) == len(b), "spec-ft-v2 must not add or remove lines"
    changed = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(changed) == 2, f"expected exactly 2 changed lines, got {len(changed)}"

    diff = "\n".join(difflib.unified_diff(a, b, lineterm=""))
    assert "full text of the paper as supplied" in diff
    assert "close paraphrase from the supplied document" in diff

    # Every coding rule is verbatim, including the ones that read oddly under a
    # full-text boundary. An odd sentence identical in BOTH arms cannot confound
    # a comparison between them; a rewritten one can.
    for rule in ("2. Separate what the text states", "4. Unverifiable is not droppable",
                 "7. Mechanism requires causal logic", "8. needs_full_text is a signal"):
        assert rule in SYSTEM_PROMPT and rule in SYSTEM_PROMPT_FULLTEXT_V2


def test_spec_ft_v2_is_derived_not_transcribed():
    """Deriving it from spec-v3 is what guarantees the diff cannot drift."""
    from aecsp.specification.llm_coder import (
        SYSTEM_PROMPT_FULLTEXT_V2,
        _build_fulltext_v2_prompt,
    )

    assert _build_fulltext_v2_prompt() == SYSTEM_PROMPT_FULLTEXT_V2


def test_all_fulltext_protocols_are_separately_cached(tmp_path: Path):
    """v1 and v2 must never share a cache; v1 stays as audit state."""
    from aecsp.specification.llm_coder import FULLTEXT_V2_PROTOCOL_ID

    dirs = {
        model_cache_dir(tmp_path, "m", p)
        for p in (PROTOCOL_ID, FULLTEXT_PROTOCOL_ID, FULLTEXT_V2_PROTOCOL_ID)
    }
    assert len(dirs) == 3


def test_protocol_registry_is_data_not_branching():
    """A new protocol should be a registry entry, not another `if`."""
    from aecsp.specification.llm_coder import PROTOCOLS, get_protocol

    from aecsp.specification.llm_coder import FULLTEXT_V2_PROTOCOL_ID

    assert set(PROTOCOLS) == {PROTOCOL_ID, FULLTEXT_PROTOCOL_ID, FULLTEXT_V2_PROTOCOL_ID}
    assert get_protocol(PROTOCOL_ID).text_field == "abstract"
    assert get_protocol(FULLTEXT_PROTOCOL_ID).text_field == "full_text"
    assert get_protocol(None).protocol_id == PROTOCOL_ID  # defaults to frozen spec-v3
    with pytest.raises(ValueError):
        get_protocol("spec-does-not-exist")


def test_registry_rebuild_is_byte_identical_for_spec_v3():
    """The refactor must not move spec-v3 output by a single character."""
    assert build_user_prompt_for(PROTOCOL_ID, PAPER) == build_user_prompt(
        "T", "A", "K", "J", "2024"
    )
    assert build_user_prompt_for(FULLTEXT_PROTOCOL_ID, PAPER) == build_user_prompt_fulltext(
        "T", PAPER["full_text"], "K", "J", "2024"
    )


def test_all_three_transports_agree_per_protocol():
    """Live, OpenAI Batch and Gemini Batch must send the same prompts.

    Live and Batch share one cache and resume each other, so if their bodies
    diverge a cache would silently mix two instruments. This is the property
    that makes the transports interchangeable, and it must hold for spec-ft-v1
    exactly as it already does for spec-v3.
    """
    from aecsp.specification.gemini_batch import request_line as gemini_line
    from aecsp.specification.openai_batch import build_body

    for protocol in (PROTOCOL_ID, FULLTEXT_PROTOCOL_ID):
        expected_system = system_prompt_for(protocol)
        expected_user = build_user_prompt_for(protocol, PAPER)

        body = build_body("m", PAPER, 4096, protocol)
        assert body["messages"][0]["content"] == expected_system
        assert body["messages"][1]["content"] == expected_user

        gem = gemini_line("m", PAPER, protocol)["request"]
        assert gem["system_instruction"]["parts"][0]["text"] == expected_system
        assert gem["contents"][0]["parts"][0]["text"] == expected_user


def test_batch_builders_default_to_spec_v3():
    """Existing call sites pass no protocol and must be unaffected."""
    from aecsp.specification.gemini_batch import request_line as gemini_line
    from aecsp.specification.openai_batch import build_body

    assert build_body("m", PAPER, 4096)["messages"][0]["content"] is SYSTEM_PROMPT
    assert gemini_line("m", PAPER)["request"]["system_instruction"]["parts"][0][
        "text"
    ] is SYSTEM_PROMPT


def test_coding_level_selection(tmp_path: Path):
    """Downstream selects a coding level; protocol ids stay internal."""
    from aecsp.specification.paths import (
        ABSTRACT_LEVEL,
        FULL_TEXT_LEVEL,
        coding_levels,
        resolve_protocol,
        specification_csv_path,
    )

    assert resolve_protocol(ABSTRACT_LEVEL) == PROTOCOL_ID
    assert resolve_protocol(None) == PROTOCOL_ID          # default is the frozen arm
    assert resolve_protocol(FULL_TEXT_LEVEL) == FULLTEXT_PROTOCOL_ID
    with pytest.raises(ValueError):
        resolve_protocol("guesswork")

    abstract_csv = specification_csv_path(tmp_path, "m", level=ABSTRACT_LEVEL)
    fulltext_csv = specification_csv_path(tmp_path, "m", level=FULL_TEXT_LEVEL)
    assert abstract_csv != fulltext_csv
    assert PROTOCOL_ID in abstract_csv.name
    assert FULLTEXT_PROTOCOL_ID in fulltext_csv.name

    # Ambiguity is a caller error, not a silent precedence rule.
    with pytest.raises(ValueError):
        specification_csv_path(tmp_path, "m", protocol=PROTOCOL_ID, level=ABSTRACT_LEVEL)

    # Availability is reported, never silently substituted: a fallback would
    # make an ablation condition indistinguishable from its control.
    levels = coding_levels(tmp_path, "m")
    assert set(levels) == {ABSTRACT_LEVEL, FULL_TEXT_LEVEL}
    assert levels[FULL_TEXT_LEVEL]["available"] is False


def test_build_paper_record_modes(tmp_path: Path):
    """The loader supports both arms from one code path."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_specification import build_paper_record

    row = {
        "paper_id": "eid:2-s2.0-000",
        "Title": "T",
        "Abstract": "A",
        "Author Keywords": "K",
        "Source title": "J",
        "Year": "2024",
    }
    abstract_mode = build_paper_record(row)
    assert "full_text" not in abstract_mode
    assert abstract_mode["abstract"] == "A"

    (tmp_path / "2-s2.0-000.md").write_text("BODY", encoding="utf-8")
    fulltext_mode = build_paper_record(row, tmp_path, FULLTEXT_PROTOCOL_ID)
    assert fulltext_mode["full_text"] == "BODY"
    # The corpus abstract is still carried; the document also holds it at its head.
    assert fulltext_mode["abstract"] == "A"

    with pytest.raises(FileNotFoundError):
        build_paper_record(
            {**row, "paper_id": "eid:missing"}, tmp_path, FULLTEXT_PROTOCOL_ID
        )


def test_build_paper_record_refuses_silent_mismatches():
    """Mismatched arguments must fail loudly, never produce the wrong text."""
    row = {
        "paper_id": "eid:2-s2.0-000",
        "Title": "T",
        "Abstract": "A",
        "Author Keywords": "K",
        "Source title": "J",
        "Year": "2024",
    }
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from aecsp.specification.llm_coder import build_paper_record as brec

    # text_dir with an abstract protocol: the directory would be ignored.
    with pytest.raises(ValueError):
        brec(row, Path("/tmp"), PROTOCOL_ID)
    # full-text protocol with no text_dir: there would be no document to code.
    with pytest.raises(ValueError):
        brec(row, None, FULLTEXT_PROTOCOL_ID)
