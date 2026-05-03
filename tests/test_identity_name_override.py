"""Regression: when identity.md has a `## Имя` section, the system
prompt preamble must include an AGENT NAME OVERRIDE block at the END
so the model treats the name as its own and doesn't deny it.

Without the override, the name lives only inside the IDENTITY block,
gets buried under the longer SOUL section + recent conversation
turns, and the agent denies being called by name ("I'm not Hrant").
"""
from __future__ import annotations

from backend.identity import IdentityManager


def test_extract_name_section_finds_body(tmp_path):
    idm = IdentityManager(base_dir=tmp_path)
    # Default identity.md doesn't include a `## Имя` section, so:
    assert idm._extract_name_section(idm.identity()) == ""


def test_extract_name_section_returns_body_when_present(tmp_path):
    idm = IdentityManager(base_dir=tmp_path)
    idm.identity_path.write_text(
        "# Identity\n\n## Имя\nMy name is Hrant.\n\n## Я\n- agent\n",
        encoding="utf-8",
    )
    assert "Hrant" in idm._extract_name_section(idm.identity())


def test_preamble_no_name_override_when_section_empty(tmp_path):
    idm = IdentityManager(base_dir=tmp_path)
    pre = idm.preamble()
    assert "AGENT NAME OVERRIDE" not in pre


def test_preamble_appends_name_override_when_set(tmp_path):
    idm = IdentityManager(base_dir=tmp_path)
    idm.identity_path.write_text(
        "# Identity\n\n## Имя\nMeня зовут Hrant.\n\n## Я\n- agent\n",
        encoding="utf-8",
    )
    pre = idm.preamble()
    assert "AGENT NAME OVERRIDE" in pre
    # Order: SOUL → IDENTITY → USER PROFILE → AGENT NAME OVERRIDE
    # (highest attention weight at the end).
    assert pre.index("# IDENTITY") < pre.index("# AGENT NAME OVERRIDE")
    assert pre.index("# USER PROFILE") < pre.index("# AGENT NAME OVERRIDE")
    body = pre.split("# AGENT NAME OVERRIDE", 1)[1]
    assert "Hrant" in body
    # Strong instruction wording so the model can't ignore it.
    assert "Do not deny" in body or "Acknowledge" in body
