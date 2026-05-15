"""Regression: agent-behavior rules accidentally written into
user_profile.md must NOT reach the system prompt.

The real incident: the memory extractor wrote
    - Respond to the name Hrant.  _(добавлено 2026-05-14)_
under `## Правила взаимодействия` of a Telegram speaker profile.
Inside the prompt's `# USER PROFILE` block that line reads as
"the user is named Hrant", and the agent then addressed the user
by its own name on follow-up turns. The fix has two layers:

  1. memory_extractor blocks such facts on the WRITE path
     (see test_memory_extractor_filter.py)
  2. IdentityManager.user_profile() strips them on the READ path
     so existing pollution doesn't reach the prompt either
     (this test)
"""
from __future__ import annotations

from backend.identity import (
    IdentityManager,
    _looks_like_agent_behavior_line,
    _strip_agent_behavior_lines,
)


def test_strips_respond_to_the_name_bullet():
    text = (
        "# User Profile\n\n"
        "## О пользователе\n"
        "- User is named Gor.\n\n"
        "## Правила взаимодействия\n"
        "- Respond to the name Hrant.\n"
        "- Always answer in Russian.\n"
    )
    out = _strip_agent_behavior_lines(text)
    assert "Respond to the name" not in out
    # Other rules survive.
    assert "answer in Russian" in out
    # User's own name survives.
    assert "User is named Gor" in out
    # Section headers untouched.
    assert "## Правила взаимодействия" in out


def test_strips_russian_variants():
    text = "## Правила\n- Откликаешься на имя Hrant.\n- Тебя зовут Hrant.\n"
    out = _strip_agent_behavior_lines(text)
    assert "Откликаешься" not in out
    assert "Тебя зовут" not in out


def test_keeps_non_bullet_text():
    """Section headers + prose paragraphs are left alone — only
    bullet items get the rule check."""
    text = (
        "# User Profile\n\n"
        "This profile tracks how to respond to the name the user "
        "asks me to use.\n"
    )
    out = _strip_agent_behavior_lines(text)
    # The prose line (no bullet marker) must survive even though it
    # contains "respond to the name" inside it.
    assert "This profile tracks how to respond to the name" in out
    # Section header survives.
    assert "# User Profile" in out


def test_keeps_user_fact_lookalikes():
    """`User's name is X` is a USER fact and must survive even
    though the prefix 'user' superficially resembles addressing
    instructions."""
    cases = [
        "- User's name is Gor.",
        "- User is named Gor.",
        "- User prefers terse answers.",
        "- User lives in Yerevan.",
    ]
    for line in cases:
        assert not _looks_like_agent_behavior_line(line), line


def test_user_profile_reader_returns_sanitized_text(tmp_path):
    idm = IdentityManager(base_dir=tmp_path)
    idm.user_path.write_text(
        "# User Profile\n\n"
        "## О пользователе\n"
        "- User is named Gor.\n\n"
        "## Правила взаимодействия\n"
        "- Respond to the name Hrant.\n",
        encoding="utf-8",
    )
    out = idm.user_profile()
    assert "Respond to the name" not in out
    assert "User is named Gor" in out


def test_preamble_does_not_show_polluted_rule(tmp_path):
    """The whole point: even with a polluted user.md on disk, the
    `# USER PROFILE` block in the system prompt must not contain the
    agent-behavior rule. Otherwise the model reads it under USER
    PROFILE and conflates the agent name with the user's name."""
    idm = IdentityManager(base_dir=tmp_path)
    idm.identity_path.write_text(
        "# Identity\n\n## Имя\nMy name is Hrant.\n",
        encoding="utf-8",
    )
    idm.user_path.write_text(
        "## О пользователе\n- User is named Gor.\n\n"
        "## Правила взаимодействия\n- Respond to the name Hrant.\n",
        encoding="utf-8",
    )
    pre = idm.preamble()
    # The polluted rule must not appear inside the USER PROFILE block.
    user_profile_block = pre.split("# USER PROFILE", 1)[1].split("# NAMES", 1)[0]
    assert "Respond to the name" not in user_profile_block
    # Hrant is still present in the prompt — but only inside the
    # NAMES block, labeled as YOUR name. And the user is Gor.
    names_block = pre.split("# NAMES", 1)[1]
    assert "Hrant" in names_block
    assert "Gor" in names_block
