"""Regression: identity preamble must clearly separate the agent's own
name from the user's name. The two names live in different files
(identity.md vs user_profile.md) but historically the prompt only
emphasised the agent name, so on follow-up turns the model started
addressing the user with the agent's name. The fix:

  - render a single `# NAMES — DO NOT CONFUSE` block at the END of the
    preamble (highest model attention)
  - state YOUR name and the USER'S name with explicit `you` / `the user`
    labels
  - add an explicit rule that 'I am X' / 'you are Y, I am X' updates
    the USER side, not the agent side
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


def test_preamble_no_names_block_when_nothing_known(tmp_path):
    idm = IdentityManager(base_dir=tmp_path)
    # Empty identity.md `## Имя` AND empty user profile → no names block.
    # (Note: the default templates seed an empty `## Имя` section so the
    # extractor returns ""; the block is only added when there's something
    # concrete to say.)
    pre = idm.preamble()
    assert "NAMES — DO NOT CONFUSE" not in pre


def test_preamble_includes_agent_name_only(tmp_path):
    """Identity has the agent name but the user profile has no name yet."""
    idm = IdentityManager(base_dir=tmp_path)
    idm.identity_path.write_text(
        "# Identity\n\n## Имя\nMeня зовут Hrant.\n\n## Я\n- agent\n",
        encoding="utf-8",
    )
    pre = idm.preamble()
    assert "NAMES — DO NOT CONFUSE" in pre
    # Order: NAMES block lives AFTER the static identity / user-profile
    # sections so the model's attention weights it highest.
    assert pre.index("# IDENTITY") < pre.index("# NAMES — DO NOT CONFUSE")
    assert pre.index("# USER PROFILE") < pre.index("# NAMES — DO NOT CONFUSE")
    body = pre.split("# NAMES — DO NOT CONFUSE", 1)[1]
    assert "Hrant" in body
    # Explicit role labels — the prior wording was strong but symmetric,
    # which the model could re-purpose as "the user is named Hrant too".
    assert "YOUR name" in body
    # And the rule that prevents flipping names on user corrections.
    assert "Do not flip" in body or "do not flip" in body.lower()


def test_preamble_includes_both_names_and_separates_roles(tmp_path):
    """Identity has Hrant; user profile says the user is Gor. Both must
    appear, each tagged with the correct role (YOUR vs USER'S)."""
    idm = IdentityManager(base_dir=tmp_path)
    idm.identity_path.write_text(
        "# Identity\n\n## Имя\nMeня зовут Hrant.\n\n## Я\n- agent\n",
        encoding="utf-8",
    )
    idm.user_path.write_text(
        "# User Profile\n\n## О пользователе\n- User's name is Gor.\n",
        encoding="utf-8",
    )
    pre = idm.preamble()
    body = pre.split("# NAMES — DO NOT CONFUSE", 1)[1]
    assert "Hrant" in body
    assert "Gor" in body
    # Hrant must be tagged as the agent's name, Gor as the user's name.
    hrant_pos = body.index("Hrant")
    gor_pos = body.index("Gor")
    your_pos = body.index("YOUR name")
    users_pos = body.index("USER'S name")
    # YOUR name block precedes USER'S name block.
    assert your_pos < users_pos
    # Hrant is in the YOUR block (between YOUR and USER'S).
    assert your_pos < hrant_pos < users_pos
    # Gor is in the USER'S block (after USER'S).
    assert gor_pos > users_pos
    # Explicit address rule.
    assert "NEVER address the user by YOUR own name" in body


def test_extract_user_name_patterns():
    """Patterns we accept in the free-form user_profile body."""
    cases = [
        ("- User's name is Gor.", "Gor"),
        ("- User is named Gor", "Gor"),
        ("- User is Gor, a 34-year-old engineer.", "Gor"),
        ("Меня зовут Gor", "Gor"),
        ("Пользователя зовут Gor.", "Gor"),
    ]
    for body, expected in cases:
        text = f"# User Profile\n\n## О пользователе\n{body}\n"
        assert IdentityManager._extract_user_name(text) == expected, body


def test_extract_user_name_empty_when_unknown():
    text = "# User Profile\n\n## О пользователе\n- likes coffee\n"
    assert IdentityManager._extract_user_name(text) == ""
