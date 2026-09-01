"""A stale speaker must be removable, and the owner must not be able to
lock themselves out doing it.

Demoting to `guest` was the only way to deal with an entry that no longer
belongs, so the roles table only ever grew — on prod it carried five
one-off audit identities that could be demoted but never cleared.
"""
import importlib

import pytest


@pytest.fixture
def roles(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend import config, roles as roles_mod
    importlib.reload(config)
    importlib.reload(roles_mod)
    return roles_mod


def test_a_stale_entry_can_be_removed(roles):
    roles.set_role("audit:claude5", "trusted", label="")
    assert "audit:claude5" in roles.list_roles()["speakers"]

    assert roles.forget_speaker("audit:claude5") is True
    assert "audit:claude5" not in roles.list_roles()["speakers"]


def test_removing_something_absent_says_so(roles):
    assert roles.forget_speaker("telegram:nobody") is False


def test_the_local_owner_cannot_be_removed(roles):
    # webui:default is pinned everywhere else for the same reason: it is
    # how the person at the keyboard owns their own box.
    with pytest.raises(ValueError, match="cannot be removed"):
        roles.forget_speaker("webui:default")


def test_an_ownerless_state_cannot_be_reached(roles):
    """There is no "last owner" guard because there cannot be a last owner.

    `_load` re-adds webui:default to the owner list on every read, so even
    a roles.json hand-edited to remove every owner comes back with one.
    A guard for that case would be unreachable code.
    """
    state = roles._load()
    state["owner_speaker_ids"] = []
    roles._save(state)
    assert roles.list_roles()["owner_speaker_ids"] == ["webui:default"]


def test_one_owner_of_several_can_go(roles):
    roles.set_role("telegram:111", "owner")
    roles.set_role("telegram:222", "owner")
    assert roles.forget_speaker("telegram:222") is True
    assert "telegram:222" not in roles.list_roles()["owner_speaker_ids"]
    assert "telegram:111" in roles.list_roles()["owner_speaker_ids"]


def test_removal_clears_the_owner_list_too(roles):
    # A speaker left in owner_speaker_ids after their entry is gone would
    # still hold owner rights while being invisible in the table.
    roles.set_role("telegram:333", "owner")
    roles.forget_speaker("telegram:333")
    assert "telegram:333" not in roles.list_roles()["owner_speaker_ids"]
    assert roles.role_of("telegram:333") == "guest"


def test_a_forgotten_speaker_comes_back_as_a_guest(roles):
    # Forgetting is not a ban. Anyone unknown is a guest by default, and
    # that must stay true for someone who was removed.
    roles.set_role("telegram:444", "trusted")
    roles.forget_speaker("telegram:444")
    assert roles.role_of("telegram:444") == "guest"
