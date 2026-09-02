"""The system prompt has to be visible to be understood.

The Pipeline screen edits prompt sections one at a time through a
dropdown and nothing showed the assembled result, so "do we have all the
system prompts in there?" had no answer anyone could look up. Measured on
prod: the thirteen rule modules a profile can override are about a third
of what the model reads.
"""
from backend import prompt_preview as pp


def test_it_shows_every_part_in_reading_order():
    d = pp.assemble()
    names = [p["name"] for p in d["parts"]]
    assert names == ["Identity", "Rules", "Permissions", "Capabilities"], names


def test_it_says_which_part_a_profile_can_change():
    """The actual answer to the question: a profile overlays RULES.
    Identity is content, edited elsewhere by design."""
    d = pp.assemble()
    by = {p["name"]: p for p in d["parts"]}
    assert by["Rules"]["profile_can_override"] is True
    assert by["Identity"]["profile_can_override"] is False
    assert "Character" in by["Identity"]["edit_in"]


def test_sizes_are_real_and_add_up():
    d = pp.assemble()
    for p in d["parts"]:
        assert p["chars"] == len(p["text"])
    assert d["total_chars"] == sum(p["chars"] for p in d["parts"])


def test_the_channel_changes_the_rules():
    """m7_format_* is channel-conditional, so the Rules body must differ."""
    tg = next(p for p in pp.assemble(channel="telegram")["parts"]
              if p["name"] == "Rules")
    web = next(p for p in pp.assemble(channel="webui")["parts"]
               if p["name"] == "Rules")
    assert tg["text"] != web["text"]


def test_the_per_turn_parts_are_named_not_omitted():
    """A preview that quietly leaves things out is how the question went
    unanswered in the first place."""
    d = pp.assemble()
    joined = " ".join(d["per_turn"]).lower()
    for expected in ("now", "recall", "conversation", "skill"):
        assert expected in joined, expected


def test_a_broken_part_does_not_break_the_preview(monkeypatch):
    import backend.identity as ident

    def boom(*a, **k):
        raise RuntimeError("soul.md unreadable")

    monkeypatch.setattr(ident.IDENTITY, "preamble", boom)
    d = pp.assemble()
    by = {p["name"]: p for p in d["parts"]}
    assert by["Identity"]["chars"] == 0
    assert by["Rules"]["chars"] > 0, "one bad part took the rest down"
