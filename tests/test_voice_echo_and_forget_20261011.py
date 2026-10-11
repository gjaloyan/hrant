"""The owner's three asks of 2026-10-11.

1. Show what a voice note was heard as, before the answer (Hermes does).
2. Typed Armenian reaches the turn in English, with the original kept.
3. A way to take a stored fact back: a memory drill had left "favourite
   colour teal" and its siblings in his long-term memory two dozen times,
   and nothing could remove them.
"""
from __future__ import annotations

import inspect
import json

import pytest

from backend import fact_search as fs
from backend import meaning_translate as mt


# ── 1. voice echo ─────────────────────────────────────────────────────

def test_the_echo_quotes_what_was_heard():
    assert mt.voice_echo("  Բարև,  ինչ ժամ է ") == "🎙 \"Բարև, ինչ ժամ է\""
    assert mt.voice_echo("") == ""
    long = mt.voice_echo("слово " * 2000)
    assert len(long) < 3600 and long.endswith("…\"")


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _fake_translation(monkeypatch, english="What time is it?"):
    calls = []

    def _render(src, *, spoken=True):
        calls.append((src, spoken))
        how = "spoken" if spoken else "written"
        return f"{english}\n\n[{how} by the user, verbatim, in the language they used: {src}]"

    monkeypatch.setattr(mt, "render_for_prompt", _render)
    return calls


def test_the_sender_sees_the_words_as_heard_before_the_turn(monkeypatch):
    calls = _fake_translation(monkeypatch)
    sent = []

    async def reply(text):
        sent.append(text)

    out = _run(mt.voice_turn_text("ինչ ժամ է", reply))
    assert sent == ["🎙 \"ինչ ժամ է\""], "the echo quotes the words as heard, not the English"
    assert out.startswith("What time is it?") and calls == [("ինչ ժամ է", True)]


def test_a_failed_echo_does_not_lose_the_message(monkeypatch):
    _fake_translation(monkeypatch)

    async def reply(text):
        raise RuntimeError("telegram down")

    assert _run(mt.voice_turn_text("ինչ ժամ է", reply)).startswith("What time")


def test_the_echo_can_be_switched_off(monkeypatch):
    from backend.config import CONFIG
    _fake_translation(monkeypatch)
    monkeypatch.setitem(CONFIG._data, "conversation", {"echo_voice_transcript": False})
    sent = []

    async def reply(text):
        sent.append(text)

    _run(mt.voice_turn_text("ինչ ժամ է", reply))
    assert sent == []


def test_the_telegram_handler_uses_both():
    import backend.channels as ch
    src = inspect.getsource(ch)
    assert "await voice_turn_text(text, msg.reply_text)" in src
    i_typed = src.index("await typed_turn_text(text)")
    i_voice = src.index("# Voice without text → use the transcript as the message body")
    assert i_typed < i_voice, "typed text only; the voice body is rendered already"


# ── 2. typed Armenian ─────────────────────────────────────────────────

def test_armenian_is_recognised_and_other_scripts_are_not():
    assert mt.is_armenian("Բարև, ինչ ժամ է հիմա Երևանում?")
    assert not mt.is_armenian("Привет, сколько времени?")
    assert not mt.is_armenian("hello Ա")


def test_typed_text_uses_the_written_brief_and_keeps_the_original(monkeypatch):
    seen = {}

    class R:
        def call(self, task_type, system, user, **kw):
            seen["system"] = system
            return "GARBLED: looks odd"  # a typed message is never "speech failed"

    import backend.llm as llm
    monkeypatch.setattr(llm, "router", lambda: R())
    out = mt.render_for_prompt("ինչ ժամ է", spoken=False)
    assert "speech recognition" not in seen["system"].lower()
    assert "[written by the user, verbatim, in the language they used: ինչ ժամ է]" in out
    assert "SPEECH RECOGNITION FAILED" not in out


def test_typed_armenian_is_rendered_and_russian_passes_through(monkeypatch):
    calls = _fake_translation(monkeypatch)
    out = _run(mt.typed_turn_text("Բարև, ինչ ժամ է հիմա?"))
    assert calls == [("Բարև, ինչ ժամ է հիմա?", False)]
    assert "[written by the user" in out
    assert _run(mt.typed_turn_text("Привет, сколько времени?")) == "Привет, сколько времени?"
    assert len(calls) == 1


def test_armenian_translation_can_be_switched_off(monkeypatch):
    from backend.config import CONFIG
    calls = _fake_translation(monkeypatch)
    monkeypatch.setitem(CONFIG._data, "conversation", {"translate_armenian_text": False})
    assert _run(mt.typed_turn_text("Բարև, ինչ ժամ է")) == "Բարև, ինչ ժամ է"
    assert calls == []


# ── 3. forgetting a fact ──────────────────────────────────────────────

@pytest.fixture()
def kb(tmp_path, monkeypatch):
    from backend import fact_forget
    from backend import knowledge_graph, memory_extractor
    from backend.graph import store as g2store
    k = tmp_path / "knowledge"
    k.mkdir(parents=True)
    fs._STORE = None
    monkeypatch.setattr(fs, "_fact_store_path", lambda: k / "fact_embeddings.json")
    monkeypatch.setattr(fs, "_facts_path", lambda: k / "memory_facts.jsonl")
    monkeypatch.setattr(fact_forget, "_archive_path", lambda: k / "self" / "forgotten.jsonl")
    kg = knowledge_graph.KnowledgeGraph(path=k / "graph.json")
    monkeypatch.setattr(knowledge_graph, "GRAPH", kg)
    monkeypatch.setattr(memory_extractor, "GRAPH", kg)
    monkeypatch.setattr(g2store, "GRAPH", g2store.Graph(root=k))
    yield k
    fs._STORE = None


TEAL = "The user's favorite color is teal."
CAR = "User owns a 2001 Subaru Forester."
STEAL = "Someone tried to steal the user's bicycle."


def _rows(k):
    return [json.loads(l) for l in (k / "memory_facts.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]


def _seed(k):
    rows = [
        {"summary": TEAL, "speaker_id": "webui:default",
         "triples": [["user", "favorite_color", "teal"]]},
        {"summary": TEAL, "speaker_id": "audit:claude", "triples": []},
        {"summary": CAR, "speaker_id": "telegram:1", "triples": []},
        {"summary": STEAL, "speaker_id": "telegram:1", "triples": []},
    ]
    (k / "memory_facts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    store = fs.get_store()
    for s in (TEAL, CAR, STEAL):
        store.add(fs.fact_id(s), [1.0, 0.0, 0.0])


def test_candidates_show_every_copy_and_change_nothing(kb):
    from backend import fact_forget
    _seed(kb)
    found = fact_forget.candidates("teal")
    assert [c["fact"] for c in found] == [TEAL], "whole words only: not 'steal'"
    assert found[0]["copies"] == 2
    assert set(found[0]["speakers"]) == {"webui:default", "audit:claude"}
    assert len(_rows(kb)) == 4


def test_forget_removes_rows_and_vectors_and_keeps_an_archive(kb):
    from backend import fact_forget
    _seed(kb)
    res = fact_forget.forget([fs.fact_id(TEAL)], by="telegram:848732236")
    assert res["forgotten"] == 2 and res["facts"] == [TEAL]
    assert [r["summary"] for r in _rows(kb)] == [CAR, STEAL]
    assert not fs.get_store().has(fs.fact_id(TEAL))
    assert fs.get_store().has(fs.fact_id(CAR))
    archived = (kb / "self" / "forgotten.jsonl").read_text(encoding="utf-8")
    assert archived.count(TEAL) == 2


def test_forget_in_one_speakers_scope_leaves_the_others_copy(kb):
    from backend import fact_forget
    _seed(kb)
    res = fact_forget.forget([fs.fact_id(TEAL)], speaker_id="webui:default")
    assert res["forgotten"] == 1
    assert [r["speaker_id"] for r in _rows(kb) if r["summary"] == TEAL] == ["audit:claude"]
    assert fs.get_store().has(fs.fact_id(TEAL)), "the other copy still needs its vector"


def test_a_forgotten_fact_leaves_graph_recall(kb):
    from backend import fact_forget, memory_extractor
    _seed(kb)
    memory_extractor.GRAPH.add_relations(
        [("user", "favorite_color", "teal")], source_note="_memory", scope="webui:default")
    mem = memory_extractor.MemoryExtractor(log_path=kb / "memory_facts.jsonl")
    assert mem.recall("teal", speaker_id="webui:default")
    fact_forget.forget([fs.fact_id(TEAL)])
    assert mem.recall("teal", speaker_id="webui:default") == []


def test_the_v2_graph_node_goes_too(kb):
    from backend import fact_forget
    from backend.graph import builder, store as g2store
    from backend.graph.model import fact_id as node_id
    _seed(kb)
    builder.add_fact(text=TEAL, related_topics=[], graph=g2store.GRAPH)
    assert g2store.GRAPH.get_node(node_id(TEAL)) is not None
    fact_forget.forget([fs.fact_id(TEAL)])
    assert g2store.GRAPH.get_node(node_id(TEAL)) is None


def test_the_last_forget_can_be_undone(kb):
    from backend import fact_forget
    _seed(kb)
    fact_forget.forget([fs.fact_id(TEAL)])
    assert fact_forget.restore(last=True)["restored"] == 2
    assert sum(r["summary"] == TEAL for r in _rows(kb)) == 2


def test_appends_wait_for_a_rewrite(tmp_path, monkeypatch):
    from backend import memory_extractor
    entered = []

    class Lock:
        def __enter__(self):
            entered.append(1)

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(memory_extractor, "FACTS_FILE_LOCK", Lock())
    mem = memory_extractor.MemoryExtractor(log_path=tmp_path / "f.jsonl")
    mem._append_log(memory_extractor.MemoryFact("x is y", [("x", "is", "y")], []))
    assert entered, "the journal append must hold the lock fact_forget rewrites under"


# ── the tool ──────────────────────────────────────────────────────────

def _as(speaker, role, monkeypatch):
    from backend import roles
    monkeypatch.setattr(roles, "current_speaker", lambda: speaker)
    monkeypatch.setattr(roles, "role_of", lambda sid: role)
    monkeypatch.setattr(roles, "is_owner", lambda sid: role == "owner")


def test_a_query_never_deletes(kb, monkeypatch):
    from backend.builtin_tools import _forget_fact_handler
    _seed(kb)
    _as("telegram:848732236", "owner", monkeypatch)
    out = json.loads(_forget_fact_handler(query="teal"))
    assert out["candidates"][0]["id"] == fs.fact_id(TEAL)
    assert len(_rows(kb)) == 4
    out = json.loads(_forget_fact_handler(fact_ids=[fs.fact_id(TEAL)]))
    assert out["forgotten"] == 2


def test_a_non_owner_forgets_only_their_own_and_a_guest_nothing(kb, monkeypatch):
    from backend.builtin_tools import _forget_fact_handler
    _seed(kb)
    _as("webui:default", "trusted", monkeypatch)
    out = json.loads(_forget_fact_handler(fact_ids=[fs.fact_id(TEAL)]))
    assert out["forgotten"] == 1
    assert json.loads(_forget_fact_handler(undo_last=True))["ok"] is False
    _as("telegram:5", "guest", monkeypatch)
    assert json.loads(_forget_fact_handler(fact_ids=[fs.fact_id(CAR)]))["ok"] is False
    assert any(r["summary"] == CAR for r in _rows(kb))


def test_the_tool_is_always_available():
    from backend.tool_bundles import BASE_TOOLS
    from backend.builtin_tools import register_builtin_tools
    from backend.tool_registry import get_registry
    register_builtin_tools()
    assert "forget_fact" in BASE_TOOLS and "forget_fact" in get_registry().tools
