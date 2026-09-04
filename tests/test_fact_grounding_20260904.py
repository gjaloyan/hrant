"""A fact the model made up is stored exactly like one the user stated.

3654 fact rows carry summary, triples, tags, category, confidence, ts,
source_turn, speaker_id, writer — and nothing that says where the claim
came from. So memory launders model output into knowledge, and recall
serves both with the same authority.

Not hypothetical. Consolidation turned "the user calls the assistant
Hrant" into "Пользователя зовут Hrant" ten times across three weeks, and
"The user's name is Alice." arrived at confidence 1.0 from a smoke-test
fixture. Asked its owner's name on 2026-09-03 the agent answered with
its own. Every one of those rows looked exactly like a fact the user had
stated in person.

`grounding` records which it is. Missing stays "unknown" rather than
being guessed: an unlabelled row must not be promoted to evidence.
"""
import json

from backend.autonomic.levers.memory_consolidation import (
    FIRE_MEMORY_CONSOLIDATION, CONSOLIDATION_SYSTEM,
)


def _rows(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def test_the_extractor_is_asked_where_each_fact_came_from():
    rules = CONSOLIDATION_SYSTEM
    assert "grounding" in rules
    for value in ("user_stated", "tool_observed", "assistant_asserted"):
        assert value in rules


def test_grounding_is_stored_when_the_extractor_supplies_it(tmp_path):
    lever = FIRE_MEMORY_CONSOLIDATION()
    path = tmp_path / "memory_facts.jsonl"
    lever._append_durable_facts(
        path,
        [{"summary": "The office is in Yerevan.", "confidence": 0.9,
          "grounding": "user_stated"}],
        set(), "s1")
    assert _rows(path)[0]["grounding"] == "user_stated"


def test_an_unlabelled_fact_is_marked_unknown_not_trusted(tmp_path):
    """The conservative direction. Guessing "user_stated" for a row the
    extractor did not classify is how an assertion becomes evidence."""
    lever = FIRE_MEMORY_CONSOLIDATION()
    path = tmp_path / "memory_facts.jsonl"
    lever._append_durable_facts(
        path, [{"summary": "Mercury boils at 356.7 C.", "confidence": 0.9}],
        set(), "s1")
    assert _rows(path)[0]["grounding"] == "unknown"


def test_a_value_the_extractor_invented_is_not_stored_verbatim(tmp_path):
    """Only the three known values, so a downstream reader can switch on
    them without a defensive branch per new word the model coins."""
    lever = FIRE_MEMORY_CONSOLIDATION()
    path = tmp_path / "memory_facts.jsonl"
    lever._append_durable_facts(
        path,
        [{"summary": "The sky is blue.", "confidence": 0.9,
          "grounding": "i read it somewhere"}],
        set(), "s1")
    assert _rows(path)[0]["grounding"] == "unknown"


def test_the_pipeline_writer_records_it_too(tmp_path, monkeypatch):
    """Two writers share this store; a field only one of them sets is a
    field nothing can rely on."""
    from backend.consolidation import pipeline

    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    (tmp_path / "knowledge").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(pipeline, "_maybe_rotate_memory_facts", lambda p: None)
    pipeline._append_memory_fact("A durable fact.", "general", 0.9, [],
                                 "2026-09-04", grounding="tool_observed")
    rows = _rows(tmp_path / "knowledge" / "memory_facts.jsonl")
    assert rows[0]["grounding"] == "tool_observed"


# --- and the reader has to be told ------------------------------------


def test_search_carries_grounding_through(tmp_path, monkeypatch):
    """A field the writer sets and the reader drops protects nobody."""
    from backend import fact_search as fs

    k = tmp_path / "knowledge"
    k.mkdir(parents=True, exist_ok=True)
    (k / "memory_facts.jsonl").write_text(
        json.dumps({"summary": "The office is in Yerevan.", "confidence": 0.9,
                    "grounding": "assistant_asserted"}) + "\n",
        encoding="utf-8")
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    fs._STORE = None
    monkeypatch.setattr(fs, "_fact_store_path", lambda: k / "fact_embeddings.json")

    class _E:
        def status(self):
            return {"backend": "fake", "dim": 3, "model": "m"}

        def embed(self, text):
            return [1.0, 0.0, 0.0]

    from unittest.mock import patch
    with patch.object(fs, "EMBEDDER", _E()):
        fs.backfill_fact_embeddings()
        hits = fs.search_facts("office", limit=3, score_floor=0.0)
    fs._STORE = None
    assert hits and hits[0]["grounding"] == "assistant_asserted"


def test_a_row_written_before_grounding_existed_reads_as_unknown(tmp_path,
                                                                 monkeypatch):
    """3654 rows predate the field. They are not evidence and not lies --
    they are unlabelled, and must say so rather than defaulting either
    way."""
    from backend import fact_search as fs

    k = tmp_path / "knowledge"
    k.mkdir(parents=True, exist_ok=True)
    (k / "memory_facts.jsonl").write_text(
        json.dumps({"summary": "An older fact.", "confidence": 0.9}) + "\n",
        encoding="utf-8")
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    fs._STORE = None
    monkeypatch.setattr(fs, "_fact_store_path", lambda: k / "fact_embeddings.json")

    class _E:
        def status(self):
            return {"backend": "fake", "dim": 3, "model": "m"}

        def embed(self, text):
            return [1.0, 0.0, 0.0]

    from unittest.mock import patch
    with patch.object(fs, "EMBEDDER", _E()):
        fs.backfill_fact_embeddings()
        hits = fs.search_facts("older", limit=3, score_floor=0.0)
    fs._STORE = None
    assert hits and hits[0]["grounding"] == "unknown"
