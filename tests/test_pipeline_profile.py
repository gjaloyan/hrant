"""PipelineProfile dataclass + file-backed store (CRUD + history).

Each profile is one JSON file under <data_dir>/pipeline_profiles/.
History snapshots live alongside under _history/<id>/<unix_ms>.json,
last 10 kept per profile."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Each test gets its own profiles root."""
    from backend import pipeline_profile as _pp
    monkeypatch.setattr(_pp, "_profiles_root", lambda: tmp_path)
    _pp.PROFILES._cache_loaded_at = 0.0
    yield tmp_path


def test_profile_dataclass_roundtrip():
    from backend.pipeline_profile import PipelineProfile
    p = PipelineProfile(
        id="benchmark",
        name="Benchmark Mode",
        description="desc",
        created_at=100.0,
        updated_at=200.0,
        engine_overrides={"router": {"tool_loop_input_budget": 80000}},
        reasoning_overrides={"routing": {"complex_solving": "medium"}},
        prompt_overrides={"sections": {"iteration_ceiling": "x"}},
        logging_overrides={"root": "INFO"},
    )
    d = p.to_dict()
    p2 = PipelineProfile.from_dict(d)
    assert p2 == p


def test_put_and_get(isolated_store):
    from backend.pipeline_profile import PROFILES, PipelineProfile
    p = PipelineProfile(
        id="x", name="X", description="",
        created_at=time.time(), updated_at=time.time(),
        engine_overrides={}, reasoning_overrides={},
        prompt_overrides={}, logging_overrides={},
    )
    PROFILES.put(p)
    loaded = PROFILES.get("x")
    assert loaded is not None
    assert loaded.id == "x"
    assert loaded.name == "X"


def test_list_excludes_history_and_active(isolated_store):
    """Force the existence of _active.json AND a _history/ dir, then
    confirm list() doesn't surface them as profiles."""
    from backend.pipeline_profile import PROFILES, PipelineProfile
    for pid in ("a", "b", "c"):
        PROFILES.put(PipelineProfile(
            id=pid, name=pid.upper(), description="",
            created_at=time.time(), updated_at=time.time(),
            engine_overrides={}, reasoning_overrides={},
            prompt_overrides={}, logging_overrides={},
        ))
    # Force _active.json to exist.
    PROFILES.set_active("a")
    # Force _history/ dir to exist by overwriting one of the profiles.
    overwritten = PROFILES.get("a")
    assert overwritten is not None
    overwritten.name = "A2"
    overwritten.updated_at = time.time() + 1
    PROFILES.put(overwritten)
    ids = {p.id for p in PROFILES.list()}
    assert ids == {"a", "b", "c"}


def test_put_existing_writes_history_snapshot(isolated_store):
    from backend.pipeline_profile import PROFILES, PipelineProfile
    p1 = PipelineProfile(
        id="x", name="v1", description="",
        created_at=time.time(), updated_at=time.time(),
        engine_overrides={}, reasoning_overrides={},
        prompt_overrides={}, logging_overrides={},
    )
    PROFILES.put(p1)
    # Modify and put again — the first version must be snapshotted.
    p1.name = "v2"
    p1.updated_at = time.time() + 1
    PROFILES.put(p1)
    history = PROFILES.history("x")
    assert len(history) == 1
    assert history[0].name == "v1"


def test_history_capped_at_ten(isolated_store):
    from backend.pipeline_profile import PROFILES, PipelineProfile
    p = PipelineProfile(
        id="x", name="0", description="",
        created_at=time.time(), updated_at=time.time(),
        engine_overrides={}, reasoning_overrides={},
        prompt_overrides={}, logging_overrides={},
    )
    PROFILES.put(p)
    for i in range(1, 13):
        p.name = str(i)
        p.updated_at = time.time() + i
        PROFILES.put(p)
    history = PROFILES.history("x")
    assert len(history) == 10
    # Newest snapshot should be the version BEFORE the current —
    # i.e. name "11" (current is "12").
    assert history[0].name == "11"


def test_delete_removes_file_and_history(isolated_store):
    from backend.pipeline_profile import PROFILES, PipelineProfile
    p = PipelineProfile(
        id="x", name="X", description="",
        created_at=time.time(), updated_at=time.time(),
        engine_overrides={}, reasoning_overrides={},
        prompt_overrides={}, logging_overrides={},
    )
    PROFILES.put(p)
    p.name = "X2"
    PROFILES.put(p)  # produces one history snapshot
    assert PROFILES.history("x") != []
    PROFILES.delete("x")
    assert PROFILES.get("x") is None
    assert PROFILES.history("x") == []


def test_active_id_read_default_when_unset(isolated_store):
    from backend.pipeline_profile import PROFILES
    # Fresh store — _active.json absent → falls back to "default".
    assert PROFILES.active_id() == "default"


def test_active_id_round_trip(isolated_store):
    from backend.pipeline_profile import PROFILES, PipelineProfile
    PROFILES.put(PipelineProfile(
        id="bench", name="B", description="",
        created_at=time.time(), updated_at=time.time(),
        engine_overrides={}, reasoning_overrides={},
        prompt_overrides={}, logging_overrides={},
    ))
    PROFILES.set_active("bench")
    assert PROFILES.active_id() == "bench"


def test_invalid_id_chars_refused():
    from backend.pipeline_profile import validate_id
    assert validate_id("benchmark")
    assert validate_id("safe-mode_2")
    assert not validate_id("../etc/passwd")
    assert not validate_id("with space")
    assert not validate_id("")
    assert not validate_id("x" * 33)  # 32 char cap


def test_put_leaves_no_tmp_file_after_success(isolated_store):
    from backend.pipeline_profile import PROFILES, PipelineProfile
    PROFILES.put(PipelineProfile(
        id="x", name="X", description="",
        created_at=time.time(), updated_at=time.time(),
        engine_overrides={}, reasoning_overrides={},
        prompt_overrides={}, logging_overrides={},
    ))
    root = Path(isolated_store)
    leftovers = list(root.glob("**/*.tmp"))
    assert leftovers == []


def test_list_skips_corrupted_json(isolated_store):
    """A malformed JSON file in the profiles dir must not crash list();
    the bad entry is silently skipped."""
    from backend.pipeline_profile import PROFILES, PipelineProfile
    PROFILES.put(PipelineProfile(
        id="good", name="OK", description="",
        created_at=time.time(), updated_at=time.time(),
        engine_overrides={}, reasoning_overrides={},
        prompt_overrides={}, logging_overrides={},
    ))
    bad = isolated_store / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    ids = {p.id for p in PROFILES.list()}
    assert ids == {"good"}


def test_history_and_restore_reject_invalid_id(isolated_store):
    """Defense-in-depth — these methods used to accept any string."""
    from backend.pipeline_profile import PROFILES
    assert PROFILES.history("../etc/passwd") == []
    assert PROFILES.restore("../foo", 12345) is None
