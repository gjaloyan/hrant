import json
from pathlib import Path

from backend.autonomic.immune import ImmuneSignature, SignatureStore


def _write_signatures(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_signature_roundtrip():
    sig = ImmuneSignature(
        id="test_v1",
        pattern={"source": "error_log", "msg_regex": "foo.*bar"},
        severity="warn",
        fix_lever="FIRE_SELF_HEAL",
        fix_params={"service": "x"},
        observed_count=0,
        success_rate=None,
    )
    d = sig.to_dict()
    assert d["id"] == "test_v1"
    assert d["fix_params"] == {"service": "x"}
    restored = ImmuneSignature.from_dict(d)
    assert restored == sig


def test_store_load_parses_jsonl(tmp_path: Path):
    p = tmp_path / "sig.jsonl"
    _write_signatures(p, [
        {"id": "a", "pattern": {"source": "error_log", "msg_regex": "x"}, "severity": "info", "fix_lever": "FIRE_ERROR_TRIAGE", "fix_params": {}, "observed_count": 0, "success_rate": None},
        {"id": "b", "pattern": {"source": "server_health", "msg_regex": "y"}, "severity": "warn", "fix_lever": "FIRE_SERVER_HEALTH", "fix_params": {}, "observed_count": 1, "success_rate": 0.5},
    ])
    store = SignatureStore(p)
    sigs = store.load()
    assert len(sigs) == 2
    assert sigs[0].id == "a"
    assert sigs[1].success_rate == 0.5


def test_store_load_missing_file_returns_empty(tmp_path: Path):
    store = SignatureStore(tmp_path / "nope.jsonl")
    assert store.load() == []


def test_store_load_skips_malformed_lines(tmp_path: Path):
    p = tmp_path / "sig.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "not-json\n"
        + json.dumps({"id": "ok", "pattern": {"source": "error_log", "msg_regex": "z"}, "severity": "info", "fix_lever": "FIRE_ERROR_TRIAGE", "fix_params": {}, "observed_count": 0, "success_rate": None})
        + "\n",
        encoding="utf-8",
    )
    store = SignatureStore(p)
    sigs = store.load()
    assert len(sigs) == 1
    assert sigs[0].id == "ok"


def test_match_returns_signature_when_regex_hits(tmp_path: Path):
    p = tmp_path / "sig.jsonl"
    _write_signatures(p, [
        {"id": "ollama_v1", "pattern": {"source": "error_log", "msg_regex": "ollama.*timeout"}, "severity": "warn", "fix_lever": "FIRE_SERVICE_REPAIR", "fix_params": {"service": "ollama"}, "observed_count": 0, "success_rate": None},
    ])
    store = SignatureStore(p)
    entry = {"source": "error_log", "message": "ollama request timeout after 30s"}
    sig = store.match(entry)
    assert sig is not None
    assert sig.id == "ollama_v1"


def test_match_returns_none_when_source_mismatches(tmp_path: Path):
    p = tmp_path / "sig.jsonl"
    _write_signatures(p, [
        {"id": "s1", "pattern": {"source": "error_log", "msg_regex": ".*"}, "severity": "info", "fix_lever": "FIRE_ERROR_TRIAGE", "fix_params": {}, "observed_count": 0, "success_rate": None},
    ])
    store = SignatureStore(p)
    assert store.match({"source": "other", "message": "anything"}) is None


def test_match_returns_none_when_no_signatures_match(tmp_path: Path):
    p = tmp_path / "sig.jsonl"
    _write_signatures(p, [
        {"id": "s1", "pattern": {"source": "error_log", "msg_regex": "abc"}, "severity": "info", "fix_lever": "FIRE_ERROR_TRIAGE", "fix_params": {}, "observed_count": 0, "success_rate": None},
    ])
    store = SignatureStore(p)
    assert store.match({"source": "error_log", "message": "xyz"}) is None


def test_record_outcome_updates_counts_and_success_rate(tmp_path: Path):
    p = tmp_path / "sig.jsonl"
    _write_signatures(p, [
        {"id": "s1", "pattern": {"source": "error_log", "msg_regex": ".*"}, "severity": "info", "fix_lever": "FIRE_ERROR_TRIAGE", "fix_params": {}, "observed_count": 0, "success_rate": None},
    ])
    store = SignatureStore(p)
    store.record_outcome("s1", success=True)
    store.record_outcome("s1", success=False)
    sigs = store.load()
    assert sigs[0].observed_count == 2
    assert sigs[0].success_rate == 0.5


def test_record_outcome_for_unknown_id_is_noop(tmp_path: Path):
    p = tmp_path / "sig.jsonl"
    _write_signatures(p, [])
    store = SignatureStore(p)
    store.record_outcome("does_not_exist", success=True)
    assert store.load() == []
