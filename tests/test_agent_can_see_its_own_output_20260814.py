"""The agent must be able to look at the images it produces.

Measured 2026-08-13/14, two consecutive DataLex turns:

    73 tool calls, 126 browser actions, $4.39 ->
      "проверил оба файла скриншотов: они существуют, но доступная проверка
       не извлекает из PNG текст или содержимое экрана"

    122 tool calls, 158 browser actions, $6.78 -> another ask_user

`analyze_image` required a `sha256` from the AttachmentStore, and the store
only ever held what the USER uploaded. So a screenshot the agent took, a
CAPTCHA it saved, a page it rendered — all invisible to it. It had a
multimodal model and no way to point it at its own output, and said so
honestly rather than inventing a reading.

The cost was not one turn. The whole local-OCR project — seven retry rounds
building Graf-J to read CAPTCHAs — existed because the vision model it
already had could not be aimed at the CAPTCHA file.
"""
import json
import tempfile
from pathlib import Path

import pytest

from backend.builtin_tools import _analyze_image_handler, _ingest_image_path

_PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00"
        b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")


@pytest.fixture
def shot(tmp_path) -> Path:
    p = tmp_path / "datalex_captcha.png"
    p.write_bytes(_PNG)
    return p


def test_a_screenshot_the_agent_took_can_be_ingested(shot):
    sha, err = _ingest_image_path(str(shot))
    assert not err
    assert len(sha) == 64


def test_the_ingested_image_is_in_the_attachment_store(shot):
    from backend.attachments import ATTACHMENTS
    sha, err = _ingest_image_path(str(shot))
    assert not err
    meta = ATTACHMENTS.get_meta(sha)
    assert meta is not None and meta.kind == "image"


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".jpeg", ".webp"])
def test_the_usual_screenshot_formats_are_accepted(tmp_path, suffix):
    p = tmp_path / f"shot{suffix}"
    p.write_bytes(_PNG)
    sha, err = _ingest_image_path(str(p))
    assert not err, err


def test_a_missing_file_says_so_plainly(tmp_path):
    sha, err = _ingest_image_path(str(tmp_path / "nope.png"))
    assert not sha and "no such file" in err


def test_a_non_image_is_refused_with_the_accepted_types(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello")
    sha, err = _ingest_image_path(str(p))
    assert not sha
    assert "not a recognised image type" in err and ".png" in err


def test_ingestion_never_raises_on_a_directory(tmp_path):
    sha, err = _ingest_image_path(str(tmp_path))
    assert not sha and err


# ── the tool contract ───────────────────────────────────────────────

def test_sha256_is_no_longer_mandatory():
    """Requiring it is exactly what made the agent blind to its own work."""
    from backend.tool_registry import get_registry
    schema = next(t for t in get_registry().to_anthropic_list()
                  if t["name"] == "analyze_image")["input_schema"]
    assert schema["required"] == ["question"]
    assert "path" in schema["properties"]


def test_a_call_with_neither_image_explains_both_options():
    out = json.loads(_analyze_image_handler(question="what is here?"))
    assert out["ok"] is False
    assert "path" in out["error"] and "sha256" in out["error"]


def test_a_question_is_still_required(shot):
    out = json.loads(_analyze_image_handler(path=str(shot), question=""))
    assert out["ok"] is False and "question" in out["error"]


def test_a_bad_path_reports_the_path_back(tmp_path):
    out = json.loads(_analyze_image_handler(
        path=str(tmp_path / "gone.png"), question="what?"))
    assert out["ok"] is False
    assert out["path"].endswith("gone.png")


def test_the_description_tells_it_not_to_reach_for_ocr():
    """It built a local OCR model for CAPTCHAs its own vision model could
    read. The description now says so before that happens again."""
    from backend.tool_registry import get_registry
    desc = next(t for t in get_registry().to_anthropic_list()
                if t["name"] == "analyze_image")["description"]
    assert "screenshot you just took" in desc
    assert "OCR" in desc
    assert "unavailable to you" in desc


def test_an_explicit_sha_still_wins_over_path(shot, monkeypatch):
    """Back-compat: callers passing a user attachment are unaffected."""
    import backend.builtin_tools as bt
    seen = {}
    monkeypatch.setattr(bt, "_analyze_image",
                        lambda sha, q: seen.setdefault("sha", sha) or "answer")
    _analyze_image_handler(sha256="a" * 64, path=str(shot), question="q")
    assert seen["sha"] == "a" * 64


# ── one classification, not two ─────────────────────────────────────

def test_delivery_is_declared_once_in_the_typed_model():
    """`endpoint_check` kept its own frozensets until 2026-08-14, and they had
    already drifted from the typed model on agent_browser, ask_user and
    sandbox_exec. Both answers now come from `tool_registry`."""
    from backend.endpoint_check import _DELIVERY_TOOLS
    from backend.tool_registry import proves_delivery
    for name in ("set_setting", "delegate", "start_background_job",
                 "agent_browser", "ask_user", "sandbox_exec", "read_file"):
        assert (name in _DELIVERY_TOOLS) is proves_delivery(name), name


def test_advances_and_delivery_are_different_questions():
    """The trap this whole change guards. `agent_browser` DOES act on the
    world — it drives a real browser, so the drift marker must count it — and
    it proves nothing: 32 calls once ended a turn with no data and the gate
    passed it. Anything reading `advances` as "delivered" reintroduces that."""
    from backend.tool_registry import get_registry, proves_delivery
    reg = get_registry()
    for name in ("agent_browser", "ask_user", "sandbox_exec"):
        sem = reg.resolve_call_semantics(name)
        assert sem.advances is True, name
        assert proves_delivery(name) is False, name


def test_the_instrument_set_is_derived_not_listed():
    """EXTERNAL minus delivery. Writing the names by hand is what drifted."""
    from backend.endpoint_check import _DELIVERY_TOOLS, _INSTRUMENT_TOOLS
    from backend.tool_registry import (
        ToolEffect, default_semantics_for_name, get_registry,
    )
    for name in get_registry().names():
        sem = default_semantics_for_name(name)
        expected = sem.effect is ToolEffect.EXTERNAL and not sem.proves_delivery
        assert (name in _INSTRUMENT_TOOLS) is expected, name
        assert not (name in _DELIVERY_TOOLS and name in _INSTRUMENT_TOOLS), name


def test_a_new_tool_defaults_to_proving_nothing():
    """Fail closed, matching how UNKNOWN already behaves in audit mode: an
    unregistered or undeclared tool must never satisfy the completion gate."""
    from backend.tool_registry import proves_delivery
    assert proves_delivery("some_future_mcp_tool") is False
