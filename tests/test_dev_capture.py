"""Dev-mode capture: redact file blobs out of LLM prompts so the
WebUI dev panel shows STRUCTURE without dumping kilobytes of file
content. Persistence to dev/<ts>_<id>.json verified end-to-end.
"""
from __future__ import annotations
import json

from backend.dev_capture import (
    redact_prompt,
    save_dev_capture,
    new_request_id,
    DEV_DIR,
)


def test_redact_replaces_soul_body_with_marker():
    prompt = (
        "# SOUL\n"
        "I am a warm and direct assistant.\n"
        "I keep replies short by default.\n"
        "\n"
        "# IDENTITY\n"
        "I am Hrant.\n"
    )
    out = redact_prompt(prompt)
    assert "warm and direct" not in out
    assert "[file: knowledge/identity/soul.md," in out
    # IDENTITY body also redacted, not leaked.
    assert "I am Hrant." not in out
    assert "[file: knowledge/identity/identity.md," in out


def test_redact_keeps_user_request_visible():
    """The actual USER REQUEST and arbitrary text we don't have a
    section header for must survive intact — that's the part the
    operator wants to see."""
    prompt = (
        "# CORE MEMORY\n"
        "secret core fact line\n"
        "\n"
        "# USER REQUEST\n"
        "what is iodine in salt?\n"
    )
    out = redact_prompt(prompt)
    assert "[file: knowledge/core_memory.md," in out
    assert "secret core fact line" not in out
    assert "# USER REQUEST" in out
    assert "what is iodine in salt?" in out


def test_redact_handles_notes_and_tool_outputs():
    prompt = (
        "# NOTES\n"
        "## Topic A\nsome A content\n"
        "## Topic B\nsome B content\n"
        "\n"
        "TOOL OUTPUTS (file contents, search results — primary evidence):\n"
        "[read_file] huge dump...\n"
    )
    out = redact_prompt(prompt)
    assert "some A content" not in out
    assert "[read_file] huge dump" not in out
    assert "[file: knowledge/profession/* (loaded notes)," in out


def test_redact_passes_through_when_no_known_sections():
    """Plain prompt with no recognised headers — return as-is."""
    s = "just a plain user message with no markdown headers."
    assert redact_prompt(s) == s


def test_redact_caps_pathological_input():
    """Pathological 200k input with no known sections must be capped."""
    big = "filler " * 50000  # ~350k chars
    out = redact_prompt(big, hard_cap_chars=8000)
    assert len(out) < 9000
    assert "truncated by dev-mode cap" in out


def test_save_dev_capture_writes_redacted_payload(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.dev_capture.DEV_DIR", tmp_path)
    rid = new_request_id()
    path = save_dev_capture(
        request_id=rid,
        question="what is iodine in salt?",
        llm_calls=[
            {
                "label": "_solve",
                "task_type": "complex_solving",
                "model": "claude-sonnet-4-5",
                "system_redacted": "# SOUL\n[file: ...]",
                "user_redacted": "# USER REQUEST\nask...",
                "response_preview": "Yes, you can test...",
                "duration_ms": 1234,
                "input_tokens": 500,
                "output_tokens": 200,
            },
        ],
        answer_preview="Yes, you can test...",
        confidence=82,
    )
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["request_id"] == rid
    assert payload["confidence"] == 82
    assert len(payload["llm_calls"]) == 1
    assert payload["llm_calls"][0]["label"] == "_solve"
    assert "[file:" in payload["llm_calls"][0]["system_redacted"]


def test_save_dev_capture_handles_disk_failure(monkeypatch):
    """If the dev/ folder can't be written, save_dev_capture returns
    None instead of raising — capture is best-effort."""
    monkeypatch.setattr("backend.dev_capture._ensure_dev_dir",
                        lambda: (_ for _ in ()).throw(OSError("read-only fs")))
    out = save_dev_capture(
        request_id="abc",
        question="q",
        llm_calls=[],
        answer_preview="",
        confidence=0,
    )
    assert out is None
