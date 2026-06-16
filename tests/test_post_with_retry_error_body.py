"""post_with_retry must treat an HTTP-200 error body as a transient failure.

OpenRouter (and some OpenAI-compat gateways) can return `200 OK` with an
`{"error": {...}}` body and NO `choices` when the upstream model rate-limits or
errors under burst. The old code returned that body straight to the caller,
which then did `data["choices"][0]` deep in the tool loop and crashed the whole
turn with `KeyError: 'choices'`. The fix: retry with backoff, then raise a
clean LLMError — never hand back a choices-less body.
"""
from __future__ import annotations

import json

import httpx
import pytest

import backend.llm as llm


class _FakeResp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.headers = {}
        self.text = json.dumps(body)
        self.request = httpx.Request("POST", "http://x")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "err", request=self.request, response=self  # type: ignore[arg-type]
            )

    def json(self):
        return self._body


def test_200_error_body_retries_then_raises(monkeypatch):
    calls = {"n": 0}

    def _fake_post(url, json, headers, timeout):
        calls["n"] += 1
        return _FakeResp(200, {"error": {"message": "rate limited", "code": 429}})

    monkeypatch.setattr(llm.httpx, "post", _fake_post)
    monkeypatch.setattr(llm.time, "sleep", lambda *_a, **_k: None)

    with pytest.raises(llm.LLMError):
        llm.post_with_retry(
            "http://x", payload={}, headers={}, provider_name="OpenAI",
            max_retries=2,
        )
    # initial attempt + 2 retries = 3 POSTs
    assert calls["n"] == 3


def test_valid_body_passes_through(monkeypatch):
    def _fake_post(url, json, headers, timeout):
        return _FakeResp(200, {"choices": [{"message": {"content": "hi"}}]})

    monkeypatch.setattr(llm.httpx, "post", _fake_post)
    out = llm.post_with_retry(
        "http://x", payload={}, headers={}, provider_name="OpenAI",
    )
    assert out["choices"][0]["message"]["content"] == "hi"


def test_error_body_recovers_when_retry_succeeds(monkeypatch):
    seq = [
        _FakeResp(200, {"error": {"message": "upstream busy"}}),
        _FakeResp(200, {"choices": [{"message": {"content": "ok"}}]}),
    ]

    def _fake_post(url, json, headers, timeout):
        return seq.pop(0)

    monkeypatch.setattr(llm.httpx, "post", _fake_post)
    monkeypatch.setattr(llm.time, "sleep", lambda *_a, **_k: None)

    out = llm.post_with_retry(
        "http://x", payload={}, headers={}, provider_name="OpenAI",
        max_retries=3,
    )
    assert out["choices"][0]["message"]["content"] == "ok"
