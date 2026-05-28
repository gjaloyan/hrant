"""Round 7 — five token-usage optimisations the agent's review caught:

  #4 P0  Self-critic retry now bails when the running request has
          already burned through `critic_retry_token_budget`. The
          guard list claimed this existed; the code was missing it.

  #3 P1  Verifier compresses tool_context to snippets around
          identifiers the answer mentions. 60-80k accumulated
          retry+subtask buffers were going into the verifier prompt
          unfiltered.

  #1 P1  _think uses compact capabilities + narrower n_conv / n_memory
          for non-self-analysis tasks. The full source map only
          matters for self-analysis where the model is going to
          read its own code.

  #2 P2  `_capabilities_block(compact=True)` is the new mode that
          drops the source map and trims tool descriptions, used
          by _think and any future caller that doesn't need the
          "where am I implemented" view.

  #5 P2  Micro-ack fast path: "ok" / "thanks" / "continue" / "понял"
          return a static "✓" without any LLM call. Plus memory
          extraction gate: short chat turns (< 30 chars) skip the
          extractor LLM call.
"""
from __future__ import annotations
from unittest.mock import patch

from backend.agent import (
    Agent,
    _capabilities_block,
)
from backend.verifier import _compress_tool_context


# --- #1 / #2: compact capabilities ----------------------------------------


def test_capabilities_compact_smaller_than_full():
    full = _capabilities_block(compact=False)
    short = _capabilities_block(compact=True)
    assert len(short) < len(full)
    # Source map presence is the canonical "is this the full block"
    # marker — compact strips it.
    assert "source map" not in short.lower()
    assert "source map" in full.lower()


def test_capabilities_compact_keeps_tool_names():
    """Compact still has to list tools — the model needs to know
    which ones it can call. Just trims descriptions and source map."""
    short = _capabilities_block(compact=True)
    assert "## Tools" in short
    # Common builtin tool names must be present
    assert "read_file" in short
    assert "calc" in short or "calc" in short.lower()


# --- #3: verifier tool_context compression --------------------------------


def test_compress_keeps_short_context_unchanged():
    """Below the keep-full threshold the body returns as-is —
    compression overhead isn't worth it for small inputs."""
    short_ctx = "tiny\n" * 20  # ~100 chars
    out = _compress_tool_context("any answer", short_ctx)
    assert out == short_ctx


def test_compress_keeps_only_snippets_around_mentioned_idents():
    """The answer mentions `TokenTracker`. The compressed body
    should keep lines around that identifier and drop the rest."""
    big_body = []
    for i in range(1000):
        big_body.append(f"line {i:04d} unrelated content " + "x" * 80)
    big_body[500] = "class TokenTracker:"
    big_body[501] = "    def record(self, x): pass"
    tool_context = "\n".join(big_body)
    answer = "I'll add a TokenTracker class to llm.py."

    out = _compress_tool_context(answer, tool_context)
    assert "class TokenTracker:" in out
    # And the bulk of unrelated lines should be dropped.
    assert len(out) < len(tool_context) // 5
    # Gap markers stitch the kept slices.
    assert "…" in out


def test_compress_falls_back_when_no_overlap():
    """Answer mentions nothing the tool output references —
    keep a head slice so verifier still has something to work with."""
    big_body = "\n".join(f"line {i}" for i in range(2000))
    answer = "The user asked about iodine in salt."
    out = _compress_tool_context(answer, big_body, max_chars=500)
    assert len(out) <= 500


# --- #4: retry token budget -----------------------------------------------


def test_critic_retry_token_budget_in_config():
    """New config key exists with a sensible default."""
    from backend.config import CONFIG
    val = CONFIG.verification.get("critic_retry_token_budget")
    assert isinstance(val, int)
    assert val > 0
    # Sanity: a single large turn must STILL be allowed, otherwise
    # we'd never retry. 10k is way too low; 1M is way too high.
    assert 10_000 <= val <= 1_000_000
