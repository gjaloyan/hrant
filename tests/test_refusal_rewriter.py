"""Tests for TSP-2 — refusal-without-attempts rewriter.

2026-05-21: the rewriter + its keyword regexes were removed when
the user asked to drop all keyword logic from the agent pipeline
("remove keyword logic fully from agent pipeline"). The whole
test module is skipped at collection time. The TSP attempt-bar
rule still lives in the system prompt — the LLM enforces it
itself.
"""
from __future__ import annotations

import pytest

pytest.skip(
    "2026-05-21: refusal rewriter + its keyword regexes removed "
    "from the pipeline; whole module skipped at collection.",
    allow_module_level=True,
)

# Original imports preserved as documentation for reverter-friendly
# reading; the symbols no longer exist in backend.unified_agent.
# from backend.unified_agent import (
#     _REFUSAL_OPENER_RE,
#     _is_policy_refusal,
#     _is_russian_dominant,
#     _count_distinct_tools_called,
#     _rewrite_refusal_without_attempts,
#     REFUSAL_ATTEMPT_BAR,
# )


# ─── regex coverage ─────────────────────────────────────────────────


@pytest.mark.parametrize("opener", [
    "Я не могу выполнить этот запрос",
    "Gor, я не могу выполнить этот запрос и вернуть результат.",
    "я не могу здесь реально 'услышать'",
    "у меня нет доступа к этому файлу",
    "среди доступных мне сейчас инструментов нет Telegram send_voice",
    "среди доступных инструментов нет такого",
    "Невозможно выполнить эту задачу",
    "К сожалению, я не могу обработать это",
    "I cannot remove the logo because I don't have video tools",
    "I can't do this without ffmpeg",
    "I'm not able to perform this task",
    "I don't have access to your filesystem",
    "I don't have a tool for that",
    "Tools are not available for this task",
    "tools are disabled",
    "This isn't supported",
    "Unable to do this without your input",
    "Sorry, I can't process this file type",
])
def test_refusal_regex_matches_known_openers(opener):
    assert _REFUSAL_OPENER_RE.search(opener), (
        f"regex should match refusal opener {opener!r}"
    )


@pytest.mark.parametrize("non_opener", [
    "Готово — логотип убран. MEDIA:/tmp/x.mp4",
    "I processed the video and saved it to outbox/clip.mp4",
    "Sure, here is the answer: 42",
    "Done. Set tts.voice to ru-Svetlana.",
    "Я применил настройки и проверил результат.",
    "I tried `read_file` and `analyze_image` — both succeeded.",
])
def test_refusal_regex_ignores_normal_answers(non_opener):
    assert not _REFUSAL_OPENER_RE.search(non_opener[:300]), (
        f"regex must NOT match normal answer {non_opener!r}"
    )


# ─── _is_russian_dominant ───────────────────────────────────────────


def test_russian_dominant_detects_russian():
    assert _is_russian_dominant("Привет, я не могу выполнить") is True


def test_russian_dominant_detects_english():
    assert _is_russian_dominant("Hello, I cannot do this") is False


def test_russian_dominant_mixed_with_more_russian():
    """Mixed text — clearly more cyrillic than latin → Russian."""
    txt = "Я пытался запустить FFmpeg через terminal_exec но команда упала"
    assert _is_russian_dominant(txt) is True


def test_russian_dominant_empty_string():
    assert _is_russian_dominant("") is False


# ─── _count_distinct_tools_called ───────────────────────────────────


def _mk_step(event, tool_name):
    """Build a fake trace step matching the agent._trace shape."""
    tc = SimpleNamespace(name=tool_name) if tool_name else None
    return SimpleNamespace(event=event, tool_call=tc)


def test_count_distinct_tools_zero():
    agent = SimpleNamespace(_trace=[])
    n, names = _count_distinct_tools_called(agent)
    assert n == 0
    assert names == set()


def test_count_distinct_tools_one():
    agent = SimpleNamespace(_trace=[
        _mk_step("tool", "read_file"),
    ])
    n, names = _count_distinct_tools_called(agent)
    assert n == 1
    assert names == {"read_file"}


def test_count_distinct_tools_dedupes_same_name():
    """Three calls to `read_file` count as ONE distinct tool — the
    bar is variety of attempts, not raw count."""
    agent = SimpleNamespace(_trace=[
        _mk_step("tool", "read_file"),
        _mk_step("tool", "read_file"),
        _mk_step("tool", "read_file"),
    ])
    n, names = _count_distinct_tools_called(agent)
    assert n == 1


def test_count_distinct_tools_counts_errors_too():
    """A tool that errored still counts as an attempt — the user
    asked for execution, the agent gave it a shot."""
    agent = SimpleNamespace(_trace=[
        _mk_step("tool_error", "terminal_exec"),
        _mk_step("tool", "read_file"),
    ])
    n, names = _count_distinct_tools_called(agent)
    assert n == 2
    assert names == {"terminal_exec", "read_file"}


def test_count_distinct_tools_handles_dict_tool_call():
    """tool_call can be a dict OR a SimpleNamespace — the rewriter
    must handle both."""
    agent = SimpleNamespace(_trace=[
        SimpleNamespace(event="tool", tool_call={"name": "fetch_url"}),
        SimpleNamespace(event="tool", tool_call={"name": "web_search"}),
    ])
    n, names = _count_distinct_tools_called(agent)
    assert n == 2
    assert names == {"fetch_url", "web_search"}


# ─── rewriter trigger conditions ────────────────────────────────────


def test_rewriter_fires_on_refusal_with_zero_tools():
    answer = "Gor, я не могу выполнить этот запрос. Это требует CAPTCHA."
    agent = SimpleNamespace(_trace=[])
    out = _rewrite_refusal_without_attempts(answer, agent)
    assert "Task Solver Process" in out
    assert "Перезапускаю" in out or "Resetting" in out


def test_rewriter_fires_on_refusal_with_one_tool():
    answer = "Я не могу обработать это видео — нет ffmpeg."
    agent = SimpleNamespace(_trace=[
        _mk_step("tool", "read_file"),
    ])
    out = _rewrite_refusal_without_attempts(answer, agent)
    assert out != answer
    assert "Task Solver Process" in out


def test_rewriter_skips_refusal_with_two_or_more_tools():
    """2+ distinct tools = TSP-compliant honest refusal, leave alone."""
    answer = "Я не могу обработать это — попробовал ffmpeg и pdftotext."
    agent = SimpleNamespace(_trace=[
        _mk_step("tool", "terminal_exec"),
        _mk_step("tool", "read_file"),
    ])
    out = _rewrite_refusal_without_attempts(answer, agent)
    assert out == answer


def test_rewriter_skips_non_refusal_answer():
    answer = "Готово — обработал видео, результат в outbox/."
    agent = SimpleNamespace(_trace=[])
    out = _rewrite_refusal_without_attempts(answer, agent)
    assert out == answer


def test_rewriter_handles_empty_answer():
    agent = SimpleNamespace(_trace=[])
    assert _rewrite_refusal_without_attempts("", agent) == ""
    assert _rewrite_refusal_without_attempts(None, agent) == ""


def test_rewriter_handles_nonstring_answer():
    agent = SimpleNamespace(_trace=[])
    assert _rewrite_refusal_without_attempts(42, agent) == ""


# ─── rewrite output shape ───────────────────────────────────────────


def test_rewrite_lists_tools_actually_called():
    answer = "Я не могу выполнить."
    agent = SimpleNamespace(_trace=[
        _mk_step("tool", "search_knowledge"),
    ])
    out = _rewrite_refusal_without_attempts(answer, agent)
    assert "search_knowledge" in out


def test_rewrite_shows_zero_tools_as_none():
    answer = "Я не могу выполнить."
    agent = SimpleNamespace(_trace=[])
    out = _rewrite_refusal_without_attempts(answer, agent)
    # "(none)" or "—" or some explicit zero-tool marker.
    assert "(none)" in out or "—" in out or "0 " in out


def test_rewrite_quotes_original_refusal_excerpt():
    """The user (or owner reading the bridge log) should see what
    the agent originally tried to say, to help diagnose."""
    answer = "Я не могу выполнить запрос — у меня нет нужного тула."
    agent = SimpleNamespace(_trace=[])
    out = _rewrite_refusal_without_attempts(answer, agent)
    assert "Я не могу выполнить запрос" in out


def test_rewrite_includes_recovery_options():
    """The rewrite tells the user how to unblock — repeat / explain /
    name blocker. Without these the rewrite is just blame."""
    answer = "Я не могу выполнить."
    agent = SimpleNamespace(_trace=[])
    out = _rewrite_refusal_without_attempts(answer, agent)
    # At least 2 of the 3 recovery hints should appear.
    hits = sum([
        ("повтори" in out.lower() or "repeat" in out.lower()),
        ("посоветуй" in out.lower() or "объясни" in out.lower() or "advise" in out.lower() or "explain" in out.lower()),
        ("блокир" in out.lower() or "blocker" in out.lower()),
    ])
    assert hits >= 2


def test_rewrite_picks_russian_for_russian_answer():
    answer = "Я не могу выполнить — нужен ffmpeg, а его нет."
    agent = SimpleNamespace(_trace=[])
    out = _rewrite_refusal_without_attempts(answer, agent)
    # Russian marker.
    assert "Перезапускаю" in out or "TSP" in out
    # Should NOT switch to English wholesale.
    assert "Resetting" not in out


def test_rewrite_picks_english_for_english_answer():
    answer = "I cannot do this without the ffmpeg binary."
    agent = SimpleNamespace(_trace=[])
    out = _rewrite_refusal_without_attempts(answer, agent)
    assert "Resetting" in out
    assert "Перезапускаю" not in out


# ─── REFUSAL_ATTEMPT_BAR constant ───────────────────────────────────


def test_attempt_bar_is_two():
    """Pin the bar at 2 distinct tools. Lowering this to 1 would
    re-admit "called read_file then refused" as acceptable — that's
    the failure mode we're closing."""
    assert REFUSAL_ATTEMPT_BAR == 2


# ─── C1: policy refusals must NOT be rewritten ──────────────────────


@pytest.mark.parametrize("policy_refusal", [
    # Real prod samples (workspace/turns/) — these would have been
    # rewritten before C1 was added, which would have been a lie.
    "Я не могу показать файл `user.md` гостевому пользователю: "
    "доступ к локальным файлам вне публичного workspace я могу давать "
    "только trusted-пользователям.",

    "Alice, I can't help find or provide a private person's phone "
    "number.\n\nI don't have verified public information about Mike's "
    "current number.",

    "I don't have verified information about a 'Vorondesh protocol' "
    "in my memory or notes.\n\nAlso, the stated invention year "
    "(2031) is in the future.",

    # Other policy / privacy variants
    "Я не могу поделиться личными данными третьих лиц.",
    "I can't share personal information about that user.",
    "Я не могу — это работа с конфиденциальной информацией.",
    "Sorry, I can't help with stalking or harassment.",
])
def test_policy_refusals_not_rewritten(policy_refusal):
    """The rewriter MUST leave policy / privacy / recall refusals
    alone — they're honest non-capability refusals, not TSP violations.
    Rewriting them would convert legitimate stances into lies."""
    from types import SimpleNamespace
    agent = SimpleNamespace(_trace=[])
    out = _rewrite_refusal_without_attempts(policy_refusal, agent)
    assert out == policy_refusal, (
        f"policy refusal was wrongly rewritten:\n"
        f"  in:  {policy_refusal[:120]!r}\n"
        f"  out: {out[:120]!r}"
    )


@pytest.mark.parametrize("capability_refusal", [
    # These DO indicate capability gaps and should still be rewritten
    # when <2 tools were called.
    "Я не могу обработать это видео — нет ffmpeg на сервере.",
    "У меня нет инструмента для конвертации DWG файлов.",
    "Среди доступных мне сейчас инструментов нет Telegram send_voice.",
    "I can't do this without the ffmpeg binary.",
    "Tools are not available for this conversion.",
    "Sorry, I can't process this file type.",
])
def test_capability_refusals_still_rewritten(capability_refusal):
    """The exclude must NOT swallow legitimate capability refusals —
    those are the whole point of the rewriter."""
    from types import SimpleNamespace
    agent = SimpleNamespace(_trace=[])
    out = _rewrite_refusal_without_attempts(capability_refusal, agent)
    assert out != capability_refusal
    assert "Task Solver Process" in out


def test_is_policy_refusal_directly():
    """The helper is the single source of truth for the exclude."""
    # Policy markers (Russian)
    assert _is_policy_refusal("это приватные данные") is True
    assert _is_policy_refusal("конфиденциальная информация") is True
    assert _is_policy_refusal("гостю не положено") is True
    assert _is_policy_refusal("чужой телефон я не могу") is True
    # Policy markers (English)
    assert _is_policy_refusal("private phone number") is True
    assert _is_policy_refusal("personal data") is True
    assert _is_policy_refusal("someone's email") is True
    assert _is_policy_refusal("third-party data") is True
    assert _is_policy_refusal("confidential") is True
    # Capability — NOT policy
    assert _is_policy_refusal("no ffmpeg installed") is False
    assert _is_policy_refusal("у меня нет тула") is False
    assert _is_policy_refusal("") is False


# ─── M1: language-detection tie-breaking goes to Russian ───────────


def test_russian_dominant_resolves_tie_to_russian():
    """Equal-count text → Russian. Pure symbols → Russian. This is
    the M1 fix: Hrant's owner is Russian-speaking, so a tie or
    symbol-heavy answer should pick the Russian rewrite, not the
    English one."""
    # Exactly tied (5 cyr / 5 lat)
    assert _is_russian_dominant("aaabb привет") is True
    # Pure symbols / digits — no letters at all
    assert _is_russian_dominant("🚀 100% ✓ !!!") is True
    assert _is_russian_dominant("12345") is True
    # Russian dominant (unchanged)
    assert _is_russian_dominant("совсем по-русски") is True
    # English dominant (unchanged)
    assert _is_russian_dominant("entirely in english here") is False
