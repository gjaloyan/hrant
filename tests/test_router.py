"""Тесты DualModelRouter: маршрутизация, fallback, budget, shift_schedule."""
from unittest.mock import patch

import pytest

from backend.llm import DualModelRouter, LLMError, TaskType


@pytest.fixture
def router(tmp_kb, monkeypatch):
    """Свежий роутер с замоканными провайдерами и env-ключом."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    r = DualModelRouter(state_path=tmp_kb.base / "router_state.json")

    # In claude_only mode CONFIG.model_b is None, which makes _ollama_available()
    # hardcode False and the model_b property raise. Tests want both providers
    # available — install a stub cfg_b so cached availability + injected fakes win.
    r.cfg_b = {"base_url": "http://localhost:11434", "model": "fake-qwen-test"}

    # По умолчанию детерминированное поведение: без shift, с fallback, с высоким бюджетом
    r.cfg_router = dict(r.cfg_router)
    r.cfg_router["auto_shift_after_finetune"] = False
    r.cfg_router["daily_api_budget_usd"] = 1000.0
    r.cfg_router["fallback_to_local"] = True

    class FakeA:
        def __init__(self):
            self.calls = 0
            self.fail = False

        def complete(self, system, user, **kw):
            self.calls += 1
            if self.fail:
                raise LLMError("A down")
            return "A:" + user[:20]

    class FakeB:
        def __init__(self):
            self.calls = 0

        def complete(self, system, user, **kw):
            self.calls += 1
            return "B:" + user[:20]

    fa, fb = FakeA(), FakeB()
    r._model_a = fa  # type: ignore
    r._model_b = fb  # type: ignore
    # По умолчанию обе модели "доступны"
    r._api_cache = (9e18, True)
    r._ollama_cache = (9e18, True)
    return r, fa, fb


# ---------- базовая маршрутизация ----------
def test_a_task_goes_to_a(router):
    r, fa, fb = router
    out = r.call(TaskType.COMPLEX_SOLVING, "sys", "hello")
    assert out.startswith("A:")
    assert fa.calls == 1
    assert fb.calls == 0


def test_b_task_goes_to_b(router):
    r, fa, fb = router
    out = r.call(TaskType.SIMPLE_LOOKUP, "sys", "hello")
    assert out.startswith("B:")
    assert fb.calls == 1
    assert fa.calls == 0


def test_verification_hard_forced_to_a(router):
    r, fa, fb = router
    # Даже если auto_shift агрессивен — verification всегда A
    r.cfg_router["auto_shift_after_finetune"] = True
    r.cfg_router["shift_schedule"] = {"v0": {"model_a_pct": 0, "model_b_pct": 100}}
    r.cfg_verification["always_use_model_a"] = True
    out = r.call(TaskType.VERIFICATION, "sys", "check")
    assert out.startswith("A:")
    assert fa.calls == 1


# ---------- fallback ----------
def test_api_down_falls_back_to_b(router):
    r, fa, fb = router
    r._api_cache = (9e18, False)  # API недоступен
    r.cfg_router["fallback_to_local"] = True
    out = r.call(TaskType.COMPLEX_SOLVING, "sys", "x")
    assert out.startswith("B:")
    assert r.state["last_reason"].startswith("Claude API unavailable")


def test_api_down_no_fallback_raises(router):
    r, fa, fb = router
    r._api_cache = (9e18, False)
    r.cfg_router["fallback_to_local"] = False
    with pytest.raises(LLMError):
        r.call(TaskType.COMPLEX_SOLVING, "sys", "x")


def test_runtime_exception_fallback(router):
    r, fa, fb = router
    fa.fail = True
    r.cfg_router["fallback_to_local"] = True
    out = r.call(TaskType.COMPLEX_SOLVING, "sys", "x")
    assert out.startswith("B:")
    assert "fallback B" in r.state["last_reason"]


# ---------- budget ----------
def test_budget_exceeded_falls_back(router):
    r, fa, fb = router
    r.cfg_router["daily_api_budget_usd"] = 1.0
    r.cfg_router["fallback_to_local"] = True
    r.state["api_cost_today"] = 2.0
    out = r.call(TaskType.COMPLEX_SOLVING, "sys", "x")
    assert out.startswith("B:")
    assert "over budget" in r.state["last_reason"]


# ---------- shift schedule ----------
def test_shift_schedule_100_pct_to_b(router):
    r, fa, fb = router
    r.cfg_router["auto_shift_after_finetune"] = True
    r.cfg_router["shift_schedule"] = {"v0": {"model_a_pct": 0, "model_b_pct": 100}}
    out = r.call(TaskType.COMPLEX_SOLVING, "sys", "x")
    assert out.startswith("B:")


def test_shift_schedule_0_pct_stays_on_a(router):
    r, fa, fb = router
    r.cfg_router["auto_shift_after_finetune"] = True
    r.cfg_router["shift_schedule"] = {"v0": {"model_a_pct": 100, "model_b_pct": 0}}
    out = r.call(TaskType.COMPLEX_SOLVING, "sys", "x")
    assert out.startswith("A:")


# ---------- state / cost accounting ----------
def test_cost_accumulates_on_a_calls(router):
    r, fa, fb = router
    r.cfg_router["estimated_cost_per_call_usd"] = 0.02
    r.call(TaskType.COMPLEX_SOLVING, "sys", "x")
    r.call(TaskType.TASK_ANALYSIS, "sys", "y")
    assert r.state["api_calls_today"] == 2
    assert abs(r.state["api_cost_today"] - 0.04) < 1e-9
    assert r.state["total_a_calls"] == 2


def test_b_calls_counted(router):
    r, fa, fb = router
    r.call(TaskType.SIMPLE_LOOKUP, "sys", "a")
    r.call(TaskType.KEYWORD_EXTRACTION, "sys", "b")
    assert r.state["model_b_calls_today"] == 2
    assert r.state["total_b_calls"] == 2


# ---------- Ollama (B) недоступен ----------
def test_ollama_down_shift_stays_on_a(router):
    """Если shift_schedule хочет уйти на B, но Ollama выключен — остаёмся на A."""
    r, fa, fb = router
    r._ollama_cache = (9e18, False)
    r.cfg_router["auto_shift_after_finetune"] = True
    r.cfg_router["shift_schedule"] = {"v0": {"model_a_pct": 0, "model_b_pct": 100}}
    out = r.call(TaskType.COMPLEX_SOLVING, "sys", "x")
    assert out.startswith("A:")
    assert fb.calls == 0


def test_ollama_down_budget_exceeded_stays_on_a(router):
    """Over-budget + Ollama down → остаёмся на A (fallback не срабатывает)."""
    r, fa, fb = router
    r._ollama_cache = (9e18, False)
    r.cfg_router["daily_api_budget_usd"] = 1.0
    r.state["api_cost_today"] = 5.0
    out = r.call(TaskType.COMPLEX_SOLVING, "sys", "x")
    assert out.startswith("A:")
    assert "no B available" in r.state["last_reason"]


def test_both_models_down_raises(router):
    """И Claude API, и Ollama недоступны → LLMError с понятным сообщением."""
    r, fa, fb = router
    r._api_cache = (9e18, False)
    r._ollama_cache = (9e18, False)
    with pytest.raises(LLMError, match="Qwen/Ollama also unavailable"):
        r.call(TaskType.COMPLEX_SOLVING, "sys", "x")


def test_b_task_escalates_to_a_when_ollama_down(router):
    """B-задача (simple_lookup) + Ollama down → эскалация на A."""
    r, fa, fb = router
    r._ollama_cache = (9e18, False)
    out = r.call(TaskType.SIMPLE_LOOKUP, "sys", "x")
    assert out.startswith("A:")
    assert "escalate A" in r.state["last_reason"]


def test_b_task_both_down_raises(router):
    """B-задача + обе модели down → LLMError."""
    r, fa, fb = router
    r._ollama_cache = (9e18, False)
    r._api_cache = (9e18, False)
    with pytest.raises(LLMError, match="Both models unavailable"):
        r.call(TaskType.SIMPLE_LOOKUP, "sys", "x")


def test_verification_still_works_without_ollama(router):
    """Главный сценарий: пользователь без Ollama — весь agent loop должен работать."""
    r, fa, fb = router
    r._ollama_cache = (9e18, False)
    # Прогоняем весь набор A-задач из agent loop
    for tt in (TaskType.TASK_ANALYSIS, TaskType.COMPLEX_SOLVING,
               TaskType.VERIFICATION, TaskType.NOTE_CREATION, TaskType.LEARNING):
        out = r.call(tt, "sys", "x")
        assert out.startswith("A:")
    assert fa.calls == 5
    assert fb.calls == 0


def test_stats_reports_availability(router):
    """stats() должен показывать статус обеих моделей."""
    r, fa, fb = router
    r._api_cache = (9e18, True)
    r._ollama_cache = (9e18, False)
    s = r.stats()
    assert s["model_a_available"] is True
    assert s["model_b_available"] is False
    assert s["model_a_id"]
    assert s["model_b_id"]


# ---------- call_json ----------
def test_call_json_parses(router):
    r, fa, fb = router
    fa.complete = lambda system, user, **kw: '```json\n{"ok": true, "n": 5}\n```'
    data = r.call_json(TaskType.TASK_ANALYSIS, "sys", "x")
    assert data == {"ok": True, "n": 5}


def test_call_json_raises_on_garbage(router):
    r, fa, fb = router
    fa.complete = lambda system, user, **kw: "not json at all"
    with pytest.raises(LLMError):
        r.call_json(TaskType.TASK_ANALYSIS, "sys", "x")
