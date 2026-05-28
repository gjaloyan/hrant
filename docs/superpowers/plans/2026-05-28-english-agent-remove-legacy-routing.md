# English Agent + Remove Dead Legacy Routing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Hrant agent respond in English (config-driven directive in the unified system prompt) and delete the dead legacy intent/preference/thinking routing code that `Agent.run` no longer calls.

**Architecture:** The live path is `unified_agent.run_unified` (single tool-loop, LLM decides everything). The legacy classifier + regex routers are unreachable dead code (`agent.py:1727-1736`). This plan adds one config-driven prompt block and removes the dead code, layer by layer, each deletion grep-gated by "no live caller" and verified by "import + full suite still green".

**Tech Stack:** Python 3.12+, FastAPI backend, pytest. Config via `backend/config.py` (`CONFIG._data` dict + `@property` accessors). System prompt assembled in `backend/identity.py::IdentityManager.preamble`, consumed by `backend/unified_agent.py`.

**Spec:** `docs/superpowers/specs/2026-05-28-english-agent-llm-intent-routing-design.md`

**Constraint:** Do NOT translate memory / user profiles / knowledge-base notes. Only the response-language directive and code change.

---

## File Structure

- `backend/config.py` — add `response_language` property (default `"en"`).
- `backend/identity.py` — `preamble()` emits the English-language block when the flag is a concrete language; skips the profile LANGUAGE OVERRIDE in that case.
- `backend/agent.py` — remove dead mixins from `Agent` bases (739-748); delete `_pick_pipeline_mode`, `_chat_fallback`, routing regexes, `_looks_like_*`, `PIPELINE_*` tier constants.
- `backend/pipeline/intent.py`, `preferences.py`, `thinking.py` — deleted.
- `backend/pipeline/__init__.py` — docstring updated.
- `backend/prompts.py` — delete `INTENT_CLASSIFIER_SYSTEM`, `PREFERENCE_EXTRACTOR_SYSTEM` + `__all__` entries.
- `tests/` — triage 11 files that reference legacy symbols.

---

## Task 1: `response_language` config flag + English directive in preamble

**Files:**
- Modify: `backend/config.py` (add property near the other `@property` accessors, ~line 437)
- Modify: `backend/identity.py` (`preamble`, ~lines 553-564)
- Test: `tests/test_response_language.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_response_language.py
"""Response-language directive: a config flag forces the agent to
answer in one language regardless of the user's input language."""
from pathlib import Path
from backend.identity import IdentityManager


def _mk(tmp_path):
    return IdentityManager(base_dir=Path(tmp_path))


def test_english_flag_injects_directive(tmp_path, monkeypatch):
    from backend.config import CONFIG
    monkeypatch.setitem(CONFIG._data, "response_language", "en")
    pre = _mk(tmp_path).preamble()
    assert "RESPONSE LANGUAGE" in pre
    assert "respond ONLY in English" in pre


def test_mirror_flag_no_directive(tmp_path, monkeypatch):
    from backend.config import CONFIG
    monkeypatch.setitem(CONFIG._data, "response_language", "mirror")
    pre = _mk(tmp_path).preamble()
    assert "RESPONSE LANGUAGE" not in pre


def test_default_is_english(tmp_path, monkeypatch):
    from backend.config import CONFIG
    monkeypatch.setitem(CONFIG._data, "response_language", None)
    # default property value is "en"
    assert CONFIG.response_language == "en"
    pre = _mk(tmp_path).preamble()
    assert "respond ONLY in English" in pre
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_response_language.py -v`
Expected: FAIL — `CONFIG.response_language` AttributeError / directive absent.

- [ ] **Step 3: Add the config property**

In `backend/config.py`, after the existing `verification` / `search` properties (~line 437), add:

```python
    @property
    def response_language(self) -> str:
        """Language the agent must answer in, regardless of the user's
        input language. "en" (default) → always English. "mirror" or
        "" → mirror the user's input language (legacy soul behavior).
        Does NOT translate stored memory/knowledge — only the reply."""
        v = self._data.get("response_language")
        return (v if v is not None else "en").strip() or "en"
```

- [ ] **Step 4: Emit the directive in `preamble`**

In `backend/identity.py::preamble`, locate the trailing block that
appends the profile LANGUAGE OVERRIDE (the `lang_body = ...` section,
~lines 553-563). Replace that section with:

```python
        from .config import CONFIG
        forced = CONFIG.response_language
        if forced and forced.lower() not in ("mirror", ""):
            # Config-pinned response language wins over both the soul's
            # "mirror the user's language" line and any profile pin.
            lang_name = {"en": "English"}.get(forced.lower(), forced)
            out += (
                "\n# RESPONSE LANGUAGE\n"
                f"Respond ONLY in {lang_name}, regardless of the language "
                "of the user's message. This OVERRIDES any soul-level rule "
                "about mirroring the user's language and any profile "
                "language pin. Stored notes/profile may be in other "
                "languages — read them, but always reply in "
                f"{lang_name}.\n"
            )
        else:
            lang_body = self._extract_language_section(profile_text)
            if lang_body:
                out += (
                    "\n# LANGUAGE OVERRIDE\n"
                    "User profile pins the response language below. "
                    "This OVERRIDES the soul-level rule about mirroring the "
                    "user's input language. Even if the current user message "
                    "is in a different language, respond in the language "
                    "specified here:\n"
                    f"{lang_body}\n"
                )
        return out
```

(Confirm the surrounding `out` assembly and the `return out` placement match the current method; preserve the NAMES block above it.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_response_language.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add backend/config.py backend/identity.py tests/test_response_language.py
git commit -m "feat(lang): config-driven English response directive in system prompt"
```

---

## Task 2: Remove dead legacy mixins (intent, preferences, thinking)

**Files:**
- Modify: `backend/agent.py:738-748` (imports + `Agent` base list)
- Delete: `backend/pipeline/intent.py`, `backend/pipeline/preferences.py`, `backend/pipeline/thinking.py`
- Modify: `backend/pipeline/__init__.py` (docstring)

- [ ] **Step 1: Prove no live caller**

Run each; all must show ONLY the definitions/comments to be deleted (no live call from `run_unified` / channels / tools):

```bash
rg -n "_classify_intent\(|_save_preference\(|self\._think\(" backend --glob '!backend/pipeline/*' --glob '!backend/agent.py'
rg -n "IntentClassifierMixin|PreferenceHandlerMixin|ThinkingMixin" backend
```
Expected: matches only in `backend/agent.py` (the import + base list + the dead-comment lines) and `backend/pipeline/`. If any other live module references them, STOP and report.

- [ ] **Step 2: Remove mixins from `Agent`**

In `backend/agent.py` delete the three import lines (739-741) and the three names in the base-class list (745-748), keeping `SelfCriticMixin` (it is live):

```python
from .pipeline.critic import SelfCriticMixin  # noqa: E402


class Agent(
    SelfCriticMixin,
):
```

(Match the exact current class signature; only the three mixins are removed.)

- [ ] **Step 3: Delete the dead pipeline files**

```bash
git rm backend/pipeline/intent.py backend/pipeline/preferences.py backend/pipeline/thinking.py
```

- [ ] **Step 4: Update `pipeline/__init__.py` docstring**

Edit `backend/pipeline/__init__.py` so the module list names only the surviving `critic.py` (SelfCriticMixin). Remove the lines describing `intent.py`, `preferences.py`, `thinking.py` and the sentence about `Agent` inheriting "every Mixin".

- [ ] **Step 5: Verify import + suite**

Run: `python -c "import backend.agent; import backend; print('ok')"`
Expected: `ok` (no ImportError).
Run: `python -m pytest -q -x -k "not subagents_store and not test_jobs"`
Expected: failures only in tests that referenced the deleted mixins/symbols (fixed in Task 5). Note them.

- [ ] **Step 6: Commit**

```bash
git add backend/agent.py backend/pipeline/__init__.py
git commit -m "refactor: drop dead legacy intent/preference/thinking mixins"
```

---

## Task 3: Remove orphaned routing code in `agent.py`

**Files:**
- Modify: `backend/agent.py` (regex constants ~270-700, `_pick_pipeline_mode`, `_chat_fallback`, `PIPELINE_*`)

- [ ] **Step 1: Prove each symbol is now orphaned**

```bash
rg -n "_pick_pipeline_mode|_chat_fallback|PIPELINE_FAST_CHAT|PIPELINE_TASK_MODE|PIPELINE_DEEP_AGENT" backend
rg -n "_ARITHMETIC_RE|_ARITHMETIC_WORDS_RE|_ARITHMETIC_DIGIT_RE|_CHITCHAT_RE|_MICRO_ACK_RE|_PROFILE_RECALL_RE|_DIRECTIVE_VERBS_RE|_SYSTEM_ATTRIBUTE_RE|_SELF_QUESTION_RE|_SELF_ANALYSIS_HINT_RE|_DEEP_AGENT_HINT_RE" backend
rg -n "_looks_like_arithmetic|_looks_like_profile_recall|_looks_like_system_directive|_looks_like_system_setting_preference|_looks_like_deep_agent_request|_looks_like_self_analysis_request|_is_self_question" backend
```
Expected after Task 2: matches only inside `backend/agent.py`. Any match elsewhere in live code → STOP and report (it must be re-homed, not deleted).

- [ ] **Step 2: Delete the orphaned definitions**

In `backend/agent.py` delete: all `_ARITHMETIC_*`, `_CHITCHAT_RE`, `_MICRO_ACK_RE`, `_PROFILE_RECALL_RE`, `_DIRECTIVE_VERBS_RE`, `_SYSTEM_ATTRIBUTE_RE`, `_SELF_QUESTION_RE`, `_SELF_ANALYSIS_HINT_RE`, `_DEEP_AGENT_HINT_RE` constants; the `_looks_like_*` and `_is_self_question` helpers; `_pick_pipeline_mode`; `_chat_fallback`; and the `PIPELINE_FAST_CHAT` / `PIPELINE_TASK_MODE` / `PIPELINE_DEEP_AGENT` constants. Keep everything still referenced by `run()` / `run_unified`.

- [ ] **Step 3: Verify import + targeted run**

Run: `python -c "import backend.agent; print('ok')"`
Expected: `ok`.
Run: `python -m pytest -q tests/test_arithmetic_routing.py tests/test_round1_self_analysis_guards.py 2>&1 | tail -5`
Expected: these legacy-routing tests now ERROR on import of deleted symbols — they are removed in Task 5.

- [ ] **Step 4: Commit**

```bash
git add backend/agent.py
git commit -m "refactor: delete orphaned legacy routing regexes + tier helpers"
```

---

## Task 4: Remove dead prompt constants

**Files:**
- Modify: `backend/prompts.py` (`INTENT_CLASSIFIER_SYSTEM`, `PREFERENCE_EXTRACTOR_SYSTEM`, `__all__`)

- [ ] **Step 1: Prove orphaned**

```bash
rg -n "INTENT_CLASSIFIER_SYSTEM|PREFERENCE_EXTRACTOR_SYSTEM" backend
```
Expected: matches only in `backend/prompts.py` (definition + `__all__`). Else STOP.

- [ ] **Step 2: Delete both constants and their `__all__` entries**

Remove the `INTENT_CLASSIFIER_SYSTEM = """..."""` and `PREFERENCE_EXTRACTOR_SYSTEM = """..."""` blocks and their names in the `__all__` list at the bottom of `backend/prompts.py`.

- [ ] **Step 3: Verify import**

Run: `python -c "import backend.prompts; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add backend/prompts.py
git commit -m "refactor: delete dead INTENT_CLASSIFIER_SYSTEM + PREFERENCE_EXTRACTOR_SYSTEM"
```

---

## Task 5: Triage legacy-path tests

**Files (11, from grep):** `tests/test_round7_token_optimisation.py`, `tests/test_tool_error_detection.py`, `tests/test_audit_t_series.py`, `tests/test_memory_extractor_filter.py`, `tests/test_identity_user_profile_sanitizer.py`, `tests/test_round9.py`, `tests/test_verifier_false_absence_detector.py`, `tests/test_round1_self_analysis_guards.py`, `tests/test_arithmetic_routing.py`, `tests/test_verifier.py`, `tests/test_attachments_grounding.py`

- [ ] **Step 1: Classify each match**

For each file run e.g. `rg -n "<deleted symbol>" tests/<file>` and decide:
- **Legacy-path test** (asserts classifier/regex-router/`_save_preference`/`_chat_fallback`/`_pick_pipeline_mode` behavior): delete the test function or file — it tests removed code. Whole-file deletion only if every test in it is legacy.
- **Incidental reference** (imports a now-deleted constant for an unrelated assertion, or tests a verifier/memory feature that survives): rewrite to not use the deleted symbol.

- [ ] **Step 2: Apply deletions/rewrites**

Edit each file per Step 1. `test_arithmetic_routing.py` and `test_round1_self_analysis_guards.py` are almost certainly whole-file legacy → `git rm` them after confirming. Verifier tests (`test_verifier*.py`) test the LIVE verifier — keep them; only fix any import of a deleted symbol.

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -q`
Expected: green except the known Windows-timing flakies (`test_subagents_store::test_list_returns_newest_first`, `test_jobs::test_api_retry_creates_new_job` — re-run to confirm they pass).

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: remove/repair legacy-routing tests after dead-code deletion"
```

---

## Task 6: Deploy + prod smoke

- [ ] **Step 1: Push**

```bash
git push origin master
```

- [ ] **Step 2: Deploy**

```bash
ssh hrant@100.124.210.21 "cd /home/hrant/hrant && /home/hrant/.local/bin/hrant update 2>&1 | tail -12"
```
Expected: `pulled … commits`, `pip install -e . ✓`, `hrant.service restarted`.

- [ ] **Step 3: Smoke — English response + tool path intact**

```bash
ssh hrant@100.124.210.21 "cd /home/hrant/hrant && /home/hrant/.local/share/pipx/venvs/agi-agent/bin/python -c \"
from backend.agent import Agent
import re
a = Agent(); ans = a.run('Привет, кто ты в двух словах?', speaker_id='webui:default', channel='webui')
txt = ans.answer or ''
print('CYRILLIC_IN_REPLY:', bool(re.search(r'[А-Яа-яЁё]', txt)))
print(txt[:300])
\" 2>&1 | tail -6"
```
Expected: `CYRILLIC_IN_REPLY: False` — a Russian question gets an English reply. (If True, the directive isn't winning over the prod soul.md; strengthen placement and re-deploy.)

- [ ] **Step 4: Done**

Report results; proceed to finishing-a-development-branch.

---

## Notes for the executor

- Deletion tasks (2-4) are dependency-ordered: consumers first, then orphans. The grep-gate in each task's Step 1 is mandatory — if a "dead" symbol has a live caller, STOP and report rather than deleting.
- `SelfCriticMixin` / `pipeline/critic.py` / the verifier / `memory_extractor` are LIVE — do not touch them here (verifier regexes are sub-project #3).
- Prod data (soul.md, profiles, notes) is NOT translated — Task 1's directive makes the agent reply in English while reading whatever language the stored data is in.
