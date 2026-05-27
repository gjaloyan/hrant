"""Content + loader-matrix tests for M2-M9.

These tests pin the discipline that each module is meant to teach
plus its conditional-load predicate. Content checks use phrase
greps rather than full-body matches so wording can evolve without
churning the test file every time.
"""
from __future__ import annotations


# ─── Registry invariants ──────────────────────────────────────────


def test_all_modules_present_in_registry():
    """All 11 modules (M1 + M2 + M3 + M4 + M5 + M6 + M7×3 + M8 + M9)
    must be registered."""
    from backend.prompt_modules import MODULES
    expected = {
        "m1_core_behavior",
        "m2_task_solver",
        "m3_tool_use",
        "m4_job_tracking",
        "m5_skill_management",
        "m6_user_interaction",
        "m7_format_webui",
        "m7_format_telegram",
        "m7_format_voice",
        "m8_safety_approval",
        "m9_small_model",
    }
    assert expected.issubset(set(MODULES.keys())), (
        f"missing modules: {expected - set(MODULES.keys())}"
    )


# ─── M2: Task Solver Policy ───────────────────────────────────────


def test_m2_loads_for_task_and_supervisor_turns():
    """M2 enforces solver discipline; chat turns skip it."""
    from backend.prompt_modules import build_prompt, TurnContext
    for tt in ("task", "supervisor"):
        out = build_prompt(TurnContext(turn_type=tt))
        assert "TASK SOLVER" in out, f"M2 missing for {tt}"


def test_m2_does_not_load_for_chat_turn():
    from backend.prompt_modules import build_prompt, TurnContext
    out = build_prompt(TurnContext(turn_type="chat"))
    assert "TASK SOLVER" not in out


def test_m2_states_react_state_machine():
    """The five legal iteration shapes — plan / execute / verify /
    ask / finalize. Catches the drift where the agent has no
    explicit per-iteration intent."""
    from backend.prompt_modules import MODULES
    body = MODULES["m2_task_solver"].body.upper()
    for verb in ("PLAN", "EXECUTE", "VERIFY", "ASK", "FINALIZE"):
        assert verb in body, f"M2 missing iteration shape {verb!r}"


def test_m2_names_inspect_without_execute_antipattern():
    from backend.prompt_modules import MODULES
    body = MODULES["m2_task_solver"].body.lower()
    # The 5+ inspect rule (anti-procrastination).
    assert "inspect" in body
    assert "5" in body  # the threshold


def test_m2_names_long_running_endpoint_protocol():
    """Long-running shell (>60s) requires define_task_endpoint THEN
    start_background_job. Pin both names + the >60s threshold."""
    from backend.prompt_modules import MODULES
    body = MODULES["m2_task_solver"].body
    assert "define_task_endpoint" in body
    assert "start_background_job" in body
    assert "60s" in body or ">60" in body


def test_m2_forbids_meta_cognitive_refusal_phrasing():
    """The 'не могу подтвердить' / 'I can't' pattern that broke on
    the 2026-05-26 terminal-bench turns must be named verbatim so
    the model self-recognises it."""
    from backend.prompt_modules import MODULES
    body = MODULES["m2_task_solver"].body
    assert "не могу подтвердить" in body or "I can't" in body
    assert "2 distinct" in body.lower() or "≥2" in body


# ─── M3: Tool Use Policy ──────────────────────────────────────────


def test_m3_is_always_on():
    from backend.prompt_modules import MODULES
    assert MODULES["m3_tool_use"].always_on is True


def test_m3_pairs_signature_to_tool():
    """The decision-table is the heart of M3. Pin a handful of the
    pairings so a refactor that drops them is caught."""
    from backend.prompt_modules import MODULES
    body = MODULES["m3_tool_use"].body
    pairings = [
        ("set_setting", "config"),
        ("save_user_fact", "trait"),
        ("start_background_job", "60s"),
        ("define_task_endpoint", "60s"),
        ("locate_symbol", "code"),
        ("MEDIA:", "file"),
        ("agent_browser", "JS"),
        ("fetch_url", "static"),
        ("search_knowledge", "knowledge"),
    ]
    for tool, near_word in pairings:
        assert tool in body, f"M3 missing tool {tool!r}"
        # near_word check is a sanity that the tool sits next to a
        # signature description, not just listed.
        idx = body.find(tool)
        nearby = body[max(0, idx - 200):idx + 200].lower()
        assert near_word.lower() in nearby, (
            f"M3 {tool!r} should be near {near_word!r} in the table"
        )


def test_m3_warns_against_verbatim_retry():
    from backend.prompt_modules import MODULES
    body = MODULES["m3_tool_use"].body.lower()
    # The rule "after a tool fails, change tool or change inputs".
    assert "fail" in body
    assert "retry" in body or "change" in body


# ─── M4: Job Tracking Policy ──────────────────────────────────────


def test_m4_loads_for_task_and_supervisor():
    """M4 must be in the prompt BEFORE the agent loads the bench
    bundle — otherwise it has the tools but not the protocol.
    That's what broke on the 2026-05-26 terminal-bench turns."""
    from backend.prompt_modules import build_prompt, TurnContext
    for tt in ("task", "supervisor"):
        out = build_prompt(TurnContext(turn_type=tt))
        assert "JOB TRACKING" in out, f"M4 missing for {tt!r}"


def test_m4_does_not_load_for_chat():
    from backend.prompt_modules import build_prompt, TurnContext
    out = build_prompt(TurnContext(turn_type="chat"))
    assert "JOB TRACKING" not in out


def test_m4_distinguishes_prerequisites_from_success_criteria():
    """The whole point of the endpoint contract is the split between
    pre-flight and post-flight gates."""
    from backend.prompt_modules import MODULES
    body = MODULES["m4_job_tracking"].body
    assert "prerequisites" in body
    assert "success_criteria" in body
    assert "BEFORE" in body
    assert "AFTER" in body


def test_m4_states_three_supervisor_outcomes():
    """RETRY / DONE / ESCALATE — pin all three so a future edit
    doesn't quietly drop one."""
    from backend.prompt_modules import MODULES
    body = MODULES["m4_job_tracking"].body
    assert "RETRY" in body
    assert "DONE" in body
    assert "ESCALATE" in body


def test_m4_warns_against_polling_for_user():
    from backend.prompt_modules import MODULES
    body = MODULES["m4_job_tracking"].body.lower()
    assert "poll" in body or "once" in body


# ─── M5: Skill Management Policy ──────────────────────────────────


def test_m5_is_always_on():
    from backend.prompt_modules import MODULES
    assert MODULES["m5_skill_management"].always_on is True


def test_m5_search_before_propose():
    """Duplicate avoidance is M5's core anti-pattern target."""
    from backend.prompt_modules import MODULES
    body = MODULES["m5_skill_management"].body.lower()
    assert "duplicate" in body or "near-duplicate" in body
    assert "search" in body or "scan" in body


def test_m5_names_universal_resolver_fallback():
    from backend.prompt_modules import MODULES
    body = MODULES["m5_skill_management"].body
    assert "universal_resolver" in body
    assert "load_skill" in body


def test_m5_explicit_skill_creator_workflow():
    from backend.prompt_modules import MODULES
    body = MODULES["m5_skill_management"].body
    assert "skill_creator" in body
    assert "propose_skill" in body


# ─── M6: User Interaction Policy ──────────────────────────────────


def test_m6_is_always_on():
    from backend.prompt_modules import MODULES
    assert MODULES["m6_user_interaction"].always_on is True


def test_m6_lists_ask_user_acceptable_reasons():
    """ask_user has exactly four legitimate triggers; pin them so
    drift back to free-text 'could you clarify?' is harder."""
    from backend.prompt_modules import MODULES
    body = MODULES["m6_user_interaction"].body.lower()
    assert "ask_user" in body
    # The four reasons (we accept phrasing drift on synonyms):
    assert "missing" in body
    assert "interpretation" in body or "interpret" in body or "ambiguous" in body
    assert "destructive" in body or "consent" in body
    assert "credential" in body or "external account" in body


def test_m6_forbids_trailing_offer():
    """The 'let me know if you need anything else' anti-pattern is
    the single most common confirmation bloat."""
    from backend.prompt_modules import MODULES
    body = MODULES["m6_user_interaction"].body.lower()
    assert "trailing" in body or "let me know" in body


# ─── M7: Channel Formatting ───────────────────────────────────────


def test_m7_webui_loads_only_for_webui():
    from backend.prompt_modules import build_prompt, TurnContext
    out_webui = build_prompt(TurnContext(channel="webui"))
    assert "OUTPUT FORMAT — WebUI" in out_webui
    out_tg = build_prompt(TurnContext(channel="telegram"))
    assert "OUTPUT FORMAT — WebUI" not in out_tg


def test_m7_telegram_loads_only_for_telegram():
    from backend.prompt_modules import build_prompt, TurnContext
    out_tg = build_prompt(TurnContext(channel="telegram"))
    assert "OUTPUT FORMAT — Telegram" in out_tg
    out_webui = build_prompt(TurnContext(channel="webui"))
    assert "OUTPUT FORMAT — Telegram" not in out_webui


def test_m7_voice_loads_only_for_voice():
    from backend.prompt_modules import build_prompt, TurnContext
    out_voice = build_prompt(TurnContext(channel="voice"))
    assert "OUTPUT FORMAT — Voice" in out_voice
    out_webui = build_prompt(TurnContext(channel="webui"))
    assert "OUTPUT FORMAT — Voice" not in out_webui


def test_m7_telegram_explains_media_convention():
    """MEDIA:/path is the only way to attach files via Telegram —
    must appear in the channel-specific module."""
    from backend.prompt_modules import MODULES
    body = MODULES["m7_format_telegram"].body
    assert "MEDIA:" in body
    assert "/absolute" in body or "/abs" in body or "absolute path" in body.lower()


def test_m7_voice_forbids_markdown():
    from backend.prompt_modules import MODULES
    body = MODULES["m7_format_voice"].body.lower()
    assert "markdown" in body
    assert "no" in body  # "No Markdown"


# ─── M8: Safety & Approval ────────────────────────────────────────


def test_m8_is_always_on():
    from backend.prompt_modules import MODULES
    assert MODULES["m8_safety_approval"].always_on is True


def test_m8_states_reversibility_tiers():
    """Reversibility-first taxonomy — local / hard-to-reverse /
    visible-to-others / third-party uploads."""
    from backend.prompt_modules import MODULES
    body = MODULES["m8_safety_approval"].body.lower()
    assert "reversibility" in body or "reversible" in body
    assert "destructive" in body or "hard-to-reverse" in body
    assert "third" in body or "3rd" in body  # third-party uploads


def test_m8_acknowledges_code_side_denylist():
    """Tells the model: don't try to bypass; the gate enforces it."""
    from backend.prompt_modules import MODULES
    body = MODULES["m8_safety_approval"].body.lower()
    assert "denylist" in body or "blocked" in body
    assert "rm -rf /" in body or "catastrophic" in body


def test_m8_warns_against_skipping_git_hooks():
    from backend.prompt_modules import MODULES
    body = MODULES["m8_safety_approval"].body
    assert "--no-verify" in body or "hooks" in body.lower()


# ─── M9: Small-Model Adaptations ──────────────────────────────────


def test_m9_loads_only_for_small_model():
    from backend.prompt_modules import build_prompt, TurnContext
    out_small = build_prompt(TurnContext(model_size="small"))
    assert "SMALL-MODEL" in out_small
    for sz in ("medium", "large"):
        out = build_prompt(TurnContext(model_size=sz))
        assert "SMALL-MODEL" not in out, f"M9 leaked into {sz}"


def test_m9_enforces_endpoint_first_line():
    """The compensatory rule: small models drift without an explicit
    endpoint anchor at the start of every turn."""
    from backend.prompt_modules import MODULES
    body = MODULES["m9_small_model"].body.lower()
    assert "done =" in body or "endpoint" in body
    assert "first line" in body or "first" in body


def test_m9_forbids_parallel_tool_calls():
    """Multi-tool parallelism overwhelms small-model attention."""
    from backend.prompt_modules import MODULES
    body = MODULES["m9_small_model"].body.lower()
    assert "one tool" in body or "single tool" in body or "parallel" in body


# ─── Composite scenarios ──────────────────────────────────────────


def test_default_task_turn_loads_expected_modules():
    """Default ctx = task, webui, no bundles, large. Includes
    M1+M2+M3+M4+M5+M6+M7_webui+M8 (M4 now always-on for task) and
    excludes M7 for other channels + M9 (large model)."""
    from backend.prompt_modules import build_prompt
    out = build_prompt()
    # Present:
    for marker in (
        "CORE AGENT BEHAVIOR",
        "TASK SOLVER",
        "TOOL USE",
        "JOB TRACKING",
        "SKILL MANAGEMENT",
        "USER INTERACTION",
        "OUTPUT FORMAT — WebUI",
        "SAFETY",
    ):
        assert marker in out, f"missing {marker!r} in default ctx"
    # Absent:
    for marker in (
        "OUTPUT FORMAT — Telegram",
        "OUTPUT FORMAT — Voice",
        "SMALL-MODEL",
    ):
        assert marker not in out, f"unexpected {marker!r} in default ctx"


def test_chat_turn_excludes_solver_and_job_tracking():
    """Chat is the cheapest turn — no solver discipline, no job
    tracking, no small-model module unless model_size is small."""
    from backend.prompt_modules import build_prompt, TurnContext
    out = build_prompt(TurnContext(turn_type="chat"))
    assert "CORE AGENT BEHAVIOR" in out
    assert "TASK SOLVER" not in out
    assert "JOB TRACKING" not in out


def test_supervisor_turn_loads_m2_and_m4():
    """Supervisor turns need both M2 discipline and M4 protocol —
    regardless of bundle state."""
    from backend.prompt_modules import build_prompt, TurnContext
    out = build_prompt(TurnContext(turn_type="supervisor"))
    assert "TASK SOLVER" in out
    assert "JOB TRACKING" in out


def test_telegram_voice_small_combo():
    """Small local model running on Telegram via voice — load M9
    + M7_voice + the rest of the always-on stack."""
    from backend.prompt_modules import build_prompt, TurnContext
    out = build_prompt(TurnContext(
        channel="voice", model_size="small",
    ))
    assert "SMALL-MODEL" in out
    assert "OUTPUT FORMAT — Voice" in out
    assert "OUTPUT FORMAT — WebUI" not in out
    assert "OUTPUT FORMAT — Telegram" not in out


def test_default_prompt_under_global_budget():
    """The whole composed default-ctx prompt should land well under
    the legacy 22 KB monolith. Current target: ~10-11 KB (M4 is
    now always-on for task turns, adding ~480 chars over the bench-
    gated design). Hard cap 12 KB to catch gross bloat early."""
    from backend.prompt_modules import build_prompt
    out = build_prompt()
    assert len(out) < 12_000, (
        f"default prompt grew to {len(out)} chars — splits or "
        "trims needed"
    )


def test_chat_prompt_is_smaller_than_task_prompt():
    """Chat-turn cost discipline: chat must be measurably cheaper
    than task (otherwise the conditional-load design has a bug)."""
    from backend.prompt_modules import build_prompt, TurnContext
    chat = build_prompt(TurnContext(turn_type="chat"))
    task = build_prompt(TurnContext(turn_type="task"))
    assert len(chat) < len(task), (
        f"chat ({len(chat)}) should be smaller than task "
        f"({len(task)}); M2 isn't being skipped"
    )
