"""Tests for H1-rev — auto-propose installs for missing required_tools.

Pinned behaviour:
  - installer supports `apt` manager: sudo -n apt-get install -y, fails
    fast if there's no passwordless sudo (no hanging on password prompts).
  - `installer.resolve_manager_for(name)` returns 'apt' for known system
    binaries (ffmpeg, libreoffice, qpdf, ...), 'pip' for everything else.
  - `installer.has_pending(packages, manager)` returns True only when a
    pending request with the same (manager, package-set) already exists.
    Order-independent comparison.
  - `SkillsManager.missing_tools_with_manager_for(skill)` returns dicts
    with the resolved manager: explicit frontmatter wins, then the hint
    map, then pip default.
  - `run_unified` auto-fires `installer.propose` for each missing tool
    when a skill matches, deduplicated by (manager, name) per turn and
    across turns (no duplicate DMs while one is still pending).
  - The system prompt carries an `AUTO-PROPOSED INSTALLS` block when
    propose was fired so the LLM can tell the user to tap Approve.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ─── installer.apt manager ──────────────────────────────────────────


@pytest.fixture
def isolated_installer(tmp_path, monkeypatch):
    """Same shape as the G2 fixture — redirect installer state into
    tmp_path so tests don't share pending lists."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend.config import CONFIG
    monkeypatch.setitem(
        CONFIG._data,
        "knowledge",
        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)},
    )
    from backend import installer
    installer.STORE._root_override = tmp_path
    saved = list(installer._ON_INSTALL_PROPOSED)
    installer._ON_INSTALL_PROPOSED.clear()
    yield installer
    installer._ON_INSTALL_PROPOSED.clear()
    installer._ON_INSTALL_PROPOSED.extend(saved)
    installer.STORE._root_override = None


def test_apt_is_supported_manager(isolated_installer):
    inst = isolated_installer
    req = inst.propose(
        packages=["ffmpeg"], manager="apt", reason="r", requester="w",
    )
    assert req is not None
    assert req.manager == "apt"


def test_apt_install_uses_sudo_n_apt_get(isolated_installer):
    """The cmd template must be `sudo -n apt-get install -y ...` so
    a missing passwordless sudo fails fast instead of hanging on a
    password prompt the Telegram bridge can't answer."""
    inst = isolated_installer
    cfg = inst._MANAGERS["apt"]
    cmd = cfg["cmd"]
    assert cmd[0] == "sudo"
    assert "-n" in cmd, "must use sudo -n (non-interactive)"
    assert "apt-get" in cmd
    assert "install" in cmd
    assert "-y" in cmd


# ─── C2: per-manager whitelist ──────────────────────────────────────


def test_apt_rejects_bracket_in_package_name(isolated_installer):
    """apt-get sees `[` `]` as shell-glob characters. The whitelist
    for apt must reject them. C2 fix."""
    inst = isolated_installer
    with pytest.raises(ValueError, match="disallowed characters"):
        inst.propose(
            packages=["foo[bar]"], manager="apt",
            reason="C2 probe", requester="w",
        )


def test_apt_rejects_glob_metacharacter(isolated_installer):
    """The exact attack shape — `foo[*]` would shell-expand to match
    filenames in CWD. Whitelist must refuse."""
    inst = isolated_installer
    with pytest.raises(ValueError, match="disallowed characters"):
        inst.propose(
            packages=["foo[*]"], manager="apt",
            reason="C2 probe", requester="w",
        )


def test_pip_still_accepts_bracket_extras(isolated_installer):
    """pip extras syntax `requests[security]` is legal — the per-
    manager whitelist must keep allowing it for pip / pipx."""
    inst = isolated_installer
    req = inst.propose(
        packages=["requests[security]"], manager="pip",
        reason="C2 probe", requester="w",
    )
    assert req is not None
    assert req.packages == ["requests[security]"]


def test_pipx_still_accepts_bracket_extras(isolated_installer):
    inst = isolated_installer
    req = inst.propose(
        packages=["foo[bar]"], manager="pipx",
        reason="C2 probe", requester="w",
    )
    assert req is not None
    assert req.packages == ["foo[bar]"]


def test_apt_accepts_normal_package_name(isolated_installer):
    """Sanity — apt with a plain alphanumeric name still works."""
    inst = isolated_installer
    req = inst.propose(
        packages=["ffmpeg"], manager="apt",
        reason="C2 probe", requester="w",
    )
    assert req is not None
    assert req.packages == ["ffmpeg"]


def test_apt_accepts_dash_dot_plus_in_name(isolated_installer):
    """Real apt names like `libreoffice-core`, `g++`, `python3.12-dev`
    must still validate."""
    inst = isolated_installer
    for ok_name in ["libreoffice-core", "g++", "python3.12-dev", "lib_foo.bar"]:
        req = inst.propose(
            packages=[ok_name], manager="apt",
            reason="C2 probe", requester="w",
        )
        assert req is not None, f"{ok_name!r} should be accepted by apt"


def test_apt_install_runs_subprocess(isolated_installer, monkeypatch):
    """approve(code) for an apt request runs sudo -n apt-get install
    via subprocess.run and journals the result."""
    inst = isolated_installer
    req = inst.propose(
        packages=["ffmpeg"], manager="apt", reason="r", requester="w",
    )
    fake_run = MagicMock()
    fake_run.return_value = type("Proc", (), {
        "returncode": 0,
        "stdout": "Setting up ffmpeg (7.0.1)...\n",
        "stderr": "",
    })()
    monkeypatch.setattr(inst.subprocess, "run", fake_run)

    res = inst.approve(req.code)
    assert res["ok"] is True
    assert res["manager"] == "apt"
    # The actual subprocess.run call must have used sudo + apt-get.
    called_cmd = fake_run.call_args.args[0]
    assert called_cmd[0] == "sudo"
    assert "apt-get" in called_cmd
    assert "ffmpeg" in called_cmd


def test_apt_install_surfaces_no_passwordless_sudo(isolated_installer, monkeypatch):
    """Without passwordless sudo, `sudo -n` returns exit 1 with a
    'sudo: a password is required' on stderr. approve() must surface
    that clearly so the owner sees why it failed."""
    inst = isolated_installer
    req = inst.propose(
        packages=["ffmpeg"], manager="apt", reason="r", requester="w",
    )
    fake_run = MagicMock()
    fake_run.return_value = type("Proc", (), {
        "returncode": 1,
        "stdout": "",
        "stderr": "sudo: a password is required\n",
    })()
    monkeypatch.setattr(inst.subprocess, "run", fake_run)

    res = inst.approve(req.code)
    assert res["ok"] is False
    assert "password" in (res["error"] or "").lower() or \
           "password" in (res["stderr_tail"] or "").lower()


# ─── resolve_manager_for ────────────────────────────────────────────


def test_resolve_manager_for_known_apt_binary():
    from backend.installer import resolve_manager_for
    assert resolve_manager_for("ffmpeg") == "apt"
    assert resolve_manager_for("libreoffice") == "apt"
    assert resolve_manager_for("qpdf") == "apt"


def test_resolve_manager_for_unknown_defaults_to_pip():
    from backend.installer import resolve_manager_for
    assert resolve_manager_for("pypdf") == "pip"
    assert resolve_manager_for("requests") == "pip"
    assert resolve_manager_for("some-unknown-pkg") == "pip"


def test_resolve_manager_for_empty_name_returns_pip():
    from backend.installer import resolve_manager_for
    assert resolve_manager_for("") == "pip"


# ─── has_pending dedup ──────────────────────────────────────────────


def test_has_pending_true_for_matching_request(isolated_installer):
    inst = isolated_installer
    inst.propose(packages=["ffmpeg"], manager="apt", reason="r", requester="w")
    assert inst.has_pending(["ffmpeg"], "apt") is True


def test_has_pending_false_for_different_packages(isolated_installer):
    inst = isolated_installer
    inst.propose(packages=["ffmpeg"], manager="apt", reason="r", requester="w")
    assert inst.has_pending(["pillow"], "apt") is False


def test_has_pending_false_for_different_manager(isolated_installer):
    inst = isolated_installer
    inst.propose(packages=["pypdf"], manager="pip", reason="r", requester="w")
    assert inst.has_pending(["pypdf"], "apt") is False


def test_has_pending_order_independent(isolated_installer):
    inst = isolated_installer
    inst.propose(
        packages=["pillow", "pypdf"], manager="pip", reason="r", requester="w",
    )
    assert inst.has_pending(["pypdf", "pillow"], "pip") is True


def test_has_pending_false_when_empty(isolated_installer):
    inst = isolated_installer
    assert inst.has_pending(["anything"], "pip") is False


def test_has_pending_ignores_empty_input(isolated_installer):
    inst = isolated_installer
    inst.propose(packages=["pillow"], manager="pip", reason="r", requester="w")
    assert inst.has_pending([], "pip") is False
    assert inst.has_pending([""], "pip") is False


# ─── SkillsManager.missing_tools_with_manager_for ──────────────────


@pytest.fixture
def skills_dir(tmp_path, monkeypatch):
    """Isolate skills root + disabled.json."""
    d = tmp_path / "skills"
    d.mkdir()
    fake_disabled = tmp_path / "skills_disabled.json"
    monkeypatch.setattr(
        "backend.skills._disabled_path",
        lambda: fake_disabled,
    )
    return d


def _write_skill(root: Path, name: str, frontmatter: str, body: str = "") -> Path:
    sk_dir = root / name
    sk_dir.mkdir()
    (sk_dir / "SKILL.md").write_text(
        f"---\n{frontmatter.strip()}\n---\n\n{body}",
        encoding="utf-8",
    )
    return sk_dir


def test_missing_with_manager_uses_explicit_hint(skills_dir, monkeypatch):
    """Frontmatter `{name: X, manager: Y}` wins over the resolve
    hint map."""
    _write_skill(skills_dir, "vid", textwrap.dedent("""
        name: vid
        description: video work
        triggers: [video]
        required_tools:
          - name: not-real-binary
            manager: pipx
    """))
    from backend.skills import SkillsManager
    from backend.tool_registry import ToolRegistry
    from backend import skills as mod
    monkeypatch.setattr(mod._shutil, "which", lambda n: None)
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    sk = sm.get("vid")
    result = sm.missing_tools_with_manager_for(sk, registry=ToolRegistry())
    assert result == [{"name": "not-real-binary", "manager": "pipx"}]


def test_missing_with_manager_falls_back_to_resolve(skills_dir, monkeypatch):
    """No explicit manager → use resolve_manager_for. ffmpeg is in
    the apt hint set; pypdf is not, defaults to pip."""
    _write_skill(skills_dir, "vid", textwrap.dedent("""
        name: vid
        description: video work
        triggers: [video]
        required_tools: [ffmpeg, pypdf]
    """))
    from backend.skills import SkillsManager
    from backend.tool_registry import ToolRegistry
    from backend import skills as mod
    monkeypatch.setattr(mod._shutil, "which", lambda n: None)
    # Make sure pypdf is also "missing" — force find_spec to return None.
    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda n: None)
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    sk = sm.get("vid")
    result = sm.missing_tools_with_manager_for(sk, registry=ToolRegistry())
    assert {"name": "ffmpeg", "manager": "apt"} in result
    assert {"name": "pypdf", "manager": "pip"} in result


def test_missing_with_manager_skips_available_tools(skills_dir):
    """Available tools don't show up in the missing list, regardless
    of manager hint."""
    _write_skill(skills_dir, "vid", textwrap.dedent("""
        name: vid
        description: video work
        triggers: [video]
        required_tools: [yaml]
    """))
    from backend.skills import SkillsManager
    from backend.tool_registry import ToolRegistry
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    sk = sm.get("vid")
    # `yaml` is importable (skills.py imports it), so not missing.
    assert sm.missing_tools_with_manager_for(sk, registry=ToolRegistry()) == []


# ─── unified_agent auto-propose integration ─────────────────────────


@pytest.fixture
def isolated_skills_with_missing_dep(tmp_path, monkeypatch):
    """Build a user-tier skill that triggers on 'autoinstallprobe'
    and requires a definitely-missing binary. Isolate skills + installer
    so the test doesn't touch real disk state."""
    from backend.config import CONFIG
    monkeypatch.setitem(
        CONFIG._data,
        "knowledge",
        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)},
    )
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))

    fake_disabled = tmp_path / "skills_disabled.json"
    monkeypatch.setattr(
        "backend.skills._disabled_path", lambda: fake_disabled,
    )

    skill_dir = tmp_path / "skills" / "autoinstallprobe-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: autoinstallprobe\n"
        "description: Trigger an auto-install probe.\n"
        "triggers: [autoinstallprobe]\n"
        "required_tools:\n"
        "  - name: definitely-not-real-xyz-789\n"
        "    manager: pip\n"
        "---\n\n"
        "# probe\nsteps go here\n",
        encoding="utf-8",
    )
    from backend import skills as sk_mod
    sk_mod.SKILLS._user_dir_override = tmp_path / "skills"
    sk_mod.SKILLS._loaded = False
    sk_mod.SKILLS.skills = []

    from backend import installer
    installer.STORE._root_override = tmp_path
    saved = list(installer._ON_INSTALL_PROPOSED)
    installer._ON_INSTALL_PROPOSED.clear()

    yield (sk_mod.SKILLS, installer)

    installer._ON_INSTALL_PROPOSED.clear()
    installer._ON_INSTALL_PROPOSED.extend(saved)
    installer.STORE._root_override = None
    sk_mod.SKILLS._user_dir_override = None
    sk_mod.SKILLS._loaded = False
    sk_mod.SKILLS.skills = []


@pytest.mark.skip(
    reason="2026-05-21: auto-propose ceremony retired — run_unified "
    "no longer fires installer.propose for matched skills' missing "
    "deps. Missing tools are now surfaced as a hint in the system "
    "prompt; agent uses terminal_exec to install when actually needed."
)
def test_run_unified_auto_proposes_for_missing_required_tool(
    isolated_skills_with_missing_dep, monkeypatch,
):
    """End-to-end: a task that triggers a skill with a missing
    required_tool causes run_unified to fire installer.propose for
    that tool. The pending list grows by one."""
    skills, installer = isolated_skills_with_missing_dep
    from backend import llm as _llm
    from backend.models import VerificationResult

    captured = {}
    fake_router = MagicMock()

    def fake_call(task_type, system, user, **kwargs):
        captured["system"] = system
        return "ack"

    fake_router.call_with_tools.side_effect = fake_call
    monkeypatch.setattr(_llm, "router", lambda: fake_router)

    from backend import verifier as _v
    monkeypatch.setattr(
        _v, "verify", lambda *a, **kw: VerificationResult(confidence=90),
    )

    pending_before = len(installer.STORE.list_pending())

    from backend.agent import Agent
    agent = Agent()
    agent.run(
        "please run autoinstallprobe analysis",
        channel="webui",
        speaker_id="webui:default",
    )

    pending_after = installer.STORE.list_pending()
    assert len(pending_after) == pending_before + 1
    req = pending_after[-1]
    assert req.packages == ["definitely-not-real-xyz-789"]
    assert req.manager == "pip"

    # And the system prompt must carry the AUTO-PROPOSED INSTALLS block.
    sys_prompt = captured.get("system") or ""
    assert "AUTO-PROPOSED INSTALLS" in sys_prompt
    assert "definitely-not-real-xyz-789" in sys_prompt


@pytest.mark.skip(
    reason="2026-05-21: auto-propose ceremony retired (see above)."
)
def test_run_unified_dedups_auto_propose_across_turns(
    isolated_skills_with_missing_dep, monkeypatch,
):
    """Second turn with the same trigger must NOT propose again while
    the first request is still pending. Owner gets one DM, not N."""
    skills, installer = isolated_skills_with_missing_dep
    from backend import llm as _llm
    from backend.models import VerificationResult

    fake_router = MagicMock()
    fake_router.call_with_tools.side_effect = lambda *a, **kw: "ack"
    monkeypatch.setattr(_llm, "router", lambda: fake_router)

    from backend import verifier as _v
    monkeypatch.setattr(
        _v, "verify", lambda *a, **kw: VerificationResult(confidence=90),
    )

    from backend.agent import Agent
    agent = Agent()
    agent.run("autoinstallprobe round 1",
              channel="webui", speaker_id="webui:default")
    agent.run("autoinstallprobe round 2",
              channel="webui", speaker_id="webui:default")
    agent.run("autoinstallprobe round 3",
              channel="webui", speaker_id="webui:default")

    pending = installer.STORE.list_pending()
    # All three turns pointed at the same missing tool — one pending
    # request, not three.
    matching = [r for r in pending
                if r.packages == ["definitely-not-real-xyz-789"]]
    assert len(matching) == 1


# ─── I2: per-turn cap on auto-propose ──────────────────────────────


@pytest.fixture
def isolated_skill_with_many_missing(tmp_path, monkeypatch):
    """Skill with MORE than AUTO_PROPOSE_CAP missing required_tools.
    Triggers on 'manymissing'."""
    from backend.config import CONFIG
    monkeypatch.setitem(
        CONFIG._data,
        "knowledge",
        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)},
    )
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))

    fake_disabled = tmp_path / "skills_disabled.json"
    monkeypatch.setattr(
        "backend.skills._disabled_path", lambda: fake_disabled,
    )

    # 8 missing tools — more than the cap (5).
    skill_dir = tmp_path / "skills" / "manymissing-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: manymissing\n"
        "description: skill with many missing deps\n"
        "triggers: [manymissing]\n"
        "required_tools:\n"
        + "\n".join(
            f"  - not-real-pkg-{i:02d}" for i in range(8)
        ) + "\n"
        "---\n\nbody\n",
        encoding="utf-8",
    )
    from backend import skills as sk_mod
    sk_mod.SKILLS._user_dir_override = tmp_path / "skills"
    sk_mod.SKILLS._loaded = False
    sk_mod.SKILLS.skills = []

    from backend import installer
    installer.STORE._root_override = tmp_path
    saved = list(installer._ON_INSTALL_PROPOSED)
    installer._ON_INSTALL_PROPOSED.clear()

    yield (sk_mod.SKILLS, installer)

    installer._ON_INSTALL_PROPOSED.clear()
    installer._ON_INSTALL_PROPOSED.extend(saved)
    installer.STORE._root_override = None
    sk_mod.SKILLS._user_dir_override = None
    sk_mod.SKILLS._loaded = False
    sk_mod.SKILLS.skills = []


@pytest.mark.skip(
    reason="2026-05-21: auto-propose ceremony retired (see above)."
)
def test_auto_propose_caps_at_AUTO_PROPOSE_CAP(
    isolated_skill_with_many_missing, monkeypatch,
):
    """A skill with 8 missing tools must produce only AUTO_PROPOSE_CAP
    (=5) actual propose calls, not 8. The remaining 3 surface as a
    deferred line in the system prompt."""
    from backend.unified_agent import AUTO_PROPOSE_CAP
    skills, installer = isolated_skill_with_many_missing
    from backend import llm as _llm
    from backend.models import VerificationResult

    captured = {}
    fake_router = MagicMock()

    def fake_call(task_type, system, user, **kwargs):
        captured["system"] = system
        return "ack"

    fake_router.call_with_tools.side_effect = fake_call
    monkeypatch.setattr(_llm, "router", lambda: fake_router)

    from backend import verifier as _v
    monkeypatch.setattr(
        _v, "verify", lambda *a, **kw: VerificationResult(confidence=90),
    )

    pending_before = len(installer.STORE.list_pending())

    from backend.agent import Agent
    agent = Agent()
    agent.run("manymissing please go",
              channel="webui", speaker_id="webui:default")

    new_pending = installer.STORE.list_pending()[pending_before:]
    # Exactly AUTO_PROPOSE_CAP DMs were fired, not 8.
    assert len(new_pending) == AUTO_PROPOSE_CAP, (
        f"expected {AUTO_PROPOSE_CAP} propose calls, got {len(new_pending)}"
    )
    # The deferred line shows up in the system prompt with the
    # remaining count (8 - 5 = 3).
    sys_prompt = captured.get("system") or ""
    assert "AUTO-PROPOSED INSTALLS" in sys_prompt
    assert "deferred" in sys_prompt
    assert "3 more" in sys_prompt or "3 mo" in sys_prompt or " 3 " in sys_prompt


def test_auto_propose_cap_constant_is_five():
    """Pin the cap at 5. Bumping it would re-admit DM-flood risk."""
    from backend.unified_agent import AUTO_PROPOSE_CAP
    assert AUTO_PROPOSE_CAP == 5


# ─── I3: requester labeling for auto-propose ───────────────────────


@pytest.mark.skip(
    reason="2026-05-21: auto-propose ceremony retired (see above)."
)
def test_auto_propose_requester_carries_auto_skill_match_suffix(
    isolated_skills_with_missing_dep, monkeypatch,
):
    """When run_unified auto-fires propose, the journal/pending entry
    must show the speaker_id WITH '(auto-skill-match)' suffix so it
    can be distinguished from an explicit propose_install call by
    the user."""
    skills, installer = isolated_skills_with_missing_dep
    from backend import llm as _llm
    from backend.models import VerificationResult

    fake_router = MagicMock()
    fake_router.call_with_tools.side_effect = lambda *a, **kw: "ack"
    monkeypatch.setattr(_llm, "router", lambda: fake_router)
    from backend import verifier as _v
    monkeypatch.setattr(
        _v, "verify", lambda *a, **kw: VerificationResult(confidence=90),
    )

    from backend.agent import Agent
    agent = Agent()
    agent.run("autoinstallprobe please",
              channel="telegram", speaker_id="telegram:649231423")

    pending = installer.STORE.list_pending()
    matching = [r for r in pending
                if r.packages == ["definitely-not-real-xyz-789"]]
    assert matching, "auto-propose must have fired"
    requester = matching[0].requester
    # Speaker id stays for audit, suffix marks it as auto-fired.
    assert "telegram:649231423" in requester
    assert "auto-skill-match" in requester


def test_run_unified_no_auto_propose_without_trigger_match(
    isolated_skills_with_missing_dep, monkeypatch,
):
    """If no skill triggers, no auto-propose fires — fuzzy semantic
    matches don't commit the owner to an install."""
    skills, installer = isolated_skills_with_missing_dep
    from backend import llm as _llm
    from backend.models import VerificationResult

    fake_router = MagicMock()
    fake_router.call_with_tools.side_effect = lambda *a, **kw: "ack"
    monkeypatch.setattr(_llm, "router", lambda: fake_router)

    from backend import verifier as _v
    monkeypatch.setattr(
        _v, "verify", lambda *a, **kw: VerificationResult(confidence=90),
    )

    from backend.agent import Agent
    agent = Agent()
    # Task text does NOT contain the trigger word — no match → no
    # auto-propose for the missing dep.
    agent.run("какая сегодня погода?",
              channel="webui", speaker_id="webui:default")

    pending = installer.STORE.list_pending()
    matching = [r for r in pending
                if r.packages == ["definitely-not-real-xyz-789"]]
    assert matching == []
