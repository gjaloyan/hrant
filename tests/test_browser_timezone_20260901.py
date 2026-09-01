"""Websites must see the owner's clock, not the server's.

Measured on prod 2026-09-01: a page evaluated
`Intl.DateTimeFormat().resolvedOptions().timeZone` and got "UTC" while it
was 13:00 in Yerevan. Anything rendering "today" -- a schedule, a booking
form, a news feed -- was four hours behind, and wrong about the DAY every
evening after 20:00 local. The owner put it plainly: "utc in armenia is not
good variant for web surfing."

The fix belongs to the browser process, not the system clock: the box stays
on UTC so logs and stored timestamps stay unambiguous.
"""
import backend.tools.agent_browser as ab


def _env_for(monkeypatch, zone="Asia/Yerevan"):
    """Run the tool far enough to capture the env it hands the browser."""
    seen = {}

    class _R:
        returncode, stdout, stderr = 0, "{}", ""

    def _fake_run(cmd, **kw):
        seen.update(kw.get("env") or {})
        return _R()

    monkeypatch.setattr(ab.subprocess, "run", _fake_run)
    monkeypatch.setattr("backend.settings.user_timezone", lambda: zone)
    monkeypatch.setattr(ab, "_resolve_binary", lambda *a, **k: "/bin/true")
    try:
        ab.run_agent_browser("open https://example.com", timeout_seconds=5)
    except Exception:
        pass
    return seen


def test_browser_gets_the_owners_zone(monkeypatch):
    env = _env_for(monkeypatch)
    assert env.get("TZ") == "Asia/Yerevan", (
        "the browser would present the server's UTC clock to every site")


def test_it_follows_the_setting_rather_than_hardcoding_yerevan(monkeypatch):
    # The owner may move, or another user may run this. The zone comes from
    # settings; a hardcoded "Asia/Yerevan" would pass the test above and be
    # wrong everywhere else.
    env = _env_for(monkeypatch, zone="Europe/Lisbon")
    assert env.get("TZ") == "Europe/Lisbon"


def test_the_server_clock_is_not_touched():
    # The fix must stay inside the child process env. Moving the system
    # clock would reinterpret every naive timestamp in the codebase.
    src = (ab.__file__ and open(ab.__file__, encoding="utf-8").read()) or ""
    assert "timedatectl" not in src
    assert 'os.environ["TZ"]' not in src
