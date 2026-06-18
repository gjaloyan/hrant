"""schedule_message must be a BASE tool, not gated behind the `admin` bundle.

Root cause of "reminders don't work / the agent hand-rolls scheduling": the
tool was in the loadable `admin` bundle, so a reminder turn didn't have it
available unless the model first called load_tool_bundle("admin"). It didn't —
it loaded the telegram skill (which says "use schedule_message") but not the
bundle, so the tool wasn't offered and the model re-implemented scheduling via
raw terminal_exec/run_python (slow, fragile, sometimes a useless outbox file).
A self-reminder is a core, common, owner-gated action — it belongs in BASE.
"""
from __future__ import annotations

from backend.tool_bundles import BASE_TOOLS, TOOL_BUNDLES


def test_schedule_message_is_base():
    assert "schedule_message" in BASE_TOOLS


def test_schedule_message_not_gated_behind_admin():
    assert "schedule_message" not in TOOL_BUNDLES.get("admin", [])
