"""Pin the concurrent_updates(True) wiring on the Telegram Application.

Root cause of the May 19 "telegram buttons dont work" prod incident:
ApplicationBuilder was constructed without `.concurrent_updates(True)`,
so PTB processed updates sequentially. When the agent was mid-tool-call
on a chat message, any inline-button tap that arrived would queue
behind the chat handler. By the time the queue drained, the callback
query had expired server-side (~15s) and every `answerCallbackQuery`
returned HTTP 400. From the user's side: buttons "don't work" —
spinner never stops.

The fix is one line: `.concurrent_updates(True)` between
`.token(...)` and `.build()`. Pin it here so it can't regress
silently.
"""
from __future__ import annotations

import textwrap


def test_telegram_application_wired_with_concurrent_updates():
    """The ApplicationBuilder call must include concurrent_updates(True).
    Greps the actual source — light-touch but catches a regression that
    would otherwise only surface when a user taps a button mid-task."""
    import inspect
    from backend import channels
    src = inspect.getsource(channels)
    assert "concurrent_updates(True)" in src, (
        "ApplicationBuilder must call .concurrent_updates(True). "
        "Without it, callback queries queue behind chat handlers and "
        "expire before the bot can answer — buttons 'don't work'."
    )


def test_telegram_concurrent_updates_comment_explains_why():
    """The next dev who reads this code needs to know WHY this flag
    exists. Without the comment it's just a magic call easy to drop."""
    import inspect
    from backend import channels
    src = inspect.getsource(channels)
    # Either the word 'concurrent_updates' appears near 'callback' or
    # we explain the symptom in the comment block.
    relevant = src[
        max(0, src.find("concurrent_updates(True)") - 600):
        src.find("concurrent_updates(True)") + 200
    ]
    low = relevant.lower()
    assert "callback" in low
    assert "expire" in low or "queue" in low or "sequential" in low, (
        "the why-block above .concurrent_updates(True) should mention "
        "the queue / expire mechanism so a future dev doesn't strip it"
    )
