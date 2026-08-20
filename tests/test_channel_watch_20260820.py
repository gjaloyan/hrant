"""Following a channel means keeping what it posted, not glancing at it.

The owner asked the agent to subscribe to a public Telegram channel,
collect the updates, and produce a daily review. A bot cannot subscribe —
Telegram pushes channel posts only to bots that administer the channel,
and this is someone else's. What a public channel does expose is
`t.me/s/<name>`, an HTML page of recent posts with stable ids.

The page holds only ~16 posts. A digest that read it once a day would
silently lose whatever a busier channel published in between, and the
summary would still look complete — which is the failure worth guarding
against, because it is invisible from the outside.

So a background poll appends new posts to a ledger, and the digest reads
the ledger. These tests pin the two properties that make that honest:
nothing already seen is stored twice, and nothing published between polls
is dropped.

Fixture HTML below is trimmed from the real COIN22T page.
"""
import json

import pytest

from backend import channel_watch as cw


PAGE = '''
<div class="tgme_widget_message_wrap">
 <div class="tgme_widget_message" data-post="COIN22T/5857">
  <div class="tgme_widget_message_text js-message_text">First post<br/>second line</div>
  <time datetime="2026-08-17T05:26:01+00:00"></time>
 </div>
</div>
<div class="tgme_widget_message_wrap">
 <div class="tgme_widget_message" data-post="COIN22T/5858">
  <div class="tgme_widget_message_text js-message_text">&#1055;&#1088;&#1080;&#1074;&#1077;&#1090; &amp; co</div>
  <time datetime="2026-08-17T09:22:26+00:00"></time>
 </div>
</div>
<div class="tgme_widget_message_wrap">
 <div class="tgme_widget_message" data-post="COIN22T/5859">
  <time datetime="2026-08-17T16:27:23+00:00"></time>
 </div>
</div>
'''


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """The ledger is a real file; tests must not touch the deployment's."""
    monkeypatch.setattr(cw, "_dir", lambda: tmp_path)
    yield


# ── reading the page ────────────────────────────────────────────────

def test_posts_are_parsed_with_id_time_and_text():
    posts = cw.parse_posts(PAGE, "COIN22T")
    assert [p["id"] for p in posts] == [5857, 5858, 5859]
    assert posts[0]["text"].startswith("First post")
    assert posts[0]["ts"] == "2026-08-17T05:26:01+00:00"


def test_line_breaks_and_entities_survive():
    posts = cw.parse_posts(PAGE, "COIN22T")
    assert "\n" in posts[0]["text"]
    assert "&" in posts[1]["text"] and "&amp;" not in posts[1]["text"]


def test_a_media_only_post_is_kept_not_dropped():
    """Silence in the ledger would make the digest disagree with the
    channel about whether anything happened."""
    posts = cw.parse_posts(PAGE, "COIN22T")
    media = [p for p in posts if p["id"] == 5859][0]
    assert media["has_media"] is True
    assert media["text"] == ""


def test_posts_carry_a_link_back_to_the_original():
    assert cw.parse_posts(PAGE, "COIN22T")[0]["link"].endswith("/COIN22T/5857")


def test_another_channels_posts_are_ignored():
    """The page embeds forwards and recommendations from other channels."""
    mixed = PAGE + '<div data-post="SOMEONEELSE/1">' \
                   '<div class="tgme_widget_message_text js-message_text">x</div></div>'
    assert all(p["channel"] == "COIN22T" for p in cw.parse_posts(mixed, "COIN22T"))


def test_a_garbage_page_yields_nothing_rather_than_raising():
    assert cw.parse_posts("<html>nope</html>", "COIN22T") == []
    assert cw.parse_posts("", "COIN22T") == []


# ── naming a channel ────────────────────────────────────────────────

@pytest.mark.parametrize("given", [
    "COIN22T", "@COIN22T", "https://t.me/COIN22T",
    "https://t.me/s/COIN22T", "t.me/COIN22T", " COIN22T ",
])
def test_every_way_of_writing_the_channel_resolves(given):
    assert cw.normalize_channel(given) == "COIN22T"


# ── collecting without duplicating or losing ────────────────────────

def _poll_with(monkeypatch, page):
    monkeypatch.setattr(cw, "fetch_page", lambda channel, before=None: page)
    return cw.poll("COIN22T")


def test_a_first_poll_stores_everything_on_the_page(monkeypatch):
    out = _poll_with(monkeypatch, PAGE)
    assert out["new"] == 3
    assert len(cw.unreviewed("COIN22T")) == 3


def test_polling_again_stores_nothing_twice(monkeypatch):
    _poll_with(monkeypatch, PAGE)
    out = _poll_with(monkeypatch, PAGE)
    assert out["new"] == 0
    assert len(cw.unreviewed("COIN22T")) == 3


def test_posts_published_between_polls_are_kept(monkeypatch):
    """The whole reason for a ledger: the page is a window, not a record."""
    _poll_with(monkeypatch, PAGE)
    later = PAGE.replace("COIN22T/5859", "COIN22T/5999")
    _poll_with(monkeypatch, later)
    ids = [p["id"] for p in cw.unreviewed("COIN22T")]
    assert 5857 in ids and 5999 in ids, ids


def test_an_unreachable_channel_does_not_raise(monkeypatch):
    def _boom(channel, before=None):
        raise OSError("network down")
    monkeypatch.setattr(cw, "fetch_page", _boom)
    out = cw.poll("COIN22T")
    assert out["new"] == 0
    assert "network down" in out["error"]


# ── what the digest sees ────────────────────────────────────────────

def test_reviewing_advances_the_watermark(monkeypatch):
    _poll_with(monkeypatch, PAGE)
    cw.mark_reviewed("COIN22T", 5858)
    assert [p["id"] for p in cw.unreviewed("COIN22T")] == [5859]


def test_reviewed_posts_are_kept_not_deleted(monkeypatch):
    """A digest that failed halfway must be re-runnable, and the owner has
    to be able to see what the agent actually had."""
    _poll_with(monkeypatch, PAGE)
    cw.mark_reviewed("COIN22T", 5859)
    assert cw.unreviewed("COIN22T") == []
    assert len(cw._read_posts("COIN22T")) == 3


def test_the_watermark_never_moves_backwards(monkeypatch):
    _poll_with(monkeypatch, PAGE)
    cw.mark_reviewed("COIN22T", 5859)
    cw.mark_reviewed("COIN22T", 5857)
    assert cw.unreviewed("COIN22T") == []


def test_digest_input_reports_unreadable_media(monkeypatch):
    _poll_with(monkeypatch, PAGE)
    out = cw.digest_input("COIN22T")
    assert out["count"] == 3
    assert out["with_media_only"] == 1
    assert out["latest_id"] == 5859


def test_nothing_new_is_an_empty_answer_not_an_error(monkeypatch):
    _poll_with(monkeypatch, PAGE)
    cw.mark_reviewed("COIN22T", 5859)
    out = cw.digest_input("COIN22T")
    assert out["count"] == 0 and out["posts"] == []


# ── the tool and the poller ─────────────────────────────────────────

def test_the_tool_marks_reviewed_so_the_next_digest_moves_on(monkeypatch):
    from backend import builtin_tools as bt
    _poll_with(monkeypatch, PAGE)
    first = json.loads(bt._channel_updates_handler(channel="COIN22T"))
    assert first["count"] == 3
    second = json.loads(bt._channel_updates_handler(channel="COIN22T"))
    assert second["count"] == 0, "the digest would repeat itself every day"


def test_the_tool_can_look_without_marking(monkeypatch):
    from backend import builtin_tools as bt
    _poll_with(monkeypatch, PAGE)
    bt._channel_updates_handler(channel="COIN22T", mark_reviewed=False)
    assert json.loads(
        bt._channel_updates_handler(channel="COIN22T"))["count"] == 3


def test_the_tool_description_warns_against_fetching_the_page_instead():
    from backend import builtin_tools as bt
    from backend.tool_registry import get_registry
    bt.register_builtin_tools()
    d = get_registry().tools["channel_updates"].description
    assert "fetch_url" in d and "silently loses the rest" in d


def test_the_poller_costs_no_llm_call():
    """Polling every ten minutes through a model would be pure waste."""
    import inspect
    from backend.autonomic.levers.channel_watch import FIRE_CHANNEL_WATCH
    src = inspect.getsource(FIRE_CHANNEL_WATCH)
    assert "router" not in src and "llm" not in src.lower()


def test_the_poller_stays_idle_when_no_channel_is_followed():
    from backend.autonomic.levers.channel_watch import FIRE_CHANNEL_WATCH
    assert FIRE_CHANNEL_WATCH().preconditions(None) is False


def test_the_lever_actually_runs_and_returns_a_valid_report(monkeypatch):
    """Executing it, not just reading its source.

    The first version built its LeverReport with a `summary=` field that
    does not exist on the dataclass. Every source-level test passed; the
    lever crashed the first time the scheduler reached it, and the daily
    digest would have had an empty ledger. Construct the report for real.
    """
    from backend.autonomic.levers.channel_watch import FIRE_CHANNEL_WATCH
    monkeypatch.setattr(cw, "watched", lambda: ["COIN22T"])
    monkeypatch.setattr(cw, "fetch_page", lambda channel, before=None: PAGE)
    report = FIRE_CHANNEL_WATCH().run({}, {})
    assert report.status.value == "success"
    assert report.outcome["new_posts"] == 3
    assert report.lever == "FIRE_CHANNEL_WATCH"


def test_the_lever_reports_a_failure_without_raising(monkeypatch):
    from backend.autonomic.levers.channel_watch import FIRE_CHANNEL_WATCH
    def _boom():
        raise RuntimeError("store unavailable")
    monkeypatch.setattr(cw, "poll_all", _boom)
    report = FIRE_CHANNEL_WATCH().run({}, {})
    assert report.status.value == "failure"
    assert "RuntimeError" in report.reason


def test_the_report_survives_a_round_trip_to_the_log():
    """It is written as JSONL and read back by the audit tooling; a field
    the dataclass rejects only shows up at that boundary."""
    from backend.autonomic.types import LeverReport
    from backend.autonomic.levers.channel_watch import FIRE_CHANNEL_WATCH
    report = FIRE_CHANNEL_WATCH().run({}, {})
    assert LeverReport.from_jsonl(report.to_jsonl()).lever == "FIRE_CHANNEL_WATCH"
