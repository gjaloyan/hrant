"""A page that MENTIONS a captcha was reported as blocked by one.

Measured 2026-09-04. `fetch_url("https://en.wikipedia.org/wiki/Duduk")`
returned "[blocked: HTTP 200 + anti-bot challenge] This is an ACCESS
failure..." — on a 200 response from which trafilatura had just
extracted 22,559 characters of article.

The bare substring "captcha" matched inside Wikipedia's own JavaScript
config: `wgConfirmEditCaptchaNeededForGenericEdit":"hcaptcha"`. So any
page whose head mentions the word — a privacy policy, a JS blob, an
article about captchas — told the agent the information was unreachable
and to try another source.

The real case this exists for stays intact: DuckDuckGo answering HTTP
202 with 14 KB of "Select all squares containing a duck" and no article
body at all. That is the discriminator — a challenge page has no
content to extract, and a page that yielded a real body is not one.
"""
from backend.tools.web_search import looks_like_bot_wall


WIKI_HEAD = (
    '<html><head><script>RLCONF={"wgConfirmEditCaptchaNeededForGenericEdit"'
    ':"hcaptcha","wgTitle":"Duduk"};</script></head><body>'
)
ARTICLE = (
    "The duduk is an ancient Armenian double reed woodwind instrument made "
    "of apricot wood. " * 40
)
CHALLENGE = (
    "<html><body><h1>Select all squares containing a duck</h1>"
    "<p>Verify you are human to continue.</p></body></html>"
)


def test_a_page_that_merely_mentions_captcha_is_not_a_wall():
    assert looks_like_bot_wall(WIKI_HEAD, extracted=ARTICLE) is False


def test_the_real_challenge_is_still_caught():
    """No article body, and the challenge text is the page."""
    assert looks_like_bot_wall(CHALLENGE, extracted=None) is True
    assert looks_like_bot_wall(CHALLENGE, extracted="") is True


def test_a_strong_marker_wins_even_with_a_body():
    """Some walls do render prose around the challenge. "Prove your
    humanity" is never incidental; "captcha" as a word often is."""
    body = "Please prove your humanity to continue. " * 30
    assert looks_like_bot_wall(
        "<p>prove your humanity</p>" + body, extracted=body) is True


def test_without_the_argument_nothing_changes():
    """Two other call sites pass raw search HTML and have no extraction
    to offer. They must behave exactly as before."""
    assert looks_like_bot_wall(CHALLENGE) is True
    assert looks_like_bot_wall("<p>ordinary page</p>") is False


def test_a_scrap_of_text_is_not_a_body():
    """A challenge page can still yield a line or two. It takes real
    content to overrule the marker."""
    assert looks_like_bot_wall(CHALLENGE, extracted="Verify you are human") is True
