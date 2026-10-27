"""A reply, a quote, a forward: what the message points at reaches the agent.

2026-10-23: the owner replied to the evening market review with "что по ПП?
покупать или нет?". The handler read only his text, so the agent never saw
the review that named PEP and PONS side by side, and asked twice which coin
he meant.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

from backend.channels import reply_context

BOT_ID = 7000
ME = NS(id=848, full_name="Gor Jaloyan", is_bot=False)
BOT = NS(id=BOT_ID, full_name="Hrant", is_bot=True)


def _msg(**kw):
    base = dict(from_user=ME, reply_to_message=None, quote=None,
                external_reply=None, forward_origin=None)
    base.update(kw)
    return NS(**base)


def test_a_reply_to_the_bot_carries_the_message_it_answers():
    review = NS(from_user=BOT, text="📊 PEP: no contract yet. PONS: −3%.", caption=None)
    ctx = reply_context(_msg(reply_to_message=review), bot_user_id=BOT_ID)
    assert "your own earlier message" in ctx and "PEP: no contract yet" in ctx


def test_a_quote_wins_over_the_whole_message():
    review = NS(from_user=BOT, text="Long review … PEP: no contract yet … more", caption=None)
    ctx = reply_context(_msg(reply_to_message=review, quote=NS(text="PEP: no contract yet")),
                        bot_user_id=BOT_ID)
    assert "quoting this part" in ctx and "«PEP: no contract yet»" in ctx
    assert "Long review" not in ctx


def test_a_reply_to_a_photo_says_it_is_attached():
    photo_msg = NS(from_user=ME, text=None, caption=None, photo=[NS(file_id="x")])
    ctx = reply_context(_msg(reply_to_message=photo_msg), bot_user_id=BOT_ID)
    assert "their own earlier message" in ctx and "contains a photo" in ctx


def test_a_forward_names_its_origin():
    origin = NS(sender_user=NS(full_name="Tigran"), sender_chat=None, chat=None)
    ctx = reply_context(_msg(forward_origin=origin), bot_user_id=BOT_ID)
    assert "originally from Tigran" in ctx


def test_a_plain_message_adds_nothing():
    assert reply_context(_msg(), bot_user_id=BOT_ID) == ""


def test_the_handler_uses_it_and_fetches_the_replied_media():
    import inspect
    from backend import channels
    src = inspect.getsource(channels)
    assert "_ctx = reply_context(" in src
    assert "msg=_replied, echo_voice=False" in src
