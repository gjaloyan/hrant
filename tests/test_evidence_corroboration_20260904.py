"""The web is not right either — so say what the evidence actually is.

The corrective asked for "a primary source over an aggregator" and
nothing checked it, which makes it a wish rather than a rule. Judging
primacy generically is not possible: cba.am is primary for an exchange
rate and useless for grain moisture meters.

What IS computable is the shape of the evidence: how many distinct
domains support a claim, and which. A claim resting on one domain is not
corroborated, and a search whose first hit is a marketplace is not the
same as one answered by a manufacturer — measured on the real search for
"прибор для измерения влажности зерна", the top hit was ozon.ru.

So the block states the facts and lets the model weigh them, rather than
ranking sources on its behalf.
"""
from backend import unified_agent as ua


HITS = (
    '[{"title": "OZON", "url": "https://www.ozon.ru/category/x/", '
    '"snippet": "прибор для влажности"},'
    ' {"title": "Fizepr", "url": "https://fizepr.ru/vlagomery-zerna", '
    '"snippet": "влагомер зерна"},'
    ' {"title": "Ozon again", "url": "https://ozon.ru/other", '
    '"snippet": "ещё"}]'
)


def test_distinct_domains_are_counted_not_hits():
    ev = ua._evidence_for_claim("влагомер", HITS)
    assert ev["domains"] == ["ozon.ru", "fizepr.ru"]


def test_www_and_subdomains_collapse_to_one_source():
    ev = ua._evidence_for_claim("x", '[{"url": "https://www.cba.am/a"},'
                                     ' {"url": "https://cba.am/b"},'
                                     ' {"url": "https://news.cba.am/c"}]')
    assert ev["domains"] == ["cba.am"]


def test_a_two_part_suffix_is_not_mistaken_for_the_domain():
    ev = ua._evidence_for_claim("x", '[{"url": "https://bbc.co.uk/a"},'
                                     ' {"url": "https://gov.co.uk/b"}]')
    assert ev["domains"] == ["bbc.co.uk", "gov.co.uk"]


def test_a_single_source_claim_is_marked_as_uncorroborated():
    block = ua._render_evidence([
        ua._evidence_for_claim("solo", '[{"url": "https://only.com/a", '
                                       '"snippet": "s"}]'),
    ])
    low = block.lower()
    assert "1 source" in low or "one source" in low
    assert "only.com" in block


def test_the_block_names_the_domains_it_found():
    block = ua._render_evidence([ua._evidence_for_claim("влагомер", HITS)])
    assert "ozon.ru" in block and "fizepr.ru" in block
    assert "влагомер" in block


def test_the_block_tells_the_model_what_to_do_with_disagreement():
    block = ua._render_evidence([ua._evidence_for_claim("x", HITS)])
    low = block.lower()
    assert "disagree" in low or "conflict" in low


def test_unparseable_search_output_yields_no_evidence_not_a_crash():
    assert ua._evidence_for_claim("x", "not json")["domains"] == []
    assert ua._evidence_for_claim("x", "")["domains"] == []
    assert ua._render_evidence([]) == ""


def test_the_full_path_corrective_says_the_same_thing():
    """Two paths giving different advice about sources is how they drift
    apart. The lane shows the domains; the full path sees them itself and
    is told the same rule."""
    from unittest.mock import patch

    with patch("backend.endpoint_check.unbacked_action_claim", return_value=""), \
         patch.object(ua, "_ungrounded_factual_claims",
                      return_value=ua._ClaimCheck(["a claim"], "checked")):
        _, corrective = ua._decide_self_correction(
            task="q", answer="a claim", turn_tools=[])
    low = corrective.lower()
    assert "corroboration" in low
    assert "disagree" in low
