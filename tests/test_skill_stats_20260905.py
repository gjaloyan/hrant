"""Reading back what the turns say about the skills they used."""
from backend.skill_stats import summarise


def _t(skills, conf, tools=()):
    return {"skills_used": list(skills), "confidence": conf,
            "tools_used": list(tools)}


def test_a_skill_is_counted_and_scored():
    rows = summarise([
        _t(["calc"], 90), _t(["calc"], 80), _t([], 60), _t([], 40),
    ])
    calc = next(r for r in rows if r["skill"] == "calc")
    assert calc["turns"] == 2
    assert calc["mean_confidence"] == 85.0
    assert calc["baseline_confidence"] == 50.0


def test_the_baseline_excludes_turns_that_used_ANY_skill():
    """Comparing against turns that used a different skill measures the
    two against each other, not against ordinary work."""
    rows = summarise([
        _t(["calc"], 90), _t(["summarize_pdf"], 10), _t([], 50),
    ])
    calc = next(r for r in rows if r["skill"] == "calc")
    assert calc["baseline_confidence"] == 50.0


def test_skills_are_ordered_by_how_much_they_were_used():
    rows = summarise([_t(["a"], 50), _t(["b"], 50), _t(["b"], 50)])
    assert [r["skill"] for r in rows] == ["b", "a"]


def test_a_skill_nobody_loaded_is_not_invented():
    assert summarise([_t([], 50)]) == []


def test_turns_with_no_confidence_do_not_poison_the_mean():
    rows = summarise([
        _t(["calc"], 80), {"skills_used": ["calc"]}, _t([], 40),
    ])
    calc = rows[0]
    assert calc["turns"] == 2
    assert calc["mean_confidence"] == 80.0
    assert calc["scored_turns"] == 1


def test_the_result_says_this_is_not_causal():
    """A skill loads on the turns that needed it, so a lower mean can
    mean a harder problem rather than a worse skill. The number must
    carry that or it will be read as a verdict."""
    rows = summarise([_t(["calc"], 90), _t([], 50)])
    assert "observational" in rows[0]["caveat"].lower()


def test_an_empty_history_is_not_an_error():
    assert summarise([]) == []
