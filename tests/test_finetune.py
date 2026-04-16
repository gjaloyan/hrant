"""Тесты для finetune store, curator, category detection."""
from backend.finetune import detect_category, store as finetune_store
from backend.finetune_curator import FinetuneDataCurator
from backend.models import FinetunePair, FinetuneMetadata, ChatMessage


# ---------- detect_category ----------
def test_detect_troubleshooting():
    cat = detect_category(
        "RS-485 не работает после 100м, что делать?",
        "проверь терминаторы 120 Ом",
    )
    assert cat == "troubleshooting"


def test_detect_procedure():
    cat = detect_category("как подключить DS18B20 к Arduino?", "1) ...")
    assert cat == "procedure"


def test_detect_factual():
    cat = detect_category("какое максимальное напряжение MAX485?", "12V")
    assert cat == "factual_qa"


def test_detect_decision():
    cat = detect_category("почему выбрали Modbus RTU а не TCP?", "потому что...")
    assert cat == "decision"


# ---------- FinetuneStore CRUD ----------
def test_store_add_and_count(tmp_kb):
    s = finetune_store()
    s.add(
        question="как подключить MAX485?",
        answer="DI к TX, RO к RX, DE+RE к D2.",
        source_notes=["MAX485", "RS-485"],
        confidence=92,
        project=None,
    )
    assert s.count() == 1
    examples = s.list_all()
    assert len(examples) == 1
    assert examples[0].metadata.confidence == 92
    assert examples[0].metadata.source_notes == ["MAX485", "RS-485"]


def test_store_id_unique(tmp_kb):
    s = finetune_store()
    p1 = s.add(question="Q1", answer="A1", source_notes=[], confidence=90, project=None)
    p2 = s.add(question="Q2", answer="A2", source_notes=[], confidence=90, project=None)
    assert p1.id != p2.id
    assert len(p1.id) == 12


def test_store_edit(tmp_kb):
    s = finetune_store()
    p = s.add(question="Q", answer="old", source_notes=[], confidence=90, project=None)
    assert s.edit(p.id, assistant="new answer")
    fetched = s.get(p.id)
    assert fetched.assistant_text() == "new answer"


def test_store_boost(tmp_kb):
    s = finetune_store()
    p = s.add(question="Q", answer="A", source_notes=[], confidence=90, project=None)
    assert s.boost(p.id)
    assert s.get(p.id).metadata.boosted is True


def test_store_delete(tmp_kb):
    s = finetune_store()
    p = s.add(question="Q", answer="A", source_notes=[], confidence=90, project=None)
    assert s.delete(p.id)
    assert s.count() == 0


def test_correction_category(tmp_kb):
    s = finetune_store()
    p = s.add_correction(
        question="напряжение датчика?",
        wrong_answer="12VDC",
        corrected_answer="24VDC",
    )
    assert p.metadata.category == "correction"
    assert p.metadata.confidence == 100
    assert p.metadata.original_wrong_answer == "12VDC"


# ---------- maybe_add_from_agent (auto-collect rules) ----------
def test_auto_collect_high_confidence(tmp_kb):
    s = finetune_store()
    pair = s.maybe_add_from_agent(
        question="детальный технический вопрос про MAX485",
        answer="ответ с источниками",
        source_notes=["MAX485"],
        confidence=92,
        is_verified=True,
        project=None,
    )
    assert pair is not None
    assert s.count() == 1


def test_auto_collect_low_confidence_skipped(tmp_kb):
    s = finetune_store()
    pair = s.maybe_add_from_agent(
        question="детальный технический вопрос про MAX485",
        answer="ответ",
        source_notes=["MAX485"],
        confidence=70,
        is_verified=True,
        project=None,
    )
    assert pair is None
    assert s.count() == 0


def test_auto_collect_no_sources_skipped(tmp_kb):
    s = finetune_store()
    pair = s.maybe_add_from_agent(
        question="детальный вопрос",
        answer="ответ",
        source_notes=[],
        confidence=95,
        is_verified=True,
        project=None,
    )
    assert pair is None


def test_auto_collect_unverified_skipped(tmp_kb):
    s = finetune_store()
    pair = s.maybe_add_from_agent(
        question="детальный вопрос",
        answer="ответ",
        source_notes=["X"],
        confidence=95,
        is_verified=False,
        project=None,
    )
    assert pair is None


# ---------- curator ----------
def _mk_pair(*, q="Q", a="A" * 100, conf=90, cat="factual_qa", boosted=False, sources=None):
    return FinetunePair(
        id="x",
        messages=[
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content=q),
            ChatMessage(role="assistant", content=a),
        ],
        metadata=FinetuneMetadata(
            confidence=conf,
            source_notes=sources or ["RS-485"],
            category=cat,
            boosted=boosted,
        ),
    )


def test_curator_quality_score():
    c = FinetuneDataCurator()
    high = _mk_pair(cat="correction", boosted=True)
    low = _mk_pair(q="x", a="y", conf=50, cat="other", sources=[])
    assert c.quality_score(high) > c.quality_score(low)
    assert c.quality_score(high) >= 0.8


def test_curator_filters_low_quality():
    c = FinetuneDataCurator()
    good = _mk_pair(cat="troubleshooting", q="problem with RS-485")
    bad = _mk_pair(q="b", a="x", conf=10, cat="other", sources=[])
    out = c.curate([good, bad])
    assert len(out) == 1
    assert out[0].metadata.category == "troubleshooting"


def test_curator_dedup():
    c = FinetuneDataCurator()
    p1 = _mk_pair(q="как подключить MAX485 к Arduino", cat="procedure")
    p2 = _mk_pair(q="как подключить MAX485 к Arduino Uno", cat="procedure")
    out = c.curate([p1, p2])
    assert len(out) == 1


def test_curator_boosting():
    c = FinetuneDataCurator()
    p = _mk_pair(cat="correction", boosted=True)
    out = c.apply_boosting([p])
    # correction + boosted → 3 копии
    assert len(out) == 3
