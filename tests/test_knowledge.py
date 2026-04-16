def test_save_and_load_note(tmp_kb):
    note = tmp_kb.save_note(
        topic="RS-485",
        body="# RS-485\n\nдифференциальный последовательный интерфейс.",
        category="profession",
        keywords=["rs485", "serial", "modbus"],
        source="test",
        confidence="verified",
    )
    assert note.frontmatter.topic == "RS-485"
    loaded = tmp_kb.get_note("RS-485")
    assert loaded is not None
    assert "дифференциальный" in loaded.body


def test_index_updates(tmp_kb):
    tmp_kb.save_note(topic="MAX485", body="transceiver IC", category="profession",
                     keywords=["max485"], source="t")
    topics = tmp_kb.list_topics()
    assert any(t.topic == "MAX485" for t in topics)


def test_access_counter(tmp_kb):
    tmp_kb.save_note(topic="OneWire", body="шина", category="profession",
                     keywords=["onewire"], source="t")
    tmp_kb.get_note("OneWire")
    tmp_kb.get_note("OneWire")
    tmp_kb.get_note("OneWire")
    assert tmp_kb.access_count("OneWire") == 3


def test_delete_note(tmp_kb):
    tmp_kb.save_note(topic="Temp", body="x", category="profession", source="t")
    assert tmp_kb.delete_note("Temp") is True
    assert tmp_kb.get_note("Temp") is None


def test_core_memory_add_and_remove(tmp_kb):
    from backend.core_memory import CoreMemory
    core = CoreMemory()
    core.add_fact("пользователь работает с промышленной автоматикой")
    assert "промышленной автоматикой" in core.read()
    core.remove_fact("промышленной автоматикой")
    assert "промышленной автоматикой" not in core.read()
