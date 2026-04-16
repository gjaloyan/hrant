"""Tests for note + identity version history.

Каждое сохранение заметки или user.md снимает предыдущую версию в
_history/. Это даёт возможность посмотреть, что было раньше, и откатиться.
"""
from __future__ import annotations
import time

from backend.identity import IdentityManager


# ---------- notes ----------
def test_first_save_creates_no_history(tmp_kb):
    tmp_kb.save_note(
        topic="RS-485",
        body="первое описание",
        category="profession",
        source="t",
    )
    # История пуста — нечего было снимать.
    assert tmp_kb.list_versions("RS-485") == []


def test_second_save_snapshots_previous_version(tmp_kb):
    tmp_kb.save_note(topic="RS-485", body="версия 1", category="profession", source="t")
    # Разнесём на осязаемый таймстамп, чтобы имя файла отличалось стабильно
    time.sleep(0.01)
    tmp_kb.save_note(topic="RS-485", body="версия 2", category="profession", source="t")

    versions = tmp_kb.list_versions("RS-485")
    assert len(versions) == 1
    # Актуальная версия — уже "версия 2" в файле.
    current = tmp_kb.get_note("RS-485")
    assert current is not None and "версия 2" in current.body

    # А в снапшоте — предыдущая.
    snapshot = tmp_kb.get_version("RS-485", index=0)
    assert snapshot is not None and "версия 1" in snapshot
    assert "версия 2" not in snapshot


def test_multiple_versions_ordered_newest_first(tmp_kb):
    tmp_kb.save_note(topic="Bus", body="v1", category="profession", source="t")
    time.sleep(0.01)
    tmp_kb.save_note(topic="Bus", body="v2", category="profession", source="t")
    time.sleep(0.01)
    tmp_kb.save_note(topic="Bus", body="v3", category="profession", source="t")

    versions = tmp_kb.list_versions("Bus")
    assert len(versions) == 2  # v1 и v2 (v3 — текущая)

    # Самая свежая запись в истории — это v2
    newest = tmp_kb.get_version("Bus", index=0)
    oldest = tmp_kb.get_version("Bus", index=1)
    assert newest is not None and "v2" in newest
    assert oldest is not None and "v1" in oldest


def test_get_version_out_of_range(tmp_kb):
    tmp_kb.save_note(topic="X", body="only", category="profession", source="t")
    assert tmp_kb.get_version("X", index=0) is None   # нет истории
    assert tmp_kb.get_version("X", index=99) is None
    assert tmp_kb.get_version("does_not_exist") is None


def test_access_log_does_not_create_history(tmp_kb):
    """log_access перезаписывает файл ради счётчика — но это не новая версия."""
    tmp_kb.save_note(topic="Y", body="content", category="profession", source="t")
    # Несколько get_note подряд увеличивают access_count и переписывают файл.
    tmp_kb.get_note("Y")
    tmp_kb.get_note("Y")
    tmp_kb.get_note("Y")
    # Но в истории ничего не появилось.
    assert tmp_kb.list_versions("Y") == []


# ---------- identity / user.md ----------
def test_user_profile_history_created_on_fact_add(tmp_kb, tmp_path):
    idm = IdentityManager(base_dir=tmp_path)
    # Исходно user.md дефолтный — истории нет.
    assert idm.list_user_versions() == []

    idm.add_user_fact("Говорит по-русски", category="language")
    # Снапшот дефолтного содержимого должен появиться.
    v1 = idm.list_user_versions()
    assert len(v1) == 1

    time.sleep(0.01)
    idm.add_user_fact("Любит краткость", category="style")
    v2 = idm.list_user_versions()
    assert len(v2) == 2

    # Старейший снапшот — это дефолтный user.md без фактов.
    oldest_path = v2[-1]["path"]
    from pathlib import Path
    oldest_text = Path(oldest_path).read_text(encoding="utf-8")
    assert "пока не указано" in oldest_text

    # Следующий — уже с первым фактом, без второго.
    middle_text = Path(v2[0]["path"]).read_text(encoding="utf-8")
    assert "Говорит по-русски" in middle_text
    assert "Любит краткость" not in middle_text

    # А сам user.md — с обоими.
    current = idm.user_profile()
    assert "Говорит по-русски" in current
    assert "Любит краткость" in current
