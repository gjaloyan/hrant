"""First-boot seeding of pipeline profiles.

On first start (empty profiles root) we drop five illustrative
starter profiles + set 'default' active. On second start the seed
function must be a no-op (don't overwrite owner edits)."""
from __future__ import annotations

import time

import pytest


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    from backend import pipeline_profile as _pp
    monkeypatch.setattr(_pp, "_profiles_root", lambda: tmp_path)
    _pp.PROFILES._cache_loaded_at = 0.0
    _pp.PROFILES._cache_valid = False
    yield tmp_path


def test_first_boot_seeds_five_profiles(isolated_store):
    from backend.pipeline_profile import PROFILES, seed_starter_profiles
    seed_starter_profiles()
    ids = {p.id for p in PROFILES.list()}
    assert "default" in ids
    assert "benchmark" in ids
    assert "development" in ids
    assert "safe" in ids
    assert "solver" in ids


def test_first_boot_sets_default_active(isolated_store):
    from backend.pipeline_profile import PROFILES, seed_starter_profiles
    seed_starter_profiles()
    assert PROFILES.active_id() == "default"


def test_second_boot_is_idempotent(isolated_store):
    from backend.pipeline_profile import PROFILES, PipelineProfile, seed_starter_profiles
    seed_starter_profiles()
    # Owner edits the default profile.
    p = PROFILES.get("default")
    assert p is not None
    p.description = "OWNER EDIT"
    p.updated_at = time.time()
    PROFILES.put(p)
    # Boot again — must NOT overwrite the owner's edit.
    seed_starter_profiles()
    p2 = PROFILES.get("default")
    assert p2 is not None
    assert p2.description == "OWNER EDIT"


def test_default_profile_has_empty_overrides(isolated_store):
    from backend.pipeline_profile import PROFILES, seed_starter_profiles
    seed_starter_profiles()
    p = PROFILES.get("default")
    assert p is not None
    assert p.engine_overrides == {}
    assert p.reasoning_overrides == {}
    assert p.prompt_overrides == {}
    assert p.logging_overrides == {}


def test_starter_profiles_pass_validation(isolated_store):
    """Each seeded profile must be a valid overlay (would survive a
    PUT through the API). A bad starter would be a footgun."""
    from backend.pipeline_profile import PROFILES, seed_starter_profiles, validate
    seed_starter_profiles()
    for p in PROFILES.list():
        errors = validate(p.to_dict())
        assert errors == [], f"profile {p.id} has errors: {errors}"
