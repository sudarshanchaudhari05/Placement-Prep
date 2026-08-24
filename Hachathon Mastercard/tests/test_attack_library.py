"""Unit tests for Attack Intelligence Library."""

import pytest
from src.attacks.attack_library import (
    AttackLibrary,
    AttackArchetype,
    ATTACK_CATALOG,
    get_default_attack_library,
)


def test_attack_catalog_count():
    """Verify library contains between 25 and 30 curated GenAI attack archetypes."""
    library = get_default_attack_library()
    assert 25 <= len(library) <= 30
    assert len(library.get_all()) == len(ATTACK_CATALOG)


def test_archetype_required_fields():
    """Verify every archetype has non-empty required fields and valid schema."""
    library = get_default_attack_library()
    valid_severities = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    for atk in library.get_all():
        assert atk.attack_id.startswith("ATK-")
        assert len(atk.name) > 0
        assert len(atk.category) > 0
        assert len(atk.description) > 15
        assert atk.severity.upper() in valid_severities
        assert 0.0 <= atk.novelty_score <= 1.0
        assert 0.0 <= atk.detectability_score <= 1.0
        assert isinstance(atk.behavioral_indicators, list)
        assert len(atk.behavioral_indicators) > 0
        assert len(atk.affected_payment_surface) > 0
        assert isinstance(atk.simulation_parameters, dict)


def test_library_lookup_by_id():
    """Test retrieval by unique attack_id."""
    library = get_default_attack_library()
    atk = library.get_by_id("ATK-001")
    assert atk is not None
    assert "Voice Clone" in atk.name
    assert library.get_by_id("NON_EXISTENT_ID") is None


def test_library_lookup_by_name():
    """Test retrieval by archetype name."""
    library = get_default_attack_library()
    atk = library.get_by_name("Deepfake Video KYC Onboarding Bypass")
    assert atk is not None
    assert atk.attack_id == "ATK-005"


def test_filtering_and_dataframe_export():
    """Test filtering by category, severity, and DataFrame export."""
    library = get_default_attack_library()
    categories = library.list_categories()
    assert len(categories) >= 5

    critical_attacks = library.filter_by_severity("CRITICAL")
    assert len(critical_attacks) > 0

    df = library.to_dataframe()
    assert len(df) == len(library)
    assert "attack_id" in df.columns
    assert "simulation_parameters" in df.columns
