"""Unit tests for city-driven agent generation."""

from __future__ import annotations

import pytest

from surrogate_safety_abm.agents import VehicleAgent
from surrogate_safety_abm.agents.vehicle import VehicleType
from surrogate_safety_abm.config.city_profiles import (
    AHMEDABAD,
    CITY_PROFILES,
    SURAT,
    VADODARA,
)
from surrogate_safety_abm.simulation.scenario import (
    generate_agents,
    make_city_behaviours,
)


def test_generate_agents_returns_requested_count() -> None:
    """Asking for N vehicles gives exactly N agents."""
    agents = generate_agents(VADODARA, seed=42, n_vehicles=20)
    assert len(agents) == 20


def test_generate_agents_deterministic_for_seed() -> None:
    """Same seed -> identical fleet (types, speeds, positions)."""
    a = generate_agents(VADODARA, seed=42, n_vehicles=15)
    b = generate_agents(VADODARA, seed=42, n_vehicles=15)
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert x.agent_id == y.agent_id
        assert x.vehicle_type == y.vehicle_type
        assert x.position.x == pytest.approx(y.position.x)
        assert x.position.y == pytest.approx(y.position.y)
        assert x.speed == pytest.approx(y.speed)


def test_generate_agents_unique_ids() -> None:
    """Every agent gets a distinct agent_id."""
    agents = generate_agents(SURAT, seed=7, n_vehicles=30)
    ids = [a.agent_id for a in agents]
    assert len(ids) == len(set(ids))


def test_generate_agents_types_match_profile() -> None:
    """With 200 agents, observed type proportions roughly match the profile."""
    agents = generate_agents(SURAT, seed=42, n_vehicles=200)
    counts: dict[VehicleType, int] = {vt: 0 for vt in VehicleType}
    for a in agents:
        assert isinstance(a, VehicleAgent)
        counts[a.vehicle_type] += 1
    total = sum(counts.values())
    for vt, expected_share in SURAT.pcu_composition.items():
        observed_share = counts[vt] / total
        # sampling noise at n=200: allow +/-0.12 slack
        assert abs(observed_share - expected_share) < 0.12, (
            f"{vt}: observed {observed_share:.2f} vs expected {expected_share:.2f}"
        )


def test_generate_agents_speeds_within_max() -> None:
    """Sampled speeds must not exceed each agent's max_speed."""
    agents = generate_agents(AHMEDABAD, seed=11, n_vehicles=40)
    for a in agents:
        assert 0.0 < a.speed <= a.max_speed


def test_make_city_behaviours_returns_standard_stack() -> None:
    """The default behaviour stack is gap-acceptance then evasive-braking."""
    from surrogate_safety_abm.behaviour import (
        EvasiveBrakingModel,
        GapAcceptanceModel,
    )

    behaviours = make_city_behaviours(VADODARA)
    assert len(behaviours) == 2
    assert isinstance(behaviours[0], GapAcceptanceModel)
    assert isinstance(behaviours[1], EvasiveBrakingModel)


def test_city_profiles_all_registered() -> None:
    """All three study cities are exposed via CITY_PROFILES."""
    assert set(CITY_PROFILES.keys()) == {"vadodara", "surat", "ahmedabad"}
    assert CITY_PROFILES["vadodara"] is VADODARA
    assert CITY_PROFILES["surat"] is SURAT
    assert CITY_PROFILES["ahmedabad"] is AHMEDABAD
