"""Unit tests for behavioural models."""

import math

import numpy as np
import pytest

from surrogate_safety_abm.agents import VehicleAgent, VehicleType
from surrogate_safety_abm.behaviour import (
    EvasiveBrakingModel,
    GapAcceptanceModel,
    sample_critical_gap_s,
    sample_speed_mps,
)
from surrogate_safety_abm.config.city_profiles import SURAT, VADODARA
from surrogate_safety_abm.environment import Intersection
from surrogate_safety_abm.geometry import ORIGIN, Vec2


class TestSpeedSampling:
    """Tests for sample_speed_mps and sample_critical_gap_s."""

    def test_speed_within_envelope(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(100):
            v = sample_speed_mps(SURAT, rng)
            v85_ms = SURAT.approach_speed_85_kph / 3.6
            assert 0.3 * v85_ms <= v <= 1.1 * v85_ms

    def test_speed_differs_between_cities(self) -> None:
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        surat_speeds = [sample_speed_mps(SURAT, rng1) for _ in range(50)]
        vad_speeds = [sample_speed_mps(VADODARA, rng2) for _ in range(50)]
        # Surat v85 = 50 kph > Vadodara v85 = 45 kph
        assert np.mean(surat_speeds) > np.mean(vad_speeds)

    def test_critical_gap_within_bounds(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(100):
            g = sample_critical_gap_s(SURAT, rng)
            assert 0.5 <= g <= 15.0

    def test_critical_gap_mean_is_reasonable(self) -> None:
        rng = np.random.default_rng(0)
        gaps = [sample_critical_gap_s(SURAT, rng) for _ in range(500)]
        # Log-normal mean should be close to profile's mean
        assert 3.0 < float(np.mean(gaps)) < 5.0


class TestEvasiveBrakingModel:
    """Tests for EvasiveBrakingModel."""

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="hard_brake_ttc_s"):
            EvasiveBrakingModel(hard_brake_ttc_s=0.0)
        with pytest.raises(ValueError, match="soft_brake_ttc_s"):
            EvasiveBrakingModel(hard_brake_ttc_s=3.0, soft_brake_ttc_s=1.0)

    def test_head_on_brakes(self) -> None:
        a = VehicleAgent("a", Vec2(-5.0, 0.0), 10.0, 0.0)
        b = VehicleAgent("b", Vec2(5.0, 0.0), 10.0, math.pi)
        model = EvasiveBrakingModel()
        model.apply([a, b], Intersection(intersection_id="test"), dt=0.1)
        # TTC = 10/20 = 0.5 s < hard_brake = 1.5 s -> strong decel
        assert a.speed < 10.0
        assert b.speed < 10.0

    def test_no_conflict_keeps_speed(self) -> None:
        a = VehicleAgent("a", Vec2(-5.0, 0.0), 10.0, 0.0)
        b = VehicleAgent("b", Vec2(5.0, 0.0), 10.0, 0.0)  # same direction
        model = EvasiveBrakingModel()
        model.apply([a, b], Intersection(intersection_id="test"), dt=0.1)
        # Same-direction pair -> no closing speed -> no brake
        assert a.speed == 10.0
        assert b.speed == 10.0

    def test_speed_never_negative(self) -> None:
        a = VehicleAgent("a", ORIGIN, 10.0, 0.0)
        b = VehicleAgent("b", Vec2(0.5, 0.0), 10.0, math.pi)
        model = EvasiveBrakingModel()
        for _ in range(50):
            model.apply([a, b], Intersection(intersection_id="test"), dt=0.1)
        assert a.speed >= 0.0
        assert b.speed >= 0.0


class TestGapAcceptanceModel:
    """Tests for GapAcceptanceModel."""

    def test_yields_to_conflicting_vehicle(self) -> None:
        # a is close to intersection; b is even closer on conflicting arm
        a = VehicleAgent("a", Vec2(-20.0, 0.0), 10.0, 0.0, critical_gap_s=5.0)
        b = VehicleAgent("b", Vec2(0.0, -5.0), 10.0, math.pi / 2)
        model = GapAcceptanceModel(approach_radius_m=30.0)
        initial_speed = a.speed
        model.apply([a, b], Intersection(intersection_id="test"), dt=0.1)
        assert a.speed < initial_speed

    def test_proceeds_when_no_conflict(self) -> None:
        a = VehicleAgent("a", Vec2(-20.0, 0.0), 10.0, 0.0)
        model = GapAcceptanceModel()
        model.apply([a], Intersection(intersection_id="test"), dt=0.1)
        assert a.speed == 10.0

    def test_no_yield_when_far(self) -> None:
        # Both far from intersection -> approach_radius not entered
        a = VehicleAgent("a", Vec2(-200.0, 0.0), 10.0, 0.0)
        b = VehicleAgent("b", Vec2(0.0, -200.0), 10.0, math.pi / 2)
        model = GapAcceptanceModel(approach_radius_m=30.0)
        model.apply([a, b], Intersection(intersection_id="test"), dt=0.1)
        assert a.speed == 10.0
        assert b.speed == 10.0


class TestScenarioGeneration:
    """Integration-level tests for scenario + behaviour pipeline."""

    def test_generate_agents_deterministic(self) -> None:
        from surrogate_safety_abm.simulation import generate_agents

        a1 = generate_agents(SURAT, seed=42, n_vehicles=8)
        a2 = generate_agents(SURAT, seed=42, n_vehicles=8)
        for x, y in zip(a1, a2, strict=True):
            assert x.agent_id == y.agent_id
            assert x.vehicle_type == y.vehicle_type
            assert abs(x.speed - y.speed) < 1e-9

    def test_generate_agents_respects_pcu(self) -> None:
        from surrogate_safety_abm.simulation import generate_agents

        agents = generate_agents(SURAT, seed=0, n_vehicles=200)
        # Surat PCU composition: 2W dominant
        two_wheelers = [a for a in agents if a.vehicle_type == VehicleType.TWO_WHEELER]
        assert len(two_wheelers) / len(agents) > 0.35

    def test_cities_produce_different_agents(self) -> None:
        from surrogate_safety_abm.simulation import generate_agents

        surat_agents = generate_agents(SURAT, seed=1, n_vehicles=30)
        vad_agents = generate_agents(VADODARA, seed=1, n_vehicles=30)
        surat_mean_v = float(np.mean([a.speed for a in surat_agents]))
        vad_mean_v = float(np.mean([a.speed for a in vad_agents]))
        # Surat v85 = 50 kph > Vadodara v85 = 45 kph
        assert surat_mean_v > vad_mean_v

    def test_make_city_behaviours_returns_two(self) -> None:
        from surrogate_safety_abm.simulation import make_city_behaviours

        stack = make_city_behaviours(SURAT)
        assert len(stack) == 2
        assert isinstance(stack[0], GapAcceptanceModel)
        assert isinstance(stack[1], EvasiveBrakingModel)


class TestCruiseSpeedDefault:
    """Covers VehicleAgent.cruise_speed default and explicit override."""

    def test_default_target_speed_is_initial_speed(self) -> None:
        v = VehicleAgent("v", ORIGIN, 10.0, 0.0)
        assert v.target_speed == 10.0
        assert v.cruise_speed == 10.0

    def test_explicit_target_speed_wins(self) -> None:
        v = VehicleAgent("v", ORIGIN, 10.0, 0.0, target_speed=8.0)
        assert v.cruise_speed == 8.0


class TestEvasiveBrakingBranches:
    """Cover remaining branches in EvasiveBrakingModel."""

    def test_zero_gap_closing_speed_guard(self) -> None:
        """_closing_speed returns 0.0 for coincident agents."""
        from surrogate_safety_abm.behaviour.evasive import _closing_speed

        a = VehicleAgent("a", ORIGIN, 10.0, 0.0)
        b = VehicleAgent("b", ORIGIN, 10.0, math.pi)
        assert _closing_speed(a, b, gap=0.0) == 0.0
        assert _closing_speed(a, b, gap=1e-10) == 0.0

    def test_soft_braking_zone(self) -> None:
        """TTC between hard and soft thresholds -> partial brake."""
        # Closing speed 10 m/s, gap 25 m -> TTC 2.5 s (between 1.5 and 3.0)
        a = VehicleAgent("a", Vec2(0.0, 0.0), 5.0, 0.0, target_speed=15.0)
        b = VehicleAgent("b", Vec2(25.0, 0.0), 5.0, math.pi, target_speed=15.0)
        model = EvasiveBrakingModel()
        model.apply([a, b], Intersection(intersection_id="test"), dt=0.1)
        assert a.speed < 15.0

    def test_pedestrian_in_fleet(self) -> None:
        """Non-vehicle agents use their current speed as cruise."""
        from surrogate_safety_abm.agents import PedestrianAgent

        p = PedestrianAgent("p", ORIGIN, 1.2, 0.0)
        model = EvasiveBrakingModel()
        model.apply([p], Intersection(intersection_id="test"), dt=0.1)
        assert p.speed == 1.2


class TestGapAcceptanceBranches:
    """Cover remaining branches in GapAcceptanceModel."""

    def test_too_close_to_intersection(self) -> None:
        """dist < 2.0 -> skip."""
        a = VehicleAgent("a", Vec2(-1.0, 0.0), 10.0, 0.0)
        model = GapAcceptanceModel(approach_radius_m=30.0)
        model.apply([a], Intersection(intersection_id="test"), dt=0.1)
        assert a.speed == 10.0

    def test_stationary_agent(self) -> None:
        """speed < 0.05 -> skip."""
        a = VehicleAgent("a", Vec2(-10.0, 0.0), 0.02, 0.0)
        model = GapAcceptanceModel()
        model.apply([a], Intersection(intersection_id="test"), dt=0.1)
        assert a.speed == 0.02

    def test_stationary_other_vehicle(self) -> None:
        """Other vehicle at rest doesn't trigger yield."""
        a = VehicleAgent("a", Vec2(-10.0, 0.0), 10.0, 0.0)
        b = VehicleAgent("b", Vec2(0.0, -10.0), 0.02, math.pi / 2)
        model = GapAcceptanceModel(approach_radius_m=30.0)
        model.apply([a, b], Intersection(intersection_id="test"), dt=0.1)
        assert a.speed == 10.0

    def test_same_arm_no_yield(self) -> None:
        """Same arm heading -> not conflicting -> no yield."""
        a = VehicleAgent("a", Vec2(-20.0, 0.0), 10.0, 0.0)
        b = VehicleAgent("b", Vec2(-10.0, 0.0), 5.0, 0.0)
        model = GapAcceptanceModel(approach_radius_m=30.0)
        model.apply([a, b], Intersection(intersection_id="test"), dt=0.1)
        assert a.speed == 10.0

    def test_other_farther_from_intersection(self) -> None:
        """Other arrival > own arrival -> no yield."""
        a = VehicleAgent(
            "a",
            Vec2(-2.0, 0.0),
            5.0,
            0.0,
            target_speed=5.0,
            critical_gap_s=5.0,
        )
        b = VehicleAgent("b", Vec2(0.0, -50.0), 5.0, math.pi / 2)
        model = GapAcceptanceModel(approach_radius_m=30.0)
        model.apply([a, b], Intersection(intersection_id="test"), dt=0.1)
        assert a.speed == 5.0

    def test_is_conflicting_arm_helper(self) -> None:
        """Same arm False; perpendicular and opposite True."""
        a = VehicleAgent("a", ORIGIN, 5.0, 0.0)
        same = VehicleAgent("b", ORIGIN, 5.0, 0.0)
        perp = VehicleAgent("c", ORIGIN, 5.0, math.pi / 2)
        opposite = VehicleAgent("d", ORIGIN, 5.0, math.pi)
        wrap = VehicleAgent("e", ORIGIN, 5.0, 3 * math.pi / 2)
        assert GapAcceptanceModel._is_conflicting_arm(a, same) is False
        assert GapAcceptanceModel._is_conflicting_arm(a, perp) is True
        assert GapAcceptanceModel._is_conflicting_arm(a, opposite) is True
        assert GapAcceptanceModel._is_conflicting_arm(a, wrap) is True


class TestEvasiveBrakingMultiAgent:
    """Covers inner-loop iterations in _min_ttc with 3+ agents."""

    def test_three_agents_mixed_conflicts(self) -> None:
        a = VehicleAgent("a", Vec2(0.0, 0.0), 10.0, 0.0, target_speed=10.0)
        b = VehicleAgent("b", Vec2(10.0, 0.0), 10.0, math.pi)
        c = VehicleAgent("c", Vec2(-5.0, 0.0), 10.0, 0.0)
        model = EvasiveBrakingModel()
        model.apply([a, b, c], Intersection(intersection_id="test"), dt=0.1)
        assert a.speed < 10.0
        assert b.speed < 10.0

    def test_far_agent_skipped_inner_loop(self) -> None:
        a = VehicleAgent("a", Vec2(0.0, 0.0), 10.0, 0.0, target_speed=10.0)
        b = VehicleAgent("b", Vec2(5.0, 0.0), 10.0, math.pi)
        c = VehicleAgent("c", Vec2(1000.0, 0.0), 10.0, math.pi)
        model = EvasiveBrakingModel()
        model.apply([a, b, c], Intersection(intersection_id="test"), dt=0.1)
        assert a.speed < 10.0


class TestGapAcceptanceMultiAgent:
    """Covers inner-loop iterations in _must_yield with 3+ agents."""

    def test_three_agents_on_conflicting_arms(self) -> None:
        a = VehicleAgent(
            "a",
            Vec2(-20.0, 0.0),
            8.0,
            0.0,
            target_speed=8.0,
            critical_gap_s=5.0,
        )
        b = VehicleAgent("b", Vec2(0.0, -3.0), 10.0, math.pi / 2, target_speed=10.0)
        c = VehicleAgent("c", Vec2(0.0, 15.0), 5.0, -math.pi / 2, target_speed=5.0)
        model = GapAcceptanceModel(approach_radius_m=30.0)
        model.apply([a, b, c], Intersection(intersection_id="test"), dt=0.1)
        assert a.speed <= 8.0

    def test_same_arm_then_conflicting_arm(self) -> None:
        # Agent b is same-arm (skip), agent c is conflicting (yield)
        a = VehicleAgent(
            "a",
            Vec2(-18.0, 0.0),
            6.0,
            0.0,
            target_speed=6.0,
            critical_gap_s=5.0,
        )
        b = VehicleAgent("b", Vec2(-30.0, 0.0), 5.0, 0.0)
        c = VehicleAgent("c", Vec2(0.0, -2.0), 10.0, math.pi / 2, target_speed=10.0)
        model = GapAcceptanceModel(approach_radius_m=30.0)
        initial = a.speed
        model.apply([a, b, c], Intersection(intersection_id="test"), dt=0.1)
        assert a.speed <= initial
