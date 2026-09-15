"""Unit tests for the simulation engine."""

import pytest

from surrogate_safety_abm.agents import VehicleAgent, VehicleType
from surrogate_safety_abm.config.city_profiles import SURAT
from surrogate_safety_abm.environment import Intersection
from surrogate_safety_abm.geometry import ORIGIN, Vec2
from surrogate_safety_abm.simulation import (
    SimulationConfig,
    SimulationEngine,
)


def _empty_intersection() -> Intersection:
    return Intersection(intersection_id="test")


class TestSimulationConfig:
    """Validation tests for SimulationConfig."""

    def test_defaults(self) -> None:
        cfg = SimulationConfig()
        assert cfg.duration_s == 60.0
        assert cfg.time_step_s == 0.1
        assert cfg.seed is None

    def test_duration_positive(self) -> None:
        with pytest.raises(ValueError, match="duration_s"):
            SimulationConfig(duration_s=0.0)

    def test_time_step_positive(self) -> None:
        with pytest.raises(ValueError, match="time_step_s"):
            SimulationConfig(time_step_s=0.0)

    def test_warm_up_non_negative(self) -> None:
        with pytest.raises(ValueError, match="warm_up_s"):
            SimulationConfig(warm_up_s=-1.0)

    def test_warm_up_less_than_duration(self) -> None:
        with pytest.raises(ValueError, match="less than duration_s"):
            SimulationConfig(warm_up_s=100.0, duration_s=60.0)

    def test_conflict_radius_positive(self) -> None:
        with pytest.raises(ValueError, match="conflict_radius_m"):
            SimulationConfig(conflict_radius_m=0.0)

    def test_ttc_threshold_positive(self) -> None:
        with pytest.raises(ValueError, match="ttc_threshold_s"):
            SimulationConfig(ttc_threshold_s=0.0)

    def test_min_gap_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_gap_m"):
            SimulationConfig(min_gap_m=-1.0)

    def test_min_gap_default_is_two_meters(self) -> None:
        assert SimulationConfig().min_gap_m == 2.0


class TestSimulationEngine:
    """Functional tests for SimulationEngine."""

    def test_duplicate_ids_rejected(self) -> None:
        a = VehicleAgent("dup", ORIGIN, 5.0, 0.0)
        b = VehicleAgent("dup", ORIGIN, 5.0, 0.0)
        with pytest.raises(ValueError, match="unique"):
            SimulationEngine(city=SURAT, intersection=_empty_intersection(), agents=[a, b])

    def test_single_agent_no_conflicts(self) -> None:
        a = VehicleAgent("a", Vec2(-10.0, 0.0), 5.0, 0.0)
        engine = SimulationEngine(
            city=SURAT,
            intersection=_empty_intersection(),
            agents=[a],
            config=SimulationConfig(duration_s=2.0, time_step_s=0.1),
        )
        result = engine.run()
        assert result.recorder.conflicts == []
        assert result.steps_completed == 20
        assert len(result.recorder.snapshots) == 20

    def test_distant_agents_no_conflicts(self) -> None:
        a = VehicleAgent("a", Vec2(-500.0, 0.0), 5.0, 0.0)
        b = VehicleAgent("b", Vec2(500.0, 0.0), 5.0, 3.14159265)
        engine = SimulationEngine(
            city=SURAT,
            intersection=_empty_intersection(),
            agents=[a, b],
            config=SimulationConfig(duration_s=1.0, time_step_s=0.1),
        )
        result = engine.run()
        assert result.recorder.conflicts == []

    def test_head_on_collision_course(self) -> None:
        a = VehicleAgent("a", Vec2(-10.0, 0.0), 10.0, 0.0)
        b = VehicleAgent("b", Vec2(10.0, 0.0), 10.0, 3.14159265)
        engine = SimulationEngine(
            city=SURAT,
            intersection=_empty_intersection(),
            agents=[a, b],
            config=SimulationConfig(duration_s=2.0, time_step_s=0.1),
        )
        result = engine.run()
        assert len(result.recorder.conflicts) > 0
        for ev in result.recorder.conflicts:
            assert ev.ttc_s >= 0.0
            assert ev.distance_m >= 0.0

    def test_warm_up_skips_recording(self) -> None:
        a = VehicleAgent("a", Vec2(-10.0, 0.0), 5.0, 0.0)
        engine = SimulationEngine(
            city=SURAT,
            intersection=_empty_intersection(),
            agents=[a],
            config=SimulationConfig(duration_s=2.0, time_step_s=0.1, warm_up_s=1.0),
        )
        result = engine.run()
        assert len(result.recorder.snapshots) == 10

    def test_overlapping_pair_is_collision_not_conflict(self) -> None:
        """Agents within min_gap_m are excluded (collisions, not conflicts)."""
        a = VehicleAgent("a", ORIGIN, 5.0, 0.0)
        b = VehicleAgent("b", Vec2(1.0, 0.0), 5.0, 3.14159265)
        engine = SimulationEngine(
            city=SURAT,
            intersection=_empty_intersection(),
            agents=[a, b],
            config=SimulationConfig(duration_s=0.5, time_step_s=0.1, min_gap_m=2.0),
        )
        result = engine.run()
        assert result.recorder.conflicts == []

    def test_zero_gap_still_skipped_by_min_gap(self) -> None:
        """Zero-distance pairs are collisions, not conflicts."""
        a = VehicleAgent("a", ORIGIN, 5.0, 0.0, VehicleType.CAR)
        b = VehicleAgent("b", ORIGIN, 5.0, 0.0, VehicleType.CAR)
        engine = SimulationEngine(
            city=SURAT,
            intersection=_empty_intersection(),
            agents=[a, b],
            config=SimulationConfig(duration_s=0.5, time_step_s=0.1, min_gap_m=2.0),
        )
        result = engine.run()
        assert result.recorder.conflicts == []

    def test_high_ttc_threshold_no_events(self) -> None:
        a = VehicleAgent("a", Vec2(-14.0, 0.0), 1.0, 0.0)
        b = VehicleAgent("b", Vec2(14.0, 0.0), 1.0, 3.14159265)
        engine = SimulationEngine(
            city=SURAT,
            intersection=_empty_intersection(),
            agents=[a, b],
            config=SimulationConfig(duration_s=0.5, time_step_s=0.1, ttc_threshold_s=0.01),
        )
        result = engine.run()
        # TTC = 28 / 2 = 14 s >> 0.01 s threshold -> no records
        assert result.recorder.conflicts == []

    def test_non_converging_pair_no_event(self) -> None:
        # Same heading and speed -> closing speed = 0 -> inf TTC
        a = VehicleAgent("a", Vec2(-5.0, 0.0), 5.0, 0.0)
        b = VehicleAgent("b", Vec2(5.0, 0.0), 5.0, 0.0)
        engine = SimulationEngine(
            city=SURAT,
            intersection=_empty_intersection(),
            agents=[a, b],
            config=SimulationConfig(duration_s=0.5, time_step_s=0.1),
        )
        result = engine.run()
        assert result.recorder.conflicts == []

    def test_result_carries_city_and_intersection(self) -> None:
        ic = Intersection.four_legged()
        engine = SimulationEngine(
            city=SURAT,
            intersection=ic,
            agents=[],
            config=SimulationConfig(duration_s=1.0, time_step_s=0.1),
        )
        result = engine.run()
        assert result.city is SURAT
        assert result.intersection is ic


class TestClosingSpeed:
    """Unit tests for the _closing_speed helper."""

    def test_zero_gap_returns_zero(self) -> None:
        """Guard branch: gap <= 1e-9 returns 0.0 immediately."""
        from surrogate_safety_abm.simulation.engine import _closing_speed

        a = VehicleAgent("a", ORIGIN, 10.0, 0.0)
        b = VehicleAgent("b", ORIGIN, 10.0, 3.14159265)
        assert _closing_speed(a, b, gap=0.0) == 0.0
        assert _closing_speed(a, b, gap=1e-10) == 0.0

    def test_head_on_approach_positive(self) -> None:
        """Head-on closure gives a positive closing speed."""
        from surrogate_safety_abm.simulation.engine import _closing_speed

        a = VehicleAgent("a", Vec2(0.0, 0.0), 10.0, 0.0)
        b = VehicleAgent("b", Vec2(10.0, 0.0), 10.0, 3.14159265)
        assert _closing_speed(a, b, gap=10.0) == pytest.approx(20.0)

    def test_same_direction_zero_closing(self) -> None:
        """Same-direction equal-speed pair has zero closing speed."""
        from surrogate_safety_abm.simulation.engine import _closing_speed

        a = VehicleAgent("a", Vec2(0.0, 0.0), 10.0, 0.0)
        b = VehicleAgent("b", Vec2(10.0, 0.0), 10.0, 0.0)
        assert _closing_speed(a, b, gap=10.0) == pytest.approx(0.0)
