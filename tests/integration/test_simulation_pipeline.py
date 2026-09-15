"""Integration test: vehicles at an intersection produce SSM values.

Simulates two vehicles converging on a single crossing conflict point
using pure kinematics (no behavioural model yet). Verifies that PET and
TTC can be computed on the recorded trajectories.
"""

from __future__ import annotations

import pytest

from surrogate_safety_abm.agents import VehicleAgent, VehicleType
from surrogate_safety_abm.environment import Intersection
from surrogate_safety_abm.geometry import ORIGIN, Vec2
from surrogate_safety_abm.ssm import compute_pet, compute_ttc


@pytest.mark.integration
class TestIntersectionSSMPipeline:
    """End-to-end SSM computation on synthetic trajectories."""

    def test_four_legged_intersection_geometry(self) -> None:
        ic = Intersection.four_legged(intersection_id="surat-synthetic")
        assert len(ic.arms) == 4
        assert len(ic.conflict_points) == 6

    def test_two_vehicles_converging_produce_ttc(self) -> None:
        # Vehicle A: approaching conflict from the west at 10 m/s
        a = VehicleAgent(
            agent_id="A",
            position=Vec2(-30.0, 0.0),
            speed=10.0,
            heading=0.0,
            vehicle_type=VehicleType.CAR,
        )
        # Vehicle B: approaching from the south at 10 m/s
        b = VehicleAgent(
            agent_id="B",
            position=Vec2(0.0, -40.0),
            speed=10.0,
            heading=1.5707963,  # pi/2, heading north
            vehicle_type=VehicleType.TWO_WHEELER,
        )
        # Advance both by 1 second
        for _ in range(10):
            a.step(0.1)
            b.step(0.1)
        # A is now at x=-20, B is at y=-30 -> gap should be ~36 m
        gap = a.position.distance_to(b.position)
        assert gap == pytest.approx(36.06, abs=0.5)
        # TTC with a closing speed of ~10 m/s
        ttc = compute_ttc(gap=gap, relative_speed=10.0)
        assert ttc.is_converging
        assert ttc.ttc_seconds == pytest.approx(3.6, abs=0.1)

    def test_conflict_zone_pet_computation(self) -> None:
        # Vehicle A passes through conflict zone [10.0, 10.5] s
        # Vehicle B arrives at conflict zone at 12.0 s
        pet = compute_pet(first_exit_time=10.5, second_entry_time=12.0)
        assert pet.pet_seconds == pytest.approx(1.5)
        assert pet.is_collision is False

    def test_collision_case_detected(self) -> None:
        # Overlapping occupancies -> collision flag
        pet = compute_pet(first_exit_time=10.5, second_entry_time=10.2)
        assert pet.is_collision is True
        assert pet.pet_seconds == 0.0

    def test_heterogeneous_pcu_aggregation(self) -> None:
        # Two 2W + one car + one HGV: expected total PCU
        vehicles = [
            VehicleAgent("a", ORIGIN, 5.0, 0.0, VehicleType.TWO_WHEELER),
            VehicleAgent("b", ORIGIN, 5.0, 0.0, VehicleType.TWO_WHEELER),
            VehicleAgent("c", ORIGIN, 5.0, 0.0, VehicleType.CAR),
            VehicleAgent("d", ORIGIN, 5.0, 0.0, VehicleType.HGV),
        ]
        total_pcu = sum(v.pcu for v in vehicles)
        # 0.5 + 0.5 + 1.0 + 3.0 = 5.0
        assert total_pcu == pytest.approx(5.0)


@pytest.mark.integration
class TestFullEngineRun:
    """End-to-end engine run on a four-legged intersection."""

    def test_full_simulation_produces_events(self, tmp_path) -> None:
        from surrogate_safety_abm.config.city_profiles import SURAT
        from surrogate_safety_abm.simulation import (
            SimulationConfig,
            SimulationEngine,
        )

        agents = [
            VehicleAgent("v1", Vec2(-40.0, 0.0), 12.0, 0.0, VehicleType.CAR),
            VehicleAgent(
                "v2",
                Vec2(0.0, -40.0),
                10.0,
                1.5707963,
                VehicleType.TWO_WHEELER,
            ),
            VehicleAgent(
                "v3",
                Vec2(40.0, 0.0),
                8.0,
                3.14159265,
                VehicleType.THREE_WHEELER,
            ),
            VehicleAgent("v4", Vec2(0.0, 40.0), 9.0, -1.5707963, VehicleType.CAR),
        ]
        intersection = Intersection.four_legged(intersection_id="demo")
        engine = SimulationEngine(
            city=SURAT,
            intersection=intersection,
            agents=agents,
            config=SimulationConfig(duration_s=20.0, time_step_s=0.2, seed=42),
        )
        result = engine.run()

        assert result.steps_completed == 100
        for v in agents:
            assert len(result.recorder.trajectory(v.agent_id)) == 100
        assert len(result.recorder.conflicts) > 0
        for ev in result.recorder.conflicts:
            assert ev.ttc_s >= 0.0
            assert ev.distance_m >= 0.0
