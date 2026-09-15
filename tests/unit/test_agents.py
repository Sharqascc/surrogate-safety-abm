"""Unit tests for agent classes."""

import pytest

from surrogate_safety_abm.agents import (
    VEHICLE_LENGTH_M,
    VEHICLE_PCU,
    VEHICLE_WIDTH_M,
    PedestrianAgent,
    VehicleAgent,
    VehicleType,
)
from surrogate_safety_abm.geometry import ORIGIN, Vec2


class TestVehicleAgent:
    """Unit tests for VehicleAgent."""

    def test_construction_car(self) -> None:
        v = VehicleAgent(
            agent_id="v1",
            position=ORIGIN,
            speed=10.0,
            heading=0.0,
            vehicle_type=VehicleType.CAR,
        )
        assert v.length_m == VEHICLE_LENGTH_M[VehicleType.CAR]
        assert v.width_m == VEHICLE_WIDTH_M[VehicleType.CAR]
        assert v.pcu == VEHICLE_PCU[VehicleType.CAR]

    def test_two_wheeler_pcu(self) -> None:
        v = VehicleAgent("v1", ORIGIN, 8.0, 0.0, VehicleType.TWO_WHEELER)
        assert v.pcu == pytest.approx(0.5)

    def test_negative_speed_rejected(self) -> None:
        with pytest.raises(ValueError, match="speed"):
            VehicleAgent("v1", ORIGIN, -1.0, 0.0)

    def test_speed_above_max_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_speed"):
            VehicleAgent("v1", ORIGIN, 20.0, 0.0, max_speed=10.0)

    def test_non_positive_max_speed_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_speed must be positive"):
            VehicleAgent("v1", ORIGIN, 0.0, 0.0, max_speed=0.0)

    def test_negative_max_speed_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_speed must be positive"):
            VehicleAgent("v1", ORIGIN, 0.0, 0.0, max_speed=-5.0)

    def test_step_integrates_position_forward(self) -> None:
        v = VehicleAgent("v1", ORIGIN, 10.0, 0.0)
        v.step(1.0)
        assert v.position.x == pytest.approx(10.0)
        assert v.position.y == pytest.approx(0.0)

    def test_step_heading_east(self) -> None:
        import math

        v = VehicleAgent("v1", ORIGIN, 10.0, math.pi / 2)
        v.step(1.0)
        assert v.position.x == pytest.approx(0.0, abs=1e-9)
        assert v.position.y == pytest.approx(10.0)

    def test_step_negative_dt_rejected(self) -> None:
        v = VehicleAgent("v1", ORIGIN, 10.0, 0.0)
        with pytest.raises(ValueError, match="dt"):
            v.step(-0.1)


class TestPedestrianAgent:
    """Unit tests for PedestrianAgent."""

    def test_construction(self) -> None:
        p = PedestrianAgent("p1", ORIGIN, 1.2, 0.0, goal=Vec2(5.0, 0.0))
        assert p.risk_propensity == 0.5
        assert p.is_crossing is False

    def test_risk_propensity_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="risk_propensity"):
            PedestrianAgent("p1", ORIGIN, 1.2, 0.0, risk_propensity=1.5)
        with pytest.raises(ValueError, match="risk_propensity"):
            PedestrianAgent("p1", ORIGIN, 1.2, 0.0, risk_propensity=-0.1)

    def test_step_advances(self) -> None:
        p = PedestrianAgent("p1", ORIGIN, 1.2, 0.0)
        p.step(2.0)
        assert p.position.x == pytest.approx(2.4)


class TestVehicleTables:
    """Sanity-check the physical tables."""

    def test_pcu_table_covers_all_types(self) -> None:
        for vt in VehicleType:
            assert vt in VEHICLE_PCU
            assert vt in VEHICLE_LENGTH_M
            assert vt in VEHICLE_WIDTH_M

    def test_two_wheeler_smallest(self) -> None:
        assert VEHICLE_LENGTH_M[VehicleType.TWO_WHEELER] <= VEHICLE_LENGTH_M[VehicleType.CAR]
        assert VEHICLE_WIDTH_M[VehicleType.TWO_WHEELER] <= VEHICLE_WIDTH_M[VehicleType.CAR]
