"""Unit tests for conflict points."""

import pytest

from surrogate_safety_abm.environment import ConflictPoint, ConflictType
from surrogate_safety_abm.geometry import Vec2


class TestConflictPoint:
    """Unit tests for ConflictPoint."""

    def test_construction(self) -> None:
        cp = ConflictPoint(
            conflict_id="c1",
            location=Vec2(1.0, 2.0),
            conflict_type=ConflictType.CROSSING,
            arm_a="N",
            arm_b="E",
        )
        assert cp.conflict_type == ConflictType.CROSSING
        assert cp.arm_a != cp.arm_b

    def test_same_arm_raises(self) -> None:
        with pytest.raises(ValueError, match="must differ"):
            ConflictPoint(
                conflict_id="c1",
                location=Vec2(0.0, 0.0),
                conflict_type=ConflictType.CROSSING,
                arm_a="N",
                arm_b="N",
            )

    def test_distance_to(self) -> None:
        a = ConflictPoint("a", Vec2(0.0, 0.0), ConflictType.CROSSING, "N", "E")
        b = ConflictPoint("b", Vec2(3.0, 4.0), ConflictType.MERGING, "S", "W")
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        cp = ConflictPoint("c", Vec2(0.0, 0.0), ConflictType.CROSSING, "N", "E")
        with pytest.raises(FrozenInstanceError):
            cp.arm_a = "S"  # type: ignore[misc]
