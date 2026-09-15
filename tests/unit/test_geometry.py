"""Unit tests for geometry primitives."""

import pytest

from surrogate_safety_abm.geometry import ORIGIN, Vec2


class TestVec2:
    """Unit tests for Vec2."""

    def test_add(self) -> None:
        assert Vec2(1.0, 2.0) + Vec2(3.0, 4.0) == Vec2(4.0, 6.0)

    def test_sub(self) -> None:
        assert Vec2(5.0, 5.0) - Vec2(1.0, 2.0) == Vec2(4.0, 3.0)

    def test_scalar_mul(self) -> None:
        assert Vec2(2.0, 3.0) * 2.0 == Vec2(4.0, 6.0)
        assert 2.0 * Vec2(2.0, 3.0) == Vec2(4.0, 6.0)

    def test_magnitude_3_4_5(self) -> None:
        assert Vec2(3.0, 4.0).magnitude() == pytest.approx(5.0)

    def test_distance_to(self) -> None:
        assert Vec2(0.0, 0.0).distance_to(Vec2(3.0, 4.0)) == pytest.approx(5.0)

    def test_origin(self) -> None:
        assert Vec2(0.0, 0.0) == ORIGIN

    def test_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        v = Vec2(1.0, 2.0)
        with pytest.raises(FrozenInstanceError):
            v.x = 9.0  # type: ignore[misc]
