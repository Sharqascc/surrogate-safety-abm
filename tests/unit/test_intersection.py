"""Unit tests for intersection geometry."""

import pytest

from surrogate_safety_abm.environment import (
    Approach,
    Intersection,
)


class TestApproach:
    """Unit tests for Approach."""

    def test_construction(self) -> None:
        a = Approach(arm_id="N", num_lanes=2, length_m=40.0, angle_rad=1.0)
        assert a.arm_id == "N"
        assert a.num_lanes == 2

    def test_zero_lanes_rejected(self) -> None:
        with pytest.raises(ValueError, match="num_lanes"):
            Approach(arm_id="N", num_lanes=0)

    def test_non_positive_length_rejected(self) -> None:
        with pytest.raises(ValueError, match="length_m"):
            Approach(arm_id="N", length_m=0.0)

    def test_entry_point_is_left_of_origin(self) -> None:
        # For angle 0 (heading east into junction), entry is to the west
        a = Approach(arm_id="E", length_m=20.0, angle_rad=0.0)
        p = a.entry_point()
        assert p.x == pytest.approx(-10.0)
        assert p.y == pytest.approx(0.0, abs=1e-9)


class TestIntersection:
    """Unit tests for Intersection."""

    def test_empty_intersection(self) -> None:
        ic = Intersection(intersection_id="test")
        assert ic.arms == {}
        assert ic.conflict_points == []

    def test_add_one_arm_no_conflicts(self) -> None:
        ic = Intersection(intersection_id="test")
        ic.add_arm(Approach("N", angle_rad=1.57))
        assert len(ic.conflict_points) == 0

    def test_four_legged_has_six_conflicts(self) -> None:
        # 4 arms -> C(4,2) = 6 pairwise crossing conflicts
        ic = Intersection.four_legged(intersection_id="test")
        assert len(ic.arms) == 4
        assert len(ic.conflict_points) == 6

    def test_conflict_ids_unique(self) -> None:
        ic = Intersection.four_legged()
        ids = [c.conflict_id for c in ic.conflict_points]
        assert len(ids) == len(set(ids))

    def test_all_arms_present(self) -> None:
        ic = Intersection.four_legged()
        assert set(ic.arms.keys()) == {"N", "E", "S", "W"}

    def test_custom_lane_count(self) -> None:
        ic = Intersection.four_legged(lanes_per_arm=3)
        for arm in ic.arms.values():
            assert arm.num_lanes == 3

    def test_add_duplicate_arm_replaces(self) -> None:
        ic = Intersection(intersection_id="test")
        ic.add_arm(Approach("N", num_lanes=1))
        ic.add_arm(Approach("N", num_lanes=2))
        assert len(ic.arms) == 1
        assert ic.arms["N"].num_lanes == 2
