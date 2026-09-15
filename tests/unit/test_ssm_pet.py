"""Unit tests for Post-Encroachment Time (PET)."""

from dataclasses import FrozenInstanceError

import pytest

from surrogate_safety_abm.ssm.pet import PETResult, compute_pet


class TestComputePET:
    """Unit tests for compute_pet."""

    def test_normal_separation(self) -> None:
        result = compute_pet(first_exit_time=10.0, second_entry_time=13.5)
        assert isinstance(result, PETResult)
        assert result.pet_seconds == pytest.approx(3.5)
        assert result.is_collision is False

    def test_touching_occupancies_flag_collision(self) -> None:
        result = compute_pet(10.0, 10.0)
        assert result.pet_seconds == 0.0
        assert result.is_collision is True

    def test_overlapping_occupancies_clamp_to_zero(self) -> None:
        result = compute_pet(12.0, 10.0)
        assert result.pet_seconds == 0.0
        assert result.is_collision is True

    def test_zero_and_positive_inputs(self) -> None:
        result = compute_pet(0.0, 5.0)
        assert result.pet_seconds == pytest.approx(5.0)

    def test_negative_first_exit_raises(self) -> None:
        with pytest.raises(ValueError, match="first_exit_time"):
            compute_pet(-1.0, 5.0)

    def test_negative_second_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="second_entry_time"):
            compute_pet(1.0, -2.0)

    def test_result_is_frozen(self) -> None:
        r = compute_pet(1.0, 2.0)
        with pytest.raises(FrozenInstanceError):
            r.pet_seconds = 99.0  # type: ignore[misc]
