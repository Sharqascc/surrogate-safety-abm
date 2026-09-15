"""Unit tests for city profiles."""

import pytest
from pydantic import ValidationError

from surrogate_safety_abm.agents import VehicleType
from surrogate_safety_abm.config import (
    AHMEDABAD,
    CITY_PROFILES,
    SURAT,
    VADODARA,
    CityProfile,
)


class TestCityProfile:
    """Unit tests for CityProfile."""

    def test_vadodara_profile(self) -> None:
        assert VADODARA.name == "Vadodara"
        assert VADODARA.state == "Gujarat"
        assert sum(VADODARA.pcu_composition.values()) == pytest.approx(1.0, abs=0.01)

    def test_surat_profile(self) -> None:
        assert SURAT.name == "Surat"
        assert SURAT.approach_speed_85_kph == 50.0

    def test_ahmedabad_profile(self) -> None:
        assert AHMEDABAD.name == "Ahmedabad"
        assert AHMEDABAD.critical_gap_mean_s == pytest.approx(4.0)

    def test_profiles_registry(self) -> None:
        assert set(CITY_PROFILES.keys()) == {"vadodara", "surat", "ahmedabad"}

    def test_pcu_sum_validation_fails(self) -> None:
        with pytest.raises(ValidationError, match=r"sum to 1\.0"):
            CityProfile(
                name="Test",
                pcu_composition={
                    VehicleType.TWO_WHEELER: 0.9,
                    VehicleType.CAR: 0.9,
                },
                critical_gap_mean_s=4.0,
                critical_gap_std_s=1.5,
                approach_speed_85_kph=45.0,
            )

    def test_negative_gap_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CityProfile(
                name="Test",
                pcu_composition={VehicleType.CAR: 1.0},
                critical_gap_mean_s=-1.0,
                critical_gap_std_s=1.5,
                approach_speed_85_kph=45.0,
            )

    def test_pcu_shares_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            CityProfile(
                name="Test",
                pcu_composition={
                    VehicleType.CAR: 1.1,
                    VehicleType.TWO_WHEELER: -0.1,
                },
                critical_gap_mean_s=4.0,
                critical_gap_std_s=1.5,
                approach_speed_85_kph=45.0,
            )
