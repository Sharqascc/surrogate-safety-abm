"""City-specific traffic and geometric parameters.

Parameter values are drawn from published studies of unsignalized
intersections in the three study cities. Each profile can be overridden
at simulation time.

References:
    Pawar, N.M., Gore, N., & Arkatkar, S. (2022). Examining crossing
    conflicts by vehicle type at unsignalized T-intersections using
    accepted gaps. Journal of Transportation Engineering, Part A, 148(6).

    Dutta, M., Jena, S., Korat, B., Bhandari, S., & Lyngdoh, G.K. (2024).
    Anticipated buffer time -- An evasive surrogate safety indicator for
    risk assessment of unsignalized intersections. Accident Analysis &
    Prevention, 208, 107796.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from surrogate_safety_abm.agents.vehicle import VehicleType


class CityProfile(BaseModel):
    """Traffic and geometric parameters for a study city.

    Attributes:
        name: City name.
        state: State where the city is located.
        intersection_type: Descriptive string for the geometry modelled.
        pcu_composition: Proportion of each vehicle type present, in PCU.
            Values must sum to approximately 1.0.
        critical_gap_mean_s: Mean accepted critical gap for through traffic
            (seconds).
        critical_gap_std_s: Standard deviation of the critical gap (seconds).
        approach_speed_85_kph: 85th-percentile approach speed (km/h).
        conflict_zone_radius_m: Radius of the modelled conflict zone.
    """

    name: str
    state: str = "Gujarat"
    intersection_type: str = "four-legged unsignalized"
    pcu_composition: dict[VehicleType, float]
    critical_gap_mean_s: float = Field(gt=0.0)
    critical_gap_std_s: float = Field(gt=0.0)
    approach_speed_85_kph: float = Field(gt=0.0)
    conflict_zone_radius_m: float = Field(default=5.0, gt=0.0)

    @field_validator("pcu_composition")
    @classmethod
    def _sum_to_one(cls, v: dict[VehicleType, float]) -> dict[VehicleType, float]:
        """Ensure PCU composition sums to 1.0 within tolerance."""
        total = sum(v.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"pcu_composition must sum to 1.0 (got {total:.3f})")
        for share in v.values():
            if share < 0.0:
                raise ValueError("pcu_composition values must be non-negative")
        return v


VADODARA = CityProfile(
    name="Vadodara",
    pcu_composition={
        VehicleType.TWO_WHEELER: 0.45,
        VehicleType.THREE_WHEELER: 0.10,
        VehicleType.CAR: 0.30,
        VehicleType.HGV: 0.10,
        VehicleType.BUS: 0.05,
    },
    critical_gap_mean_s=4.2,
    critical_gap_std_s=1.8,
    approach_speed_85_kph=45.0,
)

SURAT = CityProfile(
    name="Surat",
    pcu_composition={
        VehicleType.TWO_WHEELER: 0.55,
        VehicleType.THREE_WHEELER: 0.08,
        VehicleType.CAR: 0.27,
        VehicleType.HGV: 0.07,
        VehicleType.BUS: 0.03,
    },
    critical_gap_mean_s=3.8,
    critical_gap_std_s=1.5,
    approach_speed_85_kph=50.0,
)

AHMEDABAD = CityProfile(
    name="Ahmedabad",
    pcu_composition={
        VehicleType.TWO_WHEELER: 0.50,
        VehicleType.THREE_WHEELER: 0.12,
        VehicleType.CAR: 0.28,
        VehicleType.HGV: 0.07,
        VehicleType.BUS: 0.03,
    },
    critical_gap_mean_s=4.0,
    critical_gap_std_s=1.6,
    approach_speed_85_kph=55.0,
)

CITY_PROFILES: dict[str, CityProfile] = {
    "vadodara": VADODARA,
    "surat": SURAT,
    "ahmedabad": AHMEDABAD,
}
