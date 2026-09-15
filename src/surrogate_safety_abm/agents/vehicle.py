"""Vehicle agents for heterogeneous Indian traffic."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from surrogate_safety_abm.agents.base import Agent


class VehicleType(StrEnum):
    """Vehicle categories with distinct physical and behavioural profiles."""

    TWO_WHEELER = "2W"
    THREE_WHEELER = "3W"
    CAR = "car"
    HGV = "hgv"
    BUS = "bus"


# Physical dimensions (metres), rounded from IRC:106 and Indian vehicle specs.
VEHICLE_LENGTH_M: dict[VehicleType, float] = {
    VehicleType.TWO_WHEELER: 2.0,
    VehicleType.THREE_WHEELER: 2.6,
    VehicleType.CAR: 4.0,
    VehicleType.HGV: 7.5,
    VehicleType.BUS: 12.0,
}

VEHICLE_WIDTH_M: dict[VehicleType, float] = {
    VehicleType.TWO_WHEELER: 0.7,
    VehicleType.THREE_WHEELER: 1.3,
    VehicleType.CAR: 1.8,
    VehicleType.HGV: 2.5,
    VehicleType.BUS: 2.5,
}

# Passenger Car Unit equivalents (IRC:106 and recent Indian literature).
VEHICLE_PCU: dict[VehicleType, float] = {
    VehicleType.TWO_WHEELER: 0.5,
    VehicleType.THREE_WHEELER: 1.5,
    VehicleType.CAR: 1.0,
    VehicleType.HGV: 3.0,
    VehicleType.BUS: 3.0,
}


@dataclass
class VehicleAgent(Agent):
    """A motorised road user.

    Attributes:
        vehicle_type: Classification of the vehicle.
        max_speed: Maximum achievable speed (metres per second).
        max_accel: Maximum positive acceleration (metres per second squared).
        max_decel: Maximum magnitude of deceleration (metres per second squared).
    """

    vehicle_type: VehicleType = VehicleType.CAR
    max_speed: float = 16.7  # ~60 km/h default
    max_accel: float = 2.5
    max_decel: float = 4.5
    _length_m: float = field(init=False, repr=False)
    _width_m: float = field(init=False, repr=False)
    _pcu: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Set derived physical properties after validation."""
        super().__post_init__()
        if self.max_speed <= 0.0:
            raise ValueError("max_speed must be positive")
        if self.speed > self.max_speed:
            raise ValueError("initial speed exceeds max_speed")
        self._length_m = VEHICLE_LENGTH_M[self.vehicle_type]
        self._width_m = VEHICLE_WIDTH_M[self.vehicle_type]
        self._pcu = VEHICLE_PCU[self.vehicle_type]

    @property
    def length_m(self) -> float:
        """Return the physical length of this vehicle in metres."""
        return self._length_m

    @property
    def width_m(self) -> float:
        """Return the physical width of this vehicle in metres."""
        return self._width_m

    @property
    def pcu(self) -> float:
        """Return the Passenger Car Unit equivalent of this vehicle."""
        return self._pcu

    def step(self, dt: float) -> None:
        """Advance the vehicle by one time step at constant speed.

        A full behavioural model is applied in the simulation engine via
        the behaviour modules; this method implements only kinematic
        integration.
        """
        self.integrate(dt)
