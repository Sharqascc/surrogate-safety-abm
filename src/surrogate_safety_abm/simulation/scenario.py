"""City-driven scenario generation for simulations."""

from __future__ import annotations

import math

import numpy as np

from surrogate_safety_abm.agents import VehicleAgent, VehicleType
from surrogate_safety_abm.behaviour import (
    BehaviourModel,
    EvasiveBrakingModel,
    GapAcceptanceModel,
    sample_critical_gap_s,
    sample_speed_mps,
)
from surrogate_safety_abm.config.city_profiles import CityProfile
from surrogate_safety_abm.geometry import Vec2

_ARM_BASES: list[tuple[Vec2, float]] = [
    (Vec2(-60.0, 0.0), 0.0),  # west, heading east
    (Vec2(60.0, 0.0), math.pi),  # east, heading west
    (Vec2(0.0, -60.0), math.pi / 2.0),  # south, heading north
    (Vec2(0.0, 60.0), -math.pi / 2.0),  # north, heading south
]


def generate_agents(
    city: CityProfile,
    seed: int,
    n_vehicles: int = 12,
) -> list[VehicleAgent]:
    """Generate a heterogeneous fleet for a given city profile.

    Vehicle types are drawn from ``city.pcu_composition``; speeds and
    critical gaps are sampled from the city's v85 and gap distributions.

    Args:
        city: City profile driving the sampling distributions.
        seed: RNG seed for reproducibility.
        n_vehicles: Number of vehicles to generate.

    Returns:
        A list of :class:`VehicleAgent` positioned on the four arms.
    """
    rng = np.random.default_rng(seed)
    types = list(city.pcu_composition.keys())
    weights = np.array([city.pcu_composition[t] for t in types], dtype=float)
    weights = weights / weights.sum()

    agents: list[VehicleAgent] = []
    for idx in range(n_vehicles):
        type_idx = int(rng.choice(len(types), p=weights))
        vehicle_type: VehicleType = types[type_idx]
        speed = sample_speed_mps(city, rng)
        gap = sample_critical_gap_s(city, rng)
        base, heading = _ARM_BASES[idx % len(_ARM_BASES)]
        offset = float(rng.uniform(0.0, 30.0))
        position = Vec2(
            base.x + offset * math.cos(heading),
            base.y + offset * math.sin(heading),
        )
        agents.append(
            VehicleAgent(
                agent_id=f"v{idx + 1}",
                position=position,
                speed=speed,
                heading=heading,
                vehicle_type=vehicle_type,
                target_speed=speed,
                critical_gap_s=gap,
            )
        )
    return agents


def make_city_behaviours(city: CityProfile) -> list[BehaviourModel]:
    """Return the standard behaviour stack for a city simulation.

    Args:
        city: City profile (currently used to configure yield deceleration).

    Returns:
        A list of behaviour models in execution order.
    """
    del city  # deceleration currently uses model defaults
    return [
        GapAcceptanceModel(),
        EvasiveBrakingModel(),
    ]
