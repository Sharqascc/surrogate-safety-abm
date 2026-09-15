"""Gap-acceptance yielding at the intersection.

Each driver has a critical gap (sampled at construction from the city
profile). When approaching the intersection centre, the driver compares
their own arrival time with the earliest arrival of any conflicting-arm
vehicle. If the difference is smaller than their critical gap, they
yield by braking.

Reference: Pawar, N. M., et al. (2022). Examining crossing conflicts
by vehicle type at unsignalized T-intersections using accepted gaps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from surrogate_safety_abm.agents.base import Agent
from surrogate_safety_abm.agents.vehicle import VehicleAgent
from surrogate_safety_abm.environment.intersection import Intersection
from surrogate_safety_abm.geometry import Vec2

_SAME_ARM_ANGLE_RAD: float = math.pi / 4.0


@dataclass
class GapAcceptanceModel:
    """Brake when a conflicting-arm vehicle would arrive too close to us.

    Attributes:
        approach_radius_m: Distance from the intersection centre within
            which gap acceptance is evaluated.
        deceleration_ms2: Rate applied when yielding.
        recovery_accel_ms2: Rate applied when resuming after a yield.
    """

    approach_radius_m: float = 25.0
    deceleration_ms2: float = 3.0
    recovery_accel_ms2: float = 1.5

    def apply(
        self,
        agents: list[Agent],
        intersection: Intersection,
        dt: float,
    ) -> None:
        """Apply yield-or-proceed logic to each approaching agent."""
        centre = intersection.centre
        for i, agent in enumerate(agents):
            dist = agent.position.distance_to(centre)
            if dist > self.approach_radius_m or dist < 2.0:
                continue
            if agent.speed < 0.05:
                continue
            own_arrival = dist / agent.speed
            if self._must_yield(agent, i, agents, centre, own_arrival):
                agent.speed = max(0.0, agent.speed - self.deceleration_ms2 * dt)
            elif isinstance(agent, VehicleAgent) and (agent.speed < agent.cruise_speed):
                agent.speed = min(
                    agent.cruise_speed,
                    agent.speed + self.recovery_accel_ms2 * dt,
                )

    def _must_yield(
        self,
        agent: Agent,
        self_idx: int,
        agents: list[Agent],
        centre: Vec2,
        own_arrival: float,
    ) -> bool:
        critical_gap = getattr(agent, "critical_gap_s", 4.0)
        cx = centre.x
        cy = centre.y
        for j, other in enumerate(agents):
            if j == self_idx:
                continue
            if not self._is_conflicting_arm(agent, other):
                continue
            d_other = math.hypot(other.position.x - cx, other.position.y - cy)
            if d_other > self.approach_radius_m:
                continue
            if other.speed < 0.05:
                continue
            other_arrival = d_other / other.speed
            if other_arrival >= own_arrival:
                continue
            if (own_arrival - other_arrival) < critical_gap:
                return True
        return False

    @staticmethod
    def _is_conflicting_arm(a: Agent, b: Agent) -> bool:
        diff = abs(a.heading - b.heading) % (2.0 * math.pi)
        if diff > math.pi:
            diff = 2.0 * math.pi - diff
        return diff >= _SAME_ARM_ANGLE_RAD
