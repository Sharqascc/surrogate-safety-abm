"""Evasive braking: agents decelerate when TTC drops below thresholds."""

from __future__ import annotations

import math
from dataclasses import dataclass

from surrogate_safety_abm.agents.base import Agent
from surrogate_safety_abm.agents.vehicle import VehicleAgent
from surrogate_safety_abm.environment.intersection import Intersection


def _closing_speed(a: Agent, b: Agent, gap: float) -> float:
    """Relative speed along the a→b line (positive = approaching)."""
    if gap <= 1e-9:
        return 0.0
    ux = (b.position.x - a.position.x) / gap
    uy = (b.position.y - a.position.y) / gap
    va = a.speed * (math.cos(a.heading) * ux + math.sin(a.heading) * uy)
    vb = b.speed * (math.cos(b.heading) * ux + math.sin(b.heading) * uy)
    return va - vb


@dataclass
class EvasiveBrakingModel:
    """Brake when minimum TTC to any other vehicle drops.

    Below ``hard_brake_ttc_s`` the agent targets 0 m/s (full stop).
    Between ``hard_brake_ttc_s`` and ``soft_brake_ttc_s``, target speed
    scales linearly with available TTC.

    Attributes:
        soft_brake_ttc_s: Below this TTC the agent starts easing off.
        hard_brake_ttc_s: Below this TTC the agent targets zero speed.
        max_detection_radius_m: Pairwise TTC evaluation radius.
    """

    soft_brake_ttc_s: float = 3.0
    hard_brake_ttc_s: float = 1.5
    max_detection_radius_m: float = 25.0

    def __post_init__(self) -> None:
        """Validate thresholds."""
        if self.hard_brake_ttc_s <= 0.0:
            raise ValueError("hard_brake_ttc_s must be positive")
        if self.soft_brake_ttc_s <= self.hard_brake_ttc_s:
            raise ValueError("soft_brake_ttc_s must exceed hard_brake_ttc_s")

    def apply(
        self,
        agents: list[Agent],
        intersection: Intersection,
        dt: float,
    ) -> None:
        """Adjust each agent's speed toward a TTC-derived target."""
        del intersection  # unused but part of the protocol
        for i, agent in enumerate(agents):
            min_ttc = self._min_ttc(agent, agents, i)
            cruise = self._cruise_speed(agent)
            if min_ttc < self.hard_brake_ttc_s:
                desired = 0.0
            elif min_ttc < self.soft_brake_ttc_s:
                frac = (min_ttc - self.hard_brake_ttc_s) / (
                    self.soft_brake_ttc_s - self.hard_brake_ttc_s
                )
                desired = cruise * frac
            else:
                desired = cruise
            agent.speed = self._ramp(agent, desired, dt)

    def _min_ttc(
        self,
        agent: Agent,
        agents: list[Agent],
        self_idx: int,
    ) -> float:
        best = float("inf")
        for j, other in enumerate(agents):
            if j == self_idx:
                continue
            gap = agent.position.distance_to(other.position)
            if gap > self.max_detection_radius_m:
                continue
            rel = _closing_speed(agent, other, gap)
            if rel <= 0.0:
                continue
            ttc = gap / rel
            if ttc < best:
                best = ttc
        return best

    @staticmethod
    def _cruise_speed(agent: Agent) -> float:
        if isinstance(agent, VehicleAgent):
            return agent.cruise_speed
        return agent.speed

    @staticmethod
    def _ramp(agent: Agent, desired: float, dt: float) -> float:
        max_accel = getattr(agent, "max_accel", 2.5)
        max_decel = getattr(agent, "max_decel", 4.5)
        if desired < agent.speed:
            return max(0.0, max(desired, agent.speed - max_decel * dt))
        return max(0.0, min(desired, agent.speed + max_accel * dt))
