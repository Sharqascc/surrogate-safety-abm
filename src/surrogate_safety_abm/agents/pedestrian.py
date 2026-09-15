"""Pedestrian agent with crossing-decision state."""

from __future__ import annotations

from dataclasses import dataclass

from surrogate_safety_abm.agents.base import Agent
from surrogate_safety_abm.geometry import ORIGIN, Vec2


@dataclass
class PedestrianAgent(Agent):
    """A pedestrian moving toward a crossing goal.

    Attributes:
        goal: Position the pedestrian is attempting to reach (metres).
        risk_propensity: Willingness to accept small gaps, in [0, 1].
            Higher values model more aggressive crossing behaviour.
        is_crossing: Whether the pedestrian is currently on the roadway.
    """

    goal: Vec2 = ORIGIN
    risk_propensity: float = 0.5
    is_crossing: bool = False

    def __post_init__(self) -> None:
        """Validate the pedestrian's risk parameter after construction."""
        super().__post_init__()
        if not 0.0 <= self.risk_propensity <= 1.0:
            raise ValueError("risk_propensity must lie in [0, 1]")

    def step(self, dt: float) -> None:
        """Advance the pedestrian toward the goal by one time step."""
        self.integrate(dt)
