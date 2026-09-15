"""Protocol for behavioural models applied by the simulation engine."""

from __future__ import annotations

from typing import Protocol

from surrogate_safety_abm.agents.base import Agent
from surrogate_safety_abm.environment.intersection import Intersection


class BehaviourModel(Protocol):
    """A behavioural modifier that adjusts agents in place before each step.

    Implementations must be *pure* with respect to their public API: they
    receive the current list of agents and modify only their kinematic
    state (speed, heading, position via integration).
    """

    def apply(
        self,
        agents: list[Agent],
        intersection: Intersection,
        dt: float,
    ) -> None:
        """Mutate agent state for the upcoming time step."""
        ...
