"""Abstract base class for all traffic agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import cos, sin

from surrogate_safety_abm.geometry import Vec2


@dataclass
class Agent(ABC):
    """Abstract traffic agent with mutable kinematic state.

    Attributes:
        agent_id: Unique identifier within a simulation run.
        position: Current position in the world frame (metres).
        speed: Scalar speed along the heading (metres per second).
        heading: Direction of travel in radians (0 = +X axis,
            counter-clockwise positive).
    """

    agent_id: str
    position: Vec2
    speed: float
    heading: float

    def __post_init__(self) -> None:
        """Validate kinematic state after construction."""
        if self.speed < 0.0:
            raise ValueError("speed must be non-negative")

    def integrate(self, dt: float) -> None:
        """Advance position along the current heading for ``dt`` seconds.

        Args:
            dt: Time step in seconds (must be non-negative).

        Raises:
            ValueError: If ``dt`` is negative.
        """
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        dx = self.speed * cos(self.heading) * dt
        dy = self.speed * sin(self.heading) * dt
        self.position = Vec2(self.position.x + dx, self.position.y + dy)

    @abstractmethod
    def step(self, dt: float) -> None:
        """Advance this agent by one time step of ``dt`` seconds.

        Subclasses implement behaviour-specific logic (e.g. car-following,
        gap acceptance) and must call :meth:`integrate` to move the agent.
        """
