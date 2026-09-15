"""Trajectory and conflict-event recording for the simulation engine."""

from __future__ import annotations

from dataclasses import dataclass, field

from surrogate_safety_abm.agents.base import Agent
from surrogate_safety_abm.geometry import Vec2


@dataclass(frozen=True, slots=True)
class AgentSnapshot:
    """Immutable snapshot of one agent's kinematic state at a time instant."""

    agent_id: str
    position: Vec2
    speed: float
    heading: float
    time_s: float


@dataclass(frozen=True, slots=True)
class ConflictEvent:
    """A conflict detected between two agents at a specific time.

    Attributes:
        time_s: Simulation time of detection (seconds).
        agent_a: Identifier of the first agent.
        agent_b: Identifier of the second agent.
        ttc_s: Time-to-collision at detection (seconds).
        drac_ms2: Deceleration Rate to Avoid Crash at detection (m/s^2).
        distance_m: Separation distance between the agents (metres).
    """

    time_s: float
    agent_a: str
    agent_b: str
    ttc_s: float
    drac_ms2: float
    distance_m: float


@dataclass
class Recorder:
    """Accumulates agent snapshots and detected conflict events."""

    snapshots: list[AgentSnapshot] = field(default_factory=list)
    conflicts: list[ConflictEvent] = field(default_factory=list)

    def record_step(self, time_s: float, agents: list[Agent]) -> None:
        """Append one snapshot per agent at time ``time_s``."""
        for agent in agents:
            self.snapshots.append(
                AgentSnapshot(
                    agent_id=agent.agent_id,
                    position=agent.position,
                    speed=agent.speed,
                    heading=agent.heading,
                    time_s=time_s,
                )
            )

    def record_conflict(self, event: ConflictEvent) -> None:
        """Append a conflict event."""
        self.conflicts.append(event)

    def trajectory(self, agent_id: str) -> list[AgentSnapshot]:
        """Return all snapshots for the given agent in time order."""
        return [s for s in self.snapshots if s.agent_id == agent_id]
