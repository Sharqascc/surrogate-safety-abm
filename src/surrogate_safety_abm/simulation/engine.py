"""Discrete-time simulation engine with conflict detection.

The engine advances all agents at fixed time steps, records their
trajectories, and detects conflicts between pairs of agents that are
within a conflict radius and on a converging course.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, sin

from surrogate_safety_abm.agents.base import Agent
from surrogate_safety_abm.behaviour.base import BehaviourModel
from surrogate_safety_abm.config.city_profiles import CityProfile
from surrogate_safety_abm.environment.intersection import Intersection
from surrogate_safety_abm.simulation.recorder import ConflictEvent, Recorder
from surrogate_safety_abm.ssm import compute_drac, compute_ttc

DEFAULT_CONFLICT_RADIUS_M: float = 15.0
"""Default radius (metres) within which agent pairs are evaluated."""

DEFAULT_TTC_THRESHOLD_S: float = 5.0
"""Default TTC threshold (seconds) below which a conflict is recorded."""

DEFAULT_MIN_GAP_M: float = 2.0
"""Minimum gap (metres). Pairs closer than this are collisions,
not conflicts, and are excluded from SSM analysis."""


@dataclass
class SimulationConfig:
    """Time and detection parameters for a simulation run.

    Attributes:
        duration_s: Total simulation duration in seconds.
        time_step_s: Integration time step in seconds.
        warm_up_s: Warm-up period during which nothing is recorded.
        conflict_radius_m: Pairwise proximity threshold for detection.
        ttc_threshold_s: TTC below which a conflict event is recorded.
        min_gap_m: Minimum gap (m). Closer pairs are collisions,
            not conflicts, and are excluded from SSM analysis.
        intersection_zone_m: Half-side (m) of the square zone
            around the intersection centre. Only pairs with both
            agents inside are evaluated.
        seed: Optional RNG seed for reproducibility.
    """

    duration_s: float = 60.0
    time_step_s: float = 0.1
    warm_up_s: float = 0.0
    conflict_radius_m: float = DEFAULT_CONFLICT_RADIUS_M
    ttc_threshold_s: float = DEFAULT_TTC_THRESHOLD_S
    min_gap_m: float = DEFAULT_MIN_GAP_M
    intersection_zone_m: float = 10.0
    min_speed_mps: float = 1.4
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate the configuration values."""
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        if self.time_step_s <= 0.0:
            raise ValueError("time_step_s must be positive")
        if self.warm_up_s < 0.0:
            raise ValueError("warm_up_s must be non-negative")
        if self.warm_up_s >= self.duration_s:
            raise ValueError("warm_up_s must be less than duration_s")
        if self.conflict_radius_m <= 0.0:
            raise ValueError("conflict_radius_m must be positive")
        if self.ttc_threshold_s <= 0.0:
            raise ValueError("ttc_threshold_s must be positive")
        if self.min_gap_m < 0.0:
            raise ValueError("min_gap_m must be non-negative")
        if self.min_speed_mps < 0.0:
            raise ValueError("min_speed_mps must be non-negative")
        if self.intersection_zone_m <= 0.0:
            raise ValueError("intersection_zone_m must be positive")


@dataclass
class SimulationResult:
    """Output of a completed simulation run."""

    config: SimulationConfig
    city: CityProfile
    intersection: Intersection
    recorder: Recorder
    steps_completed: int


def _closing_speed(a: Agent, b: Agent, gap: float) -> float:
    """Return the component of relative velocity along the a->b line.

    Positive values indicate the two agents are approaching each other.
    """
    if gap <= 1e-9:
        return 0.0
    ux = (b.position.x - a.position.x) / gap
    uy = (b.position.y - a.position.y) / gap
    va_proj = a.speed * (cos(a.heading) * ux + sin(a.heading) * uy)
    vb_proj = b.speed * (cos(b.heading) * ux + sin(b.heading) * uy)
    return va_proj - vb_proj


@dataclass
class SimulationEngine:
    """Discrete-time engine over a fixed set of agents at one intersection."""

    city: CityProfile
    intersection: Intersection
    agents: list[Agent] = field(default_factory=list)
    config: SimulationConfig = field(default_factory=SimulationConfig)
    behaviours: list[BehaviourModel] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate that agent identifiers are unique."""
        ids = [a.agent_id for a in self.agents]
        if len(ids) != len(set(ids)):
            raise ValueError("agent_id values must be unique")

    def _step_agents(self, dt: float) -> None:
        for behaviour in self.behaviours:
            behaviour.apply(self.agents, self.intersection, dt)
        for agent in self.agents:
            agent.step(dt)

    def _detect_conflicts(self, time_s: float, recorder: Recorder) -> None:
        n = len(self.agents)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.agents[i], self.agents[j]
                if (a.speed < self.config.min_speed_mps
                        or b.speed < self.config.min_speed_mps):
                    continue
                cx = self.intersection.centre.x
                cy = self.intersection.centre.y
                zone = self.config.intersection_zone_m
                if (abs(a.position.x - cx) > zone
                        or abs(a.position.y - cy) > zone
                        or abs(b.position.x - cx) > zone
                        or abs(b.position.y - cy) > zone):
                    continue
                gap = a.position.distance_to(b.position)
                if gap > self.config.conflict_radius_m:
                    continue
                if gap < self.config.min_gap_m:
                    # Physically overlapping pair: collision, not conflict.
                    continue
                rel_speed = _closing_speed(a, b, gap)
                ttc = compute_ttc(gap=gap, relative_speed=rel_speed)
                if not ttc.is_converging:
                    continue
                if ttc.ttc_seconds > self.config.ttc_threshold_s:
                    continue
                drac = compute_drac(gap=gap, relative_speed=rel_speed)
                recorder.record_conflict(
                    ConflictEvent(
                        time_s=time_s,
                        agent_a=a.agent_id,
                        agent_b=b.agent_id,
                        ttc_s=ttc.ttc_seconds,
                        drac_ms2=drac.drac_ms2,
                        distance_m=gap,
                    )
                )

    def run(self) -> SimulationResult:
        """Run the simulation to completion and return the recorded result."""
        dt = self.config.time_step_s
        total_steps = round(self.config.duration_s / dt)
        warmup_steps = round(self.config.warm_up_s / dt)

        recorder = Recorder()
        for step in range(total_steps):
            time_s = (step + 1) * dt
            self._step_agents(dt)
            if step >= warmup_steps:
                recorder.record_step(time_s, self.agents)
                self._detect_conflicts(time_s, recorder)

        return SimulationResult(
            config=self.config,
            city=self.city,
            intersection=self.intersection,
            recorder=recorder,
            steps_completed=total_steps,
        )
