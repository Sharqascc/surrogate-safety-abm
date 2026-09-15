"""Collapse consecutive conflict events into distinct conflict episodes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from surrogate_safety_abm.simulation.recorder import ConflictEvent

if TYPE_CHECKING:
    from surrogate_safety_abm.analysis.severity import Severity


def _normalise_pair(a: str, b: str) -> tuple[str, str]:
    """Return the two agent ids in a canonical (sorted) order."""
    return (a, b) if a <= b else (b, a)


DEFAULT_MAX_GAP_S: float = 0.5
"""Maximum time gap (s) between same-pair events to remain one episode."""


@dataclass(frozen=True, slots=True)
class ConflictEpisode:
    """A continuous conflict between two agents."""

    agent_a: str
    agent_b: str
    start_time_s: float
    end_time_s: float
    min_ttc_s: float
    max_drac_ms2: float
    min_distance_m: float
    event_count: int

    @property
    def duration_s(self) -> float:
        """Duration of the episode in seconds."""
        return self.end_time_s - self.start_time_s

    def severity(self) -> Severity:
        """Classify the episode using its worst-case TTC and DRAC."""
        from surrogate_safety_abm.analysis.severity import (
            classify_severity,
        )

        synthetic = ConflictEvent(
            time_s=self.start_time_s,
            agent_a=self.agent_a,
            agent_b=self.agent_b,
            ttc_s=self.min_ttc_s,
            drac_ms2=self.max_drac_ms2,
            distance_m=self.min_distance_m,
        )
        return classify_severity(synthetic)


def collapse_episodes(
    events: Iterable[ConflictEvent],
    max_gap_s: float = DEFAULT_MAX_GAP_S,
) -> list[ConflictEpisode]:
    """Merge temporally contiguous same-pair events into episodes."""
    if max_gap_s < 0.0:
        raise ValueError("max_gap_s must be non-negative")

    normalized = sorted(
        ((_normalise_pair(ev.agent_a, ev.agent_b), ev) for ev in events),
        key=lambda item: (item[0], item[1].time_s),
    )

    episodes: list[ConflictEpisode] = []
    current_pair: tuple[str, str] | None = None
    current_events: list[ConflictEvent] = []

    for pair, ev in normalized:
        if current_pair is None:
            current_pair, current_events = pair, [ev]
            continue
        same_pair = pair == current_pair
        within_gap = (ev.time_s - current_events[-1].time_s) <= max_gap_s
        if same_pair and within_gap:
            current_events.append(ev)
        else:
            episodes.append(_make_episode(current_pair, current_events))
            current_pair, current_events = pair, [ev]
    if current_pair is not None:
        episodes.append(_make_episode(current_pair, current_events))

    return episodes


def _make_episode(
    pair: tuple[str, str],
    events: list[ConflictEvent],
) -> ConflictEpisode:
    """Aggregate a non-empty event list into one episode."""
    return ConflictEpisode(
        agent_a=pair[0],
        agent_b=pair[1],
        start_time_s=events[0].time_s,
        end_time_s=events[-1].time_s,
        min_ttc_s=min(e.ttc_s for e in events),
        max_drac_ms2=max(e.drac_ms2 for e in events),
        min_distance_m=min(e.distance_m for e in events),
        event_count=len(events),
    )


def episode_severity_counts(
    episodes: list[ConflictEpisode],
) -> dict[Severity, int]:
    """Count episodes by severity class, with all keys present."""
    from surrogate_safety_abm.analysis.severity import Severity

    counts: dict[Severity, int] = {s: 0 for s in Severity}
    for ep in episodes:
        counts[ep.severity()] += 1
    return counts
