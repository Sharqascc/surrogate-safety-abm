"""Unit tests for the Recorder."""

from dataclasses import FrozenInstanceError

import pytest

from surrogate_safety_abm.agents import VehicleAgent
from surrogate_safety_abm.geometry import ORIGIN
from surrogate_safety_abm.simulation import (
    AgentSnapshot,
    ConflictEvent,
    Recorder,
)


class TestRecorder:
    """Unit tests for Recorder."""

    def test_empty(self) -> None:
        r = Recorder()
        assert r.snapshots == []
        assert r.conflicts == []

    def test_record_step_appends_one_per_agent(self) -> None:
        r = Recorder()
        agents = [
            VehicleAgent("a", ORIGIN, 5.0, 0.0),
            VehicleAgent("b", ORIGIN, 5.0, 0.0),
        ]
        r.record_step(1.0, agents)
        assert len(r.snapshots) == 2
        assert all(s.time_s == 1.0 for s in r.snapshots)

    def test_trajectory_filters_by_id(self) -> None:
        r = Recorder()
        a = VehicleAgent("a", ORIGIN, 5.0, 0.0)
        b = VehicleAgent("b", ORIGIN, 5.0, 0.0)
        r.record_step(1.0, [a, b])
        r.record_step(2.0, [a, b])
        traj_a = r.trajectory("a")
        assert len(traj_a) == 2
        assert all(s.agent_id == "a" for s in traj_a)

    def test_trajectory_unknown_id_empty(self) -> None:
        r = Recorder()
        assert r.trajectory("nope") == []

    def test_record_conflict(self) -> None:
        r = Recorder()
        ev = ConflictEvent(
            time_s=1.0,
            agent_a="a",
            agent_b="b",
            ttc_s=2.0,
            drac_ms2=3.0,
            distance_m=5.0,
        )
        r.record_conflict(ev)
        assert r.conflicts == [ev]

    def test_snapshot_is_frozen(self) -> None:
        s = AgentSnapshot("a", ORIGIN, 5.0, 0.0, 1.0)
        with pytest.raises(FrozenInstanceError):
            s.speed = 9.0  # type: ignore[misc]

    def test_event_is_frozen(self) -> None:
        ev = ConflictEvent(1.0, "a", "b", 2.0, 3.0, 5.0)
        with pytest.raises(FrozenInstanceError):
            ev.ttc_s = 9.0  # type: ignore[misc]
