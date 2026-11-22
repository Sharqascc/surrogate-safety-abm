"""Property-based tests for the simulation engine."""

from hypothesis import given
from hypothesis import strategies as st

from surrogate_safety_abm.agents import VehicleAgent
from surrogate_safety_abm.config.city_profiles import SURAT
from surrogate_safety_abm.environment import Intersection
from surrogate_safety_abm.geometry import Vec2
from surrogate_safety_abm.simulation import SimulationConfig, SimulationEngine
import math

def _compare_records(a, b, path=""):
    """Yield (path, a, b) for fields where a and b differ.
    Handles dataclasses, lists, ints (exact), floats (1e-9 rel+abs),
    NaN==NaN. Falls back to != for unknown types."""
    import math
    if isinstance(a, bool) or isinstance(b, bool):
        if a != b: yield (path, a, b)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            if a != b: yield (path, a, b)
        else:
            if math.isnan(a) and math.isnan(b): return
            if abs(a-b) > 1e-9 + 1e-9*abs(a): yield (path, a, b)
    elif hasattr(a, "__dataclass_fields__"):
        for f in a.__dataclass_fields__:
            yield from _compare_records(getattr(a,f), getattr(b,f), path+"."+f)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            yield (path, "len="+str(len(a)), "len="+str(len(b))); return
        for i,(x,y) in enumerate(zip(a,b)):
            yield from _compare_records(x, y, path+"["+str(i)+"]")
    else:
        if a != b: yield (path, a, b)



_duration = st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False)
_dt = st.floats(min_value=0.05, max_value=0.5, allow_nan=False, allow_infinity=False)
_finite = st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False)


def _empty() -> Intersection:
    return Intersection(intersection_id="prop")


@given(duration=_duration, dt=_dt)
def test_steps_completed_equals_rounded_duration(duration: float, dt: float) -> None:
    """steps_completed == round(duration / dt)."""
    engine = SimulationEngine(
        city=SURAT,
        intersection=_empty(),
        agents=[],
        config=SimulationConfig(duration_s=duration, time_step_s=dt),
    )
    result = engine.run()
    assert result.steps_completed == round(duration / dt)


@given(duration=_duration)
def test_snapshot_count_equals_agents_times_steps(duration: float) -> None:
    """With warm_up=0, snapshots == n_agents * steps."""
    dt = 0.1
    agents = [
        VehicleAgent("a", Vec2(-50.0, 0.0), 5.0, 0.0),
        VehicleAgent("b", Vec2(50.0, 0.0), 5.0, 3.14159265),
    ]
    engine = SimulationEngine(
        city=SURAT,
        intersection=_empty(),
        agents=agents,
        config=SimulationConfig(duration_s=duration, time_step_s=dt),
    )
    result = engine.run()
    expected = len(agents) * round(duration / dt)
    assert len(result.recorder.snapshots) == expected


@given(x=_finite, y=_finite)
def test_all_snapshot_positions_finite(x: float, y: float) -> None:
    """Recorded positions remain finite given finite initial state."""
    a = VehicleAgent("a", Vec2(x, y), 5.0, 0.0)
    engine = SimulationEngine(
        city=SURAT,
        intersection=_empty(),
        agents=[a],
        config=SimulationConfig(duration_s=1.0, time_step_s=0.1),
    )
    result = engine.run()
    for snap in result.recorder.snapshots:
        assert abs(snap.position.x) < 1e9
        assert abs(snap.position.y) < 1e9


@given(seed=st.integers(min_value=0, max_value=10_000))
def test_run_is_deterministic(seed: int) -> None:
    """Two runs with the same seed produce identical event counts."""

    def build() -> SimulationEngine:
        return SimulationEngine(
            city=SURAT,
            intersection=_empty(),
            agents=[
                VehicleAgent("a", Vec2(-30.0, 0.0), 8.0, 0.0),
                VehicleAgent("b", Vec2(30.0, 0.0), 8.0, 3.14159265),
            ],
            config=SimulationConfig(duration_s=3.0, time_step_s=0.2, seed=seed),
        )

    r1 = build().run()
    r2 = build().run()
    assert r1.steps_completed == r2.steps_completed
    assert len(r1.recorder.snapshots) == len(r2.recorder.snapshots)
    assert len(r1.recorder.conflicts) == len(r2.recorder.conflicts)

    for i, (s1, s2) in enumerate(zip(r1.recorder.snapshots, r2.recorder.snapshots)):
        diffs = list(_compare_records(s1, s2, "snapshots["+str(i)+"]"))
        assert not diffs, (
            "snapshot " + str(i) + " differs between runs:\n"
            + "\n".join("  " + p + ": " + repr(a) + " != " + repr(b) for p,a,b in diffs)
        )
    for i, (c1, c2) in enumerate(zip(r1.recorder.conflicts, r2.recorder.conflicts)):
        diffs = list(_compare_records(c1, c2, "conflicts["+str(i)+"]"))
        assert not diffs, (
            "conflict " + str(i) + " differs between runs:\n"
            + "\n".join("  " + p + ": " + repr(a) + " != " + repr(b) for p,a,b in diffs)
        )


@given(dt=_dt)
def test_time_monotonic_in_snapshots(dt: float) -> None:
    """Recorded times are strictly increasing across steps."""
    a = VehicleAgent("a", Vec2(-5.0, 0.0), 1.0, 0.0)
    engine = SimulationEngine(
        city=SURAT,
        intersection=_empty(),
        agents=[a],
        config=SimulationConfig(duration_s=2.0, time_step_s=dt),
    )
    result = engine.run()
    times = [s.time_s for s in result.recorder.snapshots]
    unique_step_times = sorted(set(times))
    assert all(b > a for a, b in zip(unique_step_times, unique_step_times[1:]))
