"""Property-based tests for the simulation engine."""

from hypothesis import given
from hypothesis import strategies as st

from surrogate_safety_abm.agents import VehicleAgent
from surrogate_safety_abm.config.city_profiles import SURAT
from surrogate_safety_abm.environment import Intersection
from surrogate_safety_abm.geometry import Vec2
from surrogate_safety_abm.simulation import SimulationConfig, SimulationEngine

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
    assert times == sorted(times)
    assert len(times) == len(set(times))
