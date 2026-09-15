"""Property-based tests for agents and geometry."""

from hypothesis import given
from hypothesis import strategies as st

from surrogate_safety_abm.agents import PedestrianAgent, VehicleAgent
from surrogate_safety_abm.geometry import ORIGIN, Vec2

_finite = st.floats(min_value=-1_000.0, max_value=1_000.0, allow_nan=False, allow_infinity=False)
_nonneg = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)


@given(x1=_finite, y1=_finite, x2=_finite, y2=_finite)
def test_vec2_distance_symmetric(x1: float, y1: float, x2: float, y2: float) -> None:
    """Distance between two points is symmetric."""
    a, b = Vec2(x1, y1), Vec2(x2, y2)
    assert a.distance_to(b) == b.distance_to(a)


@given(x=_finite, y=_finite)
def test_vec2_distance_non_negative(x: float, y: float) -> None:
    """Distance is always non-negative."""
    assert ORIGIN.distance_to(Vec2(x, y)) >= 0.0


@given(x=_finite, y=_finite)
def test_vec2_self_distance_zero(x: float, y: float) -> None:
    """A point has zero distance to itself."""
    v = Vec2(x, y)
    assert v.distance_to(v) == 0.0


@given(
    vx=_finite,
    vy=_finite,
    s=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_vec2_scalar_mul_commutes(vx: float, vy: float, s: float) -> None:
    """Scalar multiplication commutes."""
    v = Vec2(vx, vy)
    assert v * s == s * v


@given(speed=_nonneg, dt=_nonneg)
def test_vehicle_step_advances_by_speed_times_dt(speed: float, dt: float) -> None:
    """A step along heading=0 advances x by speed*dt."""
    # Construct with max_speed >= speed so the invariant under test
    # (kinematic advance) is isolated from the max_speed constraint.
    v = VehicleAgent("v", ORIGIN, speed, 0.0, max_speed=max(speed, 1.0))
    v.step(dt)
    assert v.position.x >= -1e-9
    assert v.position.x <= speed * dt + 1e-9


@given(risk=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_pedestrian_accepts_valid_risk(risk: float) -> None:
    """Any risk_propensity in [0, 1] is accepted."""
    p = PedestrianAgent("p", ORIGIN, 1.2, 0.0, risk_propensity=risk)
    assert p.risk_propensity == risk
