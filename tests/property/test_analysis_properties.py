"""Property-based tests for the analysis layer."""

from hypothesis import given
from hypothesis import strategies as st

from surrogate_safety_abm.analysis import (
    Severity,
    classify_severity,
    severity_counts,
    summarise,
)
from surrogate_safety_abm.analysis.descriptive import _percentile
from surrogate_safety_abm.simulation import ConflictEvent

_finite = st.floats(min_value=0.0, max_value=1_000.0, allow_nan=False, allow_infinity=False)
_non_empty = st.lists(_finite, min_size=1, max_size=50)


@given(values=_non_empty)
def test_summarise_min_max_order(values: list[float]) -> None:
    s = summarise(values)
    assert s.minimum <= s.median <= s.maximum


@given(values=_non_empty)
def test_summarise_mean_in_range(values: list[float]) -> None:
    s = summarise(values)
    assert s.minimum <= s.mean <= s.maximum


@given(values=_non_empty)
def test_summarise_n_matches_input(values: list[float]) -> None:
    assert summarise(values).n == len(values)


@given(values=_non_empty)
def test_summarise_p85_in_range(values: list[float]) -> None:
    s = summarise(values)
    assert s.minimum <= s.p85 <= s.maximum


@given(values=_non_empty)
def test_percentile_monotonic_in_q(values: list[float]) -> None:
    s = sorted(values)
    p25 = _percentile(s, 0.25)
    p50 = _percentile(s, 0.50)
    p75 = _percentile(s, 0.75)
    assert p25 <= p50 <= p75


@given(ttc=_finite, drac=_finite)
def test_severity_deterministic(ttc: float, drac: float) -> None:
    ev = ConflictEvent(1.0, "a", "b", ttc, drac, 5.0)
    assert classify_severity(ev) == classify_severity(ev)


@given(
    events=st.lists(
        st.builds(
            ConflictEvent,
            time_s=st.just(1.0),
            agent_a=st.just("a"),
            agent_b=st.just("b"),
            ttc_s=_finite,
            drac_ms2=_finite,
            distance_m=st.just(5.0),
        ),
        min_size=0,
        max_size=30,
    )
)
def test_severity_counts_sum(events: list[ConflictEvent]) -> None:
    counts = severity_counts(events)
    assert sum(counts.values()) == len(events)


@given(
    events=st.lists(
        st.builds(
            ConflictEvent,
            time_s=st.just(1.0),
            agent_a=st.just("a"),
            agent_b=st.just("b"),
            ttc_s=_finite,
            drac_ms2=_finite,
            distance_m=st.just(5.0),
        ),
        min_size=0,
        max_size=10,
    )
)
def test_severity_counts_all_keys_present(events: list[ConflictEvent]) -> None:
    counts = severity_counts(events)
    assert set(counts.keys()) == {Severity.SAFE, Severity.SLIGHT, Severity.SERIOUS}
