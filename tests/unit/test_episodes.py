"""Unit + property tests for conflict-episode aggregation."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from surrogate_safety_abm.analysis import (
    Severity,
    collapse_episodes,
    episode_severity_counts,
)
from surrogate_safety_abm.simulation import ConflictEvent


def _ev(t, a, b, ttc=2.0, drac=1.0, dist=5.0):
    return ConflictEvent(
        time_s=t,
        agent_a=a,
        agent_b=b,
        ttc_s=ttc,
        drac_ms2=drac,
        distance_m=dist,
    )


class TestCollapseEpisodes:
    def test_empty(self):
        assert collapse_episodes([]) == []

    def test_single_event(self):
        eps = collapse_episodes([_ev(1.0, "a", "b")])
        assert len(eps) == 1
        assert eps[0].agent_a == "a"
        assert eps[0].agent_b == "b"
        assert eps[0].event_count == 1

    def test_consecutive_same_pair_merged(self):
        events = [_ev(t, "a", "b") for t in [1.0, 1.1, 1.2, 1.3]]
        eps = collapse_episodes(events)
        assert len(eps) == 1
        assert eps[0].event_count == 4
        assert eps[0].start_time_s == 1.0
        assert eps[0].end_time_s == 1.3

    def test_gap_splits_episode(self):
        events = [_ev(1.0, "a", "b"), _ev(1.1, "a", "b"), _ev(5.0, "a", "b"), _ev(5.1, "a", "b")]
        eps = collapse_episodes(events, max_gap_s=0.5)
        assert len(eps) == 2
        assert eps[0].event_count == 2
        assert eps[1].event_count == 2

    def test_different_pairs_not_merged(self):
        events = [_ev(1.0, "a", "b"), _ev(1.1, "c", "d")]
        eps = collapse_episodes(events)
        assert len(eps) == 2

    def test_pair_order_normalized(self):
        events = [_ev(1.0, "b", "a"), _ev(1.1, "a", "b")]
        eps = collapse_episodes(events)
        assert len(eps) == 1
        assert eps[0].agent_a == "a"
        assert eps[0].agent_b == "b"
        assert eps[0].event_count == 2

    def test_worst_case_aggregation(self):
        events = [
            _ev(1.0, "a", "b", ttc=3.0, drac=1.0, dist=10.0),
            _ev(1.1, "a", "b", ttc=0.8, drac=8.0, dist=4.0),
            _ev(1.2, "a", "b", ttc=2.0, drac=2.0, dist=6.0),
        ]
        ep = collapse_episodes(events)[0]
        assert ep.min_ttc_s == 0.8
        assert ep.max_drac_ms2 == 8.0
        assert ep.min_distance_m == 4.0

    def test_duration_property(self):
        events = [_ev(1.0, "a", "b"), _ev(1.5, "a", "b")]
        ep = collapse_episodes(events, max_gap_s=1.0)[0]
        assert ep.duration_s == pytest.approx(0.5)

    def test_negative_max_gap_raises(self):
        with pytest.raises(ValueError, match="max_gap_s"):
            collapse_episodes([_ev(1.0, "a", "b")], max_gap_s=-0.1)

    def test_zero_max_gap_only_merges_simultaneous(self):
        events = [_ev(1.0, "a", "b"), _ev(1.0, "a", "b"), _ev(1.1, "a", "b")]
        eps = collapse_episodes(events, max_gap_s=0.0)
        assert len(eps) == 2


class TestEpisodeSeverity:
    def test_serious_episode(self):
        ep = collapse_episodes([_ev(1.0, "a", "b", ttc=0.5, drac=10.0)])[0]
        assert ep.severity() == Severity.SERIOUS

    def test_slight_episode(self):
        ep = collapse_episodes([_ev(1.0, "a", "b", ttc=2.0, drac=1.0)])[0]
        assert ep.severity() == Severity.SLIGHT

    def test_safe_episode(self):
        ep = collapse_episodes([_ev(1.0, "a", "b", ttc=10.0, drac=0.5)])[0]
        assert ep.severity() == Severity.SAFE


class TestEpisodeSeverityCounts:
    def test_empty(self):
        assert episode_severity_counts([]) == {
            Severity.SAFE: 0,
            Severity.SLIGHT: 0,
            Severity.SERIOUS: 0,
        }

    def test_mixed(self):
        events = [
            _ev(1.0, "a", "b", ttc=0.5, drac=10.0),
            _ev(5.0, "c", "d", ttc=2.0, drac=1.0),
            _ev(9.0, "e", "f", ttc=10.0, drac=0.5),
        ]
        eps = collapse_episodes(events)
        counts = episode_severity_counts(eps)
        assert counts[Severity.SERIOUS] == 1
        assert counts[Severity.SLIGHT] == 1
        assert counts[Severity.SAFE] == 1

    def test_episode_is_frozen(self):
        from dataclasses import FrozenInstanceError

        ep = collapse_episodes([_ev(1.0, "a", "b")])[0]
        with pytest.raises(FrozenInstanceError):
            ep.min_ttc_s = 0.0


# ---------------- Property tests ----------------

_finite = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)


@given(
    events=st.lists(
        st.builds(
            ConflictEvent,
            time_s=st.floats(min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False),
            agent_a=st.sampled_from(["a", "b", "c"]),
            agent_b=st.sampled_from(["d", "e", "f"]),
            ttc_s=_finite,
            drac_ms2=_finite,
            distance_m=_finite,
        ),
        max_size=100,
    ),
)
def test_episode_count_leq_event_count(events):
    """Deduplication never produces more episodes than events."""
    assert len(collapse_episodes(events)) <= len(events)


@given(
    events=st.lists(
        st.builds(
            ConflictEvent,
            time_s=st.floats(min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False),
            agent_a=st.sampled_from(["a", "b", "c"]),
            agent_b=st.sampled_from(["d", "e", "f"]),
            ttc_s=_finite,
            drac_ms2=_finite,
            distance_m=_finite,
        ),
        max_size=50,
    ),
)
def test_event_count_conservation(events):
    """Every event belongs to exactly one episode."""
    eps = collapse_episodes(events)
    total = sum(ep.event_count for ep in eps)
    assert total == len(events)


@given(
    events=st.lists(
        st.builds(
            ConflictEvent,
            time_s=st.floats(min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False),
            agent_a=st.sampled_from(["a", "b"]),
            agent_b=st.sampled_from(["c", "d"]),
            ttc_s=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
            drac_ms2=_finite,
            distance_m=_finite,
        ),
        max_size=50,
    ),
)
def test_episode_bounds_are_within_events(events):
    """Episode min/max lie within the aggregated events range."""
    eps = collapse_episodes(events)
    for ep in eps:
        assert ep.min_ttc_s >= 0.0
        assert ep.max_drac_ms2 >= 0.0
        assert ep.end_time_s >= ep.start_time_s
