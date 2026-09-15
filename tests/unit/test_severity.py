"""Unit tests for conflict severity classification."""

from surrogate_safety_abm.analysis import (
    DRAC_CRITICAL_THRESHOLD_MS2,
    TTC_SERIOUS_THRESHOLD_S,
    TTC_SLIGHT_THRESHOLD_S,
    Severity,
    classify_severity,
    severity_counts,
)
from surrogate_safety_abm.simulation import ConflictEvent


def _event(ttc: float, drac: float) -> ConflictEvent:
    return ConflictEvent(
        time_s=1.0,
        agent_a="a",
        agent_b="b",
        ttc_s=ttc,
        drac_ms2=drac,
        distance_m=5.0,
    )


class TestClassifySeverity:
    """Unit tests for classify_severity."""

    def test_serious_by_low_ttc(self) -> None:
        ev = _event(ttc=1.0, drac=1.0)
        assert classify_severity(ev) == Severity.SERIOUS

    def test_serious_by_high_drac(self) -> None:
        ev = _event(ttc=5.0, drac=DRAC_CRITICAL_THRESHOLD_MS2 + 0.1)
        assert classify_severity(ev) == Severity.SERIOUS

    def test_slight(self) -> None:
        ev = _event(ttc=2.0, drac=1.0)
        assert classify_severity(ev) == Severity.SLIGHT

    def test_safe(self) -> None:
        ev = _event(ttc=10.0, drac=0.5)
        assert classify_severity(ev) == Severity.SAFE

    def test_boundary_ttc_exactly_serious_threshold(self) -> None:
        ev = _event(ttc=TTC_SERIOUS_THRESHOLD_S, drac=0.0)
        assert classify_severity(ev) == Severity.SLIGHT

    def test_boundary_ttc_exactly_slight_threshold(self) -> None:
        ev = _event(ttc=TTC_SLIGHT_THRESHOLD_S, drac=0.0)
        assert classify_severity(ev) == Severity.SAFE

    def test_boundary_drac_exactly_critical(self) -> None:
        ev = _event(ttc=10.0, drac=DRAC_CRITICAL_THRESHOLD_MS2)
        assert classify_severity(ev) == Severity.SERIOUS


class TestSeverityCounts:
    """Unit tests for severity_counts."""

    def test_empty_events(self) -> None:
        counts = severity_counts([])
        assert counts == {Severity.SAFE: 0, Severity.SLIGHT: 0, Severity.SERIOUS: 0}

    def test_mixed_events(self) -> None:
        events = [
            _event(0.5, 5.0),
            _event(2.0, 1.0),
            _event(10.0, 0.5),
            _event(0.3, 8.0),
        ]
        counts = severity_counts(events)
        assert counts[Severity.SERIOUS] == 2
        assert counts[Severity.SLIGHT] == 1
        assert counts[Severity.SAFE] == 1

    def test_counts_sum_to_total(self) -> None:
        events = [_event(1.0, 5.0) for _ in range(3)]
        counts = severity_counts(events)
        assert sum(counts.values()) == 3
