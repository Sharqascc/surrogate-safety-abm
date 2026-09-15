"""Unit tests for descriptive statistics."""

import pytest

from surrogate_safety_abm.analysis import (
    SSMSummary,
    summarise,
    summarise_drac,
    summarise_ttc,
)
from surrogate_safety_abm.analysis.descriptive import _percentile
from surrogate_safety_abm.simulation import ConflictEvent


class TestPercentile:
    """Unit tests for the internal _percentile helper."""

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _percentile([], 0.5)

    def test_out_of_range_q_raises(self) -> None:
        with pytest.raises(ValueError, match="q must lie"):
            _percentile([1.0, 2.0], 1.5)
        with pytest.raises(ValueError, match="q must lie"):
            _percentile([1.0, 2.0], -0.1)

    def test_single_value(self) -> None:
        assert _percentile([42.0], 0.85) == 42.0

    def test_median_of_odd_list(self) -> None:
        assert _percentile([1.0, 2.0, 3.0], 0.5) == pytest.approx(2.0)

    def test_median_of_even_list(self) -> None:
        assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)

    def test_q_zero_returns_min(self) -> None:
        assert _percentile([1.0, 5.0, 9.0], 0.0) == 1.0

    def test_q_one_returns_max(self) -> None:
        assert _percentile([1.0, 5.0, 9.0], 1.0) == 9.0


class TestSummarise:
    """Unit tests for summarise."""

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            summarise([])

    def test_single_value(self) -> None:
        s = summarise([7.0])
        assert isinstance(s, SSMSummary)
        assert s.n == 1
        assert s.mean == 7.0
        assert s.median == 7.0
        assert s.p85 == 7.0
        assert s.minimum == 7.0
        assert s.maximum == 7.0

    def test_known_list(self) -> None:
        s = summarise([1.0, 2.0, 3.0, 4.0, 5.0])
        assert s.n == 5
        assert s.mean == pytest.approx(3.0)
        assert s.median == pytest.approx(3.0)
        assert s.minimum == 1.0
        assert s.maximum == 5.0

    def test_p85_between_median_and_max(self) -> None:
        s = summarise(list(range(1, 101)))
        assert s.median < s.p85 < s.maximum


class TestSummariseFromEvents:
    """Unit tests for summarise_ttc / summarise_drac."""

    def _events(self) -> list[ConflictEvent]:
        return [
            ConflictEvent(1.0, "a", "b", 2.0, 1.0, 5.0),
            ConflictEvent(2.0, "a", "b", 4.0, 3.0, 5.0),
            ConflictEvent(3.0, "a", "b", 6.0, 5.0, 5.0),
        ]

    def test_summarise_ttc(self) -> None:
        s = summarise_ttc(self._events())
        assert s.n == 3
        assert s.mean == pytest.approx(4.0)

    def test_summarise_drac(self) -> None:
        s = summarise_drac(self._events())
        assert s.n == 3
        assert s.mean == pytest.approx(3.0)
