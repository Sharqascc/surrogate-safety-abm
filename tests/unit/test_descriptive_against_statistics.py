"""Regression tests for analysis.descriptive.summarise.

Every parametrized case below was hand-verified in the 2026-11-22 probe
against statistics.mean, statistics.median, np.percentile(method="linear"),
min, max. 48 cases, all match. These tests lock those values.

Two design properties pinned explicitly:

1. `summarise([0.1, 0.2, 0.3]).mean` returns exactly 0.2 — not the naive
   sum()/len() artifact 0.20000000000000004. Production code uses
   compensated/exact summation.

2. `p85` uses linear interpolation, matching np.percentile(method="linear").
   Changing to "lower", "higher", "nearest", or "midpoint" would fail
   test_p85[2,4] == 3.7 and similar.
"""
from __future__ import annotations
import pytest
from surrogate_safety_abm.analysis.descriptive import summarise

TOL = dict(rel=1e-14, abs=1e-15)


class TestSummariseContract:
    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty"):
            summarise([])


class TestN:
    @pytest.mark.parametrize("xs,expected", [
        ([1,2,3,4,5],  5),
        ([2,4],        2),
        ([10],         1),
        ([-2,-4],      2),
        ([-2,0,2],     3),
        ([0.1,0.2,0.3],3),
        ([1e6,2e6,3e6],3),
        ([1e-6,2e-6],  2),
    ])
    def test_n(self, xs, expected):
        assert summarise(xs).n == expected


class TestMean:
    @pytest.mark.parametrize("xs,expected", [
        ([1,2,3,4,5],   3.0),
        ([2,4],         3.0),
        ([10],         10.0),
        ([-2,-4],      -3.0),
        ([-2,0,2],      0.0),
        ([1e6,2e6,3e6], 2_000_000.0),
        ([1e-6,2e-6],   1.5e-6),
    ])
    def test_mean(self, xs, expected):
        assert summarise(xs).mean == pytest.approx(expected, **TOL)

    def test_mean_compensated_summation(self):
        """[0.1,0.2,0.3] must give exactly 0.2, not 0.20000000000000004.

        Regression guard on the summation method. Naive sum()/len() would
        fail this; statistics.mean and compensated summation pass.
        """
        assert summarise([0.1, 0.2, 0.3]).mean == 0.2


class TestMedian:
    @pytest.mark.parametrize("xs,expected", [
        ([1,2,3,4,5],   3.0),
        ([2,4],         3.0),
        ([10],         10.0),
        ([-2,-4],      -3.0),
        ([-2,0,2],      0.0),
        ([0.1,0.2,0.3], 0.2),
        ([1e6,2e6,3e6], 2_000_000.0),
        ([1e-6,2e-6],   1.5e-6),
    ])
    def test_median(self, xs, expected):
        assert summarise(xs).median == pytest.approx(expected, **TOL)


class TestP85:
    @pytest.mark.parametrize("xs,expected", [
        ([1,2,3,4,5],   4.4),
        ([2,4],         3.7),    # linear interp: 2 + 0.85*(4-2) = 3.7
        ([10],         10.0),
        ([-2,-4],      -2.3),
        ([-2,0,2],      1.4),
        ([0.1,0.2,0.3], 0.27),
        ([1e6,2e6,3e6], 2_700_000.0),
        ([1e-6,2e-6],   1.85e-6),
    ])
    def test_p85(self, xs, expected):
        assert summarise(xs).p85 == pytest.approx(expected, **TOL)


class TestMinimum:
    @pytest.mark.parametrize("xs,expected", [
        ([1,2,3,4,5],   1.0),
        ([2,4],         2.0),
        ([10],         10.0),
        ([-2,-4],      -4.0),
        ([-2,0,2],     -2.0),
        ([0.1,0.2,0.3], 0.1),
        ([1e6,2e6,3e6], 1_000_000.0),
        ([1e-6,2e-6],   1e-6),
    ])
    def test_minimum(self, xs, expected):
        assert summarise(xs).minimum == pytest.approx(expected, **TOL)


class TestMaximum:
    @pytest.mark.parametrize("xs,expected", [
        ([1,2,3,4,5],   5.0),
        ([2,4],         4.0),
        ([10],         10.0),
        ([-2,-4],      -2.0),
        ([-2,0,2],      2.0),
        ([0.1,0.2,0.3], 0.3),
        ([1e6,2e6,3e6], 3_000_000.0),
        ([1e-6,2e-6],   2e-6),
    ])
    def test_maximum(self, xs, expected):
        assert summarise(xs).maximum == pytest.approx(expected, **TOL)
