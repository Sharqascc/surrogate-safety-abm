"""Golden-fixture snapshot test for analysis.accuracy.compare.

The fixture in tests/fixtures/accuracy_golden.json is the frozen output of
`compare()` for a fixed observed/simulated pair. If the implementation
changes in a way that alters any field beyond 1e-9, this test fails and
forces a review of the change. That is the same role snapshots play in
other test frameworks, without pulling in a new dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from surrogate_safety_abm.analysis import compare

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "accuracy_golden.json"


def test_compare_matches_golden_fixture() -> None:
    """compare() reproduces the committed golden snapshot to 1e-9."""
    data = json.loads(_FIXTURE.read_text())
    rep = compare(data["observed"], data["simulated"])
    expected = data["expected_report"]

    assert rep.n_observed == expected["n_observed"]
    assert rep.n_simulated == expected["n_simulated"]

    for field in (
        "rmse",
        "mae",
        "pearson_r",
        "pearson_p",
        "ks_statistic",
        "ks_p",
        "mean_observed",
        "mean_simulated",
        "mean_diff",
        "mean_diff_ci95_low",
        "mean_diff_ci95_high",
    ):
        actual = getattr(rep, field)
        assert actual == pytest.approx(expected[field], abs=1e-9), field
