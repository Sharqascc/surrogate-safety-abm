"""Smoke tests for the CLI entry point."""

import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_cli_runs_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """CLI runs a full simulation, writes a CSV, and populates conflicts."""
    from surrogate_safety_abm import cli

    out = tmp_path / "conflicts.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "surrogate-safety-sim",
            "--duration",
            "8.0",
            "--dt",
            "0.1",
            "--seed",
            "42",
            "--out",
            str(out),
        ],
    )
    rc = cli.main()
    assert rc == 0
    assert out.exists()

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "time_s,agent_a,agent_b,ttc_s,drac_ms2,distance_m"
    # The 4-vehicle demo produces conflict events within 8 s.
    assert len(lines) > 1, "expected at least one conflict row"
