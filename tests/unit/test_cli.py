"""Tests for the CLI entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from surrogate_safety_abm import cli
from surrogate_safety_abm.behaviour import EvasiveBrakingModel


def _run(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["surrogate-safety-sim", *argv])
    return cli.main(argv)


@pytest.mark.integration
def test_cli_runs_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """Default (4-vehicle demo) runs, writes CSV, has expected header."""
    out = tmp_path / "conflicts.csv"
    rc = _run(
        monkeypatch,
        ["--duration", "8.0", "--dt", "0.1", "--seed", "42", "--out", str(out)],
    )
    assert rc == 0
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "time_s,agent_a,agent_b,ttc_s,drac_ms2,distance_m"
    assert len(lines) > 1


def test_cli_vehicle_count_dry_run(tmp_path: Path, monkeypatch, capsys) -> None:
    """--vehicle-count N is reflected in the dry-run header."""
    rc = _run(
        monkeypatch,
        ["--city", "vadodara", "--vehicle-count", "30", "--dry-run"],
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "agents:           30" in out
    assert "city:             Vadodara" in out
    assert not (tmp_path / "conflicts.csv").exists()


def test_cli_dry_run_writes_no_csv(tmp_path: Path, monkeypatch) -> None:
    """--dry-run exits without touching the output CSV."""
    out = tmp_path / "should_not_exist.csv"
    rc = _run(monkeypatch, ["--dry-run", "--out", str(out)])
    assert rc == 0
    assert not out.exists()


@pytest.mark.parametrize("kind", ["none", "standard", "calibrated"])
def test_cli_each_behaviour_kind_dry_runs(kind: str, monkeypatch) -> None:
    """Each --behaviours choice parses and dry-runs cleanly."""
    rc = _run(monkeypatch, ["--behaviours", kind, "--dry-run"])
    assert rc == 0


def test_cli_calibrated_uses_gentler_braking() -> None:
    """calibrated -> EvasiveBrakingModel with soft=2.0, hard=1.0."""
    behaviours = cli._resolve_behaviours("calibrated")
    evasive = [b for b in behaviours if isinstance(b, EvasiveBrakingModel)]
    assert len(evasive) == 1
    assert evasive[0].soft_brake_ttc_s == 2.0
    assert evasive[0].hard_brake_ttc_s == 1.0


def test_cli_standard_uses_default_braking() -> None:
    """standard -> EvasiveBrakingModel with its default soft/hard."""
    behaviours = cli._resolve_behaviours("standard")
    evasive = [b for b in behaviours if isinstance(b, EvasiveBrakingModel)]
    assert len(evasive) == 1
    assert evasive[0].soft_brake_ttc_s == 3.0
    assert evasive[0].hard_brake_ttc_s == 1.5


def test_cli_none_returns_empty_behaviour_list() -> None:
    """none -> no behaviour models at all."""
    assert cli._resolve_behaviours("none") == []


@pytest.mark.parametrize("city", ["vadodara", "surat", "ahmedabad"])
def test_cli_each_city_dry_runs(city: str, monkeypatch, capsys) -> None:
    """Each city is selectable and dry-runs cleanly."""
    rc = _run(monkeypatch, ["--city", city, "--dry-run"])
    assert rc == 0


def test_cli_rejects_invalid_vehicle_count(monkeypatch, capsys) -> None:
    """--vehicle-count 0 exits 2 and prints an error."""
    rc = _run(monkeypatch, ["--vehicle-count", "0"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "vehicle-count" in err.lower()


@pytest.mark.integration
def test_cli_vehicle_count_produces_csv(tmp_path: Path, monkeypatch) -> None:
    """--vehicle-count runs the full pipeline and writes a CSV."""
    out = tmp_path / "conflicts.csv"
    rc = _run(
        monkeypatch,
        [
            "--city",
            "vadodara",
            "--vehicle-count",
            "20",
            "--behaviours",
            "calibrated",
            "--duration",
            "3.0",
            "--out",
            str(out),
        ],
    )
    assert rc == 0
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "time_s,agent_a,agent_b,ttc_s,drac_ms2,distance_m"
