"""Tests for the surrogate-safety-metrics CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from surrogate_safety_abm import metrics_cli


def _write_csv(path: Path, column: str, values: list[float]) -> None:
    """Write a single-column CSV with a header."""
    lines = [column] + [str(v) for v in values]
    path.write_text("\n".join(lines) + "\n")


def _run(
    observed: Path,
    simulated: Path,
    column: str,
    out: Path,
    mode: str = "unpaired",
) -> int:
    """Invoke main() with the given arguments, return exit code."""
    return metrics_cli.main(
        [
            "--observed",
            str(observed),
            "--simulated",
            str(simulated),
            "--column",
            column,
            "--mode",
            mode,
            "--out",
            str(out),
        ]
    )


class TestHappyPath:
    """Successful runs produce the expected output schema."""

    def test_unpaired_mode_writes_expected_fields(self, tmp_path: Path) -> None:
        obs = tmp_path / "obs.csv"
        sim = tmp_path / "sim.csv"
        out = tmp_path / "out.json"
        _write_csv(obs, "ttc_s", [1.0, 2.0, 3.0, 4.0, 5.0])
        _write_csv(sim, "ttc_s", [1.5, 2.5, 3.5, 4.5, 5.5])
        rc = _run(obs, sim, "ttc_s", out)
        assert rc == 0
        payload = json.loads(out.read_text())
        # unpaired: distributional metrics only
        for key in (
            "n_observed",
            "n_simulated",
            "ks_statistic",
            "ks_p",
            "wasserstein",
            "mean_observed",
            "mean_simulated",
            "mean_diff",
            "mean_diff_ci95_low",
            "mean_diff_ci95_high",
        ):
            assert key in payload, key
        # paired metrics are absent in unpaired mode
        for key in ("rmse", "mae", "pearson_r", "pearson_p"):
            assert key not in payload, key

    def test_paired_mode_adds_paired_fields(self, tmp_path: Path) -> None:
        obs = tmp_path / "obs.csv"
        sim = tmp_path / "sim.csv"
        out = tmp_path / "out.json"
        _write_csv(obs, "ttc_s", [1.0, 2.0, 3.0, 4.0, 5.0])
        _write_csv(sim, "ttc_s", [1.1, 2.1, 3.1, 4.1, 5.1])
        rc = _run(obs, sim, "ttc_s", out, mode="paired")
        assert rc == 0
        payload = json.loads(out.read_text())
        for key in (
            "rmse",
            "mae",
            "pearson_r",
            "pearson_p",
            "ks_statistic",
            "wasserstein",
        ):
            assert key in payload, key

    def test_unequal_lengths_work_in_unpaired_mode(self, tmp_path: Path) -> None:
        obs = tmp_path / "obs.csv"
        sim = tmp_path / "sim.csv"
        out = tmp_path / "out.json"
        _write_csv(obs, "ttc_s", [1.0, 2.0, 3.0])
        _write_csv(sim, "ttc_s", [2.0, 3.0])
        rc = _run(obs, sim, "ttc_s", out)
        assert rc == 0
        payload = json.loads(out.read_text())
        assert payload["n_observed"] == 3
        assert payload["n_simulated"] == 2

    def test_output_directory_created(self, tmp_path: Path) -> None:
        obs = tmp_path / "obs.csv"
        sim = tmp_path / "sim.csv"
        out = tmp_path / "nested" / "dir" / "out.json"
        _write_csv(obs, "ttc_s", [1.0, 2.0, 3.0])
        _write_csv(sim, "ttc_s", [1.5, 2.5, 3.5])
        rc = _run(obs, sim, "ttc_s", out)
        assert rc == 0
        assert out.exists()


class TestErrors:
    """Invalid inputs exit nonzero with a clear message."""

    def test_missing_observed_file(self, tmp_path: Path) -> None:
        sim = tmp_path / "sim.csv"
        _write_csv(sim, "ttc_s", [1.0, 2.0, 3.0])
        with pytest.raises(SystemExit) as exc:
            _run(tmp_path / "missing.csv", sim, "ttc_s", tmp_path / "out.json")
        assert exc.value.code == 1

    def test_missing_simulated_file(self, tmp_path: Path) -> None:
        obs = tmp_path / "obs.csv"
        _write_csv(obs, "ttc_s", [1.0, 2.0, 3.0])
        with pytest.raises(SystemExit) as exc:
            _run(obs, tmp_path / "missing.csv", "ttc_s", tmp_path / "out.json")
        assert exc.value.code == 1

    def test_missing_column(self, tmp_path: Path, capsys) -> None:
        obs = tmp_path / "obs.csv"
        sim = tmp_path / "sim.csv"
        _write_csv(obs, "ttc_s", [1.0, 2.0, 3.0])
        _write_csv(sim, "ttc_s", [1.5, 2.5, 3.5])
        with pytest.raises(SystemExit) as exc:
            _run(obs, sim, "pet_s", tmp_path / "out.json")
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "pet_s" in captured.err

    def test_empty_csv(self, tmp_path: Path) -> None:
        obs = tmp_path / "obs.csv"
        sim = tmp_path / "sim.csv"
        obs.write_text("")
        _write_csv(sim, "ttc_s", [1.0, 2.0])
        with pytest.raises(SystemExit) as exc:
            _run(obs, sim, "ttc_s", tmp_path / "out.json")
        assert exc.value.code == 1

    def test_header_only_no_rows(self, tmp_path: Path) -> None:
        obs = tmp_path / "obs.csv"
        sim = tmp_path / "sim.csv"
        obs.write_text("ttc_s\n")
        _write_csv(sim, "ttc_s", [1.0, 2.0])
        with pytest.raises(SystemExit) as exc:
            _run(obs, sim, "ttc_s", tmp_path / "out.json")
        assert exc.value.code == 1

    def test_non_numeric_value(self, tmp_path: Path) -> None:
        obs = tmp_path / "obs.csv"
        sim = tmp_path / "sim.csv"
        obs.write_text("ttc_s\n1.0\nnot_a_number\n3.0\n")
        _write_csv(sim, "ttc_s", [1.0, 2.0, 3.0])
        with pytest.raises(SystemExit) as exc:
            _run(obs, sim, "ttc_s", tmp_path / "out.json")
        assert exc.value.code == 1

    def test_empty_value_in_row(self, tmp_path: Path) -> None:
        # DictReader skips fully blank rows, so the empty cell must be
        # inside a row that has at least one other populated column.
        obs = tmp_path / "obs.csv"
        sim = tmp_path / "sim.csv"
        obs.write_text("ttc_s,other\n1.0,a\n,b\n3.0,c\n")
        _write_csv(sim, "ttc_s", [1.0, 2.0, 3.0])
        with pytest.raises(SystemExit) as exc:
            _run(obs, sim, "ttc_s", tmp_path / "out.json")
        assert exc.value.code == 1


class TestPairedSafety:
    """--mode paired refuses to silently truncate unequal samples."""

    def test_paired_unequal_lengths_refused(self, tmp_path: Path) -> None:
        obs = tmp_path / "obs.csv"
        sim = tmp_path / "sim.csv"
        out = tmp_path / "out.json"
        _write_csv(obs, "ttc_s", [1.0, 2.0, 3.0])
        _write_csv(sim, "ttc_s", [1.5, 2.5])
        with pytest.raises(SystemExit) as exc:
            _run(obs, sim, "ttc_s", out, mode="paired")
        assert exc.value.code == 1
        # no output file written when paired mode is refused
        assert not out.exists()

    def test_paired_equal_lengths_ok(self, tmp_path: Path) -> None:
        obs = tmp_path / "obs.csv"
        sim = tmp_path / "sim.csv"
        out = tmp_path / "out.json"
        _write_csv(obs, "ttc_s", [1.0, 2.0, 3.0])
        _write_csv(sim, "ttc_s", [1.5, 2.5, 3.5])
        rc = _run(obs, sim, "ttc_s", out, mode="paired")
        assert rc == 0
        assert out.exists()
