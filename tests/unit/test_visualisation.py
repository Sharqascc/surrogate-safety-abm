"""Unit tests for publication-quality visualisations."""

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from surrogate_safety_abm.analysis.visualisation import (
    plot_conflict_heatmap,
    plot_drac_histogram,
    plot_ttc_cdf,
    plot_ttc_histogram,
)
from surrogate_safety_abm.simulation import ConflictEvent


def _events(n: int = 20) -> list[ConflictEvent]:
    return [
        ConflictEvent(
            time_s=float(i),
            agent_a="a",
            agent_b="b",
            ttc_s=float(i % 5) + 0.5,
            drac_ms2=float(i % 7),
            distance_m=10.0,
        )
        for i in range(n)
    ]


class TestPlotTTC:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            plot_ttc_histogram([])

    def test_returns_figure(self) -> None:
        fig = plot_ttc_histogram(_events())
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_saves_file(self, tmp_path) -> None:
        out = tmp_path / "ttc.png"
        fig = plot_ttc_histogram(_events(), out=out)
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close(fig)


class TestPlotDRAC:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            plot_drac_histogram([])

    def test_returns_figure(self) -> None:
        fig = plot_drac_histogram(_events())
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_saves_file(self, tmp_path) -> None:
        out = tmp_path / "drac.png"
        fig = plot_drac_histogram(_events(), out=out)
        assert out.exists()
        plt.close(fig)


class TestPlotTCCCDF:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            plot_ttc_cdf([])

    def test_returns_figure(self) -> None:
        fig = plot_ttc_cdf(_events())
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_saves_file(self, tmp_path) -> None:
        out = tmp_path / "cdf.png"
        fig = plot_ttc_cdf(_events(), out=out)
        assert out.exists()
        plt.close(fig)


class TestPlotHeatmap:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            plot_conflict_heatmap([])

    def test_returns_figure(self) -> None:
        fig = plot_conflict_heatmap(_events())
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_saves_file(self, tmp_path) -> None:
        out = tmp_path / "heatmap.png"
        fig = plot_conflict_heatmap(_events(), out=out)
        assert out.exists()
        plt.close(fig)
