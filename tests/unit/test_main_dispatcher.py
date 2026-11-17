"""Unit tests for the top-level dispatcher."""

from __future__ import annotations

from unittest.mock import MagicMock

from surrogate_safety_abm import main as dispatcher


def _patch(monkeypatch, **overrides):
    defaults = {
        "ensure_environment": MagicMock(return_value=0),
        "install_deps": MagicMock(return_value=0),
        "print_doctor_report": MagicMock(return_value=0),
        "cli_main": MagicMock(return_value=0),
        "metrics_main": MagicMock(return_value=0),
    }
    defaults.update(overrides)
    monkeypatch.setattr(dispatcher.bootstrap, "ensure_environment", defaults["ensure_environment"])
    monkeypatch.setattr(dispatcher.bootstrap, "install_deps", defaults["install_deps"])
    monkeypatch.setattr(
        dispatcher.bootstrap, "print_doctor_report", defaults["print_doctor_report"]
    )
    monkeypatch.setattr(dispatcher.cli, "main", defaults["cli_main"])
    monkeypatch.setattr(dispatcher.metrics_cli, "main", defaults["metrics_main"])
    return defaults


def test_dispatch_help_returns_zero(monkeypatch, capsys):
    _patch(monkeypatch)
    rc = dispatcher.dispatch(["--help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "surrogate-safety" in out
    assert "simulate" in out


def test_dispatch_no_args_prints_usage(monkeypatch, capsys):
    _patch(monkeypatch)
    rc = dispatcher.dispatch([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Commands:" in out


def test_dispatch_setup_invokes_install(monkeypatch):
    mocks = _patch(monkeypatch)
    rc = dispatcher.dispatch(["setup"])
    assert rc == 0
    mocks["install_deps"].assert_called_once_with(dev=True)


def test_dispatch_doctor_invokes_report(monkeypatch):
    mocks = _patch(monkeypatch)
    rc = dispatcher.dispatch(["doctor"])
    assert rc == 0
    mocks["print_doctor_report"].assert_called_once()


def test_dispatch_simulate_ensures_env_then_delegates(monkeypatch):
    mocks = _patch(monkeypatch)
    rc = dispatcher.dispatch(["simulate", "--city", "vadodara", "--dry-run"])
    assert rc == 0
    mocks["ensure_environment"].assert_called_once_with(no_install=False)
    mocks["cli_main"].assert_called_once_with(["--city", "vadodara", "--dry-run"])


def test_dispatch_simulate_skips_cli_if_env_fails(monkeypatch):
    mocks = _patch(monkeypatch, ensure_environment=MagicMock(return_value=1))
    rc = dispatcher.dispatch(["simulate", "--dry-run"])
    assert rc == 1
    mocks["cli_main"].assert_not_called()


def test_dispatch_metrics_ensures_env_then_delegates(monkeypatch):
    mocks = _patch(monkeypatch)
    rc = dispatcher.dispatch(
        [
            "metrics",
            "--observed",
            "a.csv",
            "--simulated",
            "b.csv",
            "--column",
            "ttc_s",
            "--out",
            "o.json",
        ]
    )
    assert rc == 0
    mocks["ensure_environment"].assert_called_once_with(no_install=False)
    mocks["metrics_main"].assert_called_once_with(
        ["--observed", "a.csv", "--simulated", "b.csv", "--column", "ttc_s", "--out", "o.json"]
    )


def test_dispatch_unknown_command_returns_2(monkeypatch, capsys):
    _patch(monkeypatch)
    rc = dispatcher.dispatch(["frobnicate"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown command" in err.lower()


def test_dispatch_no_install_flag_forwarded(monkeypatch):
    mocks = _patch(monkeypatch)
    rc = dispatcher.dispatch(["--no-install", "simulate", "--dry-run"])
    assert rc == 0
    mocks["ensure_environment"].assert_called_once_with(no_install=True)


def test_dispatch_no_install_flag_anywhere(monkeypatch):
    mocks = _patch(monkeypatch)
    rc = dispatcher.dispatch(["simulate", "--dry-run", "--no-install"])
    assert rc == 0
    mocks["ensure_environment"].assert_called_once_with(no_install=True)
