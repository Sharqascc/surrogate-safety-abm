"""Unit tests for bootstrap.py."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import time

from surrogate_safety_abm import bootstrap


def _fake_version_factory(present):
    def _version(name):
        if name in present:
            return "1.0.0"
        raise importlib.metadata.PackageNotFoundError(name)

    return _version


def test_check_environment_ok_when_all_present(monkeypatch):
    all_names = set(bootstrap._RUNTIME_DEPS) | set(bootstrap._DEV_DEPS)
    monkeypatch.setattr(importlib.metadata, "version", _fake_version_factory(all_names))
    rep = bootstrap.check_environment()
    assert rep.python_ok is True
    assert rep.ok is True
    assert rep.missing_runtime == []
    assert rep.missing_dev == []


def test_check_environment_reports_missing_runtime(monkeypatch):
    present = set(bootstrap._DEV_DEPS) | {
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "pydantic",
    }
    monkeypatch.setattr(importlib.metadata, "version", _fake_version_factory(present))
    rep = bootstrap.check_environment()
    assert rep.ok is False
    assert "networkx" in rep.missing_runtime


def test_check_environment_reports_missing_dev(monkeypatch):
    present = set(bootstrap._RUNTIME_DEPS) | {"pytest"}
    monkeypatch.setattr(importlib.metadata, "version", _fake_version_factory(present))
    rep = bootstrap.check_environment()
    assert rep.ok is True
    assert set(rep.missing_dev) == {"hypothesis", "mypy", "ruff"}










def test_install_deps_invokes_pip(monkeypatch, tmp_path):
    captured = {}

    def _fake_run(cmd, check=False):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(bootstrap.subprocess, "run", _fake_run)
    rc = bootstrap.install_deps(dev=True)
    assert rc == 0
    assert "-m" in captured["cmd"]
    assert "pip" in captured["cmd"]
    assert "install" in captured["cmd"]
    assert "[dev]" in captured["cmd"][-1]


def test_print_doctor_report_ok_returns_zero(monkeypatch):
    all_names = set(bootstrap._RUNTIME_DEPS) | set(bootstrap._DEV_DEPS)
    monkeypatch.setattr(importlib.metadata, "version", _fake_version_factory(all_names))
    assert bootstrap.print_doctor_report() == 0


def test_ensure_environment_no_install_when_broken(monkeypatch):
    present = set(bootstrap._DEV_DEPS)
    monkeypatch.setattr(importlib.metadata, "version", _fake_version_factory(present))
    install_called = {"value": False}

    def _fake_install(dev=True):
        install_called["value"] = True
        return 0

    monkeypatch.setattr(bootstrap, "install_deps", _fake_install)
    rc = bootstrap.ensure_environment(no_install=True)
    assert rc == 1
    assert install_called["value"] is False


def test_ensure_environment_installs_when_allowed(monkeypatch):
    state = {"phase": "broken"}
    calls = {"count": 0}

    def _fake_check():
        calls["count"] += 1
        if state["phase"] == "broken":
            return bootstrap.EnvironmentReport(
                python_ok=True, python_version="3.11.0", missing_runtime=["numpy"], missing_dev=[]
            )
        return bootstrap.EnvironmentReport(
            python_ok=True, python_version="3.11.0", missing_runtime=[], missing_dev=[]
        )

    def _fake_install(dev=True):
        state["phase"] = "ok"
        return 0

    monkeypatch.setattr(bootstrap, "check_environment", _fake_check)
    monkeypatch.setattr(bootstrap, "install_deps", _fake_install)
    rc = bootstrap.ensure_environment(no_install=False)
    assert rc == 0
    assert calls["count"] >= 2
