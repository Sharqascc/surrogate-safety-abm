"""Environment bootstrap: dependency check, install, doctor report."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

_RUNTIME_DEPS = ("numpy", "pandas", "scipy", "matplotlib", "seaborn", "pydantic", "networkx")
_DEV_DEPS = ("pytest", "hypothesis", "mypy", "ruff")
_PYTHON_MIN = (3, 11)


@dataclass
class EnvironmentReport:
    """Result of an environment check."""

    python_ok: bool
    python_version: str
    missing_runtime: list[str] = field(default_factory=list)
    missing_dev: list[str] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True iff Python and all runtime deps are present."""
        return self.python_ok and not self.missing_runtime


def _installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _check_python() -> tuple[bool, str]:
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    ok = (v.major, v.minor) >= _PYTHON_MIN
    return ok, version_str


def check_environment() -> EnvironmentReport:
    """Inspect Python + deps without touching the filesystem."""
    py_ok, py_ver = _check_python()
    missing_runtime = []
    missing_dev = []
    versions: dict[str, str] = {}
    for dep in _RUNTIME_DEPS:
        v = _installed_version(dep)
        if v is None:
            missing_runtime.append(dep)
        else:
            versions[dep] = v
    for dep in _DEV_DEPS:
        v = _installed_version(dep)
        if v is None:
            missing_dev.append(dep)
        else:
            versions[dep] = v
    return EnvironmentReport(
        python_ok=py_ok,
        python_version=py_ver,
        missing_runtime=missing_runtime,
        missing_dev=missing_dev,
        versions=versions,
    )


def repo_root() -> Path:
    """Find the repository root containing pyproject.toml."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parent


def install_deps(dev: bool = True) -> int:
    """Install the package and its deps in editable mode.

    Returns the exit code from pip. Prints progress to stdout.
    """
    root = repo_root()
    target = f"{root}[dev]" if dev else str(root)
    print(f"Installing {target} ...")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", target],
        check=False,
    )
    if r.returncode == 0:
        print("Install complete.")
    else:
        print(f"Install failed with exit code {r.returncode}.", file=sys.stderr)
    return r.returncode


def ensure_environment(no_install: bool = False) -> int:
    """Ensure Python + runtime deps are available.

    Behaviour:
    - If check passes, return 0.
    - If check fails and no_install=True, print and return 1.
    - If check fails and no_install=False, run install_deps().
      Then re-check and return 0 or 1.
    """
    report = check_environment()
    if report.ok:
        return 0
    if no_install:
        print_doctor_report(report)
        print("\nInstall deps with:  surrogate-safety setup", file=sys.stderr)
        return 1
    print(
        f"Environment incomplete: {len(report.missing_runtime)} "
        f"runtime dep(s) missing. Installing ..."
    )
    rc = install_deps(dev=True)
    if rc != 0:
        return rc
    return 0 if check_environment().ok else 1


def print_doctor_report(report: EnvironmentReport | None = None) -> int:
    """Print a version table and return 0 (ok) or 1 (broken)."""
    if report is None:
        report = check_environment()
    print(
        f"Python: {report.python_version} ({'ok' if report.python_ok else 'TOO OLD, need >= 3.11'})"
    )
    print()
    print(f"{'dep':<20} {'version':<15} {'status':<10}")
    print("-" * 45)
    for dep in _RUNTIME_DEPS:
        v = report.versions.get(dep, "(missing)")
        status = "ok" if dep not in report.missing_runtime else "MISSING"
        print(f"{dep:<20} {v:<15} {status:<10}")
    for dep in _DEV_DEPS:
        v = report.versions.get(dep, "(missing)")
        status = "ok" if dep not in report.missing_dev else "MISSING"
        print(f"{dep:<20} {v:<15} {status:<10}")
    print()
    if report.ok and not report.missing_dev:
        print("Environment: READY")
        return 0
    if report.ok:
        print(f"Environment: runtime READY, {len(report.missing_dev)} dev dep(s) missing")
        print("  (run `surrogate-safety setup` to install dev deps)")
        return 0
    print(f"Environment: NOT READY ({len(report.missing_runtime)} runtime dep(s) missing)")
    print("  Run `surrogate-safety setup` to install.")
    return 1
