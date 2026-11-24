#!/usr/bin/env python
"""Single entry point. Idempotent bootstrap + task dispatch."""
import os, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKER = ROOT / ".venv-bootstrapped"
DEV_DEPS = ["pytest>=7", "pytest-cov", "pytest-xdist", "hypothesis",
            "ruff", "mypy", "bandit"]

def _pip(*a): subprocess.run([sys.executable,"-m","pip","install","-q",*a],
                              check=True, cwd=ROOT)
def _ok():
    r = subprocess.run([sys.executable,"-c",
                        "import pytest, hypothesis, surrogate_safety_abm"],
                       capture_output=True)
    return r.returncode == 0
def _has_extras():
    try: t = (ROOT/"pyproject.toml").read_text()
    except: return False
    if "[project.optional-dependencies]" not in t: return False
    idx = t.index("[project.optional-dependencies]")
    m = re.search(r"^\[", t[idx+1:], re.M)
    sec = t[idx+1: idx+1+m.start()] if m else t[idx+1:]
    return bool(re.search(r"^dev\s*=", sec, re.M))

def bootstrap(force=False):
    if not force and MARKER.exists() and _ok(): return
    print("→ bootstrapping …")
    if _has_extras():
        try:
            _pip("-e", str(ROOT)+"[dev]")
            if _ok(): MARKER.write_text("extras"); print("✓ via [dev]"); return
        except subprocess.CalledProcessError: pass
    _pip("-e", str(ROOT)); _pip(*DEV_DEPS)
    MARKER.write_text("fallback"); print("✓ via explicit")

def cmd_bootstrap(a):
    bootstrap(force="--force" in a); return 0

def cmd_test(a):
    bootstrap()
    return subprocess.run([sys.executable,"-m","pytest",*(a or ["tests/","-v"])],
                          cwd=ROOT).returncode

def cmd_audit(_):
    bootstrap()
    print(f"{'file':<55}{'tests':>6}{'asserts':>8}")
    g = {"t":0,"a":0}
    for f in sorted((ROOT/"tests").rglob("test_*.py")):
        b = f.read_text()
        n = len(re.findall(r"^\s*(?:async\s+)?def (test_\w+)", b, re.M))
        a = len(re.findall(r"^\s*assert\b", b, re.M))
        g["t"]+=n; g["a"]+=a
        print(f"{str(f.relative_to(ROOT)):<55}{n:>6}{a:>8}")
    print(f"{'TOTAL':<55}{g['t']:>6}{g['a']:>8}")
    return 0

COMMANDS = {"bootstrap": cmd_bootstrap, "test": cmd_test, "audit": cmd_audit}
def main(argv):
    if not argv or argv[0] in ("-h","--help"):
        print("usage: python scripts/dev.py [bootstrap|test|audit]"); return 0
    return COMMANDS.get(argv[0], lambda _: 2)(argv[1:])
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
