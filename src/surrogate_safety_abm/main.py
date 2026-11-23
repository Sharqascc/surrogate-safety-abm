"""Top-level entry point for the surrogate-safety CLI.

Dispatches to:
  surrogate-safety setup        install package + dev deps
  surrogate-safety doctor       print environment report
  surrogate-safety simulate     run the ABM simulation (calls cli.main)
  surrogate-safety metrics      compare observed vs simulated (metrics_cli.main)

Auto-installs dependencies before `simulate` or `metrics` if they are
missing. Pass --no-install to disable, or run `surrogate-safety setup`
once and the check will short-circuit.
"""

from __future__ import annotations

import sys

from surrogate_safety_abm import bootstrap, cli, metrics_cli

_PROG = "surrogate-safety"
_USAGE = f"""usage: {_PROG} <command> [options]

Commands:
  setup       Install the package and its dependencies
  doctor      Print environment report (versions, missing deps)
  simulate    Run the ABM simulation
  metrics     Compute TTC/PET/DRAC accuracy metrics

Global flags (consumed anywhere):
  --no-install    Fail instead of auto-installing missing deps
  -h, --help      Show this message

Run `{_PROG} <command> --help` for subcommand help."""


def _usage_and_exit(code: int = 2) -> int:
    print(_USAGE, file=sys.stdout if code == 0 else sys.stderr)
    return code


def dispatch(argv: list[str] | None = None) -> int:
    """Route argv to the right subcommand. Returns an exit code."""
    args = list(sys.argv[1:] if argv is None else argv)

    # Global flags that may appear anywhere before the subcommand
    no_install = False
    cleaned: list[str] = []
    for a in args:
        if a == "--no-install":
            no_install = True
        else:
            cleaned.append(a)
    args = cleaned

    if not args or args[0] in ("-h", "--help", "help"):
        return _usage_and_exit(0)

    cmd = args[0]
    rest = args[1:]

    if cmd == "setup":
        return bootstrap.install_deps(dev=True)

    if cmd == "doctor":
        return bootstrap.print_doctor_report()

    if cmd == "simulate":
        rc = bootstrap.ensure_environment(no_install=no_install)
        if rc != 0:
            return rc
        return cli.main(rest)

    if cmd == "metrics":
        rc = bootstrap.ensure_environment(no_install=no_install)
        if rc != 0:
            return rc
        return metrics_cli.main(rest)

    print(f"{_PROG}: unknown command: {cmd}", file=sys.stderr)
    return _usage_and_exit(2)


def main() -> int:
    """Console-script shim."""
    return dispatch()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
