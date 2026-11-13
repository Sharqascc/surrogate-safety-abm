"""Command-line tool for TTC/PET/DRAC accuracy metrics.

Separate entry point from `surrogate-safety-sim`, per the design review:
the simulator CLI and its tests stay untouched, and this tool has no
mode flags mixed in with simulation flags. Folding both into a single
`surrogate-safety-sim simulate|metrics` subcommand structure is a
follow-up refactor after the paper ships.

Serves RESUME_HERE.md priority 1: accuracy metrics for Table 2.

Usage::

    surrogate-safety-metrics \
        --observed observed_ttc.csv \
        --simulated simulated_ttc.csv \
        --column ttc_s \
        --mode unpaired \
        --out table2_ttc.json

The `--mode` flag is deliberate:

* `unpaired` (default): distributional metrics only. KS, Wasserstein,
  mean-difference CI, and quantiles. Correct for events from two
  different populations that are not indexable against each other.
* `paired`: additionally computes RMSE, MAE, and Pearson r. Requires
  equal sample counts and assumes obs[i] corresponds to sim[i] in
  the CSV row order. The tool refuses `paired` on unequal lengths
  rather than truncating.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from surrogate_safety_abm.analysis import AccuracyReport, compare


def _read_column(path: Path, column: str) -> list[float]:
    """Read a numeric column from a CSV file.

    Args:
        path: CSV file path.
        column: Column name holding the SSM values.

    Returns:
        The column values as a list of floats.

    Raises:
        SystemExit: If the file is missing, empty, lacks the column,
            contains an empty or non-numeric cell, or has zero rows.
    """
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        raise SystemExit(1)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print(f"empty CSV (no header): {path}", file=sys.stderr)
            raise SystemExit(1)
        if column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames)
            print(
                f"column {column!r} not in {path}; available: {available}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        values: list[float] = []
        for i, row in enumerate(reader, start=2):
            raw = (row.get(column) or "").strip()
            if raw == "":
                print(
                    f"empty value at {path}:{i} column={column}",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            try:
                values.append(float(raw))
            except ValueError:
                print(
                    f"non-numeric at {path}:{i} column={column}: {raw!r}",
                    file=sys.stderr,
                )
                raise SystemExit(1) from None
    if not values:
        print(f"no rows in {path}", file=sys.stderr)
        raise SystemExit(1)
    return values


def _summary_unpaired(rep: AccuracyReport) -> dict[str, int | float]:
    """Distributional metrics: valid for any two samples.

    Args:
        rep: Report from :func:`compare`.

    Returns:
        Dict of metric name -> value. Paired metrics are omitted.
    """
    return {
        "n_observed": rep.n_observed,
        "n_simulated": rep.n_simulated,
        "ks_statistic": rep.ks_statistic,
        "ks_p": rep.ks_p,
        "wasserstein": rep.wasserstein,
        "mean_observed": rep.mean_observed,
        "mean_simulated": rep.mean_simulated,
        "mean_diff": rep.mean_diff,
        "mean_diff_ci95_low": rep.mean_diff_ci95_low,
        "mean_diff_ci95_high": rep.mean_diff_ci95_high,
    }


def _summary_paired(rep: AccuracyReport) -> dict[str, int | float]:
    """Paired metrics plus the unpaired set.

    Args:
        rep: Report from :func:`compare`.

    Returns:
        Dict of metric name -> value. Includes rmse, mae, pearson_r,
        pearson_p on top of the unpaired fields.

    Raises:
        SystemExit: If the two samples differ in length.
    """
    if rep.n_observed != rep.n_simulated:
        print(
            f"--mode paired requires equal-length samples; got "
            f"{rep.n_observed} observed vs {rep.n_simulated} simulated",
            file=sys.stderr,
        )
        raise SystemExit(1)
    out = _summary_unpaired(rep)
    out.update(
        {
            "rmse": rep.rmse,
            "mae": rep.mae,
            "pearson_r": rep.pearson_r,
            "pearson_p": rep.pearson_p,
        }
    )
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the metrics CLI.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        prog="surrogate-safety-metrics",
        description="Compare observed vs simulated SSM values.",
    )
    parser.add_argument(
        "--observed",
        type=Path,
        required=True,
        help="CSV with observed SSM values",
    )
    parser.add_argument(
        "--simulated",
        type=Path,
        required=True,
        help="CSV with simulated SSM values",
    )
    parser.add_argument(
        "--column",
        required=True,
        help="Column name holding the SSM values",
    )
    parser.add_argument(
        "--mode",
        choices=("unpaired", "paired"),
        default="unpaired",
        help=(
            "unpaired: distributional metrics only (KS, Wasserstein, "
            "mean CI); paired: also RMSE, MAE, Pearson r (requires "
            "equal n and row-wise correspondence)"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path",
    )
    args = parser.parse_args(argv)

    obs = _read_column(args.observed, args.column)
    sim = _read_column(args.simulated, args.column)
    rep = compare(obs, sim)

    summary = _summary_paired(rep) if args.mode == "paired" else _summary_unpaired(rep)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:>22}: {v:.6f}")
        else:
            print(f"  {k:>22}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
