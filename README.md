# Surrogate Safety ABM

Agent-based modelling framework for analysing surrogate safety measures (SSMs)
at unsignalized urban intersections in India, with calibrated scenarios for
Vadodara, Surat, and Ahmedabad.

## Features

- Heterogeneous traffic agents (2W, 3W, cars, HGV, pedestrians)
- Established SSMs: PET, TTC, DRAC — **verified 3-way** against independent
  implementations and trusted libraries (scipy, numpy)
- City-specific calibration profiles (IRC:106-aligned PCU weights)
- Publication-ready statistical outputs (RMSE, MAE, Pearson r, KS, Wasserstein)
- UVH-26 detector support for Indian vehicle classes

## Quickstart

    # 1) install
    pip install -e ".[dev]"

    # 2) run a simulation
    surrogate-safety simulate --city surat --duration 60 --out output/conflicts.csv

    # 3) compute accuracy metrics against observed data
    surrogate-safety metrics \
        --observed observed_ttc.csv \
        --simulated simulated_ttc.csv \
        --column ttc_s \
        --mode unpaired \
        --out table2_ttc.json

    # 4) environment check
    surrogate-safety doctor

## Commands

    surrogate-safety simulate    # run the ABM, write conflict events to CSV
    surrogate-safety metrics     # compute accuracy metrics from two CSVs
    surrogate-safety setup       # install runtime + dev dependencies
    surrogate-safety doctor      # print environment report

Global flag: `--no-install` — fail instead of auto-installing missing deps.

## Project structure

    src/surrogate_safety_abm/
      main.py          # CLI dispatcher (console-script entry point)
      cli.py           # `simulate` subcommand
      metrics_cli.py   # `metrics` subcommand
      bootstrap.py     # environment check / install
      ssm/             # TTC, PET, DRAC  (3-way verified)
      analysis/        # RMSE, MAE, Pearson r, KS, Wasserstein  (3-way verified)
      data/            # video loading, detection, homography, trajectory
      simulation/      # engine, recorder, scenario
      behaviour/       # gap acceptance, evasive braking
      agents/, environment/, geometry/, config/

    tests/
      unit/            # focused behavioural tests
      property/        # hypothesis-based invariants
      differential/    # 3-way differential tests
      integration/     # end-to-end pipeline

## Install

    pip install -e ".[dev]"

## Testing

    make test                              # full suite via scripts/dev.py
    pytest                                 # all tests with coverage
    pytest tests/unit                      # unit only
    pytest tests/property                  # hypothesis properties
    pytest tests/differential              # 3-way differential checks
    pytest -m "not slow"                   # skip slow integration

## Dev tooling

    make bootstrap    # idempotent env setup
    make test         # run pytest
    make audit        # print per-file test + assert counts
    make clean        # clear caches

## Roadmap

- Anticipated Buffer Time (ABT) -- published indicator
  [Dutta et al., 2024], implementation in progress

## License and citation

See `LICENSE` and `CITATION.cff`.
