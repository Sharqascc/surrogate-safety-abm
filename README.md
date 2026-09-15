# Surrogate Safety ABM

Agent-based modelling framework for analysing surrogate safety measures (SSMs)
at unsignalized urban intersections in India, with calibrated scenarios for
Vadodara, Surat, and Ahmedabad.

## Features

- Heterogeneous traffic agents (2W, 3W, cars, HGV, pedestrians)
- Established SSMs: PET, TTC, DRAC
- Novel indicator: Anticipated Buffer Time (ABT)
- City-specific calibration profiles
- Publication-ready statistical outputs

## Install

    pip install -e ".[dev]"

## Quality checks

    make check

## Testing

    pytest                          # all tests with coverage
    pytest tests/unit               # unit tests
    pytest tests/property           # hypothesis property tests
    pytest -m "not slow"            # skip slow integration
