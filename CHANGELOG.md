# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `LICENSE` (MIT).
- `CHANGELOG.md`.

### Changed
- `docs/results_section.md` §3.1 reframed as in-sample comparison (was
  "model validation"); §3.4 reframed as simulation-only city comparison.
- `docs/discussion_section.md` Findings 1 and §4.2 qualified: the
  calibration is in-sample against a single 62-second video; Surat and
  Ahmedabad are simulation-only.
- `docs/gap_report.md` gaps #5 (`Vec2`) and #6 (`classify_severity`)
  removed — both now covered by tests. Gap #1 clarified: `data/*` has
  0% coverage because the video pipeline cannot run without raw data.

### Fixed
- `test_time_monotonic_in_snapshots` now allows multiple agents per
  timestep (was `len(times) == len(set(times))`, which only holds for
  single-agent runs).

## [0.1.0]

### Added
- Physics property tests for SSM (TTC, PET, DRAC): non-negativity,
  monotonicity, formula consistency, non-convergence infinities.
- Golden fixture `accuracy_golden.json` pinning KS / Wasserstein outputs.
- Behavioural tests for gap-acceptance and evasive-braking models
  (30 tests), multi-agent scenarios, city-profile differentiation.
- End-to-end pipeline tests (`test_simulation_pipeline.py`).
- Property tests for accuracy, agents, analysis, and simulation engine.

### Known limitations
- `data/raw/`, `data/processed/`, `data/synthetic/` are empty —
  observed Vaidya trajectories are not committed. Table 2's comparison
  cannot be reproduced from the repo alone.
- The evasive-braking thresholds (`2.0` s soft, `1.0` s hard) are
  hardcoded in `metrics_cli.py`; no calibration code derives them from
  data. The paper should describe these as calibrated-to-match-observed
  (in-sample), not validated.
- `SimulationConfig.seed` is not read by `simulation/engine.py`;
  determinism comes entirely from `simulation/scenario.py`.
