# Gap report — test coverage and paper claims

Status: audit complete. Last verified: 2026-11-22 (HEAD 71a6697 + PRs 10-12).

## What is tested and correct

- SSM (TTC, PET, DRAC): formula, non-negativity, monotonicity, edge cases
  (zero gap, negative gap → ValueError, infinite gap, NaN propagation).
  Verified against `mpmath` arbitrary-precision on hand-worked cases.
- Accuracy metrics: `ks_test`, `wasserstein_distance`, `pearson_r`,
  `mae`, `rmse` match `scipy.stats` / `sklearn.metrics` on 5-element test.
- Descriptive stats: `summarise` returns `n`, `mean`, `median`, `p85`,
  `minimum`, `maximum` — 48 cases match `statistics.mean`,
  `statistics.median`, `np.percentile(method="linear")`.
  `mean([0.1,0.2,0.3])` returns exactly `0.2` (compensated summation).
- Engine: step counts, snapshot counts, time ordering (multi-agent fix).
- Determinism: same seed → same snapshots and conflicts (full field compare).

## What is NOT tested, with severity

| # | gap | severity |
|---|---|---|
| 1 | No observed Vaidya data in `data/raw/` — Table 2 cannot be reproduced from the repo | 🔴 paper-blocking |
| 2 | No calibration code — evasive braking thresholds (2.0s, 1.0s) are hardcoded in `metrics_cli.py`; no fit function exists | 🔴 paper-blocking |
| 3 | `SimulationConfig.seed` not read by `engine.py` — determinism comes only from `scenario.py` | 🟡 fix before stochastic behavior added |
| 4 | `extreme_value.fit_ttc_gpd` untested | 🟡 next round |
| 5 | `geometry.Vec2` class untested | 🟡 next round |
| 6 | `severity.classify_severity` untested | 🟡 no standard answer key |
| 7 | `behaviour/*` has 30 tests (branches covered) but no multi-seed sensitivity | 🟡 |

## Non-goals (would be false properties)

- TTC symmetry: `TTC(a,b) != TTC(b,a)` — direction is encoded in closing-speed sign.
- PET symmetry: `PET(exit_a, entry_b) != PET(exit_b, entry_a)`.
- Rotation invariance on SSM: functions take scalars, not coordinates.
- Calibration reproduction: no fit code exists to reproduce.

## What the paper can honestly claim

Given the above, the honest claim is:
- "The SSM formulas and accuracy metrics are tested against standard
  references (scipy, sklearn, mpmath)."
- "Reproducibility is verified within a single process."
- "The evasive braking thresholds were selected by manual inspection
  to match observed outcomes (in-sample)."

The honest claim is NOT:
- "Validated against observed data" (no data in repo).
- "Calibrated via grid search" (no fit code exists).

If a held-out validation dataset is later committed and a calibration
script is written, the claim can be upgraded.
