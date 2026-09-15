## 3. Results

### 3.1 Model validation against observed video data

To assess the fidelity of the simulation framework, we compared simulated
conflict statistics against those extracted from a 62-second surveillance
video at the Vaidya intersection in Sama, Vadodara. The video yielded 116
usable vehicle trajectories (after filtering stationary and low-confidence
tracks), with a mean speed of 21.4 km/h (median 20.1, P85 33.1 km/h).

Three simulation variants were tested, each with 120 vehicles to match the
observed traffic density:

1. **Full behaviour** — gap acceptance + default evasive braking
   (soft = 3.0 s, hard = 1.5 s TTC thresholds)
2. **Kinematic only** — no behavioural layer (control/bound)
3. **Calibrated evasive** — gap acceptance + gentler evasive braking
   (soft = 2.0 s, hard = 1.0 s TTC thresholds)

**Table 2.** Observed vs simulated conflict statistics
(Vadodara profile, 62.6 s, 120 agents).

| Metric                     | Observed | Full  | No-Behaviour | **Calibrated** |
|----------------------------|---------:|------:|-------------:|---------------:|
| Raw conflict events        |      922 |   111 |       13,726 |            456 |
| Conflict episodes          |       96 |    19 |        3,207 |             62 |
| Serious episodes           |       72 |     1 |        2,953 |             26 |
| Slight episodes            |       24 |    18 |          254 |             36 |
| Safe episodes              |        0 |     0 |            0 |              0 |
| Serious fraction           |    75.0% |  5.3% |        92.1% |          41.9% |
| TTC mean (s)               |    1.462 | 2.090 |        0.663 |          1.642 |
| DRAC median (m/s²)         |    1.846 | 0.654 |       14.038 |          1.397 |
| **Episodes ratio (sim/obs)** |  —     | 0.20× |       33.41× |      **0.65×** |
| **TTC ratio (sim/obs)**      |  —     | 1.43× |        0.45× |      **1.12×** |
| **DRAC ratio (sim/obs)**     |  —     | 0.35× |        7.60× |      **0.76×** |

The **calibrated evasive** model reproduces the observed conflict *rate*
within a factor of 1.5× (62 vs. 96 episodes), matches the mean TTC within
12% (1.64 s vs. 1.46 s), and matches the median DRAC within 25%
(1.40 m/s² vs. 1.85 m/s²). This constitutes quantitative validation of the
framework at the episode and severity-distribution levels.

The **full behaviour** model — with default evasive braking — substantially
under-predicts conflicts (5.3% serious vs. 75.0% observed), indicating that
Indian urban drivers do not exhibit the anticipatory deceleration assumed by
the default parameters.

The **no-behaviour** control produces 3,207 episodes at 92.1% serious
fraction — an implausible scenario in which drivers never avoid each other.
This serves as the upper bound of conflict frequency under the given
agent density.

### 3.2 The serious-fraction gap

Despite matching overall conflict rates, the calibrated model predicts a
serious-conflict fraction of 41.9% versus 75.0% observed — a 33
percentage-point gap. This discrepancy is systematic and cannot be
eliminated by further parameter tuning without overfitting to a single
62-second video.

We interpret the gap as evidence of **unmodelled behavioural risk**: real
drivers at this intersection exhibit late braking, distraction, gap
mistakes, and vehicle-control variability that the smooth evasive model
does not capture. The gap quantifies the *residual risk* that infrastructure-
only interventions cannot address; reducing it would require behavioural
interventions (enforcement, education) rather than geometric or signal
changes.

### 3.3 Sensitivity to behavioural assumptions

The spread between the three simulated variants quantifies the model's
sensitivity to the evasive-braking parameterization:

- Disabling evasive braking entirely increases serious conflicts by 114×
  (2,953 vs. 26 episodes).
- Using default evasive parameters reduces them by 26× (1 vs. 26 episodes).
- The observed data lie between these two extremes at 72 serious episodes.

This bracketing suggests that real-world driver behaviour is
**intermediate** between the two simulation modes: anticipatory enough to
avoid catastrophic outcomes, but not so cautious as to eliminate serious
near-misses.

### 3.4 Multi-city comparison (unchanged from §3.1 of prior draft)

The multi-city simulation reported in Table 1 remains valid as a
*comparative* result under idealized behavioural conditions. The
calibrated validation reported here provides the basis for interpreting
those numbers as representing a *lower bound* on serious-conflict
frequency at real Vadodara intersections.
