## 4. Discussion

### 4.1 Principal findings

This work makes three contributions to surrogate safety analysis at
unsignalized intersections in India:

**Finding 1 — The behavioural simulation framework can be validated
against short-duration video.** A 62-second surveillance video at a
single intersection suffices to calibrate the model's evasive-braking
parameters such that conflict rate and severity distributions match
observations within a factor of 1.5×.

**Finding 2 — The serious-conflict fraction observed in the field exceeds
what smooth kinematic models predict.** The 33-percentage-point gap
(75.0% observed vs. 41.9% calibrated) reflects the additional risk from
driver behaviour that idealized evasive models cannot capture. This is
not a limitation of the model per se — it is a quantitative statement
about the *residual risk* that behavioural interventions must address.

**Finding 3 — Kinematic-only simulation is uninformative for safety
assessment.** The no-behaviour control produces 3,207 conflict episodes
with 92% classified serious — a physically impossible scenario. Real
intersection users exhibit substantial anticipatory behaviour even if
it is not as smooth or as consistent as the default evasive model assumes.

### 4.2 Policy implications

The multi-city comparison reported in Table 1 and the calibrated
validation in Table 2 together support two policy recommendations:

1. **Speed reduction remains the highest-leverage intervention.** The
   monotonic relationship between 85th-percentile approach speed and
   serious-conflict fraction (§3.4) is robust to behavioural assumptions.

2. **Driver behaviour is a first-order contributor to injury risk.**
   Even after matching conflict rates with the calibrated model, 33
   percentage points of serious-conflict risk remains unexplained by
   infrastructure and vehicle kinematics alone. This justifies continued
   investment in enforcement, education, and real-time warning systems
   alongside geometric treatments.

### 4.3 Limitations

**Video duration.** A 62-second observation window is short relative to
the 4+ hour survey periods used in published SSM studies. Conflict-rate
statistics should be interpreted as *relative comparisons* rather than as
absolute hourly frequencies. Extending the analysis to a full peak-hour
recording (60–90 minutes) would strengthen the validation.

**Single intersection.** The validation is based on one intersection in
Vadodara. Generalization to other cities requires replication.

**Detection model specificity.** The UVH-26 fine-tuned YOLOv11-S model
used for detection delivers state-of-the-art performance on Indian
vehicles but inherits residual errors (missed 2W at distance, class
confusion between LCV and truck). Manual spot-checking estimated a
~90% detection recall for 2W and ~95% for four-wheeled vehicles.

**Homography accuracy.** The pixel-to-metre calibration relies on four
GPS-measured ground-control points. Mobile GPS uncertainty (±5 m) and
lens distortion at frame edges introduce an estimated ~10% uncertainty in
absolute speed measurements. Relative comparisons (TTC, DRAC ratios) are
less affected.

**Stationary vehicle exclusion.** The filtering step removes ~14% of raw
tracks classified as stationary or ID-swap artifacts. If this filtering
is biased toward a particular vehicle class (e.g., parked two-wheelers),
the observed class distribution may be slightly skewed.

### 4.4 Future work

The framework opens three natural extensions:

1. **Multi-intersection validation.** Applying the same pipeline to 5–10
   intersections across Gujarat would test the generality of the
   calibration approach.

2. **Pedestrian and cyclist integration.** The framework includes a
   `PedestrianAgent` class, but pedestrian–vehicle conflicts were not
   simulated. Given the high pedestrian share in Indian urban crashes,
   this is a priority extension.

3. **Real-time safety monitoring.** The inference pipeline runs at ~3.5
   FPS on a Colab GPU. A production deployment on edge hardware could
   enable real-time conflict alerts at high-risk intersections.
