"""Differential tests for data.observed_ssm (cross-checked against ssm/)."""
import math
import numpy as np
from surrogate_safety_abm.data.observed_ssm import ObservedTrack, compute_pairwise_conflicts
from surrogate_safety_abm.ssm import compute_ttc, compute_drac


def _track(tid, times, xs, ys, speeds, headings):
    return ObservedTrack(
        track_id=tid, class_name="car",
        times_s=np.array(times, float), x_m=np.array(xs, float),
        y_m=np.array(ys, float), speed_mps=np.array(speeds, float),
        heading_rad=np.array(headings, float),
    )


def test_headon_cross_check_against_ssm():
    T = [0, 0.1, 0.2]
    a = _track(1, T, [0,1,2], [0,0,0], [10,10,10], [0,0,0])
    b = _track(2, T, [10,9,8], [0,0,0], [10,10,10], [math.pi]*3)
    events = compute_pairwise_conflicts(
        [a, b], intersection_center_m=(5, 0),
        intersection_zone_m=10, conflict_radius_m=12,
        ttc_threshold_s=3.0, min_gap_m=2.0, min_speed_mps=1.4,
    )
    assert len(events) > 0
    for ev in events:
        v_rel = ev.distance_m / ev.ttc_s
        assert math.isclose(ev.ttc_s, compute_ttc(ev.distance_m, v_rel).ttc_seconds, rel_tol=1e-6)
        assert math.isclose(ev.drac_ms2, compute_drac(ev.distance_m, v_rel).drac_ms2, rel_tol=1e-6)


def test_empty_tracks_returns_no_events():
    assert compute_pairwise_conflicts([]) == []


def test_single_track_returns_no_events():
    a = _track(1, [0, 0.1], [0, 1], [0, 0], [10, 10], [0, 0])
    assert compute_pairwise_conflicts([a]) == []


def test_non_converging_returns_no_events():
    T = [0, 0.1, 0.2]
    a = _track(1, T, [0,1,2], [0,0,0], [10,10,10], [0,0,0])
    b = _track(2, T, [10,11,12], [0,0,0], [10,10,10], [0,0,0])
    assert compute_pairwise_conflicts([a, b], intersection_center_m=(6, 0)) == []


def test_below_min_speed_returns_no_events():
    T = [0, 0.1, 0.2]
    a = _track(1, T, [0,1,2], [0,0,0], [0.5]*3, [0,0,0])
    b = _track(2, T, [10,9,8], [0,0,0], [0.5]*3, [math.pi]*3)
    assert compute_pairwise_conflicts([a, b], intersection_center_m=(5, 0), min_speed_mps=1.4) == []
