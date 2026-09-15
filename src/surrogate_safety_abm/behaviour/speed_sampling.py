"""Speed sampling from city-specific 85th-percentile profiles.

Reference: IRC:106 recommends the 85th-percentile speed as the design
speed for urban roads. We sample individual speeds from a normal
distribution centred on 85% of v85 with standard deviation 15% of v85,
clipped to a physically realistic envelope.
"""

from __future__ import annotations

import numpy as np

from surrogate_safety_abm.config.city_profiles import CityProfile


def sample_speed_mps(
    city: CityProfile,
    rng: np.random.Generator,
) -> float:
    """Draw one speed (m/s) from the city's v85 profile.

    Args:
        city: City profile providing ``approach_speed_85_kph``.
        rng: Numpy random generator for reproducibility.

    Returns:
        Speed in metres per second, clipped to [0.3·v85, 1.1·v85].
    """
    v85_ms = city.approach_speed_85_kph / 3.6
    mean = 0.85 * v85_ms
    sd = 0.15 * v85_ms
    speed = float(rng.normal(mean, sd))
    return float(np.clip(speed, 0.3 * v85_ms, 1.1 * v85_ms))


def sample_critical_gap_s(
    city: CityProfile,
    rng: np.random.Generator,
) -> float:
    """Draw one critical gap (s) from the city's log-normal profile.

    Log-normal parameters are derived from the profile's mean and std.

    Args:
        city: City profile providing ``critical_gap_mean_s`` and
            ``critical_gap_std_s``.
        rng: Numpy random generator for reproducibility.

    Returns:
        Critical gap in seconds, clipped to [0.5, 15.0].
    """
    m = city.critical_gap_mean_s
    s = city.critical_gap_std_s
    # Log-normal moment matching
    mu = float(np.log(m * m / np.sqrt(m * m + s * s)))
    sigma = float(np.sqrt(np.log(1.0 + (s * s) / (m * m))))
    gap = float(rng.lognormal(mu, sigma))
    return float(np.clip(gap, 0.5, 15.0))
