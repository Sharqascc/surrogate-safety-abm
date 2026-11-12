"""Statistical analysis and visualisation of conflict-event data."""

from surrogate_safety_abm.analysis.accuracy import (
    AccuracyReport,
    compare,
    ks_test,
    mae,
    mean_diff_ci95,
    pearson_r,
    rmse,
    wasserstein_distance,
)
from surrogate_safety_abm.analysis.descriptive import (
    SSMSummary,
    summarise,
    summarise_drac,
    summarise_ttc,
)
from surrogate_safety_abm.analysis.episodes import (
    DEFAULT_MAX_GAP_S,
    ConflictEpisode,
    collapse_episodes,
    episode_severity_counts,
)
from surrogate_safety_abm.analysis.extreme_value import (
    GPDFit,
    fit_ttc_gpd,
)
from surrogate_safety_abm.analysis.severity import (
    DRAC_CRITICAL_THRESHOLD_MS2,
    TTC_SERIOUS_THRESHOLD_S,
    TTC_SLIGHT_THRESHOLD_S,
    Severity,
    classify_severity,
    severity_counts,
)

__all__ = [
    "DEFAULT_MAX_GAP_S",
    "DRAC_CRITICAL_THRESHOLD_MS2",
    "TTC_SERIOUS_THRESHOLD_S",
    "TTC_SLIGHT_THRESHOLD_S",
    "AccuracyReport",
    "ConflictEpisode",
    "GPDFit",
    "SSMSummary",
    "Severity",
    "classify_severity",
    "collapse_episodes",
    "compare",
    "episode_severity_counts",
    "fit_ttc_gpd",
    "ks_test",
    "mae",
    "mean_diff_ci95",
    "pearson_r",
    "rmse",
    "severity_counts",
    "summarise",
    "summarise_drac",
    "summarise_ttc",
    "wasserstein_distance",
]
