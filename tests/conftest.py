"""Shared pytest configuration and Hypothesis profiles."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI / Colab

from hypothesis import Verbosity, settings

settings.register_profile(
    "default",
    max_examples=200,
    deadline=None,
    verbosity=Verbosity.normal,
)
settings.register_profile(
    "nightly",
    max_examples=10_000,
    deadline=None,
    verbosity=Verbosity.normal,
)
settings.load_profile("default")
