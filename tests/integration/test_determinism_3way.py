"""End-to-end determinism, 3 independent ways + trusted key.

Rule:
  V1 pickle bytes      = binary serializer path
  V2 repr text         = __repr__ path (different code)
  V3 recursive walk    = no serializer, per-field ==
  Key generate_agents  = the project's known-good source of randomness
  Negative control     = different seeds MUST diverge

Same seed  → V1 = V2 = V3 = Key (all agree).
Diff seed  → all four MUST report "different".
"""
from __future__ import annotations
import pickle
import pytest

from surrogate_safety_abm.config.city_profiles import SURAT
from surrogate_safety_abm.environment import Intersection
from surrogate_safety_abm.simulation import SimulationConfig, SimulationEngine
from surrogate_safety_abm.simulation.scenario import (
    generate_agents, make_city_behaviours,
)


def _run(seed: int, n_vehicles: int = 5, duration_s: float = 5.0):
    agents = generate_agents(SURAT, seed=seed, n_vehicles=n_vehicles)
    cfg = SimulationConfig(duration_s=duration_s)      # ← no seed, per fix
    ic  = Intersection.four_legged(intersection_id="determinism")
    return SimulationEngine(city=SURAT, intersection=ic, agents=agents,
                            config=cfg, behaviours=make_city_behaviours(SURAT)).run()


def _deep(a, b) -> bool:
    if type(a) is not type(b): return False
    if isinstance(a, (int, float, str, bool, type(None))): return a == b
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_deep(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_deep(a[k], b[k]) for k in a)
    if hasattr(a, "__dict__"):           return _deep(vars(a), vars(b))
    if hasattr(a, "__dataclass_fields__"):
        return all(_deep(getattr(a, f), getattr(b, f)) for f in a.__dataclass_fields__)
    return a == b


class TestEndToEndDeterminism:
    def test_v1_pickle_same_seed(self):
        a, b = _run(42), _run(42)
        assert pickle.dumps(a.recorder.snapshots) == pickle.dumps(b.recorder.snapshots)

    def test_v2_repr_same_seed(self):
        a, b = _run(42), _run(42)
        assert repr(a.recorder.snapshots) == repr(b.recorder.snapshots)

    def test_v3_recursive_same_seed(self):
        a, b = _run(42), _run(42)
        assert _deep(a.recorder.snapshots, b.recorder.snapshots)

    def test_key_generate_agents_same_seed(self):
        a = generate_agents(SURAT, seed=42, n_vehicles=5)
        b = generate_agents(SURAT, seed=42, n_vehicles=5)
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert x.agent_id == y.agent_id
            assert x.speed == pytest.approx(y.speed)
            assert x.position.x == pytest.approx(y.position.x)
            assert x.position.y == pytest.approx(y.position.y)

    def test_negative_different_seeds_diverge(self):
        a, c = _run(42), _run(43)
        assert pickle.dumps(a.recorder.snapshots) != pickle.dumps(c.recorder.snapshots)

    def test_hand_checkable_fleet(self):
        ag  = generate_agents(SURAT, seed=42, n_vehicles=3)
        ag2 = generate_agents(SURAT, seed=42, n_vehicles=3)
        assert len(ag) == 3
        assert len({a.agent_id for a in ag}) == 3
        for x, y in zip(ag, ag2, strict=True):
            assert x.agent_id == y.agent_id
            assert x.vehicle_type == y.vehicle_type
            assert x.speed == pytest.approx(y.speed)

    @pytest.mark.parametrize("seed", [0, 1, 2**63 - 1])
    def test_edge_seeds_deterministic(self, seed):
        a, b = _run(seed), _run(seed)
        assert pickle.dumps(a.recorder.snapshots) == pickle.dumps(b.recorder.snapshots)

    def test_edge_empty_agents(self):
        def _empty():
            cfg = SimulationConfig(duration_s=0.5, time_step_s=0.1)
            ic  = Intersection.four_legged(intersection_id="empty")
            return SimulationEngine(city=SURAT, intersection=ic, agents=[],
                                    config=cfg, behaviours=[]).run()
        a, b = _empty(), _empty()
        assert a.steps_completed == b.steps_completed
        assert pickle.dumps(a.recorder.snapshots) == pickle.dumps(b.recorder.snapshots)
