"""Command-line entry point for running a simulation."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from surrogate_safety_abm.agents import Agent, VehicleAgent, VehicleType
from surrogate_safety_abm.behaviour import EvasiveBrakingModel, GapAcceptanceModel
from surrogate_safety_abm.behaviour.base import BehaviourModel
from surrogate_safety_abm.config.city_profiles import CITY_PROFILES
from surrogate_safety_abm.environment import Intersection
from surrogate_safety_abm.geometry import Vec2
from surrogate_safety_abm.simulation import SimulationConfig, SimulationEngine
from surrogate_safety_abm.simulation.scenario import generate_agents

_BEHAVIOURS_CHOICES = ("none", "standard", "calibrated")
_CALIBRATED_SOFT_TTC_S = 2.0
_CALIBRATED_HARD_TTC_S = 1.0


def _build_demo_agents() -> list[Agent]:
    """Create a small synthetic 4-vehicle scenario (backward-compat default)."""
    return [
        VehicleAgent("v1", Vec2(-40.0, 0.0), 12.0, 0.0, VehicleType.CAR),
        VehicleAgent("v2", Vec2(0.0, -40.0), 10.0, 1.5707963, VehicleType.TWO_WHEELER),
        VehicleAgent("v3", Vec2(40.0, 0.0), 8.0, 3.14159265, VehicleType.THREE_WHEELER),
        VehicleAgent("v4", Vec2(0.0, 40.0), 9.0, -1.5707963, VehicleType.CAR),
    ]


def _resolve_behaviours(kind: str) -> list[BehaviourModel]:
    """Map a --behaviours choice to a list of behaviour models."""
    if kind == "none":
        return []
    if kind == "standard":
        return [GapAcceptanceModel(), EvasiveBrakingModel()]
    if kind == "calibrated":
        return [
            GapAcceptanceModel(),
            EvasiveBrakingModel(
                soft_brake_ttc_s=_CALIBRATED_SOFT_TTC_S,
                hard_brake_ttc_s=_CALIBRATED_HARD_TTC_S,
            ),
        ]
    raise ValueError(f"unknown behaviours kind: {kind!r}")


def main(argv: list[str] | None = None) -> int:
    """Run the simulation and write conflict events to CSV."""
    parser = argparse.ArgumentParser(description="Run the surrogate-safety-ABM simulation.")
    parser.add_argument("--city", choices=sorted(CITY_PROFILES.keys()), default="surat")
    parser.add_argument(
        "--vehicle-count",
        type=int,
        default=None,
        help="number of agents to spawn (default: 4-vehicle demo)",
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--behaviours",
        choices=_BEHAVIOURS_CHOICES,
        default="standard",
        help=(
            "behaviour stack: none (kinematic only), "
            "standard (default braking), calibrated (gentler braking)"
        ),
    )
    parser.add_argument("--warm-up", type=float, default=0.0)
    parser.add_argument("--conflict-radius", type=float, default=15.0)
    parser.add_argument("--ttc-threshold", type=float, default=5.0)
    parser.add_argument("--min-gap", type=float, default=2.0)
    parser.add_argument("--intersection-zone", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=Path("output/conflicts.csv"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print resolved config and exit without simulating",
    )
    args = parser.parse_args(argv)

    city = CITY_PROFILES[args.city]
    intersection = Intersection.four_legged(intersection_id=args.city)

    if args.vehicle_count is not None:
        if args.vehicle_count < 1:
            print(
                f"--vehicle-count must be >= 1, got {args.vehicle_count}",
                file=sys.stderr,
            )
            return 2
        agents: list[Agent] = list(
            generate_agents(city, seed=args.seed, n_vehicles=args.vehicle_count)
        )
    else:
        agents = _build_demo_agents()

    config = SimulationConfig(
        duration_s=args.duration,
        time_step_s=args.dt,
        warm_up_s=args.warm_up,
        conflict_radius_m=args.conflict_radius,
        ttc_threshold_s=args.ttc_threshold,
        min_gap_m=args.min_gap,
        intersection_zone_m=args.intersection_zone,
        seed=args.seed,
    )

    try:
        behaviours = _resolve_behaviours(args.behaviours)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"city:             {city.name}")
        print(f"agents:           {len(agents)}")
        print(f"behaviours:       {[type(b).__name__ for b in behaviours]}")
        print(f"duration:         {config.duration_s}s (dt={config.time_step_s}s)")
        print(f"seed:             {config.seed}")
        print(f"conflict_radius:  {config.conflict_radius_m}m")
        print(f"ttc_threshold:    {config.ttc_threshold_s}s")
        print(f"out:              {args.out}")
        return 0

    engine = SimulationEngine(
        city=city,
        intersection=intersection,
        agents=agents,
        config=config,
        behaviours=behaviours,
    )
    result = engine.run()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "agent_a", "agent_b", "ttc_s", "drac_ms2", "distance_m"])
        for ev in result.recorder.conflicts:
            writer.writerow(
                [
                    ev.time_s,
                    ev.agent_a,
                    ev.agent_b,
                    ev.ttc_s,
                    ev.drac_ms2,
                    ev.distance_m,
                ]
            )

    print(f"Simulation complete: {result.steps_completed} steps")
    print(f"  Conflict events: {len(result.recorder.conflicts)}")
    print(f"  Snapshots:       {len(result.recorder.snapshots)}")
    print(f"  Output:          {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
