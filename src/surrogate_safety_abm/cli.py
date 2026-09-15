"""Command-line entry point for running a simulation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from surrogate_safety_abm.agents import Agent, VehicleAgent, VehicleType
from surrogate_safety_abm.config.city_profiles import CITY_PROFILES
from surrogate_safety_abm.environment import Intersection
from surrogate_safety_abm.geometry import Vec2
from surrogate_safety_abm.simulation import SimulationConfig, SimulationEngine


def _build_demo_agents() -> list[Agent]:
    """Create a small synthetic 4-vehicle scenario."""
    return [
        VehicleAgent("v1", Vec2(-40.0, 0.0), 12.0, 0.0, VehicleType.CAR),
        VehicleAgent("v2", Vec2(0.0, -40.0), 10.0, 1.5707963, VehicleType.TWO_WHEELER),
        VehicleAgent("v3", Vec2(40.0, 0.0), 8.0, 3.14159265, VehicleType.THREE_WHEELER),
        VehicleAgent("v4", Vec2(0.0, 40.0), 9.0, -1.5707963, VehicleType.CAR),
    ]


def main() -> int:
    """Run the simulation and write conflict events to CSV."""
    parser = argparse.ArgumentParser(description="Run the surrogate-safety-ABM simulation.")
    parser.add_argument("--city", choices=sorted(CITY_PROFILES.keys()), default="surat")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("output/conflicts.csv"))
    args = parser.parse_args()

    city = CITY_PROFILES[args.city]
    intersection = Intersection.four_legged(intersection_id=args.city)
    engine = SimulationEngine(
        city=city,
        intersection=intersection,
        agents=_build_demo_agents(),
        config=SimulationConfig(
            duration_s=args.duration,
            time_step_s=args.dt,
            seed=args.seed,
        ),
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
