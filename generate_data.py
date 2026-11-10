"""
Phase 1: Generate synthetic satellite handover scenarios.
Each row is one satellite candidate observed within a scenario.
The target preferred_satellite is 1 for the highest-scoring eligible satellite
in that scenario and 0 for all other candidates. Invisible candidates are never
labelled preferred.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "scenario_id", "satellite_id", "signal_dbm", "latency_ms",
    "bandwidth_mbps", "congestion_pct", "distance_km", "visible",
    "eligible", "quality_score"
]

def generate_dataset(n_scenarios: int = 3000, n_satellites: int = 6,
                     seed: int = 42) -> pd.DataFrame:
    if n_scenarios < 1:
        raise ValueError("n_scenarios must be at least 1")
    if n_satellites < 5:
        raise ValueError("n_satellites must be at least 5")

    rng = np.random.default_rng(seed)
    rows = []
    for scenario_id in range(n_scenarios):
        # Scenario-wide conditions create related observations among satellites.
        weather_penalty = rng.uniform(0, 12)
        network_load = rng.uniform(15, 90)
        # A changing subset is visible in each scenario; at least one usually is.
        visible_mask = rng.random(n_satellites) < 0.78
        if not visible_mask.any():
            visible_mask[rng.integers(0, n_satellites)] = True

        scenario_rows = []
        for sat_idx in range(n_satellites):
            distance = float(rng.uniform(500, 2200))
            visible = bool(visible_mask[sat_idx])
            signal = -70 - (distance - 500) / 1700 * 35
            signal += rng.normal(0, 5) - weather_penalty * 0.35
            signal = float(np.clip(signal, -125, -65))

            congestion = float(np.clip(
                0.55 * network_load + rng.normal(0, 18), 0, 100
            ))
            latency = float(np.clip(
                18 + distance / 28 + congestion * 0.42
                + weather_penalty * 0.7 + rng.normal(0, 8), 15, 250
            ))
            bandwidth = float(np.clip(
                220 * (1 - congestion / 125)
                + (signal + 125) * 1.2 + rng.normal(0, 18), 2, 300
            ))

            # Simple eligibility constraints, intended for simulation only.
            eligible = bool(
                visible and signal >= -115 and latency <= 220
                and bandwidth >= 5
            )

            # Transparent heuristic score; the model will learn these synthetic labels.
            signal_component = np.clip((signal + 125) / 60, 0, 1) * 35
            latency_component = (1 - np.clip((latency - 15) / 235, 0, 1)) * 25
            bandwidth_component = np.clip(bandwidth / 250, 0, 1) * 25
            congestion_component = (1 - congestion / 100) * 15
            score = float(
                signal_component + latency_component +
                bandwidth_component + congestion_component
            )
            if not eligible:
                score = -1.0

            scenario_rows.append({
                "scenario_id": scenario_id,
                "satellite_id": f"SAT-{sat_idx + 1:02d}",
                "signal_dbm": round(signal, 3),
                "latency_ms": round(latency, 3),
                "bandwidth_mbps": round(bandwidth, 3),
                "congestion_pct": round(congestion, 3),
                "distance_km": round(distance, 3),
                "visible": int(visible),
                "eligible": int(eligible),
                "quality_score": round(score, 4),
                "preferred_satellite": 0
            })

        eligible_indices = [
            i for i, row in enumerate(scenario_rows) if row["eligible"] == 1
        ]
        if eligible_indices:
            winner = max(eligible_indices,
                         key=lambda i: scenario_rows[i]["quality_score"])
            scenario_rows[winner]["preferred_satellite"] = 1
        rows.extend(scenario_rows)

    df = pd.DataFrame(rows)
    # Defensive validation and cleanup.
    numeric_cols = [
        "signal_dbm", "latency_ms", "bandwidth_mbps", "congestion_pct",
        "distance_km", "visible", "eligible", "quality_score",
        "preferred_satellite"
    ]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df["visible"] = df["visible"].astype(int)
    df["eligible"] = df["eligible"].astype(int)
    df["preferred_satellite"] = df["preferred_satellite"].astype(int)

    # Never permit an invisible or ineligible candidate to be the target.
    invalid_targets = (
        (df["preferred_satellite"] == 1) &
        ((df["visible"] != 1) | (df["eligible"] != 1))
    )
    if invalid_targets.any():
        raise RuntimeError("Invalid target label assigned to invisible/ineligible satellite")

    return df

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic satellite scenarios.")
    parser.add_argument("--scenarios", type=int, default=3000,
                        help="Number of distinct scenarios (default: 3000)")
    parser.add_argument("--satellites", type=int, default=6,
                        help="Satellites per scenario; minimum 5 (default: 6)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/satellite_dataset.csv")
    args = parser.parse_args()

    df = generate_dataset(args.scenarios, args.satellites, args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    winners_per_scenario = df.groupby("scenario_id")["preferred_satellite"].sum()
    print(f"Saved: {output_path.resolve()}")
    print(f"Scenarios: {df['scenario_id'].nunique():,}")
    print(f"Satellites per scenario: {df.groupby('scenario_id').size().min()}–"
          f"{df.groupby('scenario_id').size().max()}")
    print(f"Candidate rows: {len(df):,}")
    print(f"Positive preferred labels: {int(df['preferred_satellite'].sum()):,}")
    print(f"Scenarios with an eligible winner: {(winners_per_scenario == 1).sum():,}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    invalid_targets = ((df["preferred_satellite"] == 1) & ((df["visible"] == 0) | (df["eligible"] == 0))).sum()
    print(f"Invalid invisible/ineligible targets: {int(invalid_targets)}")

if __name__ == "__main__":
    main()
