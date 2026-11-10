# AI-Based Satellite Handover Decision System

## Phase 1 — Setup and synthetic data generation

This phase generates a reproducible synthetic dataset. It does not connect to real satellites or use real telemetry.

### Windows PowerShell setup

```powershell
cd satellite-handover-ai
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_data.py
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### Dataset design

Each row represents one satellite candidate in one scenario. The default run creates 3,000 scenarios × 6 satellites = 18,000 rows.

Features:
- `scenario_id`: scenario grouping identifier
- `satellite_id`: synthetic satellite name
- `signal_dbm`: simulated received signal strength (less negative is stronger)
- `latency_ms`: simulated network latency
- `bandwidth_mbps`: simulated available bandwidth
- `congestion_pct`: simulated congestion from 0 to 100
- `distance_km`: simulated distance
- `visible`: whether the satellite is visible in the scenario (0/1)
- `eligible`: whether it passes the simulation's minimum quality constraints (0/1)
- `quality_score`: transparent heuristic score used to choose the label
- `preferred_satellite`: target label; 1 for the highest-scoring eligible candidate in a scenario, otherwise 0

The labels are generated from a rule-based score, not from real handover outcomes. They are suitable for demonstrating a machine-learning workflow, not for making real-world satellite-network claims.
