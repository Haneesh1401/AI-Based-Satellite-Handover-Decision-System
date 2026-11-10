from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "handover_model.joblib"


# ============================================================
# FEATURES
# Must match the features used during model training
# ============================================================

FEATURES = [
    "signal_dbm",
    "latency_ms",
    "bandwidth_mbps",
    "congestion_pct",
    "distance_km",
    "visible",
    "eligible",
]


# ============================================================
# HANDOVER ENGINE
# ============================================================

class SatelliteHandoverEngine:

    def __init__(
        self,
        model_path=MODEL_PATH,
        seed=123,
        min_dwell_steps=3,
        min_improvement=0.035
    ):

        if not Path(model_path).exists():
            raise FileNotFoundError(
                "Trained model not found. "
                "Please run train_model.py first."
            )

        # Load trained model
        bundle = joblib.load(model_path)

        self.model = bundle["model"]

        self.feature_columns = bundle.get(
            "feature_columns",
            FEATURES
        )

        # Validate feature compatibility
        if list(self.feature_columns) != FEATURES:
            raise ValueError(
                "Model features do not match the handover engine features."
            )

        # Random generator
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Handover configuration
        self.min_dwell_steps = min_dwell_steps
        self.min_improvement = min_improvement

        # Six simulated satellites
        self.satellites = [
            f"SAT-{i:02d}"
            for i in range(1, 7)
        ]

        self.phases = np.linspace(
            0,
            2 * np.pi,
            6,
            endpoint=False
        )

        # Simulation state
        self.step_number = 0

        self.current_satellite = None

        self.steps_since_handover = 999

        self.handover_count = 0

        self.history = []

        self.events = []

        self.last_frame = pd.DataFrame()


    # ========================================================
    # RESET SIMULATION
    # ========================================================

    def reset(self):

        self.rng = np.random.default_rng(self.seed)

        self.step_number = 0

        self.current_satellite = None

        self.steps_since_handover = 999

        self.handover_count = 0

        self.history = []

        self.events = []

        self.last_frame = pd.DataFrame()


    # ========================================================
    # GENERATE SATELLITE CONDITIONS
    # ========================================================

    def generate_snapshot(self):

        t = self.step_number

        rows = []

        # Shared network conditions
        global_load = (
            48
            + 22 * np.sin(t * 0.19)
        )

        weather_penalty = max(
            0,
            5 + 4 * np.sin(t * 0.11)
        )

        # Generate each satellite
        for i, sat_id in enumerate(self.satellites):

            angle = (
                t * 0.22
                + self.phases[i]
            )

            # Simulated distance
            distance = (
                550
                + 1500
                * (
                    0.5
                    + 0.5 * np.sin(angle)
                )
            )

            # Visibility
            visible = int(
                np.sin(angle) > -0.15
            )

            # Congestion
            congestion = float(
                np.clip(
                    global_load
                    + self.rng.normal(0, 16),
                    0,
                    100
                )
            )

            # Signal strength
            signal = (
                -72
                - (distance - 550) / 1500 * 37
            )

            signal += (
                self.rng.normal(0, 3.5)
                - weather_penalty * 0.25
            )

            signal = float(
                np.clip(
                    signal,
                    -125,
                    -65
                )
            )

            # Latency
            latency = float(
                np.clip(
                    18
                    + distance / 28
                    + congestion * 0.42
                    + weather_penalty * 0.65
                    + self.rng.normal(0, 5),
                    15,
                    250
                )
            )

            # Bandwidth
            bandwidth = float(
                np.clip(
                    220
                    * (1 - congestion / 125)
                    + (signal + 125) * 1.2
                    + self.rng.normal(0, 12),
                    2,
                    300
                )
            )

            # =================================================
            # ELIGIBILITY CHECK
            # =================================================

            eligible = int(
                visible == 1
                and signal >= -115
                and latency <= 220
                and bandwidth >= 5
            )

            rows.append({

                "satellite_id": sat_id,

                "signal_dbm": round(signal, 3),

                "latency_ms": round(latency, 3),

                "bandwidth_mbps": round(bandwidth, 3),

                "congestion_pct": round(congestion, 3),

                "distance_km": round(
                    float(distance),
                    3
                ),

                "visible": visible,

                "eligible": eligible,
            })

        return pd.DataFrame(rows)


    # ========================================================
    # AI PREDICTION
    # ========================================================

    def predict_scores(self, frame):

        classifier = self.model.named_steps["classifier"]

        classes = list(
            classifier.classes_
        )

        positive_index = classes.index(1)

        probabilities = self.model.predict_proba(
            frame[FEATURES]
        )

        return probabilities[:, positive_index]


    # ========================================================
    # RECORD HANDOVER EVENT
    # ========================================================

    def record_event(
        self,
        old_id,
        new_id,
        reason,
        candidate
    ):

        self.events.append({

            "step": self.step_number,

            "from_satellite":
                old_id if old_id else "NONE",

            "to_satellite":
                new_id,

            "reason":
                reason,

            "signal_dbm":
                float(candidate["signal_dbm"]),

            "latency_ms":
                float(candidate["latency_ms"]),

            "bandwidth_mbps":
                float(candidate["bandwidth_mbps"]),

            "ai_score":
                float(candidate["ai_score"]),
        })


    # ========================================================
    # MAIN SIMULATION STEP
    # ========================================================

    def step(self):

        # Advance simulation
        self.step_number += 1

        # Generate new satellite conditions
        frame = self.generate_snapshot()

        # AI suitability prediction
        frame["ai_score"] = self.predict_scores(frame)

        self.last_frame = frame.copy()

        # Only eligible satellites can be selected
        eligible = frame[
            frame["eligible"] == 1
        ].copy()

        recommended = None

        switched = False


        # ====================================================
        # CURRENT SATELLITE STATUS
        # ====================================================

        current_rows = frame[
            frame["satellite_id"]
            == self.current_satellite
        ]

        current_available = (

            not current_rows.empty

            and int(
                current_rows.iloc[0]["eligible"]
            ) == 1
        )


        # ====================================================
        # CASE 1: NO ELIGIBLE SATELLITE
        # ====================================================

        if eligible.empty:

            reason = (
                "No eligible satellite available."
            )

            if (
                self.current_satellite is not None
                and not current_available
            ):

                reason = (
                    "Current satellite unavailable and "
                    "no eligible alternative exists."
                )


        # ====================================================
        # CASE 2: ELIGIBLE SATELLITES EXIST
        # ====================================================

        else:

            # Best AI candidate
            best = eligible.loc[
                eligible["ai_score"].idxmax()
            ]

            recommended = str(
                best["satellite_id"]
            )


            # ================================================
            # INITIAL CONNECTION
            # ================================================

            if self.current_satellite is None:

                self.current_satellite = recommended

                self.handover_count += 1

                self.steps_since_handover = 0

                switched = True

                reason = (
                    "Initial connection established "
                    "with the highest-scoring eligible satellite."
                )

                self.record_event(
                    None,
                    recommended,
                    reason,
                    best
                )


            # ================================================
            # FORCED SAFETY HANDOVER
            # ================================================

            elif not current_available:

                old_id = self.current_satellite

                self.current_satellite = recommended

                self.handover_count += 1

                self.steps_since_handover = 0

                switched = True

                reason = (
                    "Forced safety handover: "
                    "current satellite is no longer eligible."
                )

                self.record_event(
                    old_id,
                    recommended,
                    reason,
                    best
                )


            # ================================================
            # NORMAL DECISION
            # ================================================

            else:

                current = current_rows.iloc[0]

                current_score = float(
                    current["ai_score"]
                )

                best_score = float(
                    best["ai_score"]
                )

                improvement = (
                    best_score - current_score
                )


                # Current satellite is best
                if recommended == self.current_satellite:

                    reason = (
                        "Current satellite remains "
                        "the best eligible candidate."
                    )


                # Minimum dwell time
                elif (
                    self.steps_since_handover
                    < self.min_dwell_steps
                ):

                    reason = (
                        "Switch avoided: minimum dwell "
                        "time has not been reached."
                    )


                # Improvement threshold
                elif (
                    improvement
                    < self.min_improvement
                ):

                    reason = (
                        f"Switch avoided: score improvement "
                        f"({improvement:.3f}) is below "
                        f"the required threshold "
                        f"({self.min_improvement:.3f})."
                    )


                # Perform handover
                else:

                    old_id = self.current_satellite

                    self.current_satellite = recommended

                    self.handover_count += 1

                    self.steps_since_handover = 0

                    switched = True

                    reason = (
                        f"Handover triggered: alternative "
                        f"improves AI score by "
                        f"{improvement:.3f}."
                    )

                    self.record_event(
                        old_id,
                        recommended,
                        reason,
                        best
                    )


        # ====================================================
        # CURRENT SERVING METRICS
        # ====================================================

        serving = frame[
            frame["satellite_id"]
            == self.current_satellite
        ]


        if not serving.empty:

            serving_row = serving.iloc[0]

            serving_latency = float(
                serving_row["latency_ms"]
            )

            serving_signal = float(
                serving_row["signal_dbm"]
            )

            serving_visible = int(
                serving_row["visible"]
            )

        else:

            serving_latency = None

            serving_signal = None

            serving_visible = 0


        # ====================================================
        # STORE HISTORY
        # ====================================================

        result = {

            "step": self.step_number,

            "current_satellite":
                self.current_satellite,

            "recommended_satellite":
                recommended,

            "handover_triggered":
                switched,

            "handover_count":
                self.handover_count,

            "reason":
                reason,

            "serving_latency_ms":
                serving_latency,

            "serving_signal_dbm":
                serving_signal,

            "serving_visible":
                serving_visible,

            "eligible_count":
                len(eligible),
        }


        self.history.append(
            result.copy()
        )


        # Increment dwell timer
        self.steps_since_handover += 1


        return {

            **result,

            "satellites":
                frame.copy(),

            "history":
                pd.DataFrame(self.history),

            "events":
                pd.DataFrame(self.events),
        }


# ============================================================
# COMMAND LINE DEMO
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps",
        type=int,
        default=30
    )

    args = parser.parse_args()

    engine = SatelliteHandoverEngine()


    print("\nSATELLITE HANDOVER SIMULATION")
    print("=" * 70)


    for _ in range(
        max(1, args.steps)
    ):

        result = engine.step()


        status = (
            "HANDOVER"
            if result["handover_triggered"]
            else "NO SWITCH"
        )


        print(

            f"Step {result['step']:02d} | "

            f"Serving: "
            f"{str(result['current_satellite']):6s} | "

            f"Recommended: "
            f"{str(result['recommended_satellite']):6s} | "

            f"{status} | "

            f"{result['reason']}"
        )


    print("\n" + "=" * 70)

    print("HANDOVER EVENT HISTORY")

    print("=" * 70)


    if engine.events:

        events_df = pd.DataFrame(
            engine.events
        )

        print(
            events_df.to_string(
                index=False
            )
        )

    else:

        print(
            "No handovers recorded."
        )


    print(
        f"\nTotal handover events: "
        f"{engine.handover_count}"
    )


if __name__ == "__main__":

    main()