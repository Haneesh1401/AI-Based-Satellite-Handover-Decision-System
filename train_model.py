
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "satellite_dataset.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

FEATURES = [
    "signal_dbm",
    "latency_ms",
    "bandwidth_mbps",
    "congestion_pct",
    "distance_km",
    "visible",
    "eligible",
]
TARGET = "preferred_satellite"


def top1_accuracy(frame, scores):
    """Choose the highest-scoring eligible satellite in each scenario."""
    work = frame[
        ["scenario_id", "satellite_id", "eligible", TARGET]
    ].copy()
    work["selection_score"] = np.asarray(scores)

    correct = 0
    evaluated = 0

    for _, group in work.groupby("scenario_id", sort=False):
        eligible = group[group["eligible"] == 1]
        actual = eligible[eligible[TARGET] == 1]

        if eligible.empty or actual.empty:
            continue

        predicted_id = eligible.loc[
            eligible["selection_score"].idxmax(), "satellite_id"
        ]
        actual_id = actual.iloc[0]["satellite_id"]

        correct += int(predicted_id == actual_id)
        evaluated += 1

    return {
        "scenarios_evaluated": int(evaluated),
        "correct": int(correct),
        "accuracy": float(correct / evaluated) if evaluated else 0.0,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset missing. Run python generate_data.py first."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    required = FEATURES + [
        TARGET, "scenario_id", "satellite_id", "quality_score"
    ]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Missing values found in features or target.")

    invalid = (
        (df[TARGET] == 1)
        & ((df["visible"] != 1) | (df["eligible"] != 1))
    )
    if invalid.any():
        raise ValueError(
            "An invisible or ineligible satellite is labelled preferred."
        )

    # Split by whole scenario, not individual satellite rows.
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    train_idx, test_idx = next(
        splitter.split(
            df[FEATURES],
            df[TARGET],
            groups=df["scenario_id"],
        )
    )

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    overlap = set(train["scenario_id"]) & set(test["scenario_id"])
    if overlap:
        raise RuntimeError("Scenario leakage detected.")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(train[FEATURES], train[TARGET].astype(int))

    y_true = test[TARGET].astype(int)
    y_pred = model.predict(test[FEATURES])

    classes = list(model.named_steps["classifier"].classes_)
    positive_index = classes.index(1)
    probabilities = model.predict_proba(test[FEATURES])[:, positive_index]

    # Both policies are evaluated on exactly the same held-out scenarios.
    model_selection = top1_accuracy(test, probabilities)
    baseline_selection = top1_accuracy(
        test, test["quality_score"].to_numpy()
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "dataset": {
            "rows": int(len(df)),
            "scenarios": int(df["scenario_id"].nunique()),
        },
        "split": {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_scenarios": int(train["scenario_id"].nunique()),
            "test_scenarios": int(test["scenario_id"].nunique()),
            "scenario_overlap": len(overlap),
        },
        "candidate_classification": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, zero_division=0)
            ),
            "macro_f1": float(
                f1_score(y_true, y_pred, average="macro",
                        zero_division=0)
            ),
            "confusion_matrix_labels_0_then_1": cm.tolist(),
        },
        "model_top1": model_selection,
        "baseline_top1": baseline_selection,
        "feature_columns": FEATURES,
        "excluded_columns": [
            "scenario_id", "satellite_id", "quality_score"
        ],
        "interpretation": (
            "The target labels come from synthetic scoring rules. "
            "Metrics measure agreement with these labels, not real-world "
            "satellite-network performance."
        ),
    }

    # Save model and feature metadata for Phase 3.
    model_path = MODEL_DIR / "handover_model.joblib"
    joblib.dump({
        "model": model,
        "feature_columns": FEATURES,
        "target_column": TARGET,
        "model_type": "RandomForestClassifier",
    }, model_path)

    (OUTPUT_DIR / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    (OUTPUT_DIR / "classification_report.txt").write_text(
        classification_report(
            y_true, y_pred, labels=[0, 1], zero_division=0
        ),
        encoding="utf-8",
    )

    # Confusion matrix plot.
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm)
    ax.set_title("Random Forest Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], ["Not preferred", "Preferred"])
    ax.set_yticks([0, 1], ["Not preferred", "Preferred"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Compare satellite selection across held-out scenarios.
    names = ["Random Forest", "Rule-based score"]
    values = [
        model_selection["accuracy"],
        baseline_selection["accuracy"],
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, values)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Top-1 agreement with synthetic labels")
    ax.set_title("Held-out Scenario Selection Comparison")
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top1_comparison.png", dpi=150)
    plt.close(fig)

    print("\nPHASE 2 TRAINING COMPLETED")
    print("-" * 45)
    print(f"Model: {model_path}")
    print(f"Train scenarios: {metrics['split']['train_scenarios']}")
    print(f"Test scenarios: {metrics['split']['test_scenarios']}")
    print(f"Scenario overlap: {len(overlap)}")
    print("\nCandidate-level classification:")
    for name, value in metrics["candidate_classification"].items():
        print(f"  {name}: {value}")

    print(
        "\nRandom Forest top-1 accuracy: "
        f"{model_selection['accuracy']:.2%}"
    )
    print(
        "Rule-based top-1 accuracy: "
        f"{baseline_selection['accuracy']:.2%}"
    )
    print(f"\nReports and charts: {OUTPUT_DIR}")
    print(
        "\nNote: results measure agreement with synthetic labels, "
        "not real-world satellite performance."
    )


if __name__ == "__main__":
    main()
