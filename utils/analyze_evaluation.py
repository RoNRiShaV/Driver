
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def analyze(csv_path):

    if not os.path.exists(csv_path):

        print("CSV file not found:", csv_path)
        return

    df = pd.read_csv(csv_path)

    # --------------------------------------
    # BINARY DROWSINESS EVALUATION
    # --------------------------------------

    # Keep only NORMAL and DROWSY/Critical labels.
    # Distracted samples are excluded.
    df = df[
        df["ground_truth"].isin([
            "NORMAL",
            "DROWSY",
            "CRITICAL"
        ])
    ].copy()

    if df.empty:

        print("No labelled binary evaluation samples.")
        return

    # Ground truth.
    df["actual"] = df["ground_truth"].isin([
        "DROWSY",
        "CRITICAL"
    ]).astype(int)

    # Predictions.
    df["predicted"] = df["driver_state"].isin([
        "DROWSY",
        "CRITICAL"
    ]).astype(int)

    y_true = df["actual"]
    y_pred = df["predicted"]

    # --------------------------------------
    # METRICS
    # --------------------------------------

    accuracy = accuracy_score(
        y_true, y_pred
    )

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    print("\n========== BINARY DROWSINESS RESULTS ==========")

    print(f"Samples: {len(df)}")

    print(f"Accuracy:  {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall:    {recall:.4f}")

    print(f"F1-score:  {f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[
                "Non-Drowsy",
                "Drowsy"
            ],
            zero_division=0
        )
    )

    # --------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Non-Drowsy",
            "Drowsy"
        ],
        yticklabels=[
            "Non-Drowsy",
            "Drowsy"
        ]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title("Driver Drowsiness Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        "evaluation_results/confusion_matrix.png",
        dpi=300
    )

    plt.show()

    # --------------------------------------
    # TEMPORAL STATE ANALYSIS
    # --------------------------------------

    print("\n========== TEMPORAL ANALYSIS ==========")

    states = df["driver_state"].fillna(
        "UNKNOWN"
    )

    transitions = (
        states != states.shift()
    ).sum() - 1

    stability = (
        1 - transitions / len(states)
        if len(states) > 0
        else 0
    )

    print("Sampled observations:", len(states))

    print("Sampled state transitions:", transitions)

    print(f"Sampled stability: {stability:.4f}")

    # --------------------------------------
    # SAVE METRICS
    # --------------------------------------

    results = pd.DataFrame([{

        "samples": len(df),

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "sampled_transitions": transitions,

        "sampled_stability": stability

    }])

    results.to_csv(
        "evaluation_results/metrics_summary.csv",
        index=False
    )

    print("\nResults saved successfully.")


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: python utils/analyze_evaluation.py "
            "evaluation_results/your_file.csv"
        )

    else:

        analyze(sys.argv[1])
