"""Train and validate the ML classifier used for the Player 2 bonus gesture."""

from __future__ import annotations

from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from feature_extraction import FEATURE_NAMES


MIN_SAMPLES_PER_CLASS = 50


def normalize_labels(series: pd.Series) -> pd.Series | None:
    mapping = {
        "neutral": config.LABEL_NEUTRAL,
        "bonus_gesture": config.LABEL_BONUS,
        "0": config.LABEL_NEUTRAL,
        "1": config.LABEL_BONUS,
        0: config.LABEL_NEUTRAL,
        1: config.LABEL_BONUS,
    }
    normalized = series.map(lambda value: mapping.get(value, mapping.get(str(value).strip())))
    if normalized.isna().any():
        invalid = sorted(series[normalized.isna()].astype(str).unique().tolist())
        print(f"ERROR: Dataset contains unsupported labels: {invalid}")
        return None
    return normalized.astype(int)


def load_dataset() -> tuple[pd.DataFrame, pd.Series] | None:
    if not config.DATASET_PATH.exists():
        print(f"ERROR: Dataset file not found: {config.DATASET_PATH}")
        print("Run: python collect_dataset.py")
        return None

    try:
        df = pd.read_csv(config.DATASET_PATH)
    except pd.errors.EmptyDataError:
        print(f"ERROR: Dataset is empty: {config.DATASET_PATH}")
        return None

    missing = [column for column in [*FEATURE_NAMES, "label"] if column not in df.columns]
    if missing:
        print(f"ERROR: Dataset is missing columns: {missing}")
        return None

    df = df.dropna(subset=[*FEATURE_NAMES, "label"]).copy()
    if df.empty:
        print("ERROR: Dataset contains no valid rows.")
        return None

    labels = normalize_labels(df["label"])
    if labels is None:
        return None

    try:
        features = df[FEATURE_NAMES].astype(float)
    except (TypeError, ValueError) as exc:
        print(f"ERROR: Dataset contains non-numeric feature values: {exc}")
        return None
    if not np.isfinite(features.to_numpy()).all():
        print("ERROR: Dataset contains infinite feature values.")
        return None
    return features, labels


def check_sample_counts(labels: pd.Series) -> bool:
    counts = labels.value_counts().to_dict()
    neutral_count = counts.get(config.LABEL_NEUTRAL, 0)
    bonus_count = counts.get(config.LABEL_BONUS, 0)
    print(f"Samples: neutral={neutral_count}, bonus_gesture={bonus_count}")
    if neutral_count < MIN_SAMPLES_PER_CLASS or bonus_count < MIN_SAMPLES_PER_CLASS:
        print(f"WARNING: Need at least {MIN_SAMPLES_PER_CLASS} samples per class.")
        print("Training stopped. Collect more samples with: python collect_dataset.py")
        return False
    return True


def save_confusion_matrix(cm, labels: list[str]) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Bonus gesture confusion matrix",
    )

    threshold = cm.max() / 2.0 if cm.size else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                format(cm[row, col], "d"),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def main() -> int:
    loaded = load_dataset()
    if loaded is None:
        return 1
    features, labels = loaded

    if not check_sample_counts(labels):
        return 1

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(x_train.to_numpy(), y_train)
    predictions = model.predict(x_test.to_numpy())

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label=config.LABEL_BONUS, zero_division=0)
    recall = recall_score(y_test, predictions, pos_label=config.LABEL_BONUS, zero_division=0)
    f1 = f1_score(y_test, predictions, pos_label=config.LABEL_BONUS, zero_division=0)
    report = classification_report(
        y_test,
        predictions,
        labels=[config.LABEL_NEUTRAL, config.LABEL_BONUS],
        target_names=["neutral", "bonus_gesture"],
        zero_division=0,
    )
    cm = confusion_matrix(y_test, predictions, labels=[config.LABEL_NEUTRAL, config.LABEL_BONUS])

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "label_names": config.LABEL_NAMES,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": {
            "accuracy": accuracy,
            "bonus_precision": precision,
            "bonus_recall": recall,
            "bonus_f1": f1,
        },
    }
    joblib.dump(payload, config.MODEL_PATH)

    save_confusion_matrix(cm, ["neutral", "bonus_gesture"])

    warning = ""
    if precision < 0.85:
        warning = "\nWARNING: Bonus precision is below 0.85. Collect a cleaner or larger dataset before the tournament.\n"

    metrics_text = (
        "Blobby Face Controller - bonus model validation\n"
        f"Trained at: {payload['trained_at']}\n"
        f"Dataset: {config.DATASET_PATH}\n"
        f"Model: StandardScaler + SVC(kernel='rbf', probability=True)\n\n"
        f"accuracy: {accuracy:.4f}\n"
        f"bonus_precision: {precision:.4f}\n"
        f"bonus_recall: {recall:.4f}\n"
        f"bonus_f1: {f1:.4f}\n\n"
        "Classification report:\n"
        f"{report}\n"
        f"Confusion matrix:\n{cm}\n"
        f"{warning}"
    )
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "validation_metrics.txt").write_text(metrics_text, encoding="utf-8")

    print(metrics_text)
    print(f"Saved model: {config.MODEL_PATH}")
    print(f"Saved report: {config.REPORTS_DIR / 'validation_metrics.txt'}")
    print(f"Saved confusion matrix: {config.REPORTS_DIR / 'confusion_matrix.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
