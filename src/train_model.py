
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import DATA_DIR, MODEL_DIR, OUTPUT_DIR, RANDOM_SEED


def main():
    input_path = DATA_DIR / "synthetic_accident_events.parquet"

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading synthetic accident events...")
    data = pd.read_parquet(input_path)

    # Verify that the dataset contains synthetic events only.
    if "is_synthetic" not in data.columns:
        raise ValueError("Missing required column: is_synthetic")

    if not data["is_synthetic"].fillna(False).all():
        raise ValueError(
            "Expected a dataset containing only synthetic events."
        )

    target = "severity"

    if target not in data.columns:
        raise ValueError(f"Missing target column: {target}")

    # Remove rows without a target label.
    data = data.loc[data[target].notna()].copy()

    if data.empty:
        raise ValueError("No valid labelled events found.")

    # Exclude identifiers, coordinates, timestamps and source indicators.
    excluded = {
        target,
        "event_id",
        "segment_id",
        "event_datetime",
        "data_source",
        "is_synthetic",
        "start_lon",
        "start_lat",
        "end_lon",
        "end_lat",
        "centroid_lon",
        "centroid_lat",
    }

    features = data.drop(
        columns=[column for column in excluded if column in data.columns]
    ).copy()

    labels = data[target].astype(str)

    # Add the night feature before identifying numeric and categorical
    # columns. This matches the night-time rule in the data generator.
    if "hour" in features.columns:
        features["night"] = (
            (features["hour"] >= 22)
            | (features["hour"] <= 5)
        ).astype(int)

    # Detect categorical columns, including pandas nullable string columns.
    categorical = features.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    # Convert categorical columns to ordinary object dtype.
    # Replace pandas pd.NA with np.nan for sklearn compatibility.
    for column in categorical:
        features[column] = (
            features[column]
            .astype(object)
            .where(features[column].notna(), np.nan)
        )

    # Detect numeric columns.
    numeric = features.select_dtypes(
        include=["number"]
    ).columns.tolist()

    if not numeric and not categorical:
        raise ValueError("No usable feature columns were found.")

    # Replace infinite numeric values with missing values.
    for column in numeric:
        features[column] = features[column].replace(
            [np.inf, -np.inf], np.nan
        )

    # Remove columns that contain no usable values.
    usable_numeric = [
        column
        for column in numeric
        if features[column].notna().any()
    ]

    usable_categorical = [
        column
        for column in categorical
        if features[column].notna().any()
    ]

    numeric = usable_numeric
    categorical = usable_categorical

    features = features[numeric + categorical]

    if features.shape[1] == 0:
        raise ValueError("All feature columns are empty.")

    print(f"Total labelled events: {len(features):,}")
    print(f"Numeric features: {len(numeric)}")
    print(f"Categorical features: {len(categorical)}")
    print(f"Features used: {features.columns.tolist()}")

    # Split before fitting preprocessing to prevent data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    print(f"Training events: {len(X_train):,}")
    print(f"Test events: {len(X_test):,}")

    # Preprocessing pipelines.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Unknown",
            ),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    transformers = []

    if numeric:
        transformers.append(
            ("numeric", numeric_pipeline, numeric)
        )

    if categorical:
        transformers.append(
            ("categorical", categorical_pipeline, categorical)
        )

    preprocessing = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    # Random Forest classifier.
    model = Pipeline([
        ("preprocessing", preprocessing),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
        ),
    ])

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    class_names = ["Minor", "Serious", "Fatal"]

    report_text = classification_report(
        y_test,
        predictions,
        labels=class_names,
        zero_division=0,
    )

    report_dict = classification_report(
        y_test,
        predictions,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=class_names,
    )

    print("\n===== ROADWATCH NAGPUR: MODEL EVALUATION =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")

    print("\nClassification Report:")
    print(report_text)

    print("Confusion Matrix (Minor, Serious, Fatal):")
    print(matrix)

    # Save trained model.
    model_path = MODEL_DIR / "severity_model.joblib"
    joblib.dump(model, model_path)

    # Save evaluation metrics.
    metrics = {
        "model": "RandomForestClassifier",
        "data_source": "SYNTHETIC",
        "warning": (
            "These metrics measure agreement with simulated severity "
            "labels. They do not establish real-world accident "
            "prediction performance."
        ),
        "total_events": int(len(features)),
        "training_events": int(len(X_train)),
        "test_events": int(len(X_test)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
        "class_order": class_names,
        "features": features.columns.tolist(),
        "random_seed": RANDOM_SEED,
    }

    metrics_path = OUTPUT_DIR / "model_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"\nModel saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print("\nIMPORTANT: Training data contains synthetic events only.")


if __name__ == "__main__":
    main()
