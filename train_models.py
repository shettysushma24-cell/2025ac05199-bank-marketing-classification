"""Train five classifiers and create the files used by the Streamlit app.

Usage:
    python train_models.py --data data/bank-additional-full.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
MODEL_FOLDER = Path("model")


def calculate_metrics(actual, predicted, probabilities) -> dict[str, float]:
    """Calculate every metric specifically required in the assignment."""
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "AUC": roc_auc_score(actual, probabilities),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Recall": recall_score(actual, predicted, zero_division=0),
        "F1": f1_score(actual, predicted, zero_division=0),
        "MCC": matthews_corrcoef(actual, predicted),
    }


def main(data_file: str) -> None:
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Cannot find {data_path}. Download and extract the UCI file "
            "bank-additional-full.csv into the data folder first."
        )

    # The official UCI file uses semicolons rather than commas.
    raw_data = pd.read_csv(data_path, sep=";")
    if "y" not in raw_data.columns:
        raise ValueError("Expected target column 'y' was not found in the dataset.")

    features = raw_data.drop(columns="y")
    target = raw_data["y"].map({"no": 0, "yes": 1})
    if target.isna().any():
        raise ValueError("Target y must contain only 'yes' and 'no'.")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    numerical_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [col for col in features.columns if col not in numerical_columns]

    # Dense output keeps Gaussian Naive Bayes compatible with the encoded data.
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("missing_values", SimpleImputer(strategy="median")),
                        ("scaling", StandardScaler()),
                    ]
                ),
                numerical_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("missing_values", SimpleImputer(strategy="most_frequent")),
                        ("encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2500, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "kNN": KNeighborsClassifier(n_neighbors=15),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    filenames = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "kNN": "knn.joblib",
        "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
        "Random Forest": "random_forest.joblib",
    }

    MODEL_FOLDER.mkdir(exist_ok=True)
    model_results = []

    for name, classifier in classifiers.items():
        print(f"Training: {name}")
        model_pipeline = Pipeline(
            steps=[("preprocessing", preprocessing), ("classifier", classifier)]
        )
        model_pipeline.fit(X_train, y_train)

        test_predictions = model_pipeline.predict(X_test)
        test_probabilities = model_pipeline.predict_proba(X_test)[:, 1]
        scores = calculate_metrics(y_test, test_predictions, test_probabilities)
        scores["Model"] = name
        model_results.append(scores)
        joblib.dump(model_pipeline, MODEL_FOLDER / filenames[name])

    summary = pd.DataFrame(model_results)[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ].sort_values("F1", ascending=False)
    summary.to_csv(MODEL_FOLDER / "metrics_summary.csv", index=False)

    # The app uses this independent test file. It has raw feature columns and the true label.
    uploadable_test_data = X_test.copy()
    uploadable_test_data["actual_label"] = y_test.to_numpy()
    uploadable_test_data.to_csv("test_data.csv", index=False)

    with open(MODEL_FOLDER / "feature_columns.json", "w", encoding="utf-8") as file:
        json.dump(features.columns.tolist(), file, indent=2)

    print("\nRequired comparison table:")
    print(summary.round(4).to_string(index=False))
    print("\nCreated: model/*.joblib, model/metrics_summary.csv, model/feature_columns.json, test_data.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bank Marketing classification models.")
    parser.add_argument("--data", default="data/bank-additional-full.csv", help="Path to the UCI CSV file")
    arguments = parser.parse_args()
    main(arguments.data)
