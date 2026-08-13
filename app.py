"""Independent Streamlit interface for evaluating the saved models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


st.set_page_config(page_title="Bank Deposit Prediction", page_icon="🏦", layout="wide")
APP_FOLDER = Path(__file__).resolve().parent
MODEL_FOLDER = APP_FOLDER / "model"
SAMPLE_TEST_FILE = APP_FOLDER / "test_data.csv"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}


def metrics_for(actual, predicted, probabilities) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "AUC": roc_auc_score(actual, probabilities),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Recall": recall_score(actual, predicted, zero_division=0),
        "F1": f1_score(actual, predicted, zero_division=0),
        "MCC": matthews_corrcoef(actual, predicted),
    }


st.title("🏦 Bank Term Deposit Prediction")
st.caption("Use the included sample test data or upload your own CSV, then select a saved classifier and inspect its results.")

if not (MODEL_FOLDER / "metrics_summary.csv").exists():
    st.error("Model files are missing. Run train_models.py before starting the app.")
    st.stop()

saved_metrics = pd.read_csv(MODEL_FOLDER / "metrics_summary.csv")
with open(MODEL_FOLDER / "feature_columns.json", encoding="utf-8") as file:
    expected_columns = json.load(file)

left, right = st.columns([1, 2])
with left:
    selected_name = st.selectbox("Select a machine-learning model", list(MODEL_FILES))

    input_method = st.radio(
        "Choose test data",
        ["Use sample test_data.csv", "Upload your own CSV"],
        index=0,
    )

    uploaded_file = None
    if input_method == "Upload your own CSV":
        uploaded_file = st.file_uploader(
            "Upload a CSV test file",
            type="csv",
            help="Upload a CSV containing the required feature columns and actual_label (or y) for evaluation.",
        )
    elif not SAMPLE_TEST_FILE.exists():
        st.error(
            "Sample test_data.csv is not available in the deployed repository. "
            "Please upload a CSV instead."
        )

with right:
    st.subheader("Metrics saved during training")
    st.dataframe(saved_metrics.round(4), use_container_width=True, hide_index=True)

if input_method == "Use sample test_data.csv":
    if not SAMPLE_TEST_FILE.exists():
        st.stop()
    data_source = SAMPLE_TEST_FILE
    st.success("Using the sample test_data.csv included with the application.")
else:
    if uploaded_file is None:
        st.info("Upload a CSV file to continue.")
        st.stop()
    data_source = uploaded_file

# sep=None allows both ordinary comma CSVs and the original semicolon UCI CSV.
uploaded_data = pd.read_csv(data_source, sep=None, engine="python")

if "actual_label" in uploaded_data.columns:
    actual = uploaded_data.pop("actual_label")
elif "y" in uploaded_data.columns:
    actual = uploaded_data.pop("y").map({"no": 0, "yes": 1})
else:
    actual = None

missing_columns = [column for column in expected_columns if column not in uploaded_data.columns]
if missing_columns:
    st.error("This file is missing required feature columns: " + ", ".join(missing_columns))
    st.stop()

model_input = uploaded_data[expected_columns]
selected_model = joblib.load(MODEL_FOLDER / MODEL_FILES[selected_name])
predictions = selected_model.predict(model_input)
probabilities = selected_model.predict_proba(model_input)[:, 1]

predictions_table = pd.DataFrame(
    {
        "Prediction": pd.Series(predictions).map({0: "No subscription", 1: "Subscription"}),
        "Subscription probability": probabilities.round(4),
    }
)
st.subheader("Predictions")
st.dataframe(predictions_table, use_container_width=True, hide_index=True)

if actual is None:
    st.warning("No actual label was included. Predictions are available, but metrics and a confusion matrix require actual_label.")
    st.stop()

if actual.isna().any():
    st.error("The actual label column must contain only 0/1 or yes/no values.")
    st.stop()

current_metrics = metrics_for(actual, predictions, probabilities)
st.subheader(f"Evaluation metrics for {selected_name}")
st.dataframe(pd.DataFrame([current_metrics]).round(4), use_container_width=True, hide_index=True)

chart, report = st.columns(2)
with chart:
    matrix = confusion_matrix(actual, predictions)
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=axis)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(f"Confusion Matrix: {selected_name}")
    st.pyplot(figure)
with report:
    st.subheader("Classification report")
    st.code(classification_report(actual, predictions, target_names=["No", "Yes"], zero_division=0))
