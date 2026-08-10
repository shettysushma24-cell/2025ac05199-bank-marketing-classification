# Bank Term Deposit Classification

This is a learning starter for the ML Assignment 2 workflow. Replace this text with your own findings after training.

## Project structure

```text
.
├── app.py                 # Streamlit web application
├── train_models.py        # trains and saves five classifiers
├── requirements.txt       # Python dependencies
├── test_data.csv          # created after training
├── data/
│   └── bank-additional-full.csv   # download this file yourself
└── model/                 # created after training
```

## Run locally

1. Create and activate a virtual environment.
2. Run `pip install -r requirements.txt`.
3. Download the UCI Bank Marketing ZIP, extract `bank-additional-full.csv`, and place it in `data/`.
4. Run `python train_models.py`.
5. Run `streamlit run app.py`.

## Assignment README sections to complete

### Problem statement

Write this in your own words: predict whether a customer will subscribe to a term deposit after a bank marketing campaign.

### Dataset description

State the official UCI source, the number of rows and features you used, the target column `y`, and any preprocessing you performed.

### Models used

Describe the five required models. Paste your actual generated metrics table from `model/metrics_summary.csv`.

### Observations

Compare the models using F1, recall, AUC, and MCC, then identify the best model based on your own actual run.
