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

### Problem statement

The objective of this project is to predict whether a customer will subscribe to a bank term deposit after a marketing campaign. This is a binary classification problem, where the target outcome is either Yes (customer subscribes) or No (customer does not subscribe).

### Dataset description

This project uses the Bank Marketing dataset from the UCI Machine Learning Repository.

Dataset source: UCI Bank Marketing Dataset

File used: bank-additional-full.csv

Number of records: 41,188

Number of input features: 20

Target column: y

Target values: yes and no

The dataset contains customer demographic information, banking information, and marketing-campaign details. Examples of features include age, job, marital status, education, contact method, campaign duration, number of contacts, and outcome of previous campaigns.

Before training, categorical variables were encoded, numerical variables were scaled where appropriate, and the data was split into training and test sets.

### GitHub Repository Link

[View my GitHub repository](https://github.com/shettysushma24-cell/2025ac05199-bank-marketing-classification)

### Models used

The following classification models were implemented on the same dataset:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbours Classifier

Gaussian Naive Bayes Classifier

Random Forest Classifier

### Model Comparison

Required comparison table:

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8651 | 0.9438 | 0.4512 | 0.9116 | 0.6036 | 0.5813 |
| Random Forest | 0.8580 | 0.9497 | 0.4391 | 0.9407 | 0.5988 | 0.5824 |
| Decision Tree | 0.8649 | 0.9182 | 0.4493 | 0.8825 | 0.5954 | 0.5676 |
| kNN | 0.9086 | 0.9215 | 0.6606 | 0.3879 | 0.4888 | 0.4613 |
| Gaussian Naive Bayes | 0.8203 | 0.8393 | 0.3495 | 0.6907 | 0.4642 | 0.4009 |

### Observations

| ML Model | Observation about Model Performance |
|---|---|
| Logistic Regression | Highest F1-score (0.6036) with strong recall (0.9116). |
| Random Forest | Best AUC (0.9497), recall (0.9407), and MCC (0.5824). |
| Decision Tree | Reasonable performance, but lower AUC and F1-score than Logistic Regression and Random Forest. |
| kNN | Highest accuracy, but low recall; it misses many actual subscribers. |
| Gaussian Naive Bayes | Lowest overall performance on this dataset. |
| Overall Winner | **Random Forest** because it has the best AUC, recall, and MCC. |

### How to Run the Project Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 train_models.py
python3 -m streamlit run app.py
```

After the Streamlit app opens in your browser, upload `test_data.csv` to view predictions, evaluation metrics, the confusion matrix, and the classification report.
