# Machine Learning — Assignment 2
## Bank Term Deposit Prediction — Model Comparison

## 1. GitHub Repository Link

**GitHub Repository:** https://github.com/shettysushma24-cell/2025ac05199-bank-marketing-classification

The repository contains the complete source code and project artifacts:

- `train_models.py` — data loading, preprocessing, model training and evaluation
- `app.py` — Streamlit web application
- `requirements.txt` — Python dependencies
- `test_data.csv` — generated test data used for the Streamlit demonstration
- `data/bank-additional-full.csv` — UCI Bank Marketing dataset
- `model/` — trained model files, feature-column metadata and model comparison metrics

## 2. Live Streamlit App Link

**Live Streamlit App:** https://2025ac05199-ml-assignment-bank-marketing-classification.streamlit.app/

The Streamlit application allows users to:

- Use the included sample `test_data.csv` or upload your own test CSV data
- Select a machine-learning model
- Generate bank-term-deposit predictions
- View subscription probabilities
- View Accuracy, AUC, Precision, Recall, F1 Score and MCC
- View the confusion matrix
- View the classification report

## 3. Screenshot — Executed on BITS Virtual Lab

The Streamlit application was executed on the BITS Virtual Lab and the screenshot below shows the deployed interface.

![Bank Term Deposit Prediction — BITS Virtual Lab](bits_lab_screenshot.jpg)

> Streamlit application running on BITS Virtual Lab with the Bank Term Deposit Prediction interface open in the browser.

## 4. README Content

### a. Problem Statement

Banks conduct marketing campaigns to identify customers who are likely to subscribe to a term deposit. Manually identifying potential customers from a large customer base can be time-consuming and may result in inefficient targeting.

This project builds and compares five machine-learning classification models to predict whether a customer will subscribe to a term deposit after a bank marketing campaign. The prediction is based on customer demographic information, contact details, campaign information and relevant economic indicators.

An interactive Streamlit web application is also provided so that the trained models can be demonstrated and evaluated on test data.

### b. Dataset Description

**Name:** Bank Marketing Dataset — `bank-additional-full.csv`

**Source:** UCI Machine Learning Repository

**Official UCI Dataset Page:**

https://archive.ics.uci.edu/dataset/222/bank+marketing

**Direct UCI Dataset ZIP:**

https://archive.ics.uci.edu/static/public/222/bank+marketing.zip

**Dataset used in this project:**

`bank-additional-full.csv`

**Local project path:**

```text
data/bank-additional-full.csv
```

The dataset contains **41,188 instances**, **20 predictor features**, and **1 binary target column**, giving **21 columns in total**.

The target column is:

```text
y
```

Target encoding:

- `no` → `0`
- `yes` → `1`

The target distribution in the dataset is:

- `no`: 36,548 records
- `yes`: 4,640 records

This indicates a significant class imbalance, with approximately 88.7% non-subscribers and 11.3% subscribers.

#### Features

The 20 predictor features are:

- `age`
- `job`
- `marital`
- `education`
- `default`
- `housing`
- `loan`
- `contact`
- `month`
- `day_of_week`
- `duration`
- `campaign`
- `pdays`
- `previous`
- `poutcome`
- `emp.var.rate`
- `cons.price.idx`
- `cons.conf.idx`
- `euribor3m`
- `nr.employed`

#### Data preprocessing

The official UCI CSV file uses a semicolon (`;`) delimiter.

The project performs the following preprocessing:

- The target variable `y` is mapped from `no/yes` to `0/1`.
- The dataset is split into **80% training and 20% testing** using stratification.
- Numerical features are processed using:
  - Median imputation
  - StandardScaler
- Categorical features are processed using:
  - Most-frequent-value imputation
  - One-hot encoding
- Unknown categorical values are handled using `handle_unknown="ignore"`.
- The preprocessing and classifier are combined in a Scikit-learn Pipeline.

The random state is set to **42** for reproducibility.

### c. GitHub Repository Link

**GitHub Repository:** https://github.com/shettysushma24-cell/2025ac05199-bank-marketing-classification

The repository should contain the complete implementation, including the training script, Streamlit application, requirements file, dataset folder, generated test data and trained model artifacts.

### d. Models Used

Five classification models were trained using the same train/test split and preprocessing approach:

1. **Logistic Regression**
2. **Decision Tree**
3. **k-Nearest Neighbors (kNN)**
4. **Gaussian Naive Bayes**
5. **Random Forest**

Logistic Regression, Decision Tree and Random Forest use class-balanced weighting to address the imbalance between subscribers and non-subscribers.

The models were evaluated using:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8651 | 0.9438 | 0.4512 | 0.9116 | 0.6036 | 0.5813 |
| Decision Tree | 0.8642 | 0.9174 | 0.4478 | 0.8825 | 0.5941 | 0.5663 |
| kNN | 0.9086 | 0.9215 | 0.6606 | 0.3879 | 0.4888 | 0.4613 |
| Gaussian Naive Bayes | 0.8203 | 0.8393 | 0.3495 | 0.6907 | 0.4642 | 0.4009 |
| Random Forest | 0.8705 | 0.9520 | 0.4629 | 0.9332 | 0.6188 | 0.6005 |

### Observations

#### Logistic Regression

Logistic Regression provides a strong baseline, with an AUC of approximately **0.94**, recall of approximately **0.91**, and F1 Score of approximately **0.60**. Its high recall indicates that it identifies a large proportion of customers who actually subscribe to a term deposit. The relatively low precision reflects the class imbalance and the use of balanced class weights, which encourages the model to identify more positive cases.

#### Decision Tree

The Decision Tree achieves performance comparable to Logistic Regression, with an accuracy of approximately **0.86**, recall of approximately **0.88**, and F1 Score of approximately **0.59**. Its AUC of approximately **0.92** and MCC of approximately **0.57** are lower than those of Random Forest and Logistic Regression, indicating comparatively weaker overall discrimination.

#### kNN

kNN achieves the **highest accuracy (0.9086)** and **highest precision (0.6606)** among the five models. However, its recall is only **0.3879**, meaning that it misses a substantial number of actual subscribers. Consequently, its F1 Score and MCC are lower than those of Logistic Regression and Random Forest. The result demonstrates the importance of considering multiple evaluation metrics instead of accuracy alone.

#### Gaussian Naive Bayes

Gaussian Naive Bayes has the weakest overall performance, with an AUC of approximately **0.84**, precision of approximately **0.35**, F1 Score of approximately **0.46**, and MCC of approximately **0.40**. Although its recall is reasonably high at approximately **0.69**, the lower precision indicates a relatively high number of false-positive predictions.

#### Random Forest

Random Forest provides the strongest overall performance across the most important balanced metrics. It achieves the highest **AUC (0.9520)**, highest **recall (0.9332)**, highest **F1 Score (0.6188)**, and highest **MCC (0.6005)** among the evaluated models.

The precision of **0.4629** is lower than that of kNN, but this is associated with the substantially higher recall. For the bank marketing use case, identifying a large proportion of potential subscribers can be valuable, making Random Forest a strong overall choice.

### Overall Winner

**Random Forest** is selected as the overall best model for this project.

It provides the best combination of:

- AUC: **0.9520**
- Recall: **0.9332**
- F1 Score: **0.6188**
- MCC: **0.6005**

Although kNN achieves higher accuracy and precision, its recall is considerably lower. Therefore, Random Forest provides a better balance for the objective of identifying potential term-deposit subscribers.

The low precision observed for Random Forest and the other class-balanced models is mainly associated with the highly imbalanced target distribution and the emphasis on detecting the minority positive class. This creates a precision-recall trade-off and is why accuracy alone is not used to select the final model.

## Live App

**Streamlit App:** https://2025ac05199-ml-assignment-bank-marketing-classification.streamlit.app/

The application provides an interactive interface for selecting a trained model, using the included sample `test_data.csv` or uploading test data, generating predictions, viewing probabilities and inspecting evaluation metrics, confusion matrices and classification reports.
