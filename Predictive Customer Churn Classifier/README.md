# Predictive Customer Churn Classifier

## Project overview

This project develops and evaluates ensemble classifiers to identify telecommunications customers at risk of churn using the IBM Telco Customer Churn dataset. It covers exploratory data analysis, preprocessing, imbalanced classification, model development, cross-validation, held-out evaluation, and feature-importance analysis.

The dataset contains **7,043 customer records** and **19 predictors** after removing `customerID`. The target, `Churn`, is encoded as 1 (churn) and 0 (no churn). Churn represents approximately 26.5% of the records.

## Objective

Build a reproducible classification workflow using **Random Forest** and **XGBoost**, handle class imbalance with **SMOTE**, and assess model performance using accuracy, precision, recall, F1-score, and ROC-AUC. The project target was ROC-AUC greater than 80%.

## Workflow

1. **Exploratory analysis:** inspect churn distribution and examine churn patterns across contract type, tenure, services, payment method, and charges.
2. **Preprocessing:** remove the customer identifier, convert `TotalCharges` to numeric, and retain missing values for pipeline-based imputation.
3. **Feature preparation:** median-impute numerical features; most-frequent-impute and one-hot encode categorical features.
4. **Imbalanced learning:** apply ordinary SMOTE inside each imbalanced-learn pipeline, after preprocessing and only to training folds/data. The held-out test set is never resampled.
5. **Model development:** train Random Forest and XGBoost pipelines and compare them with five-fold StratifiedKFold cross-validation.
6. **Evaluation:** assess both fitted models on the same stratified 20% hold-out test set and inspect confusion matrices, ROC curves, and feature importances.

**SMOTE implementation note:** standard SMOTE interpolates the one-hot encoded feature space. Synthetic observations can therefore contain fractional values in encoded categorical columns. This is a limitation to consider when interpreting the models.

## Results

### Five-fold stratified cross-validation

Mean scores across the five validation folds:

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| Accuracy | 0.7826 | **0.8000** |
| Precision | 0.6118 | **0.6409** |
| Recall | 0.4977 | **0.5632** |
| F1-score | 0.5485 | **0.5989** |
| ROC-AUC | 0.8167 | **0.8442** |

### Held-out test set

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| Accuracy | 77.71% | **79.84%** |
| Precision | 59.74% | **63.72%** |
| Recall | 49.20% | **55.88%** |
| F1-score | 53.96% | **59.54%** |
| ROC-AUC | 82.02% | **84.40%** |

XGBoost outperformed Random Forest on all five reported metrics in both cross-validation and the held-out test evaluation. It correctly identified **209 of 374 churners**, while missing 165 and generating 119 false churn alerts. Its ROC-AUC of **0.8440** exceeded the project target of 0.80. However, its recall of 55.88% shows that the default classification threshold still misses a substantial share of customers who churn.

### Most influential XGBoost features

| Feature | Importance |
|---|---:|
| Month-to-month contract | 27.71% |
| No online security | 10.97% |
| No tech support | 7.76% |
| Electronic-check payment | 6.66% |
| Fiber-optic internet | 4.90% |

Feature importance describes how the fitted model uses predictors; it does not establish causation or, by itself, show the direction of an association.

## Repository structure

```text
Predictive Customer Churn Classifier/
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv
│   └── processed/
│       └── cleaned_telco_churn.csv
├── models/
│   ├── random_forest_churn.pkl
│   └── xgboost_churn.pkl
├── notebooks/
│   ├── 01_customer_churn_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_model_evaluation.ipynb
├── reports/
│   └── figures/
├── scripts/
│   └── export_evaluation_figures.py
├── README.md
└── requirements.txt
```

## Setup and execution

Run commands from this project directory.

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter lab
```

Run the notebooks in order:

1. `01_customer_churn_eda.ipynb`
2. `02_data_preprocessing.ipynb`
3. `03_model_development.ipynb`
4. `04_model_evaluation.ipynb`

The preprocessing notebook exports the cleaned dataset. The model-development notebook trains and saves the two pipelines. The evaluation notebook evaluates the saved models on the reproducible hold-out split.

### Regenerate report figures

After the processed dataset and saved models exist, run:

```bash
python scripts/export_evaluation_figures.py
```

This script independently regenerates the four report figures from the saved models and hold-out data. It saves each distinct figure to `reports/figures/` and closes the figure after saving.

## Dependencies

All project dependencies are listed in [requirements.txt](requirements.txt).

## Interpretation and next steps

XGBoost is the stronger of the two evaluated models on this split, but model selection for a real retention campaign should account for the relative costs of missed churners and unnecessary outreach. A threshold-tuning exercise, calibration assessment, and business-cost analysis would be sensible next steps. EDA patterns and feature importance should guide further investigation, not be treated as causal explanations.
