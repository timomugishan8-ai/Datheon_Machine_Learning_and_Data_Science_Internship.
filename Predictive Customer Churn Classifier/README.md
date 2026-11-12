# Predictive Customer Churn Classifier

## Executive Summary

This project analyzes customer churn in a telecommunications dataset and compares Random Forest and XGBoost classifiers. The work includes exploratory data analysis, preprocessing, stratified cross-validation, final model training, and evaluation on a held-out test set.

XGBoost performed better than Random Forest across all reported cross-validation and held-out test metrics. On the test set, XGBoost achieved 0.7984 accuracy, 0.6372 precision, 0.5588 recall, 0.5954 F1-score, and 0.8440 ROC-AUC. Recall remains moderate: at the default classification threshold, XGBoost missed 165 of 374 customers who churned. The results support XGBoost as the stronger of the two models evaluated, while emphasizing that churn detection still involves a missed-customer versus false-alert trade-off.

## Project Scope and Data

The source file, `data/raw/Telco-Customer-Churn.csv`, contains 7,043 customer records and 21 columns. The target is `Churn`, encoded as `1` for churn and `0` for no churn; approximately 26.5% of records represent churners.

`TotalCharges` is initially stored as text. Converting it to numeric reveals 11 blank values associated with zero-tenure customers. The EDA analysis fills these values using `MonthlyCharges * tenure`; the modeling pipeline imputes missing numerical values using the training-fold median.

## Methodology

### Exploratory analysis

The EDA notebook examines churn distribution and compares churn across contract and tenure groups, charges, internet services, payment methods, service counts, and customer characteristics. It uses visual summaries and unadjusted statistical comparisons, including chi-square tests and Welch's t-tests.

The main observed patterns are:

- Month-to-month customers, especially those in their first year, have the highest observed churn.
- Churn is higher among fiber-optic customers and internet customers without online security; lack of tech support is also prominent in the model feature-importance results.
- Electronic-check customers have higher observed churn than customers using other payment methods.
- Churners have shorter tenure on average; among month-to-month customers, churners also have higher average monthly charges.
- Senior customers have higher observed churn than non-senior customers within each contract category.

These are associations in this dataset, not evidence that any feature causes churn. Group comparisons are not adjusted for all other customer characteristics. The service-count measure also counts phone service and multiple lines separately, so it is not a count of distinct service categories.

### Preprocessing and modeling

The preprocessing workflow removes `customerID` from the predictors, converts `TotalCharges` to numeric, maps the target to binary values, and creates a cleaned dataset at `data/processed/cleaned_telco_churn.csv`. Numerical features are median-imputed; categorical features are imputed with their most frequent value and one-hot encoded with unknown categories ignored. No scaling is applied because both estimators are tree-based.

Model development uses a stratified 80/20 train-test split. Within each imbalanced-learn pipeline, preprocessing and SMOTE are fitted during training, so cross-validation resampling is restricted to each training fold. Both models are compared using five-fold stratified cross-validation on the training partition with accuracy, precision, recall, F1-score, and ROC-AUC. The pipelines are then fitted on the full training partition and saved for evaluation.

Standard SMOTE is applied after one-hot encoding. Because it interpolates encoded feature vectors, generated samples may not correspond to valid categorical combinations; this should be considered when interpreting model behavior.

### Held-out evaluation

The evaluation notebook reloads the saved pipelines and recreates the same stratified 80/20 split using the fixed random seed. It reports test metrics, classification reports, confusion matrices, ROC curves, and model-specific feature importances. The held-out test set contains 1,409 customers: 1,035 who stayed and 374 who churned.

## Results

### Five-fold cross-validation

Values are mean (standard deviation) across the five validation folds.

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| Accuracy | 0.7826 (0.0116) | **0.8000 (0.0094)** |
| Precision | 0.6118 (0.0309) | **0.6409 (0.0265)** |
| Recall | 0.4977 (0.0209) | **0.5632 (0.0266)** |
| F1-score | 0.5485 (0.0213) | **0.5989 (0.0183)** |
| ROC-AUC | 0.8167 (0.0134) | **0.8442 (0.0095)** |

### Held-out test set

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| Accuracy | 0.7771 | **0.7984** |
| Precision | 0.5974 | **0.6372** |
| Recall | 0.4920 | **0.5588** |
| F1-score | 0.5396 | **0.5954** |
| ROC-AUC | 0.8202 | **0.8440** |

XGBoost correctly identified 209 churners and missed 165; it also produced 119 false churn alerts. Random Forest identified 184 churners, missed 190, and produced 124 false alerts. XGBoost therefore improved churn detection while slightly reducing false alerts on this test split. Its ROC-AUC advantage indicates better ranking of churn risk across thresholds, but the default threshold still leaves a substantial share of actual churners undetected.

### Feature importance

XGBoost ranked month-to-month contract status as its most important feature (0.277), followed by no online security (0.110), no tech support (0.078), electronic-check payment (0.067), and fiber-optic service (0.049). Random Forest's leading features were `TotalCharges` (0.098), `tenure` (0.084), month-to-month contract status (0.080), and `MonthlyCharges` (0.079). These importance scores describe how the fitted models use features; they do not indicate causal effects or, by themselves, the direction of a feature's relationship with churn.

## Conclusion

The project completed the data exploration, preprocessing, model development, cross-validation, training, and held-out evaluation workflow. XGBoost was the stronger model among the two evaluated, outperforming Random Forest on every reported metric in both cross-validation and the held-out test results. Its test F1-score of 0.5954 and ROC-AUC of 0.8440 indicate useful predictive signal, but its recall of 0.5588 shows that many churners are still missed at the default threshold.

For churn-retention use, model selection should weigh the cost of missed churners against the cost and capacity of contacting customers who would stay. Feature-importance and EDA findings can help prioritize further investigation, but they should not be treated as causal explanations or used alone to make customer-level decisions.

## Repository Structure

```text
.
|-- data/
|   |-- raw/Telco-Customer-Churn.csv
|   `-- processed/cleaned_telco_churn.csv
|-- models/
|   |-- random_forest_churn.pkl
|   `-- xgboost_churn.pkl
|-- notebooks/
|   |-- 01_customer_churcn_eda.ipynb
|   |-- 02_data_preprocessing.ipynb
|   |-- 03_model_development.ipynb
|   `-- 04_model_evaluation.ipynb
|-- reports/figures/
|-- src/
|-- README.md
`-- requirements.txt
```

## Running the Notebooks

The notebooks use relative paths based on the `notebooks/` directory. Run them in sequence: EDA, preprocessing, model development, then evaluation. Preprocessing creates the processed CSV; model development creates the saved model pipelines used by evaluation.

Create and activate a virtual environment, then install the notebook dependencies:

```bash
python -m venv .venv
python -m pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn xgboost joblib jupyter
jupyter lab
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`; on macOS or Linux, use `source .venv/bin/activate`. The repository's `requirements.txt` is currently empty, so the dependencies are listed here explicitly.

## Notebook Reference

- [01_customer_churcn_eda.ipynb](notebooks/01_customer_churcn_eda.ipynb): data inspection and exploratory analysis.
- [02_data_preprocessing.ipynb](notebooks/02_data_preprocessing.ipynb): cleaning, feature preparation, and processed-data export.
- [03_model_development.ipynb](notebooks/03_model_development.ipynb): pipeline construction, cross-validation, model training, and model export.
- [04_model_evaluation.ipynb](notebooks/04_model_evaluation.ipynb): held-out performance comparison and feature-importance analysis.
