# California Housing Price Prediction

## Project overview

This project explores the California Housing dataset and prepares a regression workflow to predict **median house value**. The notebooks cover data inspection, missing-value treatment, exploratory analysis, feature engineering, multicollinearity assessment, and construction of a leakage-conscious preprocessing pipeline. Model fitting and held-out performance comparison are the next steps; the current regression notebook does not yet report trained-model results.

The planned model evaluation uses:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R²

## Dataset and unit of observation

The dataset contains **20,640 rows and 10 columns**. Each row represents a **California census block group**, not an individual home or household. Its features describe or summarize housing, population, income, and location in that area.

This distinction explains the median-valued columns:

- `median_income` is the median household income in a block group.
- `median_house_value` is the median home value in a block group and is the project's prediction target.

These are area-level statistics; they are not the income or price of a specific household or home, and they are not one median calculated across the full dataset.

| Feature | Description |
|---|---|
| `longitude`, `latitude` | Geographic coordinates of the block group |
| `housing_median_age` | Median age of housing in the block group |
| `total_rooms`, `total_bedrooms` | Room and bedroom counts for the block group |
| `population`, `households` | Population and household counts for the block group |
| `median_income` | Median household income for the block group |
| `median_house_value` | Median home value for the block group; prediction target |
| `ocean_proximity` | Categorical description of the block group's proximity to the ocean |

## Repository contents

- `data/raw/housing.csv` — source data used in the analysis.
- `data/processed/house_cleaned.csv` — prepared dataset used for feature and multicollinearity analysis.
- `data/cleaned/` — directory reserved for cleaned data outputs.
- `notebooks/01_data_understanding.ipynb` — data inspection, cleaning, exploratory analysis, and interpretation.
- `notebooks/02_data_preparation.ipynb` — train/test split, multicollinearity diagnostics, and feature engineering.
- `notebooks/03_regression_modeling.ipynb` — data split and preprocessing pipeline setup for regression.
- `reports/figures/` — directory for project figures.

## Data understanding and preparation findings

The initial inspection confirmed 20,640 observations and 10 features. `total_bedrooms` was the only column with missing data: **207 values, approximately 1% of the rows**. The notebook retained the feature and filled those missing entries with the column median. The median is appropriate here because bedroom counts are skewed and the median is less affected by unusually large observations than the mean. A follow-up check confirmed that no `total_bedrooms` values remained missing.

No duplicate rows were found. The target, `median_house_value`, ranges from **$14,999 to $500,001**. Its mean is approximately **$206,856**, above its median of **$179,700**, indicating that high-value observations raise the average above the midpoint.

## Exploratory analysis and interpretation

The numeric correlation analysis found that `median_income` has the strongest positive linear correlation with `median_house_value` (**r = 0.688**). The remaining numeric features have much weaker individual correlations with the target:

| Numeric feature | Correlation with `median_house_value` |
|---|---:|
| `median_income` | 0.688 |
| `total_rooms` | 0.134 |
| `housing_median_age` | 0.106 |
| `households` | 0.066 |
| `total_bedrooms` | 0.049 |
| `population` | -0.025 |
| `longitude` | -0.046 |
| `latitude` | -0.144 |

These values describe linear association, not causation. They do not measure non-linear relationships or interactions, and the numeric correlation analysis does not include the categorical `ocean_proximity` feature. The weak individual correlations therefore do not, on their own, establish that a feature should be excluded from a model.

The count features `total_rooms`, `total_bedrooms`, and `households` describe related aspects of each area. Their overlap should be assessed when developing regression models, particularly when handling multicollinearity.

## Data preparation and feature analysis

Notebook 2 reads `data/processed/house_cleaned.csv`, separates `median_house_value` from the predictors, and creates an 80/20 train/test split with `random_state=42`. It performs its correlation and variance inflation factor (VIF) analysis on numeric training predictors. The test split is kept separate for later evaluation.

The initial diagnostics indicate substantial redundancy among area-size and location variables. The strongest reported pairwise relationships include total bedrooms with households (|r| about 0.98), total rooms with total bedrooms (about 0.93), longitude with latitude (about 0.92), total rooms with households (about 0.92), and population with households (about 0.91). Initial VIF values are highest for total bedrooms (36.31), households (35.82), and total rooms (12.82); latitude (8.80), longitude (8.68), and population (6.35) are also elevated. These findings suggest that individual linear-model coefficients for some predictors may be unstable or difficult to interpret.

Notebook 2 adds four row-level predictors to both train and test splits:

- `rooms_per_household` — total rooms divided by households.
- `bedrooms_per_room` — total bedrooms divided by total rooms.
- `population_per_household` — population divided by households.
- `income_x_rooms_per_household` — median income multiplied by rooms per household.

After adding these features, the original count-feature collinearity remains because the original predictors are retained. The notebook reports VIF values of 49.94 for total bedrooms, 47.03 for households, and 18.82 for total rooms. The income-by-rooms interaction has a VIF of 17.80, and median income's VIF increases from 1.74 to 13.69. Among the engineered features, rooms per household (4.50), bedrooms per room (2.89), and population per household (1.07) have lower VIFs. These are diagnostic results, not evidence that the features improve predictive accuracy; candidate feature sets should be judged by validation performance.

## Regression preprocessing workflow

Notebook 3 reads the raw `data/raw/housing.csv`, recreates the same four engineered features, and makes a separate 80/20 split with `random_state=42`. It identifies numeric and categorical predictors, then defines a scikit-learn `ColumnTransformer` containing:

- A numeric pipeline that imputes missing values with training-set medians and standardizes the features.
- A categorical pipeline that fills missing values with the most frequent category and one-hot encodes categories, ignoring categories not seen during fitting.

Keeping imputation, scaling, and encoding inside pipelines allows each transformation to be learned from training data when the pipeline is fitted, preventing test-set statistics from influencing preprocessing. Notebook 3 currently stops after constructing the preprocessor: it does not yet fit a regression estimator, run cross-validation or hyperparameter search, or calculate test-set RMSE, MAE, or R². Therefore, this README does not present predictive performance results. The next modeling step is to attach estimators (for example, a mean baseline and linear, Ridge, and Lasso models), fit and tune them using training data, and report final metrics on the held-out test set.

## Notebooks

- [`notebooks/01_data_understanding.ipynb`](notebooks/01_data_understanding.ipynb) — dataset inspection, missing-value treatment, descriptive statistics, distributions, and exploratory relationships.
- [`notebooks/02_data_preparation.ipynb`](notebooks/02_data_preparation.ipynb) — train/test split, correlation and VIF diagnostics, engineered features, and updated multicollinearity analysis.
- [`notebooks/03_regression_modeling.ipynb`](notebooks/03_regression_modeling.ipynb) — raw-data split and leakage-conscious numeric and categorical preprocessing pipelines; model training and evaluation remain to be completed.
