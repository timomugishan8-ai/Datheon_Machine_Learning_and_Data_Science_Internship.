# California Housing Price Prediction

## Project overview

This project analyzes California housing data and develops regression models to estimate `median_house_value` for a census block group. The workflow progresses from data inspection and feature preparation through multicollinearity analysis, model development, held-out evaluation, and error interpretation. The current modeling notebook compares full and reduced predictor sets using OLS, Ridge, and Lasso regression.

## Problem statement

Can census block-group characteristics such as income, housing stock, population, location, and ocean proximity be used to estimate the area's median house value? This is a supervised regression task: the target is continuous, and each prediction represents an area-level median, not the price of an individual property.

## Dataset description and source

The local dataset, [`data/raw/housing.csv`](data/raw/housing.csv), contains **20,640 rows and 10 columns**. Each row describes a California census block group. The target is `median_house_value`; predictors describe location, housing, population, income, and ocean proximity.

| Column(s) | Description |
|---|---|
| `longitude`, `latitude` | Geographic coordinates of the block group |
| `housing_median_age` | Median age of housing in the block group |
| `total_rooms`, `total_bedrooms` | Counts of rooms and bedrooms in the area |
| `population`, `households` | Population and household counts in the area |
| `median_income` | Median household income in the block group |
| `ocean_proximity` | Categorical description of the block group's proximity to the ocean |
| `median_house_value` | Median home value in the block group; prediction target |

`median_income` and `median_house_value` are area-level medians, not values for a particular household or home. The dataset's target ranges from **$14,999 to $500,001**; the upper end is capped, which limits distinctions among the most expensive areas.

The data is the commonly used California Housing Prices dataset format, based on the 1990 U.S. Census. The canonical California housing data was distributed through [StatLib](https://lib.stat.cmu.edu/datasets/houses.zip) and is described in the [scikit-learn California Housing documentation](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset). This repository contains a 10-column CSV variant, including `ocean_proximity`; its exact download or repackaging provenance is not recorded in the project files.

## Objectives

- Inspect the data, identify missing values and duplicates, and summarize feature and target distributions.
- Explore numeric relationships with the target and diagnose predictor multicollinearity.
- Engineer household-level ratios and an income-by-rooms interaction.
- Compare full and reduced feature configurations to understand the trade-off between collinearity and predictive performance.
- Train a mean-prediction baseline and OLS, Ridge, and Lasso regressors.
- Evaluate test-set performance and inspect prediction and residual patterns.

## Technologies and libraries

- Python and Jupyter Notebook for analysis and documentation.
- NumPy and pandas for numerical computing and tabular data manipulation.
- Matplotlib and Seaborn for plots.
- scikit-learn for preprocessing, pipelines, train/test splitting, cross-validation, regression models, and metrics.
- statsmodels for variance inflation factor (VIF) calculations.

The notebooks use `OneHotEncoder(sparse_output=False)`, which requires scikit-learn 1.2 or newer. The repository does not currently include a dependency lockfile or `requirements.txt`.

## Project structure

```text
.
├── data/
│   ├── raw/housing.csv
│   ├── processed/house_cleaned.csv
│   └── cleaned/
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preparation.ipynb
│   └── 03_regression_modeling.ipynb
├── reports/figures/
└── README.md
```

`01_data_understanding.ipynb` inspects and cleans the raw data, including writing `data/processed/house_cleaned.csv`. `02_data_preparation.ipynb` reads that processed file for train/test splitting, feature engineering, and multicollinearity analysis. `03_regression_modeling.ipynb` reads the raw CSV and builds its own train/test and model-preprocessing workflow. Figures are displayed in the notebooks; `reports/figures/` is available for saved report figures.

## Data preprocessing and feature engineering

Initial data inspection found **207 missing `total_bedrooms` values** (about 1% of rows) and no duplicate rows. The data-understanding notebook fills missing bedroom counts with the median and saves the processed dataset. In the regression notebook, preprocessing is kept in scikit-learn pipelines: numeric values are median-imputed and standardized; categorical values are imputed with the most frequent category and one-hot encoded with unseen categories ignored. This lets imputation, scaling, and encoding parameters be learned from training data during fitting and cross-validation.

The notebooks create four predictors:

- `rooms_per_household` = `total_rooms / households`
- `bedrooms_per_room` = `total_bedrooms / total_rooms`
- `population_per_household` = `population / households`
- `income_x_rooms_per_household` = `median_income * rooms_per_household`

Notebook 3 uses an 80/20 split with `random_state=42`. It compares a **full** feature set containing the predictors and engineered features with a **reduced** set that removes `total_rooms`, `total_bedrooms`, and `households` while retaining the other predictors. The reduced set is an experiment to assess the effect of removing correlated raw counts, not a claim that those variables are universally unnecessary.

Notebook 2's correlation and VIF analysis found substantial overlap among area-size counts: for example, `total_bedrooms` and `households` had |r| about 0.98. In the initial feature-engineered set, VIFs were 49.94 for `total_bedrooms`, 47.03 for `households`, and 18.82 for `total_rooms`. After these raw counts were removed for the reduced-set analysis, VIF decreased for most remaining predictors, but remained elevated for `income_x_rooms_per_household` (17.23) and `median_income` (13.15). VIF is a diagnostic of predictor redundancy and coefficient stability; it does not measure predictive accuracy.

## Models developed

The notebook fits and evaluates:

- **Mean baseline:** `DummyRegressor(strategy="mean")`, used as a simple reference prediction.
- **OLS:** ordinary least-squares linear regression.
- **Ridge:** L2-regularized linear regression.
- **Lasso:** L1-regularized linear regression, which can shrink coefficients to zero.

OLS, Ridge, and Lasso are trained on both the full and reduced feature sets, producing six configurations. Ridge and Lasso are tuned separately for each configuration using five-fold `GridSearchCV` on the training data. The search evaluates 13 `alpha` values from 0.001 to 1,000 using RMSE. The preprocessing is included in each pipeline so transformations are refit within each cross-validation fold.

## Evaluation metrics and results

The notebook evaluates predictions on the same held-out 20% test observations:

- **RMSE:** errors in dollars, with larger errors penalized more; lower is better.
- **MAE:** mean absolute error in dollars; lower is better.
- **R²:** variance explained relative to predicting the test-set mean; closer to 1 is better, while a negative value is worse than that baseline.

| Feature set | Model | RMSE | MAE | R² |
|---|---|---:|---:|---:|
| Full | OLS | $68,917 | $49,710 | 0.6375 |
| Full | Lasso | $68,922 | $49,712 | 0.6375 |
| Full | Ridge | $68,926 | $49,716 | 0.6375 |
| Reduced | OLS | $71,649 | $51,567 | 0.6082 |
| Reduced | Lasso | $71,650 | $51,569 | 0.6082 |
| Reduced | Ridge | $71,656 | $51,574 | 0.6082 |

The full-feature OLS model has the lowest test RMSE (**$68,917**) and is selected as the best configuration in the notebook. The full-feature Ridge and Lasso models are effectively tied with it; their score differences are only a few dollars. Removing the three raw counts makes the scores worse on this split: reduced-set RMSE is about $2,700 higher and R² falls from 0.6375 to 0.6082. In an earlier comparison, the mean baseline scored RMSE **$114,486**, MAE **$90,607**, and R² **-0.0002**, substantially worse than the trained regression models. The reported MAE for the full models is about **$49,700**, so individual estimates can still have sizeable errors.

## Key findings

- `median_income` had the strongest positive numeric correlation with `median_house_value` in the exploratory analysis (r = 0.688); this is association, not causation.
- Raw housing-count predictors are highly collinear. Removing selected counts lowered much of the remaining VIF, but did not improve test performance in this comparison.
- The full feature set performed better than the reduced set. OLS was numerically first, but Ridge and Lasso were practically indistinguishable by these test metrics.
- The actual-versus-predicted plot shows the overall upward relationship but also substantial scatter around perfect predictions, with predictions tending toward the center of the target range. The target cap near $500,000 limits interpretation at the high end.
- The residual histogram is concentrated near zero but has long tails and a longer positive tail. Since residuals are actual minus predicted, positive residuals correspond to underpredictions. A histogram does not establish whether error variance changes with fitted values.
- The Lasso coefficient inspection found 1 of 17 transformed coefficients equal to zero in the reported fit, indicating limited sparsity for this setup. Correlated predictors also make individual coefficients less stable and they should not be interpreted causally.

OLS is the lowest-RMSE model in the current comparison, but the small gap does not establish a meaningful advantage. The notebook selects the best model using test RMSE, so the test set has influenced model selection; treat the reported scores as comparative rather than as a fully independent final estimate. For a stronger final assessment, choose the model and feature configuration with cross-validation or a validation set on training data, then evaluate once on an untouched test set. Further work could examine residuals against fitted values, investigate errors in high-value areas, and test nonlinear models using the same validation protocol.

## How to run the notebook

From the project root, create or activate a Python environment, then install the notebook dependencies:

```bash
python -m pip install "scikit-learn>=1.2" numpy pandas matplotlib seaborn statsmodels jupyter
```

Start Jupyter from the project root:

```bash
jupyter notebook
```

Open the notebooks under `notebooks/` and run cells from top to bottom. Run them in this order:

1. `01_data_understanding.ipynb` reads `data/raw/housing.csv` and creates `data/processed/house_cleaned.csv`.
2. `02_data_preparation.ipynb` reads the processed CSV and performs its split, feature engineering, and VIF analysis.
3. `03_regression_modeling.ipynb` reads the raw CSV, trains the full and reduced model configurations, tunes Ridge and Lasso, and evaluates the models.

The notebooks use paths relative to their location, so keep the repository's `data/` and `notebooks/` directories in their current relationship. The raw CSV and processed CSV should be present before running the notebooks that consume them.
