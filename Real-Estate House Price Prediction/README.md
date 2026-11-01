# California Housing Price Prediction

## Project overview

This project explores the California Housing dataset and develops regression models to predict **median house value**. The work includes data inspection, missing-value treatment, exploratory analysis, feature engineering, multicollinearity assessment, and comparison of Ordinary Least Squares (OLS), Ridge, and Lasso regression.

The project evaluates models using:

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
- `data/cleaned/` — directory reserved for cleaned data outputs.
- `notebooks/01_data_understanding.ipynb` — data inspection, cleaning, exploratory analysis, and interpretation.
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

## Modeling approach

The project will use `median_house_value` as the target and compare OLS, Ridge, and Lasso regression. Feature engineering will include interaction features, and model evaluation will use RMSE, MAE, and R².

For an unbiased model evaluation, fit preprocessing steps—including missing-value imputation—on the training data only, then apply the fitted transformations to validation or test data. Report final predictive performance using held-out data; exploratory analysis and correlation values are not model performance scores.

## Notebook

Open [`notebooks/01_data_understanding.ipynb`](notebooks/01_data_understanding.ipynb) to review the dataset inspection, missing-value treatment, descriptive statistics, distribution plots, correlation analysis, and overall interpretation.
