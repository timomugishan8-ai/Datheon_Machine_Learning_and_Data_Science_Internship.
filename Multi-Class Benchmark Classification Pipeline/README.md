# Multi-Class Benchmark Classification Pipeline

This project is a beginner-friendly classification workflow built around the classic Iris flower dataset. The work documented in the first two notebooks focuses on understanding the dataset, cleaning it, and performing exploratory data analysis to confirm which features are most informative for classifying the three species.

## Project goal

The main objective is to build a multi-class classification pipeline that can accurately distinguish between:

- Setosa
- Versicolor
- Virginica

The dataset is small but well-suited for learning core machine learning and data science concepts such as data inspection, EDA, feature relationships, and class separation.

## Dataset overview

The raw dataset is stored in `data/raw/iris.csv` and contains 150 records with 5 columns:

- sepal_length
- sepal_width
- petal_length
- petal_width
- species

The target variable is `species`, and the four numeric columns are the input features used for classification.

## Notebook 1: Data understanding

The first notebook was used to load the data and inspect its structure, quality, and general properties.

Key findings:

- The dataset contains 150 rows and 5 columns.
- All columns are present and there are no missing values.
- Feature dtypes are mostly numeric (`float64`), while the target variable is categorical (`species`).
- The dataset has a single duplicate row, which was removed during cleaning.
- After cleaning, the dataset is reduced to 149 unique observations.
- Species counts are:
  - Setosa: 50
  - Versicolor: 50
  - Virginica: 49

The cleaned dataset was saved to `data/processed/iris_cleaned.csv` for use in later analysis and modeling steps.

### Interpretation

This dataset is highly suitable for a beginner classification task because it is compact and contains a clear target variable. The class distribution is relatively balanced, which makes the dataset practical for evaluating classification models using metrics such as accuracy and confusion matrix.

## Notebook 2: Exploratory data analysis

The second notebook performs exploratory data analysis to understand how the features relate to one another and how well they separate the flower species.

### 1. Pairplot

A pairplot was created using the main numerical features grouped by species.

Findings:

- The classes overlap to some degree when examined individually, which suggests that no single feature alone is sufficient for perfect classification.
- The visual patterns indicate that multiple features together provide stronger separation between the species.
- Petal-related measurements appear to be especially informative for distinguishing the classes.

### 2. Correlation heatmap

A correlation heatmap was generated for the numerical variables.

Findings:

- Most features show strong positive correlations with each other.
- Petal length and petal width have particularly strong association.
- Sepal width appears to be weaker or less consistent in its relationship to the other variables.
- This implies that petal dimensions are likely the most discriminative features in the dataset.

### 3. Species distribution

A count plot of the target variable shows that the class distribution is fairly balanced.

Findings:

- Setosa: 50
- Versicolor: 50
- Virginica: 49

Because the dataset is nearly balanced, accuracy is a reasonable initial evaluation metric for model performance.

### 4. Boxplots of feature distributions

Boxplots were used to compare the distributions of the four numeric features across species.

Findings:

- Setosa has noticeably smaller petal dimensions and is well separated from the other two classes.
- Versicolor sits in the middle range of measurements.
- Virginica tends to have the largest petal and sepal dimensions.
- The petal measurements are especially useful for separating the species, with setosa standing out most clearly.

### Interpretation of the EDA

The exploratory analysis confirms that this is a clean and well-structured multiclass classification problem. Petal length and petal width are the strongest indicators for class separation, while sepal measurements provide additional context. These findings support moving into model building and evaluation in later notebooks.

## Repository structure

```text
Multi-Class Benchmark Classification Pipeline/
├── data/
│   ├── raw/iris.csv
│   └── processed/iris_cleaned.csv
├── Notebooks/
│   ├── 1_data understanding.ipynb
│   ├── 2_eda.ipynb
│   └── 3_modeling.ipynb
├── reports/
│   └── figures/
├── src/
│   ├── data preprocessiong.py
│   ├── features.py
│   └── modeling.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Baseline model results

The baseline classification model was evaluated on the held-out test set using the standard multiclass classification metrics.

Classification report:

- Setosa: precision = 1.00, recall = 1.00, f1-score = 1.00
- Versicolor: precision = 0.90, recall = 0.90, f1-score = 0.90
- Virginica: precision = 0.90, recall = 0.90, f1-score = 0.90
- Accuracy: 0.93
- Macro average: 0.93
- Weighted average: 0.93

The confusion matrix showed that:

- all 10 setosa samples were correctly classified,
- 9 of 10 versicolor samples were correctly classified, with 1 sample misclassified as virginica,
- 9 of 10 virginica samples were correctly classified, with 1 sample misclassified as versicolor.

This means the baseline model achieved a strong overall performance of 93% accuracy. It perfectly separated setosa from the other two species, while the only confusion occurred between versicolor and virginica, which is expected because these classes are visually and morphologically similar.

## Setup

To reproduce the analysis:

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open and run the notebooks in order:
   - `1_data understanding.ipynb`
   - `2_eda.ipynb`
   - `3_modeling.ipynb`

## Summary

The first two notebooks establish a strong foundation for the classification task:

- the dataset is clean and well-formed,
- the target variable is balanced,
- key feature relationships are understood,
- and the petal measurements are identified as the most important discriminators.

This makes the project a suitable benchmark for demonstrating a standard supervised machine learning workflow in a multiclass setting.
