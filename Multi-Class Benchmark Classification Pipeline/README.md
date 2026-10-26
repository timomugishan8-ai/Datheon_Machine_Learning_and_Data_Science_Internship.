# Multi-Class Benchmark Classification Pipeline

This project is a beginner-friendly classification workflow built around the classic Iris flower dataset. The notebooks cover understanding and cleaning the dataset, exploratory data analysis, and training and evaluating classification models.

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

## Notebook 3: Model training and evaluation

The third notebook trains and evaluates Logistic Regression and Decision Tree classifiers using a held-out test set of 30 observations. Logistic Regression is trained with standardized features, while the Decision Tree uses the original feature scales.

### Model comparison

Both models achieved the same scores on this test split:

| Model | Accuracy | Macro precision | Macro recall | Macro F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.933 | 0.933 | 0.933 | 0.933 |
| Decision Tree | 0.933 | 0.933 | 0.933 | 0.933 |

Each model correctly classified 28 of the 30 test samples. The notebook uses side-by-side confusion matrices to compare class-specific predictions and errors.

### Additional model visualizations

- **Decision Tree feature importance:** Petal length (about 0.56) and petal width (about 0.40) account for most of the tree's feature importance; the sepal features contribute much less.
- **Multiclass ROC curves:** Both models show excellent discrimination for setosa (AUC 1.00). Logistic Regression has higher AUC for versicolor and virginica (0.99 each) than the Decision Tree (0.92 each) on this test split.
- **Two-feature decision boundaries:** Plots based on petal length and petal width show clear separation for setosa and more overlap between versicolor and virginica. Logistic Regression produces smoother boundaries, while the Decision Tree creates step-like regions. These plots use models fitted on only the two petal features, for visualization.

### Interpretation

The two classifiers tie on accuracy and the reported macro-averaged metrics, while the ROC curves suggest Logistic Regression ranks versicolor and virginica more effectively on this split. The visualizations also support the EDA finding that petal measurements are especially informative. Since these results come from a single, small test set, cross-validation or evaluation on additional data is recommended before drawing conclusions about which model generalizes best.

## Logistic Regression baseline results

The Logistic Regression baseline was evaluated on the held-out test set using the standard multiclass classification metrics.

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

## Decision Tree model results

The Decision Tree was trained on the unscaled input features, since tree-based models do not require feature standardization. Its classification report on the same 30-sample test set was:

- Setosa: precision = 1.00, recall = 1.00, f1-score = 1.00 (10 samples)
- Versicolor: precision = 0.90, recall = 0.90, f1-score = 0.90 (10 samples)
- Virginica: precision = 0.90, recall = 0.90, f1-score = 0.90 (10 samples)
- Accuracy: 0.93 (28 of 30 samples)
- Macro average: precision = 0.93, recall = 0.93, f1-score = 0.93
- Weighted average: precision = 0.93, recall = 0.93, f1-score = 0.93

The Decision Tree also classified all 10 setosa samples correctly and 9 of 10 samples for each of versicolor and virginica. Its class-level report and overall scores match the Logistic Regression baseline on this split. The feature-importance plot shows that the tree relies mainly on petal length and petal width. Together, these results demonstrate comparable test-set classification performance, though the models differ in how they form decision boundaries and rank class probabilities.

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

The three notebooks establish a complete foundation and baseline modeling workflow for the classification task:

- the dataset is clean and well-formed,
- the target variable is balanced,
- key feature relationships and class distributions are explored,
- petal measurements are identified as important discriminators,
- and Logistic Regression and Decision Tree models are compared using classification metrics and visualizations.

Both models achieve 93.3% accuracy on the current test split. Further validation is needed to assess how reliably these results generalize.
