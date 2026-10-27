# Multi-Class Benchmark Classification Pipeline
This project develops and evaluates a multiclass machine-learning workflow using the Iris flower dataset. It covers data inspection and cleaning, exploratory data analysis, and a comparison of Logistic Regression and Decision Tree classifiers.

## Objective

Predict the Iris species (`setosa`, `versicolor`, or `virginica`) from four measurements: sepal length, sepal width, petal length, and petal width.

## Dataset

The raw dataset is provided in `data/raw/iris.csv` and contains 150 observations. Initial inspection found no missing values and one duplicate record. After removing the duplicate, the cleaned dataset contains 149 observations and is saved at `data/processed/iris_cleaned.csv`.

| Species | Observations after cleaning |
|---|---:|
| Setosa | 50 |
| Versicolor | 50 |
| Virginica | 49 |

## Workflow

The analysis is organized into three notebooks, intended to be run in order:

1. **Data understanding** (`Notebooks/1_data understanding.ipynb`): inspect the source data, assess data quality, remove the duplicate, and save the cleaned dataset.
2. **Exploratory analysis** (`Notebooks/2_eda.ipynb`): examine feature distributions, class balance, and relationships between measurements and species.
3. **Modeling** (`Notebooks/3_modeling.ipynb`): train and compare Logistic Regression and Decision Tree classifiers.

The modeling notebook uses a stratified 80/20 train-test split with `random_state=42`. Logistic Regression is trained on standardized features; the Decision Tree uses unscaled features. Both models are evaluated on the same held-out test set using accuracy, precision, recall, F1-score, confusion matrices, and ROC analysis.

## Results

Both classifiers correctly classified 28 of the 30 test observations. Their scores on this split were identical:

| Model | Accuracy | Macro precision | Macro recall | Macro F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.933 | 0.933 | 0.933 | 0.933 |
| Decision Tree | 0.933 | 0.933 | 0.933 | 0.933 |

Both models classified every setosa sample correctly. Each misclassified one versicolor sample as virginica and one virginica sample as versicolor. The exploratory analysis and tree feature-importance results identify petal measurements as particularly useful for separating the species. On this split, the ROC analysis also favors Logistic Regression for distinguishing versicolor and virginica, despite the equal accuracy scores.

These results describe a single split of a small dataset and should not be treated as evidence of generalization performance. Cross-validation and evaluation on additional data would provide a more robust comparison.

## Visualizations

The modeling notebook generates the following figures, saved under `reports/figures/`:

- [Model evaluation scores](reports/figures/Evaluation%20scores%20of%20the%20models.png)
- [Confusion matrix comparison](reports/figures/Confusion%20Matrix%20Comparison%20of%20the%20Models.png)
- [ROC curves](reports/figures/ROC%20curves%20model%20comparison.png)
- [Decision boundaries](reports/figures/Decision%20Boundaries%20of%20the%20models.png)

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
├── reports/figures/
├── README.md
└── requirements.txt
```

## Reproduce the analysis

From this project directory, create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
```

Activate the environment and install the requirements:

```bash
python -m pip install -r requirements.txt
```

Launch Jupyter and run the notebooks in the order listed above:

```bash
jupyter notebook
```

The notebooks read and write data using paths relative to their location, so run them from the project structure provided.

## Dependencies

The required Python packages are listed in `requirements.txt`: pandas, NumPy, scikit-learn, Matplotlib, Seaborn, and Jupyter.

## Conclusion

This project demonstrates a complete, reproducible workflow for multiclass classification, from data cleaning and exploratory analysis through model evaluation. Logistic Regression and the Decision Tree achieved the same performance on the held-out test set, with petal measurements providing strong class-separation signals. Because the evaluation uses a single split of a small dataset, cross-validation or additional data is needed before drawing broader conclusions about model performance.


