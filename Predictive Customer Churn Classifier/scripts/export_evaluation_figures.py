"""Regenerate the customer-churn evaluation figures from saved model pipelines.

Run from the project root:
    python scripts/export_evaluation_figures.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_telco_churn.csv"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")

# Recreate the exact stratified hold-out split used during model development.
data = pd.read_csv(DATA_PATH)
X = data.drop(columns=["Churn"])
y = data["Churn"].astype(int)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

models = {
    "Random Forest": joblib.load(MODEL_DIR / "random_forest_churn.pkl"),
    "XGBoost": joblib.load(MODEL_DIR / "xgboost_churn.pkl"),
}
predictions = {name: model.predict(X_test) for name, model in models.items()}
probabilities = {name: model.predict_proba(X_test)[:, 1] for name, model in models.items()}

# 1. Side-by-side confusion matrices.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, pred) in zip(axes, predictions.items()):
    matrix = confusion_matrix(y_test, pred)
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["No churn", "Churn"],
        yticklabels=["No churn", "Churn"],
    )
    ax.set_title(f"{name} — Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "confusion_matrix_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 2. ROC curves using the actual held-out prediction probabilities.
fig, ax = plt.subplots(figsize=(8, 6))
for name, score in probabilities.items():
    RocCurveDisplay.from_predictions(y_test, score, name=name, ax=ax)
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
ax.set_title("ROC Curve Comparison")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "roc_curve_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 3–4. Model-specific top-10 feature importance plots.
for name, model in models.items():
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    importances = estimator.feature_importances_

    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(10)
        .sort_values("Importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.set_title(f"Top 10 Feature Importances — {name}")
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    filename = (
        "top_10_feature_importance_random_forest.png"
        if name == "Random Forest"
        else "top_10_feature_importance_xgboost.png"
    )
    fig.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

print(f"Regenerated four evaluation figures in: {FIGURE_DIR}")
