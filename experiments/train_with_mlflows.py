import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df = pd.read_csv("data/processed/heart_disease_clean.csv")
X = df.drop(columns="target")
y = df["target"]

# ------------------------------------------------------------
# Feature groups
# ------------------------------------------------------------
numerical_features = ["age","trestbps","chol","thalach","oldpeak","ca"]
categorical_features = ["sex","cp","fbs","restecg","exang","slope","thal"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", "passthrough", categorical_features)
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ------------------------------------------------------------
# MLflow setup
# ------------------------------------------------------------
mlflow.set_experiment("Heart Disease Classification")

# ============================================================
# Logistic Regression
# ============================================================
with mlflow.start_run(run_name="Logistic_Regression"):

    logreg = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000, solver="liblinear"))
    ])

    param_grid = {
        "model__C": [0.01, 0.1, 1, 10],
        "model__penalty": ["l1", "l2"]
    }

    grid = GridSearchCV(
        logreg, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    grid.fit(X, y)

    best_model = grid.best_estimator_

    scores = cross_validate(
        best_model, X, y, cv=cv,
        scoring=["accuracy","precision","recall","roc_auc"]
    )

    # Log parameters
    mlflow.log_params(grid.best_params_)

    # Log metrics
    mlflow.log_metrics({
        "accuracy": scores["test_accuracy"].mean(),
        "precision": scores["test_precision"].mean(),
        "recall": scores["test_recall"].mean(),
        "roc_auc": scores["test_roc_auc"].mean()
    })

    # Log model
    mlflow.sklearn.log_model(best_model, "logistic_regression_model")

# ============================================================
# Random Forest
# ============================================================
with mlflow.start_run(run_name="Random_Forest"):

    rf = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5]
    }

    grid = GridSearchCV(
        rf, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    grid.fit(X, y)

    best_model = grid.best_estimator_

    scores = cross_validate(
        best_model, X, y, cv=cv,
        scoring=["accuracy","precision","recall","roc_auc"]
    )

    mlflow.log_params(grid.best_params_)

    mlflow.log_metrics({
        "accuracy": scores["test_accuracy"].mean(),
        "precision": scores["test_precision"].mean(),
        "recall": scores["test_recall"].mean(),
        "roc_auc": scores["test_roc_auc"].mean()
    })

    mlflow.sklearn.log_model(best_model, "random_forest_model")
