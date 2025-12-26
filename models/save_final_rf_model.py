import os

# CI-safe MLflow storage
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

# --------------------------------------------------
# Load cleaned data
# --------------------------------------------------
df = pd.read_csv("data/processed/heart_disease_clean.csv")
X = df.drop(columns="target")
y = df["target"]

# --------------------------------------------------
# Feature groups
# --------------------------------------------------
numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", "passthrough", categorical_features)
])

rf_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

param_grid = {
    "model__n_estimators": [200],
    "model__max_depth": [10],
    "model__min_samples_split": [2]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X, y)
final_model = grid.best_estimator_

# --------------------------------------------------
# MLflow logging (CI-safe)
# --------------------------------------------------
mlflow.set_experiment("Heart Disease Final Model")

with mlflow.start_run(run_name="RandomForest_Final"):
    mlflow.log_params(grid.best_params_)
    mlflow.sklearn.log_model(final_model, artifact_path="model")

# --------------------------------------------------
# Save pickle (local)
# --------------------------------------------------
os.makedirs("models/artifacts", exist_ok=True)
joblib.dump(final_model, "models/artifacts/random_forest_final.pkl")

print("CI training + MLflow logging completed successfully.")
