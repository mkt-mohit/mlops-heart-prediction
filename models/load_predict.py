import mlflow
import mlflow.sklearn
import pandas as pd

# --------------------------------------------------
# Correct experiment name (MUST match save script)
# --------------------------------------------------
EXPERIMENT_NAME = "Heart Disease Final Model"

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found")

# --------------------------------------------------
# Get latest run
# --------------------------------------------------
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)

latest_run_id = runs.iloc[0].run_id

# --------------------------------------------------
# Load model from latest run
# --------------------------------------------------
model = mlflow.sklearn.load_model(
    f"runs:/{latest_run_id}/model"
)

# --------------------------------------------------
# Sample inference
# --------------------------------------------------
sample = pd.DataFrame([{
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 1,
    "ca": 0,
    "thal": 2
}])

print("Prediction:", model.predict(sample))
