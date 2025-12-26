import joblib
import pandas as pd
from fastapi import FastAPI
from app.schemas import HeartDiseaseInput, PredictionOutput

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0"
)

# Load trained pipeline
MODEL_PATH = "models/artifacts/random_forest_final.pkl"
model = joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Heart Disease Model is running"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: HeartDiseaseInput):

    input_df = pd.DataFrame(
        [[
            data.age,
            data.sex,
            data.cp,
            data.trestbps,
            data.chol,
            data.fbs,
            data.restecg,
            data.thalach,
            data.exang,
            data.oldpeak,
            data.slope,
            data.ca,
            data.thal
        ]],
        columns=[
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal"
        ]
    )

    # prediction
    prediction = int(model.predict(input_df)[0])

    # probability (SAFE)
    proba = model.predict_proba(input_df)
    confidence = float(proba[0, 1])   # probability of class "1"

    return PredictionOutput(
        prediction=prediction,
        confidence=round(confidence, 4)
    )
