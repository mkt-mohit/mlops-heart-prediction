❤️ Heart Disease Prediction — End-to-End MLOps Project
📌 Introduction

This project demonstrates a complete end-to-end MLOps pipeline for building, validating, deploying, and monitoring a machine learning model that predicts heart disease risk using the UCI Heart Disease dataset.

The project follows modern MLOps best practices, including:

Reproducible data preprocessing and model training

Experiment tracking

Automated CI/CD

Containerized model serving

Cloud-native production deployment

Centralized logging and monitoring

The final model is exposed as a REST API deployed on Google Cloud Run.

🎯 Problem Statement

Given patient health attributes (age, cholesterol, blood pressure, ECG results, etc.), predict whether the patient has a risk of heart disease.

Target variable

0 → No heart disease

1 → Heart disease present

🏗️ Architecture Overview
GitHub Repository
        │
        │  (git push)
        ▼
GitHub Actions (CI/CD)
 ├─ Linting (flake8)
 ├─ Unit tests (pytest)
 ├─ Model training & validation
 └─ Cloud Run deployment
        │
        ▼
Google Cloud Run
 ├─ FastAPI REST API
 ├─ Docker container
 ├─ Auto-scaling
 └─ Cloud Logging & Metrics

🧰 Tech Stack
Category	Tools
Language	Python 3.12
Machine Learning	scikit-learn
Experiment Tracking	MLflow
API Framework	FastAPI
Containerization	Docker
CI/CD	GitHub Actions
Cloud Platform	Google Cloud (Cloud Run)
Monitoring & Logging	Cloud Logging, Cloud Run Metrics
📂 Project Structure
mlops_heart_disease/
├── app/
│   ├── main.py               # FastAPI application
│   └── schemas.py            # Request/response schemas
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── artifacts/
│       └── random_forest_final.pkl
├── tests/
│   ├── test_data.py
│   └── test_model.py
├── .github/workflows/
│   └── ci.yaml               # CI/CD pipeline
├── Dockerfile
├── requirements.txt
├── README.md

🔄 MLOps Workflow
Step-1: Data Acquisition & EDA

Dataset downloaded from UCI repository

Missing values handled

Target converted to binary

EDA performed

Clean dataset saved for reuse

Step-2: Feature Engineering & Model Training

Feature scaling and encoding

Logistic Regression and Random Forest trained

Cross-validation used for model selection

Step-3: Experiment Tracking

MLflow used to log parameters, metrics, and models

Enables reproducibility and comparison

Step-4: Model Packaging

Final model saved as a pipeline (preprocessing + model)

Prevents training-serving skew

Step-5: CI/CD Automation

GitHub Actions pipeline includes:

Linting

Unit tests

Model training (sanity check)

Step-6: Model Containerization

FastAPI-based inference service

Dockerized for portability

/predict endpoint returns prediction and confidence

Step-7: Production Deployment

Deployed to Google Cloud Run

Fully serverless and auto-scaling

Deployment triggered automatically via CI/CD

Step-8: Monitoring & Logging

Application logs emitted via Python logging

Logs captured in Cloud Logging

Metrics visible via Cloud Run dashboard

🚀 Deployment Instructions
Prerequisites

Google Cloud account

GitHub account

Billing enabled on GCP

GitHub repository cloned

1️⃣ Clone the Repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2️⃣ CI/CD & Cloud Deployment (Recommended)

No local dependency installation is required

All dependencies listed in requirements.txt are automatically installed during the CI/CD pipeline and container build.

Simply push code to the main branch:

git push origin main


GitHub Actions will:

Run tests and linting

Build the container

Deploy the API to Google Cloud Run

3️⃣ Local Development (Optional)

If you want to run the API locally for testing or debugging:

python -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
uvicorn app.main:app --reload


API will be available at:

http://localhost:8000

🌐 Using the API
Health Check
GET /


Response:

{
  "status": "ok",
  "message": "Heart Disease Model is running"
}

Prediction Endpoint
POST /predict

Sample Request
curl -X POST https://<cloud-run-url>/predict \
-H "Content-Type: application/json" \
-d '{
  "age": 67,
  "sex": 1,
  "cp": 3,
  "trestbps": 180,
  "chol": 320,
  "fbs": 1,
  "restecg": 2,
  "thalach": 90,
  "exang": 1,
  "oldpeak": 3.5,
  "slope": 0,
  "ca": 3,
  "thal": 3
}'

Sample Response
{
  "prediction": 1,
  "confidence": 0.84
}

🧪 Testing

Run unit tests locally:

pytest


Run linting:

flake8

📊 Monitoring & Logs
Logs

Google Cloud Console → Cloud Run → Service → Logs

Or Logs Explorer with:

resource.type="cloud_run_revision"

Metrics

Cloud Run → Service → Metrics

View request rate, latency, CPU, and memory usage

🔐 Security & Design Notes

Cloud Run infrastructure logs do not capture request bodies by design

Application logs record sanitized summaries of inputs

Full payload logging is intentionally avoided to prevent PII exposure

CI validates code correctness; deployment is automated separately
