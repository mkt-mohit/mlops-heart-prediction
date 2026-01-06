# ❤️ Heart Disease Prediction — End-to-End MLOps Project
"Commit for Demo"
## 1. Introduction

This project demonstrates a **complete end-to-end MLOps pipeline** for building, validating, deploying, and monitoring a machine learning model that predicts **heart disease risk** using the **UCI Heart Disease dataset**.

The project follows **industry-aligned MLOps best practices**, including:

- Reproducible data preprocessing and model training  
- Experiment tracking  
- Automated CI/CD  
- Containerized model serving  
- Cloud-native production deployment  
- Centralized logging and monitoring  

The final model is exposed as a **REST API** deployed on **Google Cloud Run**.

---

## 2. Problem Statement

Given patient health attributes such as age, cholesterol, blood pressure, and ECG results, predict whether the patient is at risk of heart disease.

**Target Variable**

- `0` → No heart disease  
- `1` → Heart disease present  

---

## 3. Architecture Overview
```
graph LR
    A[GitHub Repository] -->|git push| B{GitHub Actions}
    
    subgraph "CI/CD Pipeline"
        B --> C[Linting<br/>flake8]
        B --> D[Unit Tests<br/>pytest]
        B --> E[Model Training<br/>sanity check]
        B --> F[Deploy to<br/>Cloud Run]
    end
    
    subgraph "Google Cloud Run"
        F --> G[FastAPI<br/>REST API]
        F --> H[Docker<br/>Container]
        F --> I[Auto-scaling]
        F --> J[Cloud Logging<br/>& Metrics]
    end
    
    style A fill:#e1f5ff,stroke:#01579b
    style B fill:#fff9c4,stroke:#f57f17
    style G fill:#c8e6c9,stroke:#2e7d32
    style H fill:#c8e6c9,stroke:#2e7d32
    style I fill:#c8e6c9,stroke:#2e7d32
    style J fill:#c8e6c9,stroke:#2e7d32
```
## 4. Technology Stack

| Category | Tools |
|--------|------|
| Language | Python 3.12 |
| ML | scikit-learn |
| Experiment Tracking | MLflow |
| API Framework | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud Platform | Google Cloud Run |
| Monitoring | Cloud Logging, Cloud Run Metrics |

---

## 5. Project Structure

mlops_heart_disease/
├── app/
│ ├── main.py
│ └── schemas.py
├── data/
│ ├── raw/
│ └── processed/
├── models/
│ └── artifacts/
│ └── random_forest_final.pkl
├── tests/
│ ├── test_data.py
│ └── test_model.py
├── .github/workflows/
│ └── ci.yaml
├── Dockerfile
├── requirements.txt
├── README.md


---

## 6. MLOps Workflow

### Step 1: Data Acquisition & EDA
- Dataset downloaded from UCI repository  
- Missing values handled  
- Target converted to binary  
- Exploratory Data Analysis performed  

### Step 2: Feature Engineering & Model Training
- Feature scaling and encoding  
- Logistic Regression and Random Forest trained  
- Cross-validation used for model selection  

### Step 3: Experiment Tracking
- MLflow used to log parameters, metrics, and models  

### Step 4: Model Packaging
- Final model saved as a pipeline (preprocessing + model)  

### Step 5: CI/CD Automation
- GitHub Actions pipeline performs:
  - Linting
  - Unit testing
  - Model training sanity check  

### Step 6: Model Containerization
- FastAPI-based inference service  
- Dockerized application  
- `/predict` endpoint returns prediction and confidence  

### Step 7: Production Deployment
- Deployed on **Google Cloud Run**
- Fully serverless and auto-scaling  
- Deployment triggered via CI/CD  

### Step 8: Monitoring & Logging
- Application logs emitted using Python logging  
- Logs captured in Cloud Logging  
- Metrics visible in Cloud Run dashboard  

---

## 7. Deployment Instructions

### 7.1 Prerequisites

- Google Cloud account  
- GitHub account  
- Billing enabled on GCP  

---

### 7.2 Clone the Repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
7.3 Cloud Deployment (Recommended)

No local dependency installation required

All dependencies from requirements.txt are installed automatically during the CI/CD pipeline and container build.

Deploy by pushing to main branch:

git push origin main

7.4 Local Development (Optional)

If you want to run the API locally:

python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
uvicorn app.main:app --reload


API will be available at:

http://localhost:8000

# 8. Using the API
# 8.1 Health Check
GET /


Response:

{
  "status": "ok",
  "message": "Heart Disease Model is running"
}

# 8.2 Prediction Endpoint

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

