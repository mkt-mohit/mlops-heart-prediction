import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

def test_model_pipeline_runs():
    df = pd.read_csv("data/processed/heart_disease_clean.csv")
    X = df.drop(columns="target")
    y = df["target"]

    num_features = ["age","trestbps","chol","thalach","oldpeak","ca"]
    cat_features = ["sex","cp","fbs","restecg","exang","slope","thal"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", "passthrough", cat_features)
    ])

    model = Pipeline([
        ("preprocess", preprocessor),
        ("rf", RandomForestClassifier(n_estimators=10, random_state=42))
    ])

    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(y)
