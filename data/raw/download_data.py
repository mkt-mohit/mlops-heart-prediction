import pandas as pd
import os

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLUMNS = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]

os.makedirs("data/raw", exist_ok=True)

df = pd.read_csv(URL, header=None, names=COLUMNS)

# Binary target (0 = no disease, 1 = disease)
df["target"] = (df["target"] > 0).astype(int)

df.to_csv("data/raw/heart_disease_raw.csv", index=False)
print("Downloaded: data/raw/heart_disease_raw.csv")
