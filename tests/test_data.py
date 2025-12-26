import pandas as pd

def test_cleaned_data_exists():
    df = pd.read_csv("data/processed/heart_disease_clean.csv")
    assert not df.empty

def test_target_column_present():
    df = pd.read_csv("data/processed/heart_disease_clean.csv")
    assert "target" in df.columns

def test_no_missing_values():
    df = pd.read_csv("data/processed/heart_disease_clean.csv")
    assert df.isnull().sum().sum() == 0
