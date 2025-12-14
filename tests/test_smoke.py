"""
Smoke test ensuring training and prediction work together.
"""

import pandas as pd
from pathlib import Path
from src.train import train
from src.pipeline import build_pipeline

def test_training_and_prediction(tmp_path):
    df = pd.DataFrame([{
        "employee_id": i,
        "age": 30,
        "gender": "Male",
        "marital_status": "Single",
        "salary": 50000,
        "employment_type": "Full-time",
        "region": "West",
        "has_dependents": "No",
        "tenure_years": 2.0,
        "enrolled": i % 2
    } for i in range(20)])

    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    out_dir = tmp_path / "models"
    train(str(data_path), str(out_dir))

    assert (out_dir / "model.joblib").exists()
