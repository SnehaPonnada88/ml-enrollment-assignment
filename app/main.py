"""
FastAPI inference service.

Loads a trained sklearn Pipeline and exposes prediction endpoints.
"""

from fastapi import FastAPI, HTTPException
from pathlib import Path
import joblib

from pydantic import BaseModel
from typing import List

MODEL_PATH = "models/model.joblib"
app = FastAPI(title="Enrollment Prediction API")

class EmployeeInput(BaseModel):
    age: float
    gender: str
    marital_status: str
    salary: float
    employment_type: str
    region: str
    has_dependents: str
    tenure_years: float

class PredictRequest(BaseModel):
    records: List[EmployeeInput]

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"model_loaded": Path(MODEL_PATH).exists()}

@app.post("/predict")
def predict(req: PredictRequest):
    """Return predictions for one or more records."""
    if not Path(MODEL_PATH).exists():
        raise HTTPException(status_code=400, detail="Model not trained")

    model = joblib.load(MODEL_PATH)

    # Convert list[EmployeeInput] -> list[dict] -> DataFrame for sklearn
    records = [r.dict() for r in req.records]

    import pandas as pd
    X = pd.DataFrame(records)

    preds = model.predict(X).tolist()

    # Predict probabilities if supported
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)[:, 1].tolist()
    else:
        probas = [None] * len(preds)

    return [{"enrolled": int(p), "probability": probas[i]} for i, p in enumerate(preds)]
