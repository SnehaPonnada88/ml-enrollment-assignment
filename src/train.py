"""
Model training entrypoint.

Steps:
1. Load and validate data
2. Train/test split
3. Fit pipeline
4. Save model artifact and metrics
"""

import argparse
import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from .data import load_data
from .pipeline import build_pipeline, split_xy

def train(data_path: str, out_dir: str):
    df = load_data(data_path)
    X, y = split_xy(df)

    # Stratified split keeps class distribution stable
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred)
    }

    if hasattr(pipeline, "predict_proba"):
        metrics["roc_auc"] = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

    metrics["classification_report"] = classification_report(y_test, y_pred, output_dict=True)

    # Save artifacts
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    joblib.dump(pipeline, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()
    train(args.data, args.out_dir)

if __name__ == "__main__":
    main()
