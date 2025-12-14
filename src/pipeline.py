"""
Model pipeline definition.

This module builds a **single sklearn Pipeline** that includes:
1. Preprocessing (numeric + categorical)
2. Classification model

Saving this pipeline avoids training/serving skew.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from .config import FEATURE_COLUMNS, TARGET_COLUMN

NUMERIC_FEATURES = ["age", "salary", "tenure_years"]
CATEGORICAL_FEATURES = [c for c in FEATURE_COLUMNS if c not in NUMERIC_FEATURES]

def build_pipeline(random_state: int = 42) -> Pipeline:
    """Create preprocessing + model pipeline."""

    # Numeric preprocessing:
    # - Median imputation handles missing values robustly
    # - Scaling helps Logistic Regression converge
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing:
    # - Most frequent imputation
    # - One-hot encoding (ignore unseen categories at inference)
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    # Baseline classifier
    model = LogisticRegression(max_iter=2000, random_state=random_state)

    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

def split_xy(df: pd.DataFrame):
    """Split dataframe into features (X) and target (y)."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).to_numpy()
    return X, y
