"""
Data loading and validation utilities.

Responsibilities:
- Read CSV file
- Validate schema (required columns)
- Perform minimal, safe type coercions

Heavy transformations are intentionally left to the sklearn Pipeline.
"""

import pandas as pd
from .config import REQUIRED_COLUMNS

def load_data(path: str) -> pd.DataFrame:
    """Load CSV file and validate its schema."""
    df = pd.read_csv(path)
    validate_schema(df)
    return df

def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate required columns and enforce basic types.

    Raises:
        ValueError: if required columns are missing
    """
    # Check column presence
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure numeric columns are numeric
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    df["tenure_years"] = pd.to_numeric(df["tenure_years"], errors="coerce")

    # Target must be binary integer
    df["enrolled"] = pd.to_numeric(df["enrolled"], errors="raise").astype(int)

    # Normalize categorical text fields
    for col in ["gender", "marital_status", "employment_type", "region", "has_dependents"]:
        df[col] = df[col].astype(str).str.strip()
