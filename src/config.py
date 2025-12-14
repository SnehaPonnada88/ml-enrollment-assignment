"""
Central configuration for column names.

Keeping column definitions in one place ensures that:
- Training
- Evaluation
- Inference

all use the same feature set and target definition.
"""

# Columns that must exist in the dataset
REQUIRED_COLUMNS = [
    "employee_id",
    "age",
    "gender",
    "marital_status",
    "salary",
    "employment_type",
    "region",
    "has_dependents",
    "tenure_years",
    "enrolled",
]

# Features used for training (exclude identifier and target)
FEATURE_COLUMNS = [
    "age",
    "gender",
    "marital_status",
    "salary",
    "employment_type",
    "region",
    "has_dependents",
    "tenure_years",
]

TARGET_COLUMN = "enrolled"
