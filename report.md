# Report: Employee Enrollment Prediction

## 1. Data Observations

The dataset contains employee-level demographic and employment information with the objective of predicting whether an employee is enrolled (`enrolled = 1`) or not (`enrolled = 0`).

### Dataset characteristics
- Total records: **2,000**
- Target variable: `enrolled` (binary classification)
- Class distribution:
  - Enrolled (1): **1,235 (~62%)**
  - Not enrolled (0): **765 (~38%)**

This indicates a **moderate class imbalance**, but not severe enough to require aggressive resampling techniques.

### Feature types
- **Numerical features**:
  - `age`
  - `salary`
  - `tenure_years`
- **Categorical features**:
  - `gender`
  - `marital_status`
  - `employment_type`
  - `region`
  - `has_dependents`

### Data quality observations
- Some numerical features contain missing values.
- Categorical features are well-defined but require encoding.
- `employee_id` is a unique identifier and was excluded from modeling to prevent data leakage.

### Data processing decisions
- Schema validation was performed to ensure required columns are present.
- Numerical values were coerced to appropriate numeric types.
- Missing values were handled using imputation within the modeling pipeline.
- All preprocessing was included in the model pipeline to ensure consistency between training and inference.

---

## 2. Model Choice & Rationale

### Selected model: Logistic Regression

A **Logistic Regression** classifier was selected as the baseline model and implemented using a scikit-learn `Pipeline` that combines preprocessing and modeling.

#### Rationale for Logistic Regression
- Well-suited for binary classification problems
- Strong baseline for tabular datasets
- Interpretable coefficients that can explain feature impact
- Lower risk of overfitting compared to complex non-linear models
- Fast training and reproducible results

Given the dataset size and structure, Logistic Regression offers an optimal balance between **performance, simplicity, and interpretability**.

### Preprocessing strategy
- **Numerical features**:
  - Median imputation (robust to outliers)
  - Standard scaling
- **Categorical features**:
  - Most-frequent imputation
  - One-hot encoding with `handle_unknown="ignore"`

The preprocessing steps were included in the same pipeline as the model to avoid training–serving skew.

---

## 3. Evaluation Results

The model was evaluated using a **stratified 80/20 train-test split**, preserving the original class distribution.

### Overall metrics
- **Accuracy**: **89.65%**
- **ROC-AUC**: **0.97**

The high ROC-AUC indicates excellent separability between enrolled and non-enrolled employees.

### Class-wise performance

#### Enrolled employees (Class 1)
- Precision: **0.91**
- Recall: **0.92**
- F1-score: **0.92**

The model correctly identifies the majority of enrolled employees while maintaining a low false-positive rate.

#### Not enrolled employees (Class 0)
- Precision: **0.87**
- Recall: **0.85**
- F1-score: **0.86**

Performance remains balanced across both classes, indicating that the model does not disproportionately favor the majority class.

---

## 4. Key Takeaways & Next Steps

### Key takeaways
- A well-constructed baseline model can achieve strong performance without unnecessary complexity.
- Logistic Regression, combined with appropriate preprocessing, is sufficient to model enrollment behavior effectively.
- The model demonstrates robust and balanced performance across classes despite moderate imbalance.
- Clear separation of training, evaluation, and inference logic improves maintainability and reliability.

### Next steps with more time
- Tune the classification threshold based on business objectives.
- Perform cross-validation to assess model stability.
- Analyze feature coefficients to understand drivers of enrollment.
- Benchmark tree-based models (Random Forest, Gradient Boosting) to evaluate marginal gains.
- Conduct fairness and bias checks across demographic segments.
- Define monitoring and retraining strategies for production deployment.

---

## Conclusion

This project demonstrates a complete, production-aware machine learning workflow—from data validation and modeling to evaluation and inference—emphasizing clarity, correctness, and maintainability.