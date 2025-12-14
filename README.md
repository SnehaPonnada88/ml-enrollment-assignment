# Employee Enrollment Prediction (ML + FastAPI)

This repository contains an end-to-end **machine learning workflow**:
1. Load and validate tabular employee data
2. Train a reproducible ML model using a sklearn Pipeline
3. Evaluate model performance
4. Serve predictions using FastAPI

The project is structured and commented to clearly demonstrate:
- Data processing
- Model development
- Functional, runnable code
- Code quality and readability

---

## Project Structure

```
src/        -> Data processing, training, evaluation logic
app/        -> FastAPI inference service
tests/      -> Automated smoke tests
models/     -> Generated model artifacts
```

---

## How to Run

### 1. Setup (Windows PowerShell)

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Train
Place `employee_data.csv` in the project root.

```bash
python -m src.train --data employee_data.csv --out_dir models
```

### 3. Run API
```bash
uvicorn app.main:app --reload
```

Visit:
- `http://localhost:8000/docs`
- `http://localhost:8000/health`
