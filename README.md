# Explainable Anomaly Detection — Household Power Consumption

End-to-end anomaly detection system on the UCI Individual Household Electric
Power Consumption dataset with visual, model-agnostic explanations.

## Dataset

UCI ML Repository #235 — ~2 M minute-resolution records, Dec 2006 – Nov 2010.  
7 numeric features: active power, reactive power, voltage, current, 3 sub-meters.

## Project Structure

```
household-anomaly/
├── data/
│   ├── raw/           # downloaded dataset (git-ignored)
│   ├── processed/     # cleaned parquet (git-ignored)
│   ├── models/        # trained artifacts (git-ignored)
│   └── explanations/  # cached XAI outputs (git-ignored)
├── src/
│   ├── data/          # download + cleaning pipeline
│   ├── features/      # feature engineering (Phase 2)
│   ├── models/        # IF + LSTM-AE training (Phase 3)
│   ├── xai/           # SHAP, LIME, DiCE (Phase 4)
│   └── dashboard/     # Dash app (Phase 5)
├── notebooks/         # EDA and experiments
└── tests/             # pytest suite
```

## Phase 1 — Data Pipeline Setup

### Prerequisites

```
python 3.11
pip install -r requirements.txt
```

> Windows GPU users: replace `tensorflow==2.17.0` with `tensorflow-cpu==2.17.0`
> in requirements.txt if CUDA is not configured.

### Steps

```bash
# 1. Download raw dataset (~132 MB zip, extracts to ~130 MB txt)
python src/data/download.py

# 2. Clean and save as parquet (~70 MB)
python src/data/load_clean.py

# 3. Run EDA notebook
jupyter notebook notebooks/01_eda.ipynb

# 4. Run tests
pytest tests/ -v
```

## Phases 2–6 (upcoming)

| Phase | Description                        |
|-------|------------------------------------|
| 2     | Feature engineering (lags, rolling stats, calendar features) |
| 3     | Model training (Isolation Forest + LSTM Autoencoder) |
| 4     | Explanations (SHAP, LIME, DiCE counterfactuals)       |
| 5     | Interactive Dash dashboard                            |
| 6     | Evaluation + report                                   |
