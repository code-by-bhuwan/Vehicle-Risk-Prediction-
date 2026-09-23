# Vehicle Risk & Fault Prediction System

Predicts vehicle fault risk from OBD-II sensor data using Random Forest / XGBoost / LSTM,
with SHAP explainability, a live robustness-testing lab, and an AI-generated report layer.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Live app
(add your deployed Streamlit Cloud link here once deployed)

## Files
- `app.py` — Streamlit GUI (Dashboard, Predict, Model Comparison, Robustness Lab, SHAP, AI Report)
- `vehicle_risk_pipeline.py` — adaptable preprocessing + prediction pipeline (handles new brands/columns)
- `llm_report.py` — table reader + LLM/template report generator
- `vehicle_risk_pipeline.pkl` — pre-trained Random Forest pipeline
- `clean_dataset.csv` — cleaned OBD-II training data (47,093 rows, 14 vehicles)

## Data source
OBD-II Dataset (da Silveira Barreto, Kaggle) — real telemetry from 14 vehicles across 10 brands.
