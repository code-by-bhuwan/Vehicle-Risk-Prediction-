# Vehicle Risk & Fault Prediction System

Predicts vehicle fault risk from OBD-II sensor data using Random Forest / XGBoost / LSTM,
with SHAP explainability, a live robustness-testing lab, an AI-generated report, and a
conversational "Ask the AI" assistant grounded in the project's own data and findings.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Features
- 🏠 Overview — project summary and key findings
- 📊 Dashboard — interactive data exploration (Plotly)
- 🔮 Predict — upload any CSV/Excel, even unseen brands / renamed columns
- ⚖️ Model Comparison — RF vs XGBoost vs LSTM
- 🧪 Robustness Lab — live noise/missing-data sliders
- 🔍 Explainability — global + per-prediction SHAP
- 🤖 AI Report — LLM-generated plain-English risk report (OpenAI/Anthropic/template)
- 💬 Ask the AI — chat about the project, your data, or your results

## AI setup
Select a backend (OpenAI/Anthropic) and paste an API key in the sidebar to enable real
LLM responses. Without a key, both AI features fall back to safe, working alternatives
(template report / keyword-matched answers) — the app never crashes without a key.

## Data source
OBD-II Dataset (da Silveira Barreto, Kaggle) — real telemetry from 14 vehicles, 10 brands.
