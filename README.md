# AutoSentry AI — Vehicle Risk & Fault Prediction System

Predicts vehicle fault risk from OBD-II sensor data using Random Forest / XGBoost / LSTM,
with SHAP explainability, a live robustness-testing lab, an AI-generated report (PDF/TXT
export), a vehicle comparison tool, and a conversational "Ask the AI" assistant.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

The custom dark theme is locked via `.streamlit/config.toml` — keep that folder alongside
app.py or the branded look (and the light/dark-mode text-visibility fix) will not apply.

## Pages
- 🏠 Overview · 📊 Dashboard · 🔮 Predict · 🆚 Compare Vehicles · ⚖️ Model Comparison
- 🧪 Robustness Lab · 🔍 Explainability · 🤖 AI Report · 💬 Ask the AI · 📚 Data & Research

## AI backends
Google Gemini (free, no card — aistudio.google.com/apikey), OpenAI, Anthropic, or a
template/keyword fallback that always works with zero setup.

## Data source
OBD-II Dataset (da Silveira Barreto, Kaggle). DOI: 10.34740/KAGGLE/DSV/83155
https://www.kaggle.com/dsv/83155 — the exact dataset cited as ref [99] in:
Mahale, Kolhar & More (2025), Discover Applied Sciences 7:243.
