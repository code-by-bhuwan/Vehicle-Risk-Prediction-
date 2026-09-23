"""
Vehicle Risk/Fault Prediction — Full GUI Application
=======================================================
Run:  streamlit run app.py

Features:
  - Dashboard: dataset overview, fault distribution
  - Predict: upload any CSV/Excel (even new brands, renamed columns) -> risk predictions
  - Model Comparison: RF vs XGBoost vs LSTM metrics
  - Robustness Lab: live noise/missing-data injection, watch accuracy degrade
  - Explainability: SHAP breakdown for any row
  - AI Report: LLM-generated (or template) plain-English risk report
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from vehicle_risk_pipeline import VehicleRiskPipeline, map_columns, NUMERIC_FEATURES, ALL_FEATURES
from llm_report import TableReader, generate_report

st.set_page_config(page_title="Vehicle Risk & Fault Prediction", layout="wide", page_icon="🚗")

# ---------------------------------------------------------------------------
# Cached loaders (run once per session, not on every interaction)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    return VehicleRiskPipeline.load("vehicle_risk_pipeline.pkl")

@st.cache_data
def load_dataset():
    return pd.read_csv("clean_dataset.csv")

@st.cache_resource
def get_train_test_and_models():
    df = load_dataset()
    pipeline = load_pipeline()
    X_raw = df.copy()
    X_raw['MARK_ENC'] = pipeline.mark_encoder.transform(X_raw['MARK'].str.lower())
    X_raw['MODEL_ENC'] = pipeline.model_encoder.transform(X_raw['MODEL'].str.lower())
    X = X_raw[ALL_FEATURES]
    y = df['FAULT']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = pipeline.model  # already trained
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42,
                         eval_metric='logloss', scale_pos_weight=(y_train==0).sum()/(y_train==1).sum())
    xgb.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, rf, xgb

@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

LSTM_METRICS = {"accuracy": 0.9878, "precision": 0.9884, "recall": 0.9756, "f1": 0.9820, "roc_auc": 0.9994}

pipeline = load_pipeline()
df = load_dataset()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🚗 Vehicle Risk System")
page = st.sidebar.radio("Navigate", [
    "📊 Dashboard",
    "🔮 Predict",
    "⚖️ Model Comparison",
    "🧪 Robustness Lab",
    "🔍 Explainability (SHAP)",
    "🤖 AI Report"
])
st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {len(df):,} rows | {df['MARK'].nunique()} brands | "
                    f"Fault rate: {df['FAULT'].mean():.1%}")

# ---------------------------------------------------------------------------
# PAGE: Dashboard
# ---------------------------------------------------------------------------
if page == "📊 Dashboard":
    st.title("Vehicle Predictive Maintenance — Dashboard")
    st.caption("Real OBD-II telemetry from 14 vehicles, cleaned and labeled for fault prediction.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Brands", df['MARK'].nunique())
    c3.metric("Fault Rate", f"{df['FAULT'].mean():.1%}")
    c4.metric("Features", len(NUMERIC_FEATURES))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fault Rate by Brand")
        fault_by_brand = df.groupby('MARK')['FAULT'].mean().sort_values(ascending=False) * 100
        fig, ax = plt.subplots(figsize=(6,4))
        fault_by_brand.plot(kind='bar', ax=ax, color='#e74c3c')
        ax.set_ylabel("Fault Rate (%)"); ax.set_xlabel("")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        st.info("⚠️ Fault rate is heavily brand-concentrated — see Explainability tab for why this matters.")

    with col2:
        st.subheader("Records per Brand")
        counts = df['MARK'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6,4))
        counts.plot(kind='bar', ax=ax2, color='#3498db')
        ax2.set_ylabel("Rows"); ax2.set_xlabel("")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig2)

    st.subheader("Sensor Feature Distributions")
    feat = st.selectbox("Choose a sensor feature", NUMERIC_FEATURES)
    fig3, ax3 = plt.subplots(figsize=(10,3))
    df[df.FAULT==0][feat].plot(kind='hist', bins=40, alpha=0.6, label='No Fault', ax=ax3, color='#2ecc71')
    df[df.FAULT==1][feat].plot(kind='hist', bins=40, alpha=0.6, label='Fault', ax=ax3, color='#e74c3c')
    ax3.legend(); ax3.set_xlabel(feat)
    st.pyplot(fig3)

# ---------------------------------------------------------------------------
# PAGE: Predict
# ---------------------------------------------------------------------------
elif page == "🔮 Predict":
    st.title("Predict Vehicle Risk / Fault")
    st.caption("Upload any OBD-style CSV/Excel — even new brands or renamed columns are handled automatically.")

    upload = st.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx', 'xls'])
    use_sample = st.checkbox("Use built-in sample (Indian + European brands, messy columns)", value=not bool(upload))

    if upload is not None:
        input_df = pd.read_csv(upload) if upload.name.endswith('.csv') else pd.read_excel(upload)
    elif use_sample:
        input_df = pd.DataFrame({
            'brand':        ['Volkswagen', 'Toyota', 'Maruti Suzuki', 'Tata', 'Peugeot', 'Hyundai'],
            'model':        ['Polo', 'Corolla', 'Swift', 'Nexon', '208', 'Creta'],
            'rpm':          [2200, 1500, '1,800', 1600, 2400, 1900],
            'coolant_temp': [98, 82, 85, '90%', 101, 89],
            'throttle_pos': [15, 12, 10, 14, 22, 16],
            'speed_kmh':    [50, 30, 0, 40, 60, 35],
        })
    else:
        st.warning("Upload a file or check the sample-data box.")
        st.stop()

    st.subheader("Input Data Preview")
    st.dataframe(input_df, use_container_width=True)

    profile = TableReader.profile(input_df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", profile['n_rows'])
    c2.metric("Recognized Features", len(profile['recognized_columns']))
    c3.metric("Brands Detected", len(profile['brands_seen']))

    unknown_brands = set(b.lower() for b in profile['brands_seen']) - set(pipeline.mark_encoder.classes_)
    if unknown_brands:
        st.warning(f"🆕 New brand(s) not in training data — using safe fallback: {', '.join(unknown_brands)}")

    if st.button("Run Prediction", type="primary"):
        preds = pipeline.predict(input_df)
        result = pd.concat([input_df.reset_index(drop=True), preds], axis=1)

        def highlight_risk(row):
            color = '#5c1a1a' if row['FAULT_PREDICTION'] == 1 else ''
            return [f'background-color: {color}'] * len(row)

        st.subheader("Prediction Results")
        st.dataframe(result.style.apply(highlight_risk, axis=1), use_container_width=True)

        fault_count = preds['FAULT_PREDICTION'].sum()
        st.metric("Vehicles Flagged At-Risk", f"{fault_count} / {len(preds)}")
        st.session_state['last_input'] = input_df
        st.session_state['last_result'] = result

# ---------------------------------------------------------------------------
# PAGE: Model Comparison
# ---------------------------------------------------------------------------
elif page == "⚖️ Model Comparison":
    st.title("Model Comparison: RF vs XGBoost vs LSTM")
    with st.spinner("Training / loading models..."):
        X_train, X_test, y_train, y_test, rf, xgb = get_train_test_and_models()

    def eval_model(model, X, y):
        pred = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        return {"accuracy": accuracy_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, proba)}

    rf_m = eval_model(rf, X_test, y_test)
    xgb_m = eval_model(xgb, X_test, y_test)
    lstm_m = LSTM_METRICS

    metrics_df = pd.DataFrame({
        "Random Forest": rf_m, "XGBoost": xgb_m,
        "LSTM (precomputed*)": {"accuracy": lstm_m['accuracy'], "f1": lstm_m['f1'], "roc_auc": lstm_m['roc_auc']}
    }).T * 100

    st.dataframe(metrics_df.round(2).style.background_gradient(cmap='Greens', axis=0), use_container_width=True)
    st.caption("*LSTM uses time-windowed sequences and was trained offline for speed; not retrained live here.")

    fig, ax = plt.subplots(figsize=(9,4))
    metrics_df.plot(kind='bar', ax=ax)
    ax.set_ylabel("%"); ax.set_ylim(90, 100)
    ax.legend(title="Metric")
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# PAGE: Robustness Lab
# ---------------------------------------------------------------------------
elif page == "🧪 Robustness Lab":
    st.title("Robustness to Noisy / Missing Data")
    st.caption("Live experiment: corrupt the test set and watch model performance degrade.")

    X_train, X_test, y_train, y_test, rf, xgb = get_train_test_and_models()
    train_medians = X_train.median()
    rng = np.random.RandomState(42)

    col1, col2 = st.columns(2)
    with col1:
        missing_rate = st.slider("Missing-data rate", 0.0, 0.8, 0.0, 0.05)
    with col2:
        noise_level = st.slider("Sensor noise (fraction of std-dev)", 0.0, 2.0, 0.0, 0.1)

    Xc = X_test.copy()
    if missing_rate > 0:
        mask = rng.rand(*Xc.shape) < missing_rate
        vals = Xc.values.astype(float)
        vals[mask] = np.nan
        Xc = pd.DataFrame(vals, columns=Xc.columns, index=Xc.index).fillna(train_medians)
    if noise_level > 0:
        for col in Xc.columns:
            Xc[col] = Xc[col] + rng.normal(0, noise_level * X_train[col].std(), size=len(Xc))

    results = {}
    for name, model in [("Random Forest", rf), ("XGBoost", xgb)]:
        pred = model.predict(Xc)
        proba = model.predict_proba(Xc)[:, 1]
        results[name] = {"Accuracy": accuracy_score(y_test, pred)*100,
                          "F1": f1_score(y_test, pred)*100,
                          "ROC-AUC": roc_auc_score(y_test, proba)*100}

    res_df = pd.DataFrame(results).T
    c1, c2 = st.columns(2)
    c1.metric("RF F1 Score", f"{results['Random Forest']['F1']:.1f}%")
    c2.metric("XGBoost F1 Score", f"{results['XGBoost']['F1']:.1f}%")

    fig, ax = plt.subplots(figsize=(8,4))
    res_df.plot(kind='bar', ax=ax)
    ax.set_ylim(0, 100); ax.set_ylabel("%")
    ax.set_title(f"Performance at {missing_rate:.0%} missing, {noise_level:.1f}σ noise")
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# PAGE: Explainability
# ---------------------------------------------------------------------------
elif page == "🔍 Explainability (SHAP)":
    st.title("Explainability — Why did the model predict this?")
    X_train, X_test, y_train, y_test, rf, xgb = get_train_test_and_models()
    explainer = get_shap_explainer(rf)

    st.subheader("Global Feature Importance")
    sample = X_test.sample(min(500, len(X_test)), random_state=1)
    sv = explainer.shap_values(sample)
    sv1 = sv[:, :, 1] if isinstance(sv, np.ndarray) and sv.ndim == 3 else sv[1]

    fig, ax = plt.subplots(figsize=(8,6))
    shap.summary_plot(sv1, sample, show=False, plot_size=None)
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader("Explain a Single Prediction")
    idx = st.slider("Pick a test-set row", 0, len(X_test)-1, 0)
    row = X_test.iloc[[idx]]
    row_sv = explainer.shap_values(row)
    row_sv1 = row_sv[:, :, 1] if isinstance(row_sv, np.ndarray) and row_sv.ndim == 3 else row_sv[1]

    proba = rf.predict_proba(row)[0,1]
    st.metric("Predicted Risk Probability", f"{proba:.1%}")

    contrib = pd.Series(row_sv1[0], index=row.columns).sort_values()
    fig2, ax2 = plt.subplots(figsize=(8,4))
    colors = ['#e74c3c' if v>0 else '#2ecc71' for v in contrib.values]
    contrib.plot(kind='barh', ax=ax2, color=colors)
    ax2.set_xlabel("SHAP value (impact on risk)")
    st.pyplot(fig2)

# ---------------------------------------------------------------------------
# PAGE: AI Report
# ---------------------------------------------------------------------------
elif page == "🤖 AI Report":
    st.title("AI-Generated Risk Report")
    st.caption("Turns predictions + SHAP explanations into a plain-English report for a fleet manager.")

    backend_choice = st.radio("Report engine", ["Template (no key needed)", "OpenAI", "Anthropic"], horizontal=True)
    api_key = None
    backend = None
    if backend_choice == "OpenAI":
        api_key = st.text_input("OpenAI API key", type="password")
        backend = 'openai'
    elif backend_choice == "Anthropic":
        api_key = st.text_input("Anthropic API key", type="password")
        backend = 'anthropic'

    data_source = st.session_state.get('last_input')
    if data_source is None:
        st.info("Go to the Predict tab first and run a prediction, or use the sample below.")
        data_source = pd.DataFrame({
            'brand': ['Volkswagen','Toyota','Maruti Suzuki'], 'model': ['Polo','Corolla','Swift'],
            'rpm': [2200,1500,1800], 'coolant_temp': [98,82,85], 'throttle_pos': [15,12,10], 'speed_kmh':[50,30,0]
        })

    if st.button("Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            report = generate_report(pipeline, data_source, backend=backend, api_key=api_key or None)
        st.success(f"Report source: {report['source']}")
        st.text(report['report_text'])
