"""
Vehicle Risk/Fault Prediction — Full GUI Application
=======================================================
Run:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from vehicle_risk_pipeline import VehicleRiskPipeline, map_columns, NUMERIC_FEATURES, ALL_FEATURES
from llm_report import TableReader, generate_report, llm_chat, PROJECT_CONTEXT

st.set_page_config(page_title="Vehicle Risk & Fault Prediction", layout="wide", page_icon="🚗",
                    initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Professional theming
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
    .stMetric { background-color: #1a1d24; padding: 12px; border-radius: 10px; border: 1px solid #2d323d; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #4fc3f7; }
    .hero { padding: 1.5rem 2rem; background: linear-gradient(135deg, #1a2332, #0e1117);
            border-radius: 14px; border: 1px solid #2d3748; margin-bottom: 1.2rem; }
    .badge { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.75rem;
             background: #234; color: #7dd3fc; margin-right: 6px; }
    .stChatMessage { border-radius: 10px; }
    section[data-testid="stSidebar"] { background-color: #12151c; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached loaders
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
    rf = pipeline.model
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

# session state defaults
for key, default in [('last_input', None), ('last_result', None), ('last_report', None),
                      ('chat_history', []), ('backend', None), ('api_key', '')]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🚗 Vehicle Risk System")
st.sidebar.caption("AI-powered predictive maintenance")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 Dashboard",
    "🔮 Predict",
    "⚖️ Model Comparison",
    "🧪 Robustness Lab",
    "🔍 Explainability (SHAP)",
    "🤖 AI Report",
    "💬 Ask the AI",
])

st.sidebar.markdown("---")
st.sidebar.markdown("**AI Engine Settings**")
backend_choice = st.sidebar.selectbox("LLM Backend", ["Template (no key)", "OpenAI", "Anthropic"])
backend_map = {"Template (no key)": None, "OpenAI": "openai", "Anthropic": "anthropic"}
st.session_state['backend'] = backend_map[backend_choice]
if st.session_state['backend']:
    st.session_state['api_key'] = st.sidebar.text_input(f"{backend_choice} API Key", type="password",
                                                          value=st.session_state['api_key'])
else:
    st.session_state['api_key'] = None
st.sidebar.caption("Used by both 'AI Report' and 'Ask the AI' tabs.")

st.sidebar.markdown("---")
st.sidebar.caption(f"📁 {len(df):,} records | {df['MARK'].nunique()} brands | "
                    f"Fault rate {df['FAULT'].mean():.1%}")

# ===========================================================================
# PAGE: Overview
# ===========================================================================
if page == "🏠 Overview":
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.title("🚗 Vehicle Risk & Fault Prediction System")
    st.markdown("""
<span class="badge">Random Forest</span><span class="badge">XGBoost</span><span class="badge">LSTM</span>
<span class="badge">SHAP Explainability</span><span class="badge">Robustness Testing</span>
<span class="badge">LLM Reporting</span><span class="badge">Adaptable to New Brands</span>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Records", f"{len(df):,}")
    c2.metric("Vehicle Brands", df['MARK'].nunique())
    c3.metric("Best Model Accuracy", "99.6%")
    c4.metric("Fault Rate", f"{df['FAULT'].mean():.1%}")

    st.markdown("### What this system does")
    st.markdown("""
    This project predicts **vehicle fault/risk** from real OBD-II sensor telemetry (engine RPM,
    coolant temperature, throttle position, fuel trim, etc.) combined with vehicle metadata
    (brand, model, year, engine power). It was built end-to-end: real data → cleaning →
    feature engineering → three trained models → robustness testing → explainability → an
    AI reporting layer that can also answer follow-up questions.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔑 Key Finding")
        st.info(
            "Exploratory analysis + SHAP revealed the fault label was heavily concentrated in "
            "just 2 of 10 brands (Volkswagen ~98%, Peugeot ~94%) due to persistent uncleared "
            "trouble codes — **not driving conditions**. A vehicle-held-out test confirmed sensor "
            "readings alone can't predict fault for unseen vehicles (ROC-AUC 0.19), an honest "
            "documented limitation rather than an inflated accuracy claim."
        )
    with col2:
        st.markdown("### 📚 Built on Literature")
        st.success(
            "Extends Mahale, Kolhar & More (2025) — a 94-paper systematic review with **no "
            "original experiments**. This project implements real models against the paper's "
            "own named datasets, open challenges, and all five proposed future-research themes: "
            "Multimodal Data, Adaptive Systems, Interpretability, Cost Reduction, and "
            "Generative AI."
        )

    st.markdown("### How to use this app")
    st.markdown("""
    1. **📊 Dashboard** — explore the training data
    2. **🔮 Predict** — upload your own vehicle data (any brand, even unseen ones) for risk scoring
    3. **⚖️ Model Comparison** & **🧪 Robustness Lab** — see how models compare and degrade under noisy/missing data
    4. **🔍 Explainability** — see exactly *why* a prediction was made (SHAP)
    5. **🤖 AI Report** — generate a plain-English maintenance report
    6. **💬 Ask the AI** — chat about the project, the data, or your results
    """)

# ===========================================================================
# PAGE: Dashboard
# ===========================================================================
elif page == "📊 Dashboard":
    st.title("Dataset Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Brands", df['MARK'].nunique())
    c3.metric("Fault Rate", f"{df['FAULT'].mean():.1%}")
    c4.metric("Features", len(NUMERIC_FEATURES))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fault Rate by Brand")
        fault_by_brand = (df.groupby('MARK')['FAULT'].mean()*100).sort_values(ascending=False).reset_index()
        fig = px.bar(fault_by_brand, x='MARK', y='FAULT', color='FAULT',
                     color_continuous_scale='Reds', labels={'FAULT': 'Fault Rate (%)', 'MARK': 'Brand'})
        fig.update_layout(template='plotly_dark', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("⚠️ Concentrated in 2 brands — see Explainability tab for why this matters.")

    with col2:
        st.subheader("Records per Brand")
        counts = df['MARK'].value_counts().reset_index()
        counts.columns = ['MARK', 'count']
        fig2 = px.bar(counts, x='MARK', y='count', color='count', color_continuous_scale='Blues',
                      labels={'count': 'Rows', 'MARK': 'Brand'})
        fig2.update_layout(template='plotly_dark', showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Sensor Feature Explorer")
    feat = st.selectbox("Choose a sensor feature", NUMERIC_FEATURES)
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=df[df.FAULT==0][feat], name='No Fault', opacity=0.6, marker_color='#2ecc71'))
    fig3.add_trace(go.Histogram(x=df[df.FAULT==1][feat], name='Fault', opacity=0.6, marker_color='#e74c3c'))
    fig3.update_layout(barmode='overlay', template='plotly_dark', xaxis_title=feat, height=350)
    st.plotly_chart(fig3, use_container_width=True)

# ===========================================================================
# PAGE: Predict
# ===========================================================================
elif page == "🔮 Predict":
    st.title("Predict Vehicle Risk / Fault")
    st.caption("Upload any OBD-style CSV/Excel — new brands and renamed columns are handled automatically.")

    upload = st.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx', 'xls'])
    use_sample = st.checkbox("Use built-in sample (mixed known + Indian brands, messy columns)",
                              value=not bool(upload))

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
        cA, cB = st.columns([1,1])
        cA.metric("Vehicles Flagged At-Risk", f"{fault_count} / {len(preds)}")
        csv_bytes = result.to_csv(index=False).encode('utf-8')
        cB.download_button("⬇️ Download results as CSV", csv_bytes, "risk_predictions.csv", "text/csv")

        st.session_state['last_input'] = input_df
        st.session_state['last_result'] = result

# ===========================================================================
# PAGE: Model Comparison
# ===========================================================================
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
    metrics_df = (pd.DataFrame({
        "Random Forest": rf_m, "XGBoost": xgb_m,
        "LSTM (precomputed*)": {"accuracy": LSTM_METRICS['accuracy'], "f1": LSTM_METRICS['f1'],
                                 "roc_auc": LSTM_METRICS['roc_auc']}
    }).T * 100)

    st.dataframe(metrics_df.round(2).style.background_gradient(cmap='Greens', axis=0), use_container_width=True)
    st.caption("*LSTM uses time-windowed sequences, trained offline for speed; not retrained live here.")

    fig = go.Figure()
    for col in metrics_df.columns:
        fig.add_trace(go.Bar(name=col, x=metrics_df.index, y=metrics_df[col]))
    fig.update_layout(barmode='group', template='plotly_dark', yaxis_range=[90,100], yaxis_title="%")
    st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# PAGE: Robustness Lab
# ===========================================================================
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

    fig = go.Figure()
    for col in res_df.columns:
        fig.add_trace(go.Bar(name=col, x=res_df.index, y=res_df[col]))
    fig.update_layout(barmode='group', template='plotly_dark', yaxis_range=[0,100], yaxis_title="%",
                       title=f"Performance at {missing_rate:.0%} missing, {noise_level:.1f}σ noise")
    st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# PAGE: Explainability
# ===========================================================================
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
    fig2 = go.Figure(go.Bar(
        x=contrib.values, y=contrib.index, orientation='h',
        marker_color=['#e74c3c' if v > 0 else '#2ecc71' for v in contrib.values]
    ))
    fig2.update_layout(template='plotly_dark', xaxis_title="SHAP value (impact on risk)", height=400)
    st.plotly_chart(fig2, use_container_width=True)

# ===========================================================================
# PAGE: AI Report
# ===========================================================================
elif page == "🤖 AI Report":
    st.title("AI-Generated Risk Report")
    st.caption("Turns predictions + SHAP explanations into a plain-English report for a fleet manager.")

    backend = st.session_state['backend']
    api_key = st.session_state['api_key']
    if backend:
        st.info(f"Using **{backend}** — key {'✅ set' if api_key else '❌ not set (add it in the sidebar)'}")
    else:
        st.info("Using deterministic **template** mode (no API key) — select OpenAI/Anthropic in the sidebar for a real AI-written report.")

    data_source = st.session_state.get('last_input')
    if data_source is None:
        st.info("💡 Tip: go to the Predict tab first for a real result, or use the sample below.")
        data_source = pd.DataFrame({
            'brand': ['Volkswagen','Toyota','Maruti Suzuki'], 'model': ['Polo','Corolla','Swift'],
            'rpm': [2200,1500,1800], 'coolant_temp': [98,82,85], 'throttle_pos': [15,12,10], 'speed_kmh':[50,30,0]
        })
        st.dataframe(data_source, use_container_width=True)

    if st.button("📝 Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            report = generate_report(pipeline, data_source, backend=backend, api_key=api_key or None)
        st.session_state['last_report'] = report
        st.success(f"Report source: {report['source']}")
        st.markdown(report['report_text'])
        st.download_button("⬇️ Download report (.txt)", report['report_text'], "risk_report.txt")

# ===========================================================================
# PAGE: Ask the AI
# ===========================================================================
elif page == "💬 Ask the AI":
    st.title("Ask the AI")
    st.caption("Ask about the project, the data, your last prediction, the generated report, "
               "or the source literature this project builds on.")

    backend = st.session_state['backend']
    api_key = st.session_state['api_key']
    if not backend:
        st.warning("Currently in keyword-fallback mode (no LLM key). Select OpenAI/Anthropic + "
                   "paste a key in the sidebar for full conversational answers.")

    for msg in st.session_state['chat_history']:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    user_q = st.chat_input("Ask a question...")
    if user_q:
        st.session_state['chat_history'].append({'role': 'user', 'content': user_q})
        with st.chat_message('user'):
            st.markdown(user_q)

        context_blocks = [PROJECT_CONTEXT]
        if st.session_state.get('last_result') is not None:
            context_blocks.append(f"LATEST PREDICTION RESULTS:\n{st.session_state['last_result'].to_string()}")
        if st.session_state.get('last_report') is not None:
            context_blocks.append(f"LATEST GENERATED REPORT:\n{st.session_state['last_report']['report_text']}")

        history_for_llm = [{"role": m['role'], "content": m['content']}
                            for m in st.session_state['chat_history'][:-1]]

        with st.chat_message('assistant'):
            with st.spinner("Thinking..."):
                answer = llm_chat(user_q, history_for_llm, context_blocks, backend=backend, api_key=api_key or None)
            st.markdown(answer)
        st.session_state['chat_history'].append({'role': 'assistant', 'content': answer})

    if st.session_state['chat_history']:
        if st.button("🗑️ Clear conversation"):
            st.session_state['chat_history'] = []
            st.rerun()
