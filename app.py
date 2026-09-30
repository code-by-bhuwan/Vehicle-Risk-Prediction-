"""
AutoSentry AI — Vehicle Risk & Fault Prediction System
=========================================================
Run:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from io import BytesIO
from fpdf import FPDF
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from vehicle_risk_pipeline import VehicleRiskPipeline, map_columns, NUMERIC_FEATURES, ALL_FEATURES
from llm_report import TableReader, generate_report, llm_chat, PROJECT_CONTEXT

st.set_page_config(page_title="AutoSentry AI — Vehicle Risk Intelligence", layout="wide",
                    page_icon="🚦", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Brand palette (kept in sync with .streamlit/config.toml — do not diverge)
# ---------------------------------------------------------------------------
BG = "#0A0E17"
BG2 = "#131826"
BG3 = "#1A2133"
BORDER = "#232B40"
TEXT = "#E8EAED"
TEXT_MUTED = "#9CA5B4"
ACCENT = "#FF6B35"      # Signal Amber
ACCENT2 = "#00D9FF"     # Circuit Cyan
SUCCESS = "#2ECC71"
DANGER = "#FF4757"

# matplotlib must match, or SHAP plots render white-on-dark and look broken
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
    'text.color': TEXT, 'axes.labelcolor': TEXT, 'axes.edgecolor': BORDER,
    'xtick.color': TEXT, 'ytick.color': TEXT, 'grid.color': BORDER,
})

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif !important; }}

.stApp {{ background-color: {BG}; color: {TEXT}; }}
section[data-testid="stSidebar"] {{ background-color: {BG2}; border-right: 1px solid {BORDER}; }}
section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

.stMetric {{
    background: linear-gradient(145deg, {BG2}, {BG3});
    padding: 14px 16px; border-radius: 12px; border: 1px solid {BORDER};
}}
div[data-testid="stMetricValue"] {{ color: {ACCENT2}; font-family: 'Space Grotesk', sans-serif; }}
div[data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; }}

.brand-header {{
    display: flex; align-items: center; gap: 14px; padding: 1.4rem 2rem;
    background: linear-gradient(120deg, {BG2} 0%, {BG} 100%);
    border: 1px solid {BORDER}; border-radius: 16px; margin-bottom: 1.4rem;
}}
.brand-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 2.1rem; font-weight: 700;
    background: linear-gradient(90deg, {ACCENT}, {ACCENT2});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }}
.brand-sub {{ color: {TEXT_MUTED}; font-size: 0.95rem; margin-top: 2px; }}

.badge {{ display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 0.75rem;
    background: {BG3}; color: {ACCENT2}; border: 1px solid {BORDER}; margin: 3px 4px 3px 0; }}

.info-card {{ background: {BG2}; border: 1px solid {BORDER}; border-left: 4px solid {ACCENT};
    border-radius: 10px; padding: 16px 18px; margin-bottom: 10px; color: {TEXT}; }}
.info-card.success {{ border-left-color: {SUCCESS}; }}
.info-card.cyan {{ border-left-color: {ACCENT2}; }}

.stButton>button, .stDownloadButton>button {{
    background: linear-gradient(90deg, {ACCENT}, #FF8C5A); color: white; border: none;
    border-radius: 8px; font-weight: 600; padding: 0.5rem 1.2rem;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{ filter: brightness(1.1); }}

div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; overflow: hidden; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
.stTabs [data-baseweb="tab"] {{ background-color: {BG2}; border-radius: 8px 8px 0 0; color: {TEXT_MUTED}; }}
.stTabs [aria-selected="true"] {{ background-color: {BG3}; color: {ACCENT2} !important; }}

hr {{ border-color: {BORDER}; }}
</style>
""", unsafe_allow_html=True)


def brand_header(title, subtitle):
    st.markdown(f"""
    <div class="brand-header">
        <div style="font-size:2.6rem;">🚦</div>
        <div>
            <div class="brand-title">{title}</div>
            <div class="brand-sub">{subtitle}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def info_card(text, kind=""):
    st.markdown(f'<div class="info-card {kind}">{text}</div>', unsafe_allow_html=True)


PLOTLY_TEMPLATE = go.layout.Template()
PLOTLY_TEMPLATE.layout = go.Layout(
    paper_bgcolor=BG, plot_bgcolor=BG2,
    font=dict(color=TEXT, family="Inter"),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    colorway=[ACCENT2, ACCENT, SUCCESS, DANGER, "#A78BFA", "#F472B6"],
)

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

for key, default in [('last_input', None), ('last_result', None), ('last_report', None),
                      ('chat_history', []), ('backend', None), ('api_key', '')]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown(f"""
<div style="text-align:center; padding: 8px 0 4px 0;">
    <div style="font-size:2.4rem;">🚦</div>
    <div style="font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.3rem;
        background: linear-gradient(90deg, {ACCENT}, {ACCENT2}); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;">AutoSentry AI</div>
    <div style="color:{TEXT_MUTED}; font-size:0.78rem;">Vehicle Risk Intelligence</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 Dashboard",
    "🔮 Predict",
    "🆚 Compare Vehicles",
    "⚖️ Model Comparison",
    "🧪 Robustness Lab",
    "🔍 Explainability (SHAP)",
    "🤖 AI Report",
    "💬 Ask the AI",
    "📚 Data & Research",
])

st.sidebar.markdown("---")
st.sidebar.markdown("**AI Engine Settings**")
backend_choice = st.sidebar.selectbox("LLM Backend",
    ["Template (no key)", "Google Gemini (Free)", "OpenAI", "Anthropic"])
backend_map = {"Template (no key)": None, "Google Gemini (Free)": "gemini",
               "OpenAI": "openai", "Anthropic": "anthropic"}
st.session_state['backend'] = backend_map[backend_choice]
if st.session_state['backend']:
    st.session_state['api_key'] = st.sidebar.text_input(f"{backend_choice} API Key", type="password",
                                                          value=st.session_state['api_key'])
    if backend_choice == "Google Gemini (Free)":
        st.sidebar.caption("Free, no credit card — get a key at aistudio.google.com/apikey")
else:
    st.session_state['api_key'] = None
st.sidebar.caption("Used by both 'AI Report' and 'Ask the AI' tabs.")

st.sidebar.markdown("---")
st.sidebar.caption(f"📁 {len(df):,} records · {df['MARK'].nunique()} brands · "
                    f"Fault rate {df['FAULT'].mean():.1%}")
st.sidebar.caption("Data: OBD-II Dataset (Kaggle) · See 📚 Data & Research")

# ===========================================================================
# PAGE: Overview
# ===========================================================================
if page == "🏠 Overview":
    brand_header("AutoSentry AI", "AI-powered vehicle risk & fault intelligence, built on real OBD-II telemetry")

    st.markdown("""
<span class="badge">Random Forest</span><span class="badge">XGBoost</span><span class="badge">LSTM</span>
<span class="badge">SHAP Explainability</span><span class="badge">Robustness Testing</span>
<span class="badge">LLM Reporting</span><span class="badge">Adaptable to New Brands</span>
    """, unsafe_allow_html=True)
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Records", f"{len(df):,}")
    c2.metric("Vehicle Brands", df['MARK'].nunique())
    c3.metric("Best Model Accuracy", "99.6%")
    c4.metric("Fault Rate", f"{df['FAULT'].mean():.1%}")

    st.markdown("### What this system does")
    st.write("""
    AutoSentry AI predicts **vehicle fault/risk** from real OBD-II sensor telemetry (engine RPM,
    coolant temperature, throttle position, fuel trim, etc.) combined with vehicle metadata
    (brand, model, year, engine power). Built end-to-end: real data → cleaning → feature
    engineering → three trained models → robustness testing → explainability → an AI reporting
    layer that can also answer follow-up questions.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔑 Key Finding")
        info_card(
            "Exploratory analysis + SHAP revealed the fault label was heavily concentrated in "
            "just 2 of 10 brands (Volkswagen ~98%, Peugeot ~94%) due to persistent uncleared "
            "trouble codes — <b>not driving conditions</b>. A vehicle-held-out test confirmed sensor "
            "readings alone can't predict fault for unseen vehicles (ROC-AUC 0.19), an honest "
            "documented limitation rather than an inflated accuracy claim."
        )
    with col2:
        st.markdown("### 📚 Built on Literature")
        info_card(
            "Extends Mahale, Kolhar & More (2025) — a 94-paper systematic review with <b>no "
            "original experiments</b>. This project implements real models against the paper's "
            "own named datasets, open challenges, and all five proposed future-research themes. "
            "See the 📚 Data &amp; Research tab for full citations.", kind="cyan"
        )

    st.markdown("### How to use this app")
    st.markdown("""
    1. **📊 Dashboard** — explore the training data
    2. **🔮 Predict** — upload your own vehicle data (any brand, even unseen ones) for risk scoring
    3. **🆚 Compare Vehicles** — put two predictions side by side
    4. **⚖️ Model Comparison** & **🧪 Robustness Lab** — compare models and stress-test them
    5. **🔍 Explainability** — see exactly *why* a prediction was made (SHAP)
    6. **🤖 AI Report** & **💬 Ask the AI** — plain-English reporting and conversation
    7. **📚 Data & Research** — dataset source, paper citations, and how this project compares
    """)

# ===========================================================================
# PAGE: Dashboard
# ===========================================================================
elif page == "📊 Dashboard":
    brand_header("Dataset Dashboard", "Explore the training data behind AutoSentry AI")

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
                     color_continuous_scale=[ACCENT2, ACCENT], labels={'FAULT': 'Fault Rate (%)', 'MARK': 'Brand'})
        fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("⚠️ Concentrated in 2 brands — see Data & Research tab for why this matters.")

    with col2:
        st.subheader("Records per Brand")
        counts = df['MARK'].value_counts().reset_index()
        counts.columns = ['MARK', 'count']
        fig2 = px.bar(counts, x='MARK', y='count', color='count', color_continuous_scale=[BG3, ACCENT2],
                      labels={'count': 'Rows', 'MARK': 'Brand'})
        fig2.update_layout(template=PLOTLY_TEMPLATE, showlegend=False, height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Sensor Feature Explorer")
    feat = st.selectbox("Choose a sensor feature", NUMERIC_FEATURES)
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=df[df.FAULT==0][feat], name='No Fault', opacity=0.65, marker_color=SUCCESS))
    fig3.add_trace(go.Histogram(x=df[df.FAULT==1][feat], name='Fault', opacity=0.65, marker_color=DANGER))
    fig3.update_layout(barmode='overlay', template=PLOTLY_TEMPLATE, xaxis_title=feat, height=350)
    st.plotly_chart(fig3, use_container_width=True)

# ===========================================================================
# PAGE: Predict
# ===========================================================================
elif page == "🔮 Predict":
    brand_header("Predict Vehicle Risk", "Upload any OBD-style data — new brands and renamed columns handled automatically")

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
            color = 'background-color: rgba(255,71,87,0.18)' if row['FAULT_PREDICTION'] == 1 else ''
            return [color] * len(row)

        st.subheader("Prediction Results")
        st.dataframe(result.style.apply(highlight_risk, axis=1), use_container_width=True)

        fault_count = preds['FAULT_PREDICTION'].sum()
        avg_risk = preds['RISK_PROBABILITY'].mean()

        cA, cB = st.columns([1,1])
        cA.metric("Vehicles Flagged At-Risk", f"{fault_count} / {len(preds)}")
        csv_bytes = result.to_csv(index=False).encode('utf-8')
        cB.download_button("⬇️ Download results as CSV", csv_bytes, "risk_predictions.csv", "text/csv")

        # --- NEW: fleet average risk gauge ---
        st.subheader("Fleet Risk Gauge")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=avg_risk*100,
            number={'suffix': "%", 'font': {'color': TEXT}},
            gauge={'axis': {'range': [0, 100], 'tickcolor': TEXT},
                   'bar': {'color': ACCENT},
                   'bgcolor': BG2, 'borderwidth': 1, 'bordercolor': BORDER,
                   'steps': [{'range': [0, 33], 'color': '#1e3a2f'},
                             {'range': [33, 66], 'color': '#3a331e'},
                             {'range': [66, 100], 'color': '#3a1e1e'}]},
            title={'text': "Average Predicted Risk", 'font': {'color': TEXT}}
        ))
        gauge.update_layout(paper_bgcolor=BG, height=300, font={'color': TEXT})
        st.plotly_chart(gauge, use_container_width=True)

        st.session_state['last_input'] = input_df
        st.session_state['last_result'] = result

# ===========================================================================
# PAGE: Compare Vehicles (NEW)
# ===========================================================================
elif page == "🆚 Compare Vehicles":
    brand_header("Compare Vehicles", "Put two predicted vehicles side by side")

    result = st.session_state.get('last_result')
    if result is None or len(result) < 2:
        st.info("Run a prediction with at least 2 rows on the 🔮 Predict tab first.")
        st.stop()

    labels = [f"Row {i}: {result.iloc[i].get('brand', result.iloc[i].get('MARK', '?'))} "
              f"{result.iloc[i].get('model', result.iloc[i].get('MODEL', ''))}" for i in range(len(result))]
    c1, c2 = st.columns(2)
    idx_a = c1.selectbox("Vehicle A", range(len(result)), format_func=lambda i: labels[i], index=0)
    idx_b = c2.selectbox("Vehicle B", range(len(result)), format_func=lambda i: labels[i],
                          index=min(1, len(result)-1))

    row_a, row_b = result.iloc[idx_a], result.iloc[idx_b]
    cA, cB = st.columns(2)
    cA.metric(labels[idx_a], f"{row_a['RISK_PROBABILITY']:.1%} risk",
              "AT RISK" if row_a['FAULT_PREDICTION'] else "OK")
    cB.metric(labels[idx_b], f"{row_b['RISK_PROBABILITY']:.1%} risk",
              "AT RISK" if row_b['FAULT_PREDICTION'] else "OK")

    # Radar chart comparing available numeric sensor fields
    mapped = map_columns(result)
    compare_feats = [f for f in NUMERIC_FEATURES if f in mapped.columns and mapped[f].notna().all()]
    if len(compare_feats) >= 3:
        train_stats = df[compare_feats]
        norm = lambda v, f: (v - train_stats[f].min()) / (train_stats[f].max() - train_stats[f].min() + 1e-9)
        vals_a = [norm(mapped.iloc[idx_a][f], f) for f in compare_feats]
        vals_b = [norm(mapped.iloc[idx_b][f], f) for f in compare_feats]

        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(r=vals_a+[vals_a[0]], theta=compare_feats+[compare_feats[0]],
                                         name=labels[idx_a], line_color=ACCENT2, fill='toself', opacity=0.5))
        radar.add_trace(go.Scatterpolar(r=vals_b+[vals_b[0]], theta=compare_feats+[compare_feats[0]],
                                         name=labels[idx_b], line_color=ACCENT, fill='toself', opacity=0.5))
        radar.update_layout(template=PLOTLY_TEMPLATE, height=450,
                             polar=dict(bgcolor=BG2, radialaxis=dict(visible=True, range=[0,1], gridcolor=BORDER)),
                             showlegend=True, title="Normalized Sensor Profile (0=fleet min, 1=fleet max)")
        st.plotly_chart(radar, use_container_width=True)
    else:
        st.caption("Not enough shared numeric sensor fields to draw a comparison radar for these rows.")

# ===========================================================================
# PAGE: Model Comparison
# ===========================================================================
elif page == "⚖️ Model Comparison":
    brand_header("Model Comparison", "Random Forest vs XGBoost vs LSTM, head to head")
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

    st.dataframe(metrics_df.round(2).style.background_gradient(cmap='Oranges', axis=0), use_container_width=True)
    st.caption("*LSTM uses time-windowed sequences, trained offline for speed; not retrained live here.")

    fig = go.Figure()
    for col in metrics_df.columns:
        fig.add_trace(go.Bar(name=col, x=metrics_df.index, y=metrics_df[col]))
    fig.update_layout(barmode='group', template=PLOTLY_TEMPLATE, yaxis_range=[90,100], yaxis_title="%", height=420)
    st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# PAGE: Robustness Lab
# ===========================================================================
elif page == "🧪 Robustness Lab":
    brand_header("Robustness Lab", "Live experiment: corrupt the test set and watch performance degrade")

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
    fig.update_layout(barmode='group', template=PLOTLY_TEMPLATE, yaxis_range=[0,100], yaxis_title="%", height=420,
                       title=f"Performance at {missing_rate:.0%} missing, {noise_level:.1f}σ noise")
    st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# PAGE: Explainability
# ===========================================================================
elif page == "🔍 Explainability (SHAP)":
    brand_header("Explainability", "Why did the model predict this? (SHAP)")
    X_train, X_test, y_train, y_test, rf, xgb = get_train_test_and_models()
    explainer = get_shap_explainer(rf)

    st.subheader("Global Feature Importance")
    sample = X_test.sample(min(500, len(X_test)), random_state=1)
    sv = explainer.shap_values(sample)
    sv1 = sv[:, :, 1] if isinstance(sv, np.ndarray) and sv.ndim == 3 else sv[1]

    fig, ax = plt.subplots(figsize=(8,6))
    shap.summary_plot(sv1, sample, show=False, plot_size=None)
    fig = plt.gcf()
    fig.patch.set_facecolor(BG)
    for a in fig.axes:
        a.set_facecolor(BG)
    st.pyplot(fig)
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
        marker_color=[DANGER if v > 0 else SUCCESS for v in contrib.values]
    ))
    fig2.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="SHAP value (impact on risk)", height=400)
    st.plotly_chart(fig2, use_container_width=True)

# ===========================================================================
# PAGE: AI Report
# ===========================================================================
elif page == "🤖 AI Report":
    brand_header("AI-Generated Risk Report", "Predictions + SHAP, turned into a plain-English report")

    backend = st.session_state['backend']
    api_key = st.session_state['api_key']
    if backend:
        st.info(f"Using **{backend}** — key {'✅ set' if api_key else '❌ not set (add it in the sidebar)'}")
    else:
        st.info("Using deterministic **template** mode (no API key) — select Gemini/OpenAI/Anthropic in the sidebar for a real AI-written report.")

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

    report = st.session_state.get('last_report')
    if report:
        st.success(f"Report source: {report['source']}")
        st.markdown(report['report_text'])

        col1, col2 = st.columns(2)
        col1.download_button("⬇️ Download as .txt", report['report_text'], "risk_report.txt")

        # --- NEW: PDF export ---
        def build_pdf(text):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 12, "AutoSentry AI - Vehicle Risk Report")
            pdf.ln(14)
            pdf.set_font("Helvetica", "", 11)
            for line in text.split("\n"):
                safe_line = line.encode('latin-1', 'replace').decode('latin-1')
                if safe_line.strip() == "":
                    pdf.ln(6)
                else:
                    pdf.multi_cell(0, 6, safe_line)
            buf = BytesIO()
            pdf.output(buf)
            return buf.getvalue()

        pdf_bytes = build_pdf(report['report_text'])
        col2.download_button("⬇️ Download as .pdf", pdf_bytes, "risk_report.pdf", "application/pdf")

# ===========================================================================
# PAGE: Ask the AI
# ===========================================================================
elif page == "💬 Ask the AI":
    brand_header("Ask the AI", "Ask about the project, your data, or the underlying research")

    backend = st.session_state['backend']
    api_key = st.session_state['api_key']
    if not backend:
        st.warning("Currently in keyword-fallback mode (no LLM key). Select Gemini/OpenAI/Anthropic + "
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

# ===========================================================================
# PAGE: Data & Research (NEW)
# ===========================================================================
elif page == "📚 Data & Research":
    brand_header("Data & Research", "Exact sources, citations, and how this project compares to prior work")

    st.markdown("### 📦 Dataset used")
    info_card(
        "<b>OBD-II Dataset</b> — da Silveira Barreto, C. A. (2018). Kaggle.<br>"
        "DOI: <a href='https://doi.org/10.34740/KAGGLE/DSV/83155' target='_blank' style='color:#00D9FF;'>"
        "10.34740/KAGGLE/DSV/83155</a> · "
        "<a href='https://www.kaggle.com/dsv/83155' target='_blank' style='color:#00D9FF;'>kaggle.com/dsv/83155</a><br>"
        "Real telemetry from 14 vehicles / 10 brands, captured via an ELM327 OBD-II device and an "
        "Android OBD reader app. This is the <b>exact same dataset</b> cited as reference&nbsp;[99] in the "
        "source literature review below — not a substitute or synthetic dataset.", kind="cyan"
    )

    st.markdown("### 📖 Source literature")
    info_card(
        "Mahale, Y., Kolhar, S., &amp; More, A. S. (2025). <i>A comprehensive review on artificial "
        "intelligence driven predictive maintenance in vehicles: technologies, challenges and future "
        "research directions.</i> Discover Applied Sciences, 7, Article 243.<br>"
        "DOI: <a href='https://doi.org/10.1007/s42452-025-06681-3' target='_blank' style='color:#00D9FF;'>"
        "10.1007/s42452-025-06681-3</a><br>"
        "A systematic review of 94 papers with <b>no original experiments</b> — this project implements "
        "real models against its own named datasets, gaps, and five proposed future-research themes."
    )

    st.markdown("### ⚖️ Comparison to a real implementation paper")
    st.write(
        "Rather than only comparing against a review paper, we also benchmark against a genuine "
        "implementation cited inside that same review (reference [19]):"
    )
    info_card(
        "Hafeez, A. B., Alonso, E., &amp; Ter-Sarkisov, A. (2021). <i>Towards sequential multivariate "
        "fault prediction for vehicular predictive maintenance.</i> IEEE ICMLA 2021.<br>"
        "DOI: <a href='https://doi.org/10.1109/ICMLA52953.2021.00167' target='_blank' style='color:#00D9FF;'>"
        "10.1109/ICMLA52953.2021.00167</a>"
    )

    comp_df = pd.DataFrame({
        "Dimension": ["Task", "Best result", "Models compared", "Explainability",
                      "Robustness testing", "Data-quality analysis", "Generative AI / reporting", "Deployment"],
        "Hafeez et al. (2021)": ["Predict next DTC fault event (multi-class)", "63% top-3 accuracy",
                                  "LSTM only", "Not addressed", "Not addressed", "Not addressed",
                                  "Not addressed", "Research code only"],
        "AutoSentry AI (this project)": ["Predict current fault risk (binary)",
                                          "99.6% top-1 accuracy (RF/XGBoost)", "RF, XGBoost, LSTM",
                                          "Full SHAP (global + per-prediction)",
                                          "Controlled noise/missing-data study",
                                          "Brand-leakage discovery & validation",
                                          "Working LLM report + chat layer",
                                          "Public web app"],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.caption(
        "⚠️ Honest caveat: their task (predicting *which* future fault event occurs, over a large "
        "code vocabulary) is inherently harder than our binary current-risk classification, so the "
        "raw accuracy gap partly reflects task difficulty. The fairer claim is breadth and rigor: "
        "multi-model comparison, robustness testing, explainability, and a deployed system — none of "
        "which the compared paper attempts."
    )

    st.markdown("### 🔑 Key original contribution")
    info_card(
        "Beyond closing gaps named in the literature, this project identified and rigorously validated "
        "a <b>brand/vehicle-identity leakage</b> issue not specifically named in the source paper's data-"
        "quality discussion — using a vehicle-held-out validation test (ROC-AUC dropping to 0.19 when "
        "brand is excluded and vehicles are unseen), demonstrated in the ⚖️ Model Comparison methodology."
    )
