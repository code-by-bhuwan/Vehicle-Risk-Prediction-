"""
Table Reader + LLM Explanation Layer for the Vehicle Risk Pipeline
--------------------------------------------------------------------
- TableReader: ingests any CSV/Excel, profiles it, maps columns.
- generate_report(): runs predictions + SHAP, then narrates results either via
  an LLM (OpenAI or Anthropic, if an API key is set) or a template fallback
  (always works, no key needed -- this is what will run during your demo).
"""
import os
import pandas as pd
import numpy as np
import shap
from vehicle_risk_pipeline import VehicleRiskPipeline, map_columns, NUMERIC_FEATURES


class TableReader:
    """Reads any tabular file and produces a quick data profile."""
    @staticmethod
    def read(path):
        if path.endswith('.csv'):
            df = pd.read_csv(path, low_memory=False)
        elif path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(path)
        else:
            raise ValueError("Unsupported file type — use .csv or .xlsx")
        return df

    @staticmethod
    def profile(df):
        mapped = map_columns(df)
        known = [c for c in NUMERIC_FEATURES + ['MARK', 'MODEL'] if c in mapped.columns]
        n_recognized_original_cols = len(set(df.columns) & set(mapped.columns)) + \
            sum(1 for orig, new in zip(df.columns, mapped.columns) if orig != new)
        missing_pct = mapped[known].isna().mean().mul(100).round(1).to_dict() if known else {}
        return {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'recognized_columns': known,
            'unrecognized_columns_count': max(0, len(df.columns) - len(known)),
            'missing_pct': missing_pct,
            'brands_seen': sorted(mapped['MARK'].dropna().unique().tolist()) if 'MARK' in mapped.columns else []
        }


def _template_narrative(profile, pred_summary, top_factors):
    lines = []
    lines.append(f"DATA SUMMARY: {profile['n_rows']} rows, {len(profile['recognized_columns'])} recognized OBD features.")
    if profile['unrecognized_columns_count']:
        lines.append(f"Note: {profile['unrecognized_columns_count']} column(s) not recognized and were ignored.")
    lines.append(f"Brands present: {', '.join(profile['brands_seen']) if profile['brands_seen'] else 'none detected'}.")
    lines.append("")
    lines.append(f"RISK ASSESSMENT: {pred_summary['fault_count']} of {pred_summary['n']} vehicles "
                 f"flagged at elevated fault risk ({pred_summary['fault_rate']:.1%}).")
    lines.append(f"Average predicted risk probability: {pred_summary['avg_prob']:.1%}.")
    lines.append("")
    lines.append("TOP CONTRIBUTING FACTORS (SHAP):")
    for feat, val in top_factors:
        direction = "increases" if val > 0 else "decreases"
        lines.append(f"  - {feat}: {direction} predicted risk (avg impact {val:+.3f})")
    lines.append("")
    lines.append("RECOMMENDATION: Prioritize inspection for vehicles flagged FAULT=1, "
                 "especially where coolant temperature or fuel trim readings are abnormal.")
    return "\n".join(lines)


def _llm_narrative(profile, pred_summary, top_factors, backend, api_key):
    context = (
        f"{PROJECT_CONTEXT}\n\n"
        f"Vehicle fleet data summary: {profile}\n\n"
        f"Prediction summary: {pred_summary}\n\n"
        f"Top SHAP factors (feature: avg impact on fault risk): {top_factors}\n\n"
        "Write a concise, plain-English maintenance risk report for a fleet manager, "
        "explaining what's driving the risk predictions and what to check first. "
        "Use short paragraphs and a few bullet points. Keep it under 250 words."
    )
    try:
        if backend == 'openai':
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": context}],
                max_tokens=500,
                temperature=0.4,
            )
            return resp.choices[0].message.content
        elif backend == 'anthropic':
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=600,
                messages=[{"role": "user", "content": context}]
            )
            return resp.content[0].text
    except Exception as e:
        err = str(e).lower()
        if "auth" in err or "api_key" in err or "401" in err:
            note = f"⚠️ {backend} API key was rejected — check it's correct and has billing enabled."
        elif "rate" in err or "429" in err:
            note = f"⚠️ {backend} rate limit hit — wait a moment and try again."
        else:
            note = f"⚠️ {backend} call failed ({e})."
        return note + "\n\nFalling back to template report:\n\n" + _template_narrative(profile, pred_summary, top_factors)


PROJECT_CONTEXT = """
PROJECT: Vehicle Risk/Fault Prediction System
- Trained on real OBD-II telemetry from 14 vehicles across 10 brands (Volkswagen, Peugeot, Citroen,
  Chevrolet, Fiat, Ford, Nissan, Honda, Renault, Toyota), 47,093 cleaned rows.
- Models: Random Forest (~99.6% acc), XGBoost (~99.6% acc), LSTM (~98.8% acc) on a fleet-monitoring
  framing (brand/vehicle known). Explainability via SHAP.
- KEY FINDING: fault label is heavily concentrated in only 2-3 vehicles (VW, Peugeot, Citroen) due to
  persistent uncleared trouble codes, not driving conditions -- a vehicle-blind generalization test
  (unseen cars, sensors only) collapses to ROC-AUC 0.19, exposing a real data-quality limitation.
- Robustness lab: Random Forest degrades more gracefully under missing data; XGBoost degrades more
  gracefully under sensor noise.
- Adaptable pipeline: gracefully handles brand-new vehicle brands (e.g. Maruti Suzuki, Tata, Mahindra,
  Hyundai) never seen in training, renamed columns, and missing fields, via an UNKNOWN-category
  fallback and a retrain_with_new_data() method for incorporating real data later.
- Based on / extends gaps identified in: Mahale, Kolhar & More (2025), "A comprehensive review on
  AI driven predictive maintenance in vehicles," Discover Applied Sciences 7:243. That paper is a
  94-paper literature review with no original experiments; this project implements and validates
  against its own named datasets, challenges, and five proposed research themes (multimodal data,
  adaptive/resilient systems, interpretability, cost reduction, generative AI).
"""


def _keyword_fallback_answer(user_message, context_blocks):
    """Very lightweight, no-API-key fallback: surfaces the most keyword-relevant
    lines from the available context so the chat still returns something useful."""
    text = "\n".join(context_blocks)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    words = set(w.lower().strip(".,:;()%") for w in user_message.split() if len(w) > 3)
    scored = []
    for line in lines:
        line_words = set(w.lower().strip(".,:;()%") for w in line.split())
        overlap = len(words & line_words)
        if overlap > 0:
            scored.append((overlap, line))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return ("I don't have an LLM API key configured, so I can only do simple keyword matching "
                "right now. I couldn't find anything closely related to that question in the current "
                "report/context. Try asking about: fault rate, model accuracy, SHAP, robustness, "
                "brands, or the source paper.")
    top_lines = [l for _, l in scored[:5]]
    return ("[No LLM key configured -- keyword-based answer]\n\n" + "\n".join(f"- {l}" for l in top_lines))


def llm_chat(user_message, history, context_blocks, backend=None, api_key=None):
    """
    General-purpose chat function grounded in whatever context is available
    (project facts, last report, last predictions, dataset profile).
    history: list of {'role': 'user'|'assistant', 'content': str}
    context_blocks: list of strings to include as background context
    """
    system_prompt = (
        "You are a helpful assistant embedded in a vehicle risk/fault prediction application. "
        "Answer questions about the project, the data, the models, the generated report, and "
        "predictions, using the context provided. Be concise, concrete, and honest about "
        "limitations. If asked something outside this context, answer using general automotive "
        "knowledge but say so.\n\nCONTEXT:\n" + "\n\n".join(context_blocks)
    )

    if not (backend and api_key):
        return _keyword_fallback_answer(user_message, context_blocks)

    try:
        if backend == 'openai':
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            messages = [{"role": "system", "content": system_prompt}] + history + \
                       [{"role": "user", "content": user_message}]
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
            return resp.choices[0].message.content
        elif backend == 'anthropic':
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            messages = history + [{"role": "user", "content": user_message}]
            resp = client.messages.create(model="claude-sonnet-4-5", max_tokens=800,
                                           system=system_prompt, messages=messages)
            return resp.content[0].text
    except Exception as e:
        return f"[LLM call failed: {e}]\n\n" + _keyword_fallback_answer(user_message, context_blocks)


def generate_report(pipeline: VehicleRiskPipeline, df_raw, backend=None, api_key=None):
    """
    backend: None -> template fallback (always works)
             'openai' or 'anthropic' -> uses that LLM if api_key is provided (env var or arg)
    """
    profile = TableReader.profile(df_raw)

    X = pipeline._prepare_features(df_raw, fit=False)
    proba = pipeline.model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    pred_summary = {
        'n': len(pred),
        'fault_count': int(pred.sum()),
        'fault_rate': float(pred.mean()),
        'avg_prob': float(proba.mean())
    }

    explainer = shap.TreeExplainer(pipeline.model)
    sv = explainer.shap_values(X)
    sv1 = sv[:, :, 1] if isinstance(sv, np.ndarray) and sv.ndim == 3 else sv[1]
    mean_shap = pd.Series(np.mean(sv1, axis=0), index=X.columns).sort_values(key=abs, ascending=False)
    top_factors = list(mean_shap.head(5).items())

    api_key = api_key or os.environ.get(f'{(backend or "").upper()}_API_KEY')

    if backend and api_key:
        narrative = _llm_narrative(profile, pred_summary, top_factors, backend, api_key)
        source = f"LLM ({backend})"
    else:
        narrative = _template_narrative(profile, pred_summary, top_factors)
        source = "template fallback (no API key set)"

    return {
        'source': source,
        'profile': profile,
        'prediction_summary': pred_summary,
        'top_factors': top_factors,
        'report_text': narrative
    }
