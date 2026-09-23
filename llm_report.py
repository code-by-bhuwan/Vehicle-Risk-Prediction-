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
import sys
sys.path.insert(0, '/home/claude/pipeline')
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
        f"Vehicle fleet data summary: {profile}\n\n"
        f"Prediction summary: {pred_summary}\n\n"
        f"Top SHAP factors (feature: avg impact on fault risk): {top_factors}\n\n"
        "Write a concise, plain-English maintenance risk report for a fleet manager, "
        "explaining what's driving the risk predictions and what to check first."
    )
    try:
        if backend == 'openai':
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": context}]
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
        return f"[LLM call failed ({e}) — falling back to template]\n\n" + _template_narrative(profile, pred_summary, top_factors)


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
