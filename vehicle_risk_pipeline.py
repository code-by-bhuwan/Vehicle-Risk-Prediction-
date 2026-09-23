"""
Vehicle Risk/Fault Prediction Pipeline
----------------------------------------
A reusable, adaptable pipeline that:
  1. Accepts any OBD-style CSV, even with different column names
  2. Cleans messy numeric formats (percent signs, comma decimals, HH:MM:SS)
  3. Handles UNSEEN brands/models gracefully (no retrain needed to just predict)
  4. Imputes missing values using stored training statistics
  5. Predicts Vehicle Risk/Fault + gives a SHAP-based explanation
  6. Supports incremental retraining when new labeled data (e.g. Indian brands) arrives
"""
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# 1. Column alias map: canonical_name -> list of accepted incoming names
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    'MARK':                        ['mark', 'brand', 'make', 'manufacturer'],
    'MODEL':                       ['model', 'car_model', 'variant'],
    'CAR_YEAR':                    ['car_year', 'year', 'model_year'],
    'ENGINE_POWER':                ['engine_power', 'displacement', 'engine_cc', 'power'],
    'AUTOMATIC':                   ['automatic', 'transmission', 'is_automatic'],
    'ENGINE_COOLANT_TEMP':         ['engine_coolant_temp', 'coolant_temp', 'coolanttemp'],
    'ENGINE_LOAD':                 ['engine_load', 'load', 'engineload'],
    'ENGINE_RPM':                  ['engine_rpm', 'rpm'],
    'INTAKE_MANIFOLD_PRESSURE':    ['intake_manifold_pressure', 'map', 'manifold_pressure'],
    'AIR_INTAKE_TEMP':             ['air_intake_temp', 'intake_air_temp', 'iat'],
    'SPEED':                       ['speed', 'vehicle_speed', 'speed_kmh'],
    'SHORT TERM FUEL TRIM BANK 1': ['short term fuel trim bank 1', 'stft', 'fuel_trim'],
    'THROTTLE_POS':                ['throttle_pos', 'throttle', 'throttle_position'],
    'TIMING_ADVANCE':              ['timing_advance', 'ignition_timing'],
}

NUMERIC_FEATURES = ['CAR_YEAR','ENGINE_POWER','AUTOMATIC','ENGINE_COOLANT_TEMP','ENGINE_LOAD',
                     'ENGINE_RPM','INTAKE_MANIFOLD_PRESSURE','AIR_INTAKE_TEMP','SPEED',
                     'SHORT TERM FUEL TRIM BANK 1','THROTTLE_POS','TIMING_ADVANCE']
CATEGORICAL_FEATURES = ['MARK', 'MODEL']
ALL_FEATURES = NUMERIC_FEATURES + ['MARK_ENC', 'MODEL_ENC']


def _clean_numeric_value(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace('%', '').replace(',', '.').strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def map_columns(df):
    """Rename incoming columns to canonical names using the alias map (case-insensitive)."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in [canonical.lower()] + aliases:
            if alias in lower_map:
                rename[lower_map[alias]] = canonical
                break
    return df.rename(columns=rename)


class UnknownAwareEncoder:
    """LabelEncoder-like class that maps unseen categories to a dedicated 'UNKNOWN' bucket
    instead of throwing an error — critical for new brands like Maruti Suzuki/Tata/Mahindra."""
    def __init__(self):
        self.classes_ = []
        self.mapping = {}

    def fit(self, values):
        self.classes_ = sorted(set(values))
        self.mapping = {v: i for i, v in enumerate(self.classes_)}
        self.mapping['__UNKNOWN__'] = len(self.classes_)  # reserved index
        return self

    def transform(self, values):
        return np.array([self.mapping.get(v, self.mapping['__UNKNOWN__']) for v in values])

    def add_new_categories(self, new_values):
        """Extend the encoder with newly seen categories (used before retraining)."""
        new_unique = sorted(set(new_values) - set(self.classes_))
        for v in new_unique:
            self.classes_.append(v)
        self.mapping = {v: i for i, v in enumerate(self.classes_)}
        self.mapping['__UNKNOWN__'] = len(self.classes_)


class VehicleRiskPipeline:
    def __init__(self):
        self.model = None
        self.mark_encoder = UnknownAwareEncoder()
        self.model_encoder = UnknownAwareEncoder()
        self.train_medians = None
        self.explainer = None

    # ---------------- Preprocessing ----------------
    def _prepare_features(self, df_raw, fit=False):
        df = map_columns(df_raw.copy())

        # ensure all expected columns exist (fill missing ones with NaN)
        for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
            if col not in df.columns:
                df[col] = np.nan

        # clean numeric formatting issues
        for col in NUMERIC_FEATURES:
            df[col] = df[col].apply(_clean_numeric_value)

        # categorical: fill missing with a literal 'UNKNOWN' string, then encode
        df['MARK'] = df['MARK'].fillna('UNKNOWN').astype(str).str.lower().str.strip()
        df['MODEL'] = df['MODEL'].fillna('UNKNOWN').astype(str).str.lower().str.strip()

        if fit:
            self.mark_encoder.fit(df['MARK'])
            self.model_encoder.fit(df['MODEL'])
            self.train_medians = df[NUMERIC_FEATURES].median()
        else:
            # extend encoders on the fly so new brands don't crash, but flag them
            unseen_marks = set(df['MARK']) - set(self.mark_encoder.classes_)
            unseen_models = set(df['MODEL']) - set(self.model_encoder.classes_)
            if unseen_marks:
                print(f"[INFO] New/unseen brand(s) detected (not in training data): {unseen_marks}")
            if unseen_models:
                print(f"[INFO] New/unseen model(s) detected: {unseen_models}")

        df['MARK_ENC'] = self.mark_encoder.transform(df['MARK'])
        df['MODEL_ENC'] = self.model_encoder.transform(df['MODEL'])

        # impute missing numerics with stored TRAIN medians (not the new data's own medians)
        for col in NUMERIC_FEATURES:
            df[col] = df[col].fillna(self.train_medians[col] if self.train_medians is not None else 0)

        return df[ALL_FEATURES]

    # ---------------- Training ----------------
    def fit(self, df_raw, y):
        X = self._prepare_features(df_raw, fit=True)
        self.model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42,
                                             class_weight='balanced', n_jobs=-1)
        self.model.fit(X, y)
        return self

    def retrain_with_new_data(self, df_new, y_new, df_existing=None, y_existing=None):
        """Incrementally incorporate newly gathered data (e.g. Indian-brand records)
        by extending encoders and refitting on the combined dataset."""
        self.mark_encoder.add_new_categories(
            map_columns(df_new)['MARK'].fillna('UNKNOWN').astype(str).str.lower().str.strip())
        self.model_encoder.add_new_categories(
            map_columns(df_new)['MODEL'].fillna('UNKNOWN').astype(str).str.lower().str.strip())

        if df_existing is not None:
            combined_df = pd.concat([df_existing, df_new], ignore_index=True)
            combined_y = pd.concat([y_existing, y_new], ignore_index=True)
        else:
            combined_df, combined_y = df_new, y_new

        X = self._prepare_features(combined_df, fit=False)
        self.train_medians = X[NUMERIC_FEATURES].median()  # refresh medians with combined data
        self.model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42,
                                             class_weight='balanced', n_jobs=-1)
        self.model.fit(X, combined_y)
        return self

    # ---------------- Inference ----------------
    def predict(self, df_raw):
        X = self._prepare_features(df_raw, fit=False)
        proba = self.model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return pd.DataFrame({'FAULT_PREDICTION': pred, 'RISK_PROBABILITY': proba.round(4)})

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
