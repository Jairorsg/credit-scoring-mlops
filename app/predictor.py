import joblib
import json
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────
BASE     = Path(__file__).parent.parent
cfg      = yaml.safe_load(open(BASE / "config.yaml"))
UMBRAL   = cfg["model"]["decision_threshold"]

MODEL_PATH  = BASE / cfg["model"]["artifacts"]["model_path"]
PREP_PATH   = BASE / cfg["model"]["artifacts"]["preprocessor_path"]
FEAT_PATH   = BASE / cfg["model"]["artifacts"]["features_path"]
METRICS_PATH= BASE / "model/artifacts/metrics.json"

# ── Carga en memoria (una sola vez al iniciar) ────────────
model        = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREP_PATH)

with open(FEAT_PATH) as f:
    feature_names = json.load(f)

with open(METRICS_PATH) as f:
    metrics = json.load(f)

MODELO_VERSION = f"lgbm_v1_auc{metrics['roc_auc_test']}"


def _calcular_features(data: dict) -> pd.DataFrame:
    """Construye el vector de features igual que en 02_features.ipynb."""
    row = data.copy()

    # Ratios financieros
    row["PAYMENT_RATE"]         = row["AMT_ANNUITY"]  / (row["AMT_INCOME_TOTAL"] or np.nan)
    row["CREDIT_INCOME_RATIO"]  = row["AMT_CREDIT"]   / (row["AMT_INCOME_TOTAL"] or np.nan)
    row["CREDIT_GOODS_RATIO"]   = row["AMT_CREDIT"]   / (row["AMT_GOODS_PRICE"]  or np.nan)
    row["ANNUITY_CREDIT_RATIO"] = row["AMT_ANNUITY"]  / (row["AMT_CREDIT"]       or np.nan)

    # Features derivadas
    row["AGE_YEARS"]          = abs(row["DAYS_BIRTH"]) / 365
    row["EMPLOYMENT_YEARS"]   = abs(min(row["DAYS_EMPLOYED"], 0)) / 365
    row["INCOME_PER_PERSON"]  = row["AMT_INCOME_TOTAL"] / (row["CNT_FAM_MEMBERS"] or np.nan)
    row["CREDIT_TERM_MONTHS"] = row["AMT_CREDIT"] / (row["AMT_ANNUITY"] or np.nan)

    df = pd.DataFrame([row])

    # Alinear con las features del entrenamiento
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan
    df = df[feature_names]

    return df


def _nivel_riesgo(prob: float) -> str:
    if prob < 0.15:   return "BAJO"
    elif prob < 0.30: return "MEDIO"
    elif prob < 0.50: return "ALTO"
    else:             return "MUY ALTO"


def predecir(data: dict) -> dict:
    df_input  = _calcular_features(data)
    X         = preprocessor.transform(df_input)
    proba     = float(model.predict_proba(X)[0][1])  # type: ignore
    decision  = "RECHAZADO" if proba >= UMBRAL else "APROBADO"
    score     = int((1 - proba) * 1000)

    return {
        "decision":       decision,
        "probabilidad":   round(proba, 4),
        "score":          score,
        "nivel_riesgo":   _nivel_riesgo(proba),
        "umbral_usado":   UMBRAL,
        "modelo_version": MODELO_VERSION,
    }