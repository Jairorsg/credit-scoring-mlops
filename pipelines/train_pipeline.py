"""
╔══════════════════════════════════════════════════════════════╗
║         Credit Scoring MLOps — Pipeline de entrenamiento    ║
║   Ejecutar: python pipelines/train_pipeline.py              ║
╚══════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any

import mlflow                          # type: ignore[import]
import mlflow.lightgbm                 # type: ignore[import]
import lightgbm as lgb                 # type: ignore[import]
import scipy.sparse as sp

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

# ─────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
cfg  = yaml.safe_load(open(BASE / "config.yaml"))


# ─────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}]  {msg}")


def cargar_datos() -> pd.DataFrame:
    log("Cargando datos...")
    df = pd.read_csv(BASE / cfg["data"]["raw_path"])
    log(f"Dataset: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df


def feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:  # type: ignore[type-arg]
    log("Construyendo features...")
    TARGET = cfg["project"]["target_column"]
    data   = df.copy()

    for ratio in cfg["features"]["ratios"]:
        data[ratio["name"]] = (
            data[ratio["numerator"]] / data[ratio["denominator"]].replace(0, np.nan)
        )

    data["AGE_YEARS"]          = data["DAYS_BIRTH"].abs() / 365
    data["EMPLOYMENT_YEARS"]   = data["DAYS_EMPLOYED"].clip(upper=0).abs() / 365
    data["INCOME_PER_PERSON"]  = data["AMT_INCOME_TOTAL"] / data["CNT_FAM_MEMBERS"].replace(0, np.nan)
    data["CREDIT_TERM_MONTHS"] = data["AMT_CREDIT"] / data["AMT_ANNUITY"].replace(0, np.nan)

    drop_cols = cfg["features"]["drop_columns"] + [TARGET]
    X: pd.DataFrame = data.drop(columns=drop_cols, errors="ignore")
    y: pd.Series = data[TARGET]  # type: ignore[assignment]

    log(f"Features construidas: {X.shape[1]}")
    return X, y


def construir_preprocesador(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg["features"]["imputation"]["numeric_strategy"]))
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg["features"]["imputation"]["categorical_strategy"])),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def to_dense(arr: Any) -> np.ndarray:  # type: ignore[type-arg]
    """Convierte sparse matrix a numpy array denso si es necesario."""
    if sp.issparse(arr):
        return arr.toarray()  # type: ignore[union-attr]
    return np.array(arr)


def ejecutar_pipeline() -> None:
    print("\n" + "═" * 58)
    print("  CREDIT SCORING — PIPELINE DE ENTRENAMIENTO")
    print("═" * 58 + "\n")

    mlflow_uri = "file:///" + str(BASE / cfg["mlflow"]["tracking_uri"]).replace("\\", "/")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M')}"):

        # ── 1. Datos ──────────────────────────────────────
        df       = cargar_datos()
        X, y     = feature_engineering(df)
        feat_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=cfg["project"]["random_seed"],
            stratify=y
        )

        # ── 2. Preprocesamiento ───────────────────────────
        log("Preprocesando...")
        preprocessor = construir_preprocesador(X_train)
        X_train_proc: np.ndarray = to_dense(preprocessor.fit_transform(X_train))
        X_test_proc:  np.ndarray = to_dense(preprocessor.transform(X_test))

        # ── 3. SMOTE ──────────────────────────────────────
        log("Aplicando SMOTE...")
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(
            sampling_strategy=cfg["smote"]["sampling_strategy"],
            k_neighbors=cfg["smote"]["k_neighbors"],
            random_state=cfg["smote"]["random_state"]
        )
        X_bal, y_bal = smote.fit_resample(X_train_proc, y_train)  # type: ignore[misc]
        y_bal_arr: np.ndarray = np.array(y_bal)
        log(f"Tras SMOTE: {X_bal.shape[0]:,} filas ({float(y_bal_arr.mean())*100:.1f}% morosos)")

        # ── 4. Entrenamiento ──────────────────────────────
        log("Entrenando LightGBM...")
        params = cfg["model"]["params"].copy()
        params["random_state"] = cfg["project"]["random_seed"]
        model = lgb.LGBMClassifier(**params)

        cv = StratifiedKFold(
            n_splits=cfg["model"]["cv_folds"],
            shuffle=True,
            random_state=cfg["project"]["random_seed"]
        )
        cv_scores: np.ndarray = cross_val_score(
        model, X_bal, y_bal_arr,  # type: ignore[arg-type]
        cv=cv, scoring="roc_auc", n_jobs=-1
        )
        model.fit(X_bal, y_bal_arr)  # type: ignore[union-attr]

        # ── 5. Evaluación ─────────────────────────────────
        log("Evaluando...")
        y_proba: np.ndarray = np.array(model.predict_proba(X_test_proc))[:, 1]  # type: ignore[union-attr]
        y_pred  = (y_proba >= cfg["model"]["decision_threshold"]).astype(int)
        auc     = float(roc_auc_score(y_test, y_proba))

        # ── 6. Registrar en MLflow ────────────────────────
        log("Registrando en MLflow...")
        mlflow.log_params(params)
        mlflow.log_param("smote_ratio",  cfg["smote"]["sampling_strategy"])
        mlflow.log_param("cv_folds",     cfg["model"]["cv_folds"])
        mlflow.log_param("n_features",   len(feat_names))
        mlflow.log_param("threshold",    cfg["model"]["decision_threshold"])

        mlflow.log_metric("roc_auc_test",    auc)
        mlflow.log_metric("roc_auc_cv_mean", float(cv_scores.mean()))
        mlflow.log_metric("roc_auc_cv_std",  float(cv_scores.std()))
        mlflow.log_metric("n_train",         float(X_bal.shape[0]))
        mlflow.log_metric("n_test",          float(X_test_proc.shape[0]))

        mlflow.lightgbm.log_model(          # type: ignore[attr-defined]
            model,
            artifact_path="model",
            registered_model_name=cfg["mlflow"]["model_name"]
        )

        # ── 7. Guardar artefactos ─────────────────────────
        log("Guardando artefactos...")
        artifacts_path = BASE / "model/artifacts"
        artifacts_path.mkdir(parents=True, exist_ok=True)

        joblib.dump(model,        artifacts_path / "lgbm_model.joblib")
        joblib.dump(preprocessor, artifacts_path / "preprocessor.joblib")

        with open(artifacts_path / "feature_names.json", "w") as f:
            json.dump(feat_names, f, indent=2)

        metrics = {
            "roc_auc_test":       round(auc, 4),
            "roc_auc_cv_mean":    round(float(cv_scores.mean()), 4),
            "roc_auc_cv_std":     round(float(cv_scores.std()), 4),
            "n_train":            int(X_bal.shape[0]),
            "n_test":             int(X_test_proc.shape[0]),
            "n_features":         len(feat_names),
            "decision_threshold": cfg["model"]["decision_threshold"],
            "timestamp":          datetime.now().isoformat(),
        }
        with open(artifacts_path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # ── Resumen final ─────────────────────────────────
        run_id = mlflow.active_run().info.run_id  # type: ignore[union-attr]
        print("\n" + "═" * 58)
        print("  RESULTADOS")
        print("═" * 58)
        print(f"  ROC-AUC Test  : {auc:.4f}")
        print(f"  CV medio      : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  MLflow Run    : {run_id}")
        print("═" * 58 + "\n")
        print(classification_report(y_test, y_pred,
              target_names=["Pagador", "Moroso"]))


if __name__ == "__main__":
    ejecutar_pipeline()