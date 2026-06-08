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

import mlflow                          # type: ignore[import]
import mlflow.sklearn                  # type: ignore[import]  ← sklearn, no lightgbm
import lightgbm as lgb                 # type: ignore[import]

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

# ── CAMBIO 1: ImbPipeline en vez de sklearn Pipeline para el flujo completo ──
from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore[import]
from imblearn.over_sampling import SMOTE               # type: ignore[import]

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

    # "median" crea una copia completa del masked array para ordenarlo
    # → OOM en folds grandes (196k filas × 112 cols × 8 bytes ≈ 168 MiB por fold).
    # "mean" es O(n) sin copias extra y estadísticamente equivalente a esta escala.
    num_strategy = cfg["features"]["imputation"]["numeric_strategy"]
    if num_strategy == "median":
        num_strategy = "mean"

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=num_strategy))
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg["features"]["imputation"]["categorical_strategy"])),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def ejecutar_pipeline() -> None:
    print("\n" + "═" * 58)
    print("  CREDIT SCORING — PIPELINE DE ENTRENAMIENTO")
    print("═" * 58 + "\n")

    mlflow_uri = f"sqlite:///{BASE / 'mlruns.db'}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M')}"):

        # ── 1. Datos ──────────────────────────────────────
        df         = cargar_datos()
        X, y       = feature_engineering(df)
        feat_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=cfg["project"]["random_seed"],
            stratify=y
        )

        # ── 2. Construir ImbPipeline ───────────────────────
        # CAMBIO 2: SMOTE va DENTRO del pipeline, no antes del CV.
        # Así cada fold de CV aplica SMOTE solo en sus datos de
        # entrenamiento → CV honesto, sin data leakage.
        log("Construyendo ImbPipeline (preprocesador + SMOTE + modelo)...")
        params = cfg["model"]["params"].copy()
        params["random_state"] = cfg["project"]["random_seed"]

        full_pipeline = ImbPipeline([
            ("preprocessor", construir_preprocesador(X_train)),
            ("smote", SMOTE(
                sampling_strategy=cfg["smote"]["sampling_strategy"],
                k_neighbors=cfg["smote"]["k_neighbors"],
                random_state=cfg["smote"]["random_state"]
            )),
            ("model", lgb.LGBMClassifier(**params))
        ])

        # ── 3. CV honesto ──────────────────────────────────
        log("Validación cruzada (SMOTE dentro de cada fold)...")
        cv = StratifiedKFold(
            n_splits=cfg["model"]["cv_folds"],
            shuffle=True,
            random_state=cfg["project"]["random_seed"]
        )
        cv_scores: np.ndarray = cross_val_score(
            full_pipeline, X_train, y_train,
            cv=cv, scoring="roc_auc",
            n_jobs=1   # serial: evita multiplicar RAM × n_folds en paralelo
        )
        log(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # ── 4. Entrenamiento final ─────────────────────────
        log("Entrenando pipeline final sobre todo el train set...")
        full_pipeline.fit(X_train, y_train)
        log(f"Pipeline fit completo. Train original: {len(y_train):,} filas")

        # ── 5. Evaluación + threshold óptimo ──────────────
        log("Evaluando en test set...")
        y_proba: np.ndarray = full_pipeline.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_proba))

        # CAMBIO 3: threshold óptimo por F1 en vez del valor fijo
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        f1_arr = (2 * precisions[:-1] * recalls[:-1]
                  / (precisions[:-1] + recalls[:-1] + 1e-8))
        optimal_idx       = int(np.argmax(f1_arr))
        optimal_threshold = float(thresholds[optimal_idx])
        y_pred = (y_proba >= optimal_threshold).astype(int)

        log(f"Threshold óptimo (F1): {optimal_threshold:.4f}  "
            f"[default era {cfg['model']['decision_threshold']}]")

        # ── 6. Registrar en MLflow ────────────────────────
        log("Registrando en MLflow...")
        mlflow.log_params(params)
        mlflow.log_param("smote_ratio",       cfg["smote"]["sampling_strategy"])
        mlflow.log_param("cv_folds",          cfg["model"]["cv_folds"])
        mlflow.log_param("n_features",        len(feat_names))
        mlflow.log_param("threshold_default", cfg["model"]["decision_threshold"])
        mlflow.log_param("threshold_optimal", round(optimal_threshold, 4))

        mlflow.log_metric("roc_auc_test",    auc)
        mlflow.log_metric("roc_auc_cv_mean", float(cv_scores.mean()))
        mlflow.log_metric("roc_auc_cv_std",  float(cv_scores.std()))
        mlflow.log_metric("n_train",         float(len(y_train)))
        mlflow.log_metric("n_test",          float(len(y_test)))

        # CAMBIO 4: mlflow.sklearn en vez de mlflow.lightgbm,
        #           name= en vez del deprecado artifact_path=
        mlflow.sklearn.log_model(           # type: ignore[attr-defined]
            full_pipeline,
            name="model",
            registered_model_name=cfg["mlflow"]["model_name"]
        )

        # ── 7. Guardar artefactos ─────────────────────────
        log("Guardando artefactos...")
        artifacts_path = BASE / "model/artifacts"
        artifacts_path.mkdir(parents=True, exist_ok=True)

        # Un solo joblib con todo el pipeline (preprocesador + SMOTE + modelo)
        joblib.dump(full_pipeline, artifacts_path / "lgbm_pipeline.joblib")

        with open(artifacts_path / "feature_names.json", "w") as f:
            json.dump(feat_names, f, indent=2)

        metrics = {
            "roc_auc_test":       round(auc, 4),
            "roc_auc_cv_mean":    round(float(cv_scores.mean()), 4),
            "roc_auc_cv_std":     round(float(cv_scores.std()), 4),
            "n_train":            int(len(y_train)),
            "n_test":             int(len(y_test)),
            "n_features":         len(feat_names),
            "threshold_default":  cfg["model"]["decision_threshold"],
            "threshold_optimal":  round(optimal_threshold, 4),
            "timestamp":          datetime.now().isoformat(),
        }
        with open(artifacts_path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # ── Resumen final ─────────────────────────────────
        run_id = mlflow.active_run().info.run_id  # type: ignore[union-attr]
        print("\n" + "═" * 58)
        print("  RESULTADOS")
        print("═" * 58)
        print(f"  ROC-AUC Test       : {auc:.4f}")
        print(f"  CV medio (honesto) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Threshold óptimo   : {optimal_threshold:.4f}")
        print(f"  MLflow Run         : {run_id}")
        print("═" * 58 + "\n")
        print(classification_report(y_test, y_pred,
              target_names=["Pagador", "Moroso"]))


if __name__ == "__main__":
    ejecutar_pipeline()
