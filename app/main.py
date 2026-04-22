from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time

from app.schemas   import SolicitudCredito, RespuestaCredito
from app.predictor import predecir, metrics, MODELO_VERSION

# ── Metadata de la API ────────────────────────────────────
app = FastAPI(
    title="Credit Scoring API",
    description=(
        "API de riesgo crediticio — LightGBM + SHAP\n\n"
        "**Modelo:** LightGBM con SMOTE para clases desbalanceadas\n\n"
        f"**ROC-AUC Test:** {metrics['roc_auc_test']}  |  "
        f"**CV medio:** {metrics['roc_auc_cv_mean']} ± {metrics['roc_auc_cv_std']}"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Status"])
def root():
    return {
        "api":     "Credit Scoring MLOps",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Status"])
def health():
    return {
        "status":          "ok",
        "modelo_version":  MODELO_VERSION,
        "roc_auc_test":    metrics["roc_auc_test"],
        "roc_auc_cv_mean": metrics["roc_auc_cv_mean"],
    }


@app.post("/predict", response_model=RespuestaCredito, tags=["Predicción"])
def predict(solicitud: SolicitudCredito):
    try:
        t0      = time.time()
        result  = predecir(solicitud.model_dump())
        latency = round((time.time() - t0) * 1000, 2)
        result["latencia_ms"] = latency  # type: ignore
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))