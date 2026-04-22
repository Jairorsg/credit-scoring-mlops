# ── Base image ────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="Credit Scoring MLOps"
LABEL description="API de riesgo crediticio — LightGBM + FastAPI"

# ── Variables de entorno ──────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# ── Directorio de trabajo ─────────────────────────────────
WORKDIR /app

# ── Dependencias del sistema ──────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencias Python ───────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi \
        uvicorn[standard] \
        pydantic \
        lightgbm \
        scikit-learn \
        pandas \
        numpy \
        joblib \
        pyyaml \
        python-dotenv \
        shap

# ── Copiar artefactos del modelo ──────────────────────────
COPY config.yaml .
COPY model/artifacts/ ./model/artifacts/
COPY app/ ./app/

# ── Puerto ────────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Comando de inicio ─────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]