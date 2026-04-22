# Credit Scoring MLOps

**Modelamiento Predictivo de Riesgo Crediticio y Explicabilidad Financiera mediante Algoritmos de Ensamble (LightGBM) y Valores SHAP en Entornos de Clases Desbalanceadas**

---

## Arquitectura del pipeline

```
Fuentes de datos           Orquestador (Airflow / GitHub Actions)
─────────────────   ┌──────────────────────────────────────────────────┐
                    │                                                  │
PostgreSQL · CSV ───┤→  SMOTE  │  LightGBM  │  SHAP                  │
                    │          Pipeline de entrenamiento               │
APIs externas    ───┤                                                  │
                    │→  MLflow Registry  (Versioning · Staging → Prod) │
                    │                                                  │
                    │→  FastAPI + Docker  (/predict · /explain)        │
                    └──────────────────────────────────────────────────┘
                                         │
                            Analistas · Core bancario
```

## Stack tecnológico

| Capa               | Herramienta              | Propósito                          |
|--------------------|--------------------------|------------------------------------|
| ML Core            | LightGBM 4.3             | Modelo de clasificación            |
| Desbalance         | imbalanced-learn (SMOTE) | Oversampling de morosos            |
| Explicabilidad     | SHAP 0.45                | Valores SHAP por cliente           |
| Experiment Tracking| MLflow 2.13              | Versioning y registro de modelos   |
| API                | FastAPI + Uvicorn        | Servicio de predicciones           |
| Contenedor         | Docker                   | Despliegue reproducible            |
| Orquestación       | Airflow / GitHub Actions | Reentrenamiento automatizado       |

## Estructura del proyecto

```
credit_scoring_pro/
├── data/
│   ├── raw/                 ← application_train.csv (no versionar)
│   ├── processed/           ← features preprocesadas (.parquet)
│   └── interim/             ← pasos intermedios
├── notebooks/
│   ├── 01_exploracion.ipynb ← EDA y desbalance de clases
│   ├── 02_features.ipynb    ← Ratios financieros y feature engineering
│   └── 03_modelo_lgbm.ipynb ← SMOTE + LightGBM + SHAP + exportar
├── model/
│   ├── artifacts/           ← lgbm_model.joblib, preprocessor.joblib
│   └── reports/             ← curvas ROC, gráficas SHAP
├── app/
│   ├── main.py              ← Servidor FastAPI
│   ├── predictor.py         ← Carga del modelo y lógica de predicción
│   └── schemas.py           ← Validación de inputs con Pydantic
├── pipelines/
│   └── train_pipeline.py    ← Script automatizado de reentrenamiento
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── config.yaml              ← Parámetros centrales del proyecto
├── requirements.txt
└── setup_project.py         ← Setup automatizado
```

## Instalación

```bash
# 1. Clonar / abrir carpeta
cd credit_scoring_pro

# 2. Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate          # Mac/Linux
# .\venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar entorno
python setup_project.py --verify
```

## Inicio rápido

```bash
# Abrir laboratorio Jupyter
jupyter lab

# Ejecutar pipeline de entrenamiento completo
python pipelines/train_pipeline.py

# Iniciar API
uvicorn app.main:app --reload

# Ver experimentos MLflow
mlflow ui
```

## Fases del proyecto

| Fase | Estado | Descripción |
|------|--------|-------------|
| 1 — Mesa de trabajo  | ✅ | Entorno, carpetas, dependencias |
| 2 — Laboratorio      | 🔄 | Notebooks EDA, features, modelo |
| 3 — Ingeniería de SW | ⏳ | FastAPI: schemas, predictor, server |
| 4 — MLOps            | ⏳ | Pipeline automatizado + MLflow |

## Feature engineering (ratios financieros)

El modelo incorpora ratios derivados para capturar la capacidad de pago real:

- **PAYMENT_RATE** = Cuota anual / Ingreso total (carga del servicio de deuda)
- **CREDIT_INCOME_RATIO** = Monto del préstamo / Ingreso anual
- **CREDIT_GOODS_RATIO** = Monto del préstamo / Valor del bien financiado
- **ANNUITY_CREDIT_RATIO** = Cuota / Préstamo (proxy inverso del plazo)

---

*Proyecto académico — datos sintéticos / anonimizados. No usar en producción sin validación regulatoria.*