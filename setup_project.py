"""
╔══════════════════════════════════════════════════════════════╗
║          Credit scoring MLOps — setup_project.py            ║
║   Crea la estructura completa del proyecto en un comando.   ║
╚══════════════════════════════════════════════════════════════╝

Uso:
    python setup_project.py              # Setup completo
    python setup_project.py --verify     # Solo verificar entorno
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Colores ANSI (funcionan en VS Code terminal y Linux/Mac)
class C:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    GRAY   = "\033[90m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

    @classmethod
    def ok(cls, msg):    return f"{cls.GREEN}✓{cls.RESET}  {msg}"
    @classmethod
    def warn(cls, msg):  return f"{cls.YELLOW}⚠{cls.RESET}  {msg}"
    @classmethod
    def info(cls, msg):  return f"{cls.CYAN}→{cls.RESET}  {msg}"
    @classmethod
    def dim(cls, msg):   return f"{cls.GRAY}{msg}{cls.RESET}"

# Estructura del proyecto (árbol completo)
PROJECT_TREE = {
    "data": {
        "raw":       None,
        "processed": None,
        "interim":   None,
    },
    "notebooks": None,
    "model": {
        "artifacts": None,
        "reports":   None,
    },
    "app": {
        "tests": None,
    },
    "pipelines":    None,
    "mlruns":       None,
    "tests":        None,
    ".github": {
        "workflows": None,
    },
    "docker":       None,
}


# Templates de archivos iniciales
GITIGNORE_CONTENT = """\
# Entornos virtuales ──────────────────────────────────────────────
venv/
.env
*.env
.env.*

# Python ──────────────────────────────────────────────────────────
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/

# Jupyter ─────────────────────────────────────────────────────────
.ipynb_checkpoints/

# Datos sensibles (NUNCA versionar datos personales) ──────────────
data/raw/
data/processed/
data/interim/

# Artefactos de modelos (pesados) ─────────────────────────────────
model/artifacts/*.joblib
model/artifacts/*.pkl

# MLflow ──────────────────────────────────────────────────────────
mlruns/

# IDEs ────────────────────────────────────────────────────────────
.vscode/settings.json
.idea/
*.swp

# Sistema operativo ───────────────────────────────────────────────
.DS_Store
Thumbs.db
"""

ENV_TEMPLATE = """\
# Credenciales de base de datos (NUNCA subir a Git) ──
DB_HOST=localhost
DB_PORT=5432
DB_NAME=credit_scoring
DB_USER=postgres
DB_PASSWORD=your_password_here

# MLflow ──
MLFLOW_TRACKING_URI=mlruns/

# API ──
API_SECRET_KEY=change_this_in_production
"""

NOTEBOOKS_PLAN = """\
# Plan de notebooks — credit_scoring_pro
# 01_exploracion.ipynb     → EDA, distribuciones, desbalance de clases
# 02_features.ipynb        → Feature engineering: ratios financieros
# 03_modelo_lgbm.ipynb     → SMOTE + LightGBM + SHAP + exportar .joblib
# Crear con: jupyter lab
"""

# Funciones
def print_header():
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    print(f"\n{C.BOLD}╔══════════════════════════════════════════════════════════╗")
    print( "║       Credit scoring MLOps — Project Setup               ║")
    print(f"║       {C.GRAY}{ts}{C.BOLD}                        ║")
    print( "╚══════════════════════════════════════════════════════════╝\n" + C.RESET)


def check_python_version() -> bool:
    major, minor = sys.version_info[:2]
    version_str = f"Python {major}.{minor}.{sys.version_info[2]}"
    if major < 3 or (major == 3 and minor < 10):
        print(C.warn(f"{version_str} — se recomienda 3.10+"))
        return False
    print(C.ok(version_str))
    return True


def check_venv() -> bool:
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    if in_venv:
        print(C.ok(f"Entorno virtual activo"))
        print(C.dim(f"   {sys.prefix}"))
    else:
        print(C.warn("No se detectó entorno virtual activo."))
        print(C.dim("   Ejecuta: python -m venv venv"))
        if sys.platform == "win32":
            print(C.dim("            .\\venv\\Scripts\\activate"))
        else:
            print(C.dim("            source venv/bin/activate"))
    return in_venv


def check_pip() -> bool:
    req = Path("requirements.txt")
    if req.exists():
        print(C.ok("requirements.txt encontrado"))
        return True
    print(C.warn("requirements.txt no encontrado en el directorio actual"))
    return False


def create_tree(base: Path, tree: dict, indent: int = 0, prefix: str = ""):
    """Crea carpetas recursivamente e imprime el árbol."""
    items = list(tree.items())
    for i, (name, children) in enumerate(items):
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")

        folder = base / name
        existed = folder.exists()
        folder.mkdir(parents=True, exist_ok=True)

        status = C.dim("ya existe") if existed else C.GREEN + "creado" + C.RESET
        print(f"  {prefix}{connector}{C.CYAN}{name}/{C.RESET}  {status}")

        if children is None:

            gitkeep = folder / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()
        else:
            create_tree(base, children, indent + 1, child_prefix)


def create_aux_files(base: Path):
    """Crea archivos auxiliares de configuración."""
    files = {
        ".gitignore":        GITIGNORE_CONTENT,
        ".env.example":      ENV_TEMPLATE,
        "notebooks/PLAN.md": NOTEBOOKS_PLAN,
    }
    for rel_path, content in files.items():
        path = base / rel_path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(C.ok(rel_path))
        else:
            print(C.dim(f"  · {rel_path}  ya existe"))


def print_next_steps(base: Path):
    print(f"\n{C.BOLD}Próximos pasos{C.RESET}\n")
    steps = [
        ("1", "Mueve tu CSV",       "data/raw/application_train.csv"),
        ("2", "Instala el stack",   "pip install -r requirements.txt"),
        ("3", "Abre el laboratorio","jupyter lab"),
        ("4", "Primer notebook",    "notebooks/01_exploracion.ipynb"),
    ]
    for num, action, detail in steps:
        print(f"  {C.CYAN}{num}.{C.RESET} {action}")
        print(C.dim(f"     {detail}"))
    print(f"\n{C.dim('  Proyecto en: ' + str(base.resolve()))}\n")

# Main
def main():
    parser = argparse.ArgumentParser(description="Setup del proyecto Credit scoring MLOps")
    parser.add_argument("--verify", action="store_true", help="Solo verificar entorno, no crear carpetas")
    args = parser.parse_args()

    print_header()
    base = Path(".")

    print(f"{C.BOLD}Verificando entorno{C.RESET}")
    py_ok    = check_python_version()
    venv_ok  = check_venv()
    check_pip()

    if args.verify:
        status = "OK" if (py_ok and venv_ok) else "REVISAR"
        print(f"\n  Estado del entorno: {C.BOLD}{status}{C.RESET}\n")
        return

    print(f"\n{C.BOLD}Creando estructura de carpetas {C.RESET}")
    create_tree(base, PROJECT_TREE)

    print(f"\n{C.BOLD}Archivos de configuración{C.RESET}")
    create_aux_files(base)

    print_next_steps(base)


if __name__ == "__main__":
    main()