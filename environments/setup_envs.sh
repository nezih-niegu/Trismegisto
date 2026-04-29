#!/usr/bin/env bash
# =============================================================================
# setup_envs.sh — Create both virtual environments for Trismegisto
# =============================================================================
# Usage:
#   bash setup_envs.sh
#
# Requires Python 3.9 and Python 3.12 to be installed on the system.
# On Linux/macOS you can install them via pyenv, conda, or your package manager.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo " Trismegisto — Environment Setup"
echo "========================================"

# ---------------------------------------------------------------------------
# 1. Python 3.9 environment (pyradiomics)
# ---------------------------------------------------------------------------
echo ""
echo "[1/2] Creating Python 3.9 environment for pyradiomics..."

PY39=$(command -v python3.9 || command -v python3.9 2>/dev/null || echo "")
if [ -z "$PY39" ]; then
    echo "ERROR: python3.9 not found. Install it via pyenv or your package manager."
    exit 1
fi

python3.9 -m venv "$ROOT_DIR/.venv-py39"
source "$ROOT_DIR/.venv-py39/bin/activate"

echo "  Installing pyradiomics dependencies (order matters)..."
pip install --upgrade pip --quiet

# Install in the exact order required to avoid breakage
pip install pydicom ipykernel pyradiomics --quiet
pip install SimpleITK --quiet
pip install "opencv-python==4.9.0.80" --quiet
pip install "numpy<2.0.0" --quiet
pip install GDCM --quiet
pip install pylibjpeg pylibjpeg-libjpeg "numpy==1.26.4" --quiet
pip install "pandas==2.2.1" --quiet

deactivate
echo "  ✓  .venv-py39 ready"

# ---------------------------------------------------------------------------
# 2. Python 3.12 environment (features extraction)
# ---------------------------------------------------------------------------
echo ""
echo "[2/2] Creating Python 3.12 environment for features extraction..."

PY312=$(command -v python3.12 || command -v python3.12 2>/dev/null || echo "")
if [ -z "$PY312" ]; then
    echo "ERROR: python3.12 not found. Install it via pyenv or your package manager."
    exit 1
fi

python3.12 -m venv "$ROOT_DIR/.venv-py312"
source "$ROOT_DIR/.venv-py312/bin/activate"

pip install --upgrade pip --quiet
pip install -r "$ROOT_DIR/environments/requirements_py312.txt" --quiet

deactivate
echo "  ✓  .venv-py312 ready"

# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo " Setup complete!"
echo ""
echo " Activate environments:"
echo "   Pyradiomics  →  source .venv-py39/bin/activate"
echo "   Features     →  source .venv-py312/bin/activate"
echo "========================================"
