@echo off
REM =============================================================================
REM  setup_envs.bat — Create both virtual environments for Trismegisto (Windows)
REM =============================================================================
REM  Requires Python 3.9 and Python 3.12 installed and on PATH.
REM  Run from the repo root:  environments\setup_envs.bat
REM =============================================================================

echo ========================================
echo  Trismegisto - Environment Setup
echo ========================================

SET ROOT=%~dp0..

REM ---------------------------------------------------------------------------
REM 1. Python 3.9 environment (pyradiomics)
REM ---------------------------------------------------------------------------
echo.
echo [1/2] Creating Python 3.9 environment for pyradiomics...

py -3.9 -m venv "%ROOT%\.venv-py39"
CALL "%ROOT%\.venv-py39\Scripts\activate.bat"

echo   Installing pyradiomics dependencies (order matters)...
pip install --upgrade pip --quiet
pip install pydicom ipykernel pyradiomics --quiet
pip install SimpleITK --quiet
pip install "opencv-python==4.9.0.80" --quiet
pip install "numpy<2.0.0" --quiet
pip install GDCM --quiet
pip install pylibjpeg pylibjpeg-libjpeg "numpy==1.26.4" --quiet
pip install "pandas==2.2.1" --quiet

CALL deactivate
echo   OK  .venv-py39 ready

REM ---------------------------------------------------------------------------
REM 2. Python 3.12 environment (features extraction)
REM ---------------------------------------------------------------------------
echo.
echo [2/2] Creating Python 3.12 environment for features extraction...

py -3.12 -m venv "%ROOT%\.venv-py312"
CALL "%ROOT%\.venv-py312\Scripts\activate.bat"

pip install --upgrade pip --quiet
pip install -r "%ROOT%\environments\requirements_py312.txt" --quiet

CALL deactivate
echo   OK  .venv-py312 ready

echo.
echo ========================================
echo  Setup complete!
echo.
echo  Activate environments:
echo    Pyradiomics  -^>  .venv-py39\Scripts\activate
echo    Features     -^>  .venv-py312\Scripts\activate
echo ========================================
pause
