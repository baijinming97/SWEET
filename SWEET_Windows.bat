@echo off
chcp 65001 >nul 2>&1
title SWEET - SAM Widget for Edge Evaluation Tool

echo.
echo ============================================================
echo                SWEET - Windows Edition
echo      SAM Widget for Edge Evaluation Tool
echo ============================================================
echo.

cd /d "%~dp0"

set "VENV_PYTHON=python\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo ERROR: Python environment not found!
    echo.
    echo Please run setup first:
    echo    python install
    echo.
    pause
    exit /b 1
)

echo Platform: Windows
echo.
echo Testing Python environment...
"%VENV_PYTHON%" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Environment: OK')" 2>nul
if errorlevel 1 (
    echo Environment test failed - some packages may be missing
    echo SWEET will still attempt to run...
)
echo.

echo Please select language:
echo.
echo [1] English
echo [2] Chinese
echo.
set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo Starting SWEET in English...
    "%VENV_PYTHON%" src\sam_annotator_english.py
) else if "%choice%"=="2" (
    echo.
    echo Starting SWEET in Chinese...
    "%VENV_PYTHON%" src\sam_annotator_chinese.py
) else (
    echo.
    echo Invalid choice. Defaulting to English...
    "%VENV_PYTHON%" src\sam_annotator_english.py
)

echo.
echo Check logs\sam_annotator.log for detailed performance analysis
echo.
pause