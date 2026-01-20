@echo off
REM YoloGen Equipment Manuals - Setup Script
REM Run this after placing your PDF files in data\manuals\

echo ============================================================
echo   YoloGen Equipment Manuals Setup
echo ============================================================
echo.

REM Check if Ollama is running
echo Checking Ollama connection...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo ERROR: Ollama is not running!
    echo Please start Ollama first: ollama serve
    pause
    exit /b 1
)
echo OK - Ollama is running

REM Check for PDFs
echo.
echo Checking for PDF files in data\manuals\...
dir /b data\manuals\*.pdf >nul 2>&1
if errorlevel 1 (
    echo ERROR: No PDF files found in data\manuals\
    echo Please add your PDF manuals to that folder
    pause
    exit /b 1
)
for /f %%a in ('dir /b data\manuals\*.pdf 2^>nul ^| find /c /v ""') do set PDF_COUNT=%%a
echo Found %PDF_COUNT% PDF files

REM Step 1: Extract images
echo.
echo ============================================================
echo Step 1: Extracting images from PDFs...
echo ============================================================
python scripts\extract_pdf_images.py --pdf-dir data\manuals --output-dir data\processed --min-size 150
if errorlevel 1 (
    echo ERROR: Failed to extract images
    pause
    exit /b 1
)

REM Clean existing labels
echo.
echo Cleaning any existing label files...
del /q data\processed\labels\train\* 2>nul
del /q data\processed\labels\val\* 2>nul

REM Step 2: Auto-label training images
echo.
echo ============================================================
echo Step 2: Auto-labeling training images with LLaVA...
echo This may take 30-60 minutes depending on image count
echo ============================================================
python scripts\auto_label_images.py --images-dir data\processed\images\train --model "hf.co/cjpais/llava-1.6-mistral-7b-gguf:Q4_K_M"
if errorlevel 1 (
    echo ERROR: Failed to label training images
    pause
    exit /b 1
)

REM Step 3: Auto-label validation images
echo.
echo ============================================================
echo Step 3: Auto-labeling validation images with LLaVA...
echo ============================================================
python scripts\auto_label_images.py --images-dir data\processed\images\val --model "hf.co/cjpais/llava-1.6-mistral-7b-gguf:Q4_K_M"
if errorlevel 1 (
    echo ERROR: Failed to label validation images
    pause
    exit /b 1
)

REM Summary
echo.
echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo Your data is ready in: data\processed\
echo.
echo Next steps:
echo   1. Verify labels: python -c "from pathlib import Path; from collections import Counter; ..."
echo   2. Train YOLO: python train.py --config configs\equipment_manuals.yaml
echo.
pause
