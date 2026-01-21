@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  VLM YOLO Detector - Automated Installation Script
echo ============================================================
echo.

:: Change to the script's directory
cd /d "%~dp0"

:: Check for Python
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%
echo.

:: Check for Ollama (required for VLM descriptions)
echo [2/6] Checking Ollama installation...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Ollama is not installed or not in PATH.
    echo Ollama is required for generating VLM image descriptions.
    echo Please install Ollama from https://ollama.com/download
    echo.
    echo Continuing installation, but VLM features will not work until Ollama is installed.
    echo.
) else (
    for /f "tokens=4" %%i in ('ollama --version 2^>^&1') do set OLLAMA_VERSION=%%i
    echo Found Ollama !OLLAMA_VERSION!
)
echo.

:: Install uv package manager
echo [3/6] Installing uv package manager...
pip install uv --quiet
if errorlevel 1 (
    echo ERROR: Failed to install uv package manager.
    pause
    exit /b 1
)
echo uv installed successfully.
echo.

:: Install Python dependencies
echo [4/6] Installing Python dependencies...
uv sync
if errorlevel 1 (
    echo uv sync failed. Trying pip install...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install Python dependencies.
        pause
        exit /b 1
    )
)

:: Install additional required packages
echo Installing additional packages...
pip install pymupdf sentence-transformers --quiet
if errorlevel 1 (
    echo WARNING: Some optional packages failed to install.
)
echo Python dependencies installed successfully.
echo.

:: Pull Ollama VLM model (optional but recommended)
echo [5/6] Setting up Ollama VLM model...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo Skipping VLM model pull (Ollama not installed).
) else (
    echo Checking if Ollama is running...
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if errorlevel 1 (
        echo WARNING: Ollama is not running. Start it with: ollama serve
        echo Skipping VLM model pull.
    ) else (
        echo Pulling LLaVA 13B model for VLM descriptions (this may take a while)...
        ollama pull llava:13b
        if errorlevel 1 (
            echo WARNING: Failed to pull LLaVA model. You can pull it manually later.
            echo Run: ollama pull llava:13b
        ) else (
            echo LLaVA model pulled successfully.
        )
    )
)
echo.

:: Check for PDF manuals and process if available
echo [6/6] Checking for PDF manuals...
echo.
dir /b data\manuals\*.pdf >nul 2>&1
if errorlevel 1 (
    echo No PDF files found in data\manuals\
    echo.
    echo To set up image data:
    echo   1. Place your PDF manuals in data\manuals\
    echo   2. Run the processing commands below
    echo.
) else (
    for /f %%a in ('dir /b data\manuals\*.pdf 2^>nul ^| find /c /v ""') do set PDF_COUNT=%%a
    echo Found !PDF_COUNT! PDF files in data\manuals\
    echo.
    
    :: Check if embeddings already exist
    if exist "data\processed\image_embeddings.npy" (
        echo Image embeddings already exist. Skipping processing.
        echo To regenerate, delete data\processed\ and run the processing commands manually.
    ) else (
        echo Starting automatic image extraction and processing...
        echo This may take 30-60 minutes depending on the number of PDFs.
        echo.
        
        choice /C YN /M "Do you want to process PDFs now"
        if errorlevel 2 (
            echo Skipping PDF processing. You can run it manually later.
        ) else (
            echo.
            echo Step 1: Extracting images from PDFs...
            python scripts\extract_all_images.py --pdf-dir data\manuals --output-dir data\processed --min-size 150
            if errorlevel 1 (
                echo ERROR: Failed to extract images.
                echo You can try manually: python scripts\extract_all_images.py --pdf-dir data\manuals --output-dir data\processed
                goto :skip_processing
            )
            
            echo.
            echo Step 2: Generating VLM descriptions...
            curl -s http://localhost:11434/api/tags >nul 2>&1
            if errorlevel 1 (
                echo WARNING: Ollama not running. Skipping VLM descriptions.
                echo Start Ollama and run: python scripts\describe_images_vlm.py --index data\processed\image_index.json
            ) else (
                python scripts\describe_images_vlm.py --index data\processed\image_index.json
                if errorlevel 1 (
                    echo WARNING: VLM description generation had issues.
                )
            )
            
            echo.
            echo Step 3: Building semantic embeddings...
            python scripts\build_embeddings.py
            if errorlevel 1 (
                echo ERROR: Failed to build embeddings.
                echo You can try manually: python scripts\build_embeddings.py
            )
        )
    )
)

:skip_processing
echo.
echo ============================================================
echo  Verifying Installation
echo ============================================================
echo.

echo Checking for required data files...
if exist "data\processed\image_index.json" (
    echo   [OK] image_index.json
) else (
    echo   [MISSING] image_index.json
)

if exist "data\processed\image_embeddings.npy" (
    echo   [OK] image_embeddings.npy
) else (
    echo   [MISSING] image_embeddings.npy
)

if exist "data\processed\embedding_mapping.json" (
    echo   [OK] embedding_mapping.json
) else (
    echo   [MISSING] embedding_mapping.json
)
echo.

echo ============================================================
echo  Installation Complete!
echo ============================================================
echo.
echo This repository provides image data for the agentic-rag system.
echo.
echo Directory structure:
echo   data\manuals\          - Place PDF manuals here
echo   data\processed\        - Processed images and embeddings
echo   scripts\               - Processing scripts
echo.
echo Manual processing commands:
echo   1. Extract images:
echo      python scripts\extract_all_images.py --pdf-dir data\manuals --output-dir data\processed
echo   2. Generate VLM descriptions (requires Ollama with llava:13b):
echo      python scripts\describe_images_vlm.py --index data\processed\image_index.json
echo   3. Build embeddings:
echo      python scripts\build_embeddings.py
echo.
echo For agentic-rag integration:
echo   Ensure this repository is cloned alongside agentic-rag in the same parent directory.
echo   The agentic-rag system will automatically find the embeddings at:
echo   ..\vlm-yolo-detector\data\processed\
echo.
pause
