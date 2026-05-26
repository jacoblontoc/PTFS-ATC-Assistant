@echo off
setlocal

echo ============================================================
echo  PTFS ATC Assistant - First-time setup
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not on PATH.
    echo         Download from https://www.python.org/downloads/
    pause & exit /b 1
)

echo [1/3] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed.
    pause & exit /b 1
)

echo.
echo [2/3] Checking Ollama...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama not found on PATH.
    echo           Download and install from https://ollama.ai
    echo           Then run this script again, or manually run:
    echo             ollama pull llama3.2:3b
) else (
    echo [3/3] Pulling llama3.2:3b model (this may take a few minutes)...
    ollama pull llama3.2:3b
)

echo.
echo ============================================================
echo  Setup complete!
echo  Start the assistant with:  python main.py
echo ============================================================
echo.
pause
