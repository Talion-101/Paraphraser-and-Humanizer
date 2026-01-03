@echo off
REM Paraphraser & Humanizer - Quick Start Script for Windows

echo.
echo ╔════════════════════════════════════════════╗
echo ║                                            ║
echo ║   Paraphraser & Humanizer - Tkinter       ║
echo ║                                            ║
echo ╚════════════════════════════════════════════╝
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

REM Install NLTK if not already installed
echo Checking dependencies...
python -m pip install nltk -q >nul 2>&1

REM Download NLTK data if needed
echo Downloading language data (first run only)...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('omw-1.4', quiet=True)" >nul 2>&1

echo.
echo ╔════════════════════════════════════════════╗
echo ║     Starting Paraphraser Application...    ║
echo ╚════════════════════════════════════════════╝
echo.

REM Run the main application
python main.py

pause
