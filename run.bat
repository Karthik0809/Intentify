@echo off
echo Setting up AI Virtual Assistant...

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.8 or higher.
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install/upgrade pip
python -m pip install --upgrade pip

:: Install requirements
echo Installing dependencies...
pip install -r requirements.txt

:: Install PyAudio for Windows
echo Installing PyAudio...
pipwin install pyaudio

:: Create necessary directories
if not exist "models" mkdir models
if not exist "data" mkdir data

:: Check if model exists
if not exist "models\intent_classifier_best.pt" (
    echo Training model...
    python train.py
)

:: Start the application
echo Starting AI Virtual Assistant...
streamlit run main.py

:: Deactivate virtual environment on exit
call venv\Scripts\deactivate.bat

:: Run the batch file
.\run.bat