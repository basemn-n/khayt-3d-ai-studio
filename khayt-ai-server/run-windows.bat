@echo off
setlocal
cd /d %~dp0
if not exist .venv (
  echo [1/3] Creating Python environment...
  py -3 -m venv .venv
  if errorlevel 1 (echo Python 3 is required. Install Python 3.11+ and try again.& pause & exit /b 1)
)
call .venv\Scripts\activate.bat
if not exist .env (
  copy .env.example .env >nul
  echo.
  echo Created .env. Open it and add your OpenAI API key, then run this file again.
  start notepad .env
  pause
  exit /b 0
)
echo [2/3] Installing/updating dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (echo Dependency installation failed.& pause & exit /b 1)
echo [3/3] Starting Khayt 3D AI server...
start "Khayt 3D" http://127.0.0.1:8000
python -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
