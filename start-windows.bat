@echo off
cd /d "%~dp0"
echo [1/2] Installing local AI dependency (first time may take a while)...
py -3 -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed. Check Python 3.10+ and your network connection.
  pause
  exit /b 1
)
echo [2/2] Starting daily reading buddy...
start "Reading Buddy Server" cmd /k "py -3 server.py"
timeout /t 2 /nobreak >nul
start "" "http://localhost:8765"
echo The reading buddy is running in the other window. Close that window to stop it.
