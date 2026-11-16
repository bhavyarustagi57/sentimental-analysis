# Starts the FastAPI backend (port 8000) and the Vite frontend (port 5173) together.
# Usage (from the project folder):  .\run-dev.ps1
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$env:PYTHONUTF8 = "1"
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
  Write-Host "Missing .venv. Run: python -m venv .venv ; .\.venv\Scripts\python.exe -m pip install -r requirements.txt" -ForegroundColor Red
  exit 1
}
Start-Process powershell -ArgumentList "-NoExit","-Command","Set-Location '$root'; .\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000"
Start-Process powershell -ArgumentList "-NoExit","-Command","Set-Location '$root'; npm run dev"
Write-Host "Backend: http://localhost:8000/docs   Frontend: http://localhost:5173" -ForegroundColor Green
