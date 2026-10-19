# One-command local setup + run (Windows PowerShell). Usage: .\run_local.ps1
$ErrorActionPreference = "Stop"
if (-not (Test-Path .venv)) { py -3.12 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -q -r backend\requirements.txt
if (-not (Test-Path backend\models\model_bundle.joblib)) { .\.venv\Scripts\python.exe ml\train.py }
if (-not (Test-Path frontend\node_modules)) { Push-Location frontend; npm ci; Pop-Location }
Start-Process powershell -ArgumentList "-NoExit","-Command","cd '$PWD'; .\.venv\Scripts\python.exe backend\app.py"
Start-Process powershell -ArgumentList "-NoExit","-Command","cd '$PWD\frontend'; npm start"
Write-Host "Flask: http://127.0.0.1:5000/health   App: http://localhost:4200"
