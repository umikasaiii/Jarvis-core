# Start JARVIS Core in production-like mode (no reload) (Windows).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& ".venv\Scripts\Activate.ps1"
pip install --quiet -r requirements.txt

if (-not (Test-Path ".env")) {
    Write-Host "No .env found, copying .env.example -> .env"
    Copy-Item ".env.example" ".env"
}

$hostAddr = if ($env:SERVER_HOST) { $env:SERVER_HOST } else { "127.0.0.1" }
$port = if ($env:SERVER_PORT) { $env:SERVER_PORT } else { "8000" }

uvicorn app.main:app --host $hostAddr --port $port --workers 1
