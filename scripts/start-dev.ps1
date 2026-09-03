# Start JARVIS Core with auto-reload for local development (Windows).
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

# Load .env into the process environment. Without this, editing .env alone
# never changes what --host/--port below actually bind to: the app's own
# Settings object reads .env directly (via pydantic-settings) and logs the
# right value, but that log line is informational only - it does not control
# the uvicorn socket. A real bug found live: SERVER_HOST=0.0.0.0 in .env,
# yet uvicorn's own startup banner still printed "http://127.0.0.1:8000"
# because $env:SERVER_HOST was never set here, so it silently fell back to
# the loopback-only default and the phone could never reach the server.
Get-Content ".env" | ForEach-Object {
    if ($_ -match '^\s*([^#=\s][^=]*)\s*=\s*(.*)\s*$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
    }
}

$hostAddr = if ($env:SERVER_HOST) { $env:SERVER_HOST } else { "127.0.0.1" }
$port = if ($env:SERVER_PORT) { $env:SERVER_PORT } else { "8000" }

uvicorn app.main:app --reload --host $hostAddr --port $port
