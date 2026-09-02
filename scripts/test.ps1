# Run the test suite (Windows).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& ".venv\Scripts\Activate.ps1"
pip install --quiet -r requirements.txt

pytest @args
