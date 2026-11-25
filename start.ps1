# Starts the vision-tagger web UI on Windows.
# Usage (from the repository folder):  powershell -ExecutionPolicy Bypass -File .\start.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed. Install it with:" -ForegroundColor Yellow
    Write-Host '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    exit 1
}

uv sync --extra gpu
if (-not (Test-Path "config.toml")) {
    uv run vision-tagger init
    Write-Host "Created config.toml - adjust image_roots / model settings if needed." -ForegroundColor Green
}
uv run vision-tagger serve --open
