[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$url = 'http://127.0.0.1:5002'
try {
    $health = Invoke-RestMethod "$url/health" -TimeoutSec 2
    if ($health.ok -and $health.service -eq 'jepa-demo') {
        Write-Host "JEPA demo is already running: $url"
        return
    }
} catch { }
$python = (Get-Command python -ErrorAction Stop).Source
& $python -c 'import cv2, numpy, flask'
if ($LASTEXITCODE -ne 0) {
    throw 'The existing Python runtime must have cv2, numpy, and Flask. This launcher does not install dependencies.'
}
Write-Host "JEPA demo: $url"
Write-Host 'This is a localhost-only demo. Keep this terminal open; press Ctrl+C to stop.'
Push-Location -LiteralPath $root
try { & $python -u (Join-Path $root 'code\demo_app.py') } finally { Pop-Location }