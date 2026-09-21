param([switch]$DeepSeek, [int]$Port = 8765)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { $python = 'python' }
if ($DeepSeek) {
    $secret = Read-Host 'DeepSeek API Key (not saved)' -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
    try { $env:PAPER_AGENT_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
    $env:PAPER_AGENT_BASE_URL = 'https://api.deepseek.com'
    $env:PAPER_AGENT_MODEL = 'deepseek-flash'
}
try { & $python -m paper_agent serve --port $Port }
finally { if ($DeepSeek) { Remove-Item Env:PAPER_AGENT_API_KEY -ErrorAction SilentlyContinue } }
