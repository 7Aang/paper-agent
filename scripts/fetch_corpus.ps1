$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$corpusDir = Join-Path $projectRoot 'data/corpus'
New-Item -ItemType Directory -Force -Path $corpusDir | Out-Null
$papers = Get-Content -Raw -LiteralPath (Join-Path $projectRoot 'data_manifest.json') | ConvertFrom-Json
foreach ($paper in $papers) {
    $destination = Join-Path $corpusDir ($paper.id + '.pdf')
    if (-not (Test-Path -LiteralPath $destination)) {
        Invoke-WebRequest -Uri ('https://arxiv.org/pdf/' + $paper.id) -OutFile $destination -TimeoutSec 90
        Start-Sleep -Milliseconds 3100
    }
    Write-Output ($paper.id + ' downloaded')
}
python (Join-Path $PSScriptRoot 'fetch_corpus.py')
