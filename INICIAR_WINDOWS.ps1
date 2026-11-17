$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    Write-Host 'Criando ambiente virtual Python...' -ForegroundColor Cyan
    python -m venv .venv
}
Write-Host 'Instalando dependencias...' -ForegroundColor Cyan
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar requirements.txt' }
& $py -m pip install -e .
if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar pacote' }
if (-not (Test-Path 'data\processed\config.json')) {
    if (-not (Test-Path 'data\raw\olist_order_reviews_dataset.csv')) {
        Write-Host 'Baixando base Olist...' -ForegroundColor Cyan
        & $py -m voc_rag.cli download
        if ($LASTEXITCODE -ne 0) { throw 'Falha ao baixar base' }
    }
    Write-Host 'Gerando indice (somente na primeira execucao)...' -ForegroundColor Cyan
    & $py -m voc_rag.cli build --backend lite
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao gerar indice' }
}
Write-Host 'Iniciando VoxInsight...' -ForegroundColor Green
& $py -m streamlit run app.py
