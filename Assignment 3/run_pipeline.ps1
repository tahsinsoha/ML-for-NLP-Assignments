# Run the full MT pipeline
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "Step 1: Preprocessing"
python -u preprocessing.py
Write-Host ""

Write-Host "Step 2: Training (forward & reverse models)"
python -u training.py
Write-Host ""

Write-Host "Step 3: Viterbi alignment & symmetrization"
python -u alignment.py
Write-Host ""

Write-Host "Step 4: Phrase extraction & scoring"
python -u phrase_extraction.py
