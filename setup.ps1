# MH26-01: Smart Admission Document Validation - Setup Script
# PowerShell script for Windows 10/11

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " MH26-01 · Smart Admission Document Validation Setup" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Create virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "[1/4] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    try {
        py -m venv .venv
    } catch {
        python -m venv .venv
    }
    if (-not (Test-Path ".venv")) {
        Write-Host "Trying python directly..." -ForegroundColor Yellow
        python -m venv .venv
    }
} else {
    Write-Host "[1/4] Virtual environment (.venv) already exists." -ForegroundColor Green
}

# 2. Activate virtual environment
Write-Host "[2/4] Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# 3. Install requirements
Write-Host "[3/4] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 4. Generate synthetic dataset
Write-Host "[4/4] Generating 90+ synthetic mark sheets, photos & ground truth..." -ForegroundColor Yellow
& .\.venv\Scripts\python.exe -m data_gen.generate

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host " Setup complete! Run .\run.ps1 to launch Streamlit UI" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

