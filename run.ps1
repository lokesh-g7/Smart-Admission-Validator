# MH26-01: Smart Admission Document Validation - Launch Script
# PowerShell script for Windows 10/11

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Launching Smart Admission Document Validation UI..." -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & .\.venv\Scripts\Activate.ps1
    & .\.venv\Scripts\python.exe -m streamlit run app.py
} else {
    Write-Host "Virtual environment not found! Running setup.ps1 first..." -ForegroundColor Yellow
    & .\setup.ps1
    & .\.venv\Scripts\python.exe -m streamlit run app.py
}

