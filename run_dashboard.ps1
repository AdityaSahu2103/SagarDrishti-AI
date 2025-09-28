# FloatChat Dashboard Launcher
# Uses the full Python path to ensure compatibility

Write-Host "Starting FloatChat Dashboard..." -ForegroundColor Green
Write-Host ""
Write-Host "Using Python path: C:/Users/Jarvis/AppData/Local/Microsoft/WindowsApps/python3.11.exe" -ForegroundColor Yellow
Write-Host "Dashboard will be available at: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the dashboard" -ForegroundColor Red
Write-Host ""

# Activate virtual environment if it exists
if (Test-Path ".venv/Scripts/Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .venv/Scripts/Activate.ps1
}

# Run the dashboard with full Python path
& "C:/Users/Jarvis/AppData/Local/Microsoft/WindowsApps/python3.11.exe" -m streamlit run dashboard.py

Write-Host ""
Write-Host "Dashboard stopped." -ForegroundColor Green
