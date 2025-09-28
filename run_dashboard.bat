@echo off
echo Starting FloatChat Dashboard...
echo.
echo Using Python path: C:/Users/Jarvis/AppData/Local/Microsoft/WindowsApps/python3.11.exe
echo Dashboard will be available at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard
echo.

C:/Users/Jarvis/AppData/Local/Microsoft/WindowsApps/python3.11.exe -m streamlit run dashboard.py

pause
