@echo off
echo ============================================================
echo Live Multi-Language Translator
echo ============================================================
echo.
echo Starting application...
echo.

REM Open browser after a short delay to let the server start
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5000"

python app.py

pause
