@echo off
echo ============================================================
echo Live Multi-Language Translator
echo ============================================================
echo.
echo Verifying GPU setup...
echo.

REM Verify CUDA/GPU is properly configured
python verify_gpu.py
if errorlevel 1 (
    echo.
    echo GPU verification failed! Please fix the issues above.
    pause
    exit /b 1
)

echo.
echo Starting application...
echo.

REM Open browser after a short delay to let the server start
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5000"

python app.py

pause
