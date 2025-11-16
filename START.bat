@echo off
echo ============================================================
echo Polyglot - Live Multi-Language Translator
echo ============================================================
echo.

REM Check if .env file exists
if not exist .env (
    echo WARNING: .env file not found!
    echo.
    echo Creating .env from template...
    if exist .env.example (
        copy .env.example .env
        echo .env file created.
        echo.
        echo IMPORTANT: Edit .env and add your HuggingFace token!
        echo Press any key to open .env in notepad...
        pause >nul
        notepad .env
    ) else (
        echo ERROR: .env.example not found!
        echo Please run INSTALL.bat first.
        pause
        exit /b 1
    )
)

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
echo Starting Cloudflare Tunnel...
echo.

REM Check if cloudflared tunnel service is installed
sc query cloudflared >nul 2>&1
if %errorlevel% neq 0 (
    echo Cloudflare tunnel service not found. Installing...
    for /f "tokens=3" %%a in ('findstr /C:"CLOUDFLARE_TUNNEL_TOKEN" .env') do set TUNNEL_TOKEN=%%a
    if defined TUNNEL_TOKEN (
        cloudflared.exe service install %TUNNEL_TOKEN%
        echo Cloudflare tunnel service installed successfully.
    ) else (
        echo WARNING: CLOUDFLARE_TUNNEL_TOKEN not found in .env file.
        echo Skipping Cloudflare tunnel setup.
    )
) else (
    echo Cloudflare tunnel service already installed.
)

echo.
echo Starting Cloudflare tunnel service...
net start cloudflared 2>nul
if %errorlevel% equ 0 (
    echo Cloudflare tunnel started successfully.
) else (
    echo Cloudflare tunnel already running or failed to start.
)

echo.
echo Starting application...
echo.

REM Open browser after a short delay to let the server start
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5000"

python app.py

pause
