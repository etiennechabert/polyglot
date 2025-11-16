@echo off
echo ================================================================================
echo Polyglot Installation Script
echo ================================================================================
echo.
echo This script will install all dependencies in the correct order to avoid
echo dependency resolution issues.
echo.
pause

echo.
echo ================================================================================
echo Step 1/4: Installing PyTorch (stable 2.9.1 with CUDA support)
echo ================================================================================
echo.
echo This may take a few minutes...
echo Installing torch 2.9.1 and torchaudio 2.9.1...
pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ERROR: PyTorch installation failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Step 2/4: Installing core dependencies
echo ================================================================================
echo.
pip install flask==3.0.0 flask-socketio==5.3.5 python-socketio==5.10.0
if errorlevel 1 (
    echo ERROR: Core dependencies installation failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Step 3/4: Installing audio and ML libraries
echo ================================================================================
echo.
pip install sounddevice==0.4.6 pyaudiowpatch==0.2.12.7 resampy==0.4.3 soundfile==0.12.1
pip install "numpy<2.0.0" langdetect python-dotenv
pip install transformers accelerate sentencepiece protobuf
if errorlevel 1 (
    echo ERROR: Audio/ML libraries installation failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Step 4/4: Installing speaker diarization
echo ================================================================================
echo.
pip install pyannote.audio pyannote.core
if errorlevel 1 (
    echo ERROR: Speaker diarization installation failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Creating .env file
echo ================================================================================
echo.
if not exist .env (
    copy .env.example .env
    echo .env file created! Please edit it with your HuggingFace token.
) else (
    echo .env file already exists, skipping...
)

echo.
echo ================================================================================
echo Installation Complete!
echo ================================================================================
echo.
echo Next steps:
echo 1. Get a HuggingFace token at: https://huggingface.co/settings/tokens
echo 2. Accept model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1
echo 3. Edit .env file and add your token: HF_TOKEN=your_token_here
echo 4. Run the app: python app.py
echo.
echo See QUICK_SETUP.md for detailed instructions
echo.
pause
