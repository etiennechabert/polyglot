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
echo Step 1/4: Installing PyTorch (2.8.0 with CUDA 12.9 support)
echo ================================================================================
echo.
echo This may take a few minutes...
echo Installing torch 2.8.0+cu129 and torchaudio 2.8.0+cu129...
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
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
pip install sounddevice==0.4.6 pyaudiowpatch==0.2.12.7 resampy==0.4.3 soundfile==0.13.1
pip install numpy==2.3.4 langdetect==1.0.9 python-dotenv==1.2.1
pip install transformers==4.57.1 accelerate==1.11.0 sentencepiece==0.2.1 protobuf==6.33.1
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
pip install pyannote.audio==4.0.1 pyannote.core==6.0.1
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
