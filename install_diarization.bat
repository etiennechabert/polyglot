@echo off
echo ================================================================================
echo Installing Speaker Diarization Dependencies
echo ================================================================================
echo.

echo Installing all required dependencies...
pip install -r requirements.txt

echo.
echo Creating .env file from template...
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
