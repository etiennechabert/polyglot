# Manual Installation Guide

If the automated installer fails or you prefer manual installation, follow these steps **in order**.

## Why This Order Matters

PyTorch and its dependencies can cause pip to hang when resolving complex dependency trees. Installing in stages avoids this issue.

## Installation Steps

### Step 1: Install PyTorch (CUDA 12.9 for RTX 5080)

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu129
```

**Wait for this to complete before proceeding!**

### Step 2: Install Core Web Framework

```bash
pip install flask==3.0.0 flask-socketio==5.3.5 python-socketio==5.10.0
```

### Step 3: Install Audio Processing Libraries

```bash
pip install sounddevice==0.4.6 pyaudiowpatch==0.2.12.7 resampy==0.4.3 soundfile==0.12.1
```

### Step 4: Install Utilities

```bash
pip install "numpy<2.0.0" langdetect python-dotenv
```

### Step 5: Install Transformers and Dependencies

```bash
pip install transformers accelerate sentencepiece protobuf
```

### Step 6: Install Speaker Diarization

```bash
pip install pyannote.audio pyannote.core
```

### Step 7: Set Up Environment Variables

1. Copy the template:
   ```bash
   copy .env.example .env
   ```

2. Edit `.env` and add your HuggingFace token:
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxx
   ```

### Step 8: Get HuggingFace Token

1. Sign up: https://huggingface.co/join
2. Create token: https://huggingface.co/settings/tokens
3. Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1

### Step 9: Run the App

```bash
python app.py
```

## Troubleshooting

### "pip is taking forever / hanging"

**Problem**: pip is trying to resolve complex dependencies

**Solution**:
1. Kill pip (Ctrl+C)
2. Follow the manual steps above **in exact order**
3. Don't skip steps or install all at once

### "CUDA not available"

**Problem**: PyTorch CPU version installed instead of GPU version

**Solution**:
```bash
pip uninstall torch
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu129
```

### "Module not found: pyannote"

**Problem**: pyannote.audio not installed

**Solution**:
```bash
pip install pyannote.audio pyannote.core
```

### "Dependency conflict"

**Problem**: Version conflicts between packages

**Solution**:
1. Create a fresh virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. Follow installation steps from the beginning

## Why Not Use requirements.txt Directly?

The `requirements.txt` file is provided for reference, but installing all at once can cause pip to hang due to complex dependency resolution. The staged approach above is more reliable.

If you want to try anyway:
```bash
pip install -r requirements.txt
```

But be prepared to wait a long time or encounter issues!
