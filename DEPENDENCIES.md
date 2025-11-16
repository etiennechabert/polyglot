# Dependency Installation Guide

## Why You Need INSTALL.bat (or Manual Installation)

**CRITICAL:** You **CANNOT** use `pip install -r requirements.txt` alone for this project!

### The PyTorch CUDA Version Problem

The `requirements.txt` file lists package versions but **cannot specify CUDA versions**. PyTorch requires special installation commands with the `--index-url` flag to get the correct CUDA-enabled builds.

**What happens if you just run `pip install -r requirements.txt`:**
- You get the CPU-only version of PyTorch (no GPU acceleration)
- OR you get the wrong CUDA version (incompatible with your GPU)
- Result: The app will either be extremely slow or fail to run

## Tested Working Configuration

This configuration has been tested and verified to work on RTX 5080 (16GB VRAM):

```
torch==2.8.0+cu129
torchaudio==2.8.0+cu129
pyannote.audio==4.0.1
pyannote.core==6.0.1
transformers==4.57.1
accelerate==1.11.0
numpy==2.3.4
soundfile==0.13.1
```

See [requirements-frozen.txt](requirements-frozen.txt) for the complete list of all package versions.

**GPU Requirements:**
- NVIDIA GPU with CUDA 12.9 support
- Minimum 8GB VRAM (16GB recommended)
- NVIDIA drivers 525.60.13 or newer

## Installation Methods

### Method 1: Use INSTALL.bat (Recommended for Windows)

```bash
INSTALL.bat
```

This script installs everything in the correct order:
1. PyTorch 2.8.0 with CUDA 12.9 support
2. Core Flask dependencies
3. Audio processing libraries
4. Speaker diarization models

### Method 2: Manual Installation (Linux/Mac or Advanced Users)

```bash
# Step 1: Install PyTorch with CUDA 12.9
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129

# Step 2: Install other dependencies
pip install -r requirements.txt
```

**Note:** For different CUDA versions, check the [PyTorch installation page](https://pytorch.org/get-started/locally/)

## Why These Specific Versions?

### PyTorch 2.8.0+cu129
- **Stable release** compatible with pyannote.audio 3.1.1
- **CUDA 12.9** support for latest NVIDIA GPUs
- Tested with RTX 40-series and RTX 50-series GPUs

### pyannote.audio 4.0.1
- Latest stable version (as of 2025-11-16)
- Compatible with PyTorch 2.8.0+cu129
- Required for speaker diarization feature

### Avoid These Versions
- ❌ **torch nightly (2.10.0+)**: Has compatibility issues with torchaudio
- ❌ **pyannote.audio 3.x**: Older version, use 4.0.1 instead
- ❌ **numpy 2.0.0-2.3.3**: May have compatibility issues, use 2.3.4+

## Verifying Your Installation

After installation, verify you have the correct PyTorch with CUDA:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

**Expected output:**
```
PyTorch: 2.8.0+cu129
CUDA Available: True
CUDA Version: 12.9
```

If CUDA is not available, you installed the wrong PyTorch version. Reinstall using INSTALL.bat or the manual method above.

## Troubleshooting

### "No CUDA-capable device is detected"
- Install/update NVIDIA drivers
- Reinstall PyTorch with correct CUDA version using INSTALL.bat

### "ImportError: cannot import name 'X' from 'torch'"
- Version mismatch between torch and torchaudio
- Reinstall both using INSTALL.bat

### "RuntimeError: CUDA error: out of memory"
- Your GPU doesn't have enough VRAM
- Try a smaller Whisper model in config.py (change from "turbo" to "medium" or "base")
- Reduce WHISPER_CHUNK_LENGTH in config.py

### "pyannote.audio fails to load"
- Make sure you installed pyannote.audio AFTER PyTorch
- Reinstall: `pip uninstall pyannote.audio && pip install pyannote.audio==3.1.1`

## Alternative: CPU-Only Installation

If you don't have an NVIDIA GPU, you can run on CPU (very slow):

```bash
# Install CPU-only PyTorch
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

**Warning:** CPU mode will be 10-50x slower. Real-time translation will not be possible.
