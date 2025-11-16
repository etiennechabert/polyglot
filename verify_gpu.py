"""
Verify GPU/CUDA setup for Polyglot
Run this after pip install to ensure CUDA is properly configured
"""

import sys

def verify_cuda():
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed!")
        print("Run: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu129")
        return False

    print(f"PyTorch version: {torch.__version__}")

    # Check if it's CPU-only version
    if "+cpu" in torch.__version__:
        print("\n" + "=" * 80)
        print("CRITICAL ERROR: CPU-only PyTorch detected!")
        print("This application requires GPU acceleration.")
        print("\nTo fix:")
        print("1. pip uninstall -y torch torchvision torchaudio")
        print("2. pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu129")
        print("=" * 80 + "\n")
        return False

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("WARNING: CUDA is not available!")
        print(f"PyTorch version: {torch.__version__}")
        print("\nPossible causes:")
        print("1. NVIDIA drivers not installed or outdated")
        print("2. Wrong PyTorch version installed")
        print("3. CUDA toolkit mismatch")
        print("=" * 80 + "\n")
        return False

    # Check version is >= 2.6.0 for transformers compatibility
    from packaging import version
    torch_version = torch.__version__.split("+")[0]
    if version.parse(torch_version) < version.parse("2.6.0"):
        print("\n" + "=" * 80)
        print(f"WARNING: PyTorch {torch_version} is too old!")
        print("transformers requires PyTorch >= 2.6.0")
        print("\nTo fix:")
        print("pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu129")
        print("=" * 80 + "\n")
        return False

    # All checks passed
    print(f"CUDA available: True")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}0")
    print("\nGPU setup is correct!")
    return True

if __name__ == "__main__":
    success = verify_cuda()
    sys.exit(0 if success else 1)
