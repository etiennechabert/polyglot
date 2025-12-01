"""
Configuration for Polyglot - Real-time Multi-Language Audio Translator
"""

import os
from pathlib import Path

import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration"""

    # Secrets and API Keys
    # HuggingFace token for speaker diarization model
    HF_TOKEN = os.getenv("HF_TOKEN")

    # Admin password for admin access
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

    # Viewer password - dynamically generated 3-word passphrase
    # This is generated at startup and displayed to admin
    VIEWER_PASSWORD = None  # Set at runtime by generate_viewer_password()

    # Summarization settings
    # Set to False to disable live summary feature
    ENABLE_SUMMARIZATION = os.getenv("ENABLE_SUMMARIZATION", "True").lower() in ("true", "1", "yes")
    # Generate summary once per minute
    SUMMARY_INTERVAL_SECONDS = int(os.getenv("SUMMARY_INTERVAL_SECONDS", "60"))

    # Model rotation for summarization (experimental)
    # When enabled, offloads Whisper/Translation/Diarization models to CPU before running summarization
    # This frees up VRAM for larger summarization models, then reloads models after
    # Audio recording continues during rotation - transcription is queued and processed after reload
    # WARNING: This adds latency (~10-20s) during model swap
    # Set to True if you have limited VRAM and want to use larger summarization models
    ENABLE_MODEL_ROTATION = os.getenv("ENABLE_MODEL_ROTATION", "True").lower() in ("true", "1", "yes")

    # Local summarization model
    # Options (sorted by speed, all with 128K context):
    #   - "microsoft/Phi-3-mini-128k-instruct" (3.8B, ~3GB VRAM, 128K context, FAST) - RECOMMENDED for Windows
    #   - "microsoft/Phi-3-small-128k-instruct" (7B, ~5GB VRAM, 128K context) - Linux only (requires Triton)
    #   - "meta-llama/Llama-3.2-3B-Instruct" (3B, ~2.5GB VRAM, 128K context, FAST)
    #   - "microsoft/Phi-3-medium-128k-instruct" (14B, ~8GB VRAM, 128K context, SLOW but high quality)
    #   - "Qwen/Qwen2.5-7B-Instruct" (7B, ~14GB VRAM, 128K context, SLOW but excellent quality)
    SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "microsoft/Phi-3-mini-128k-instruct")

    # Summarization model context size (in tokens)
    # This should match your chosen model's context window
    # Used to calculate how much transcript we can fit in the prompt
    # Common values:
    #   - 128000 for Phi-3-medium-128k, Qwen2.5-7B
    #   - 32000 for Mistral-7B-Instruct-v0.3
    #   - 4096 for smaller models
    # Rule of thumb: 1 token ≈ 4 characters, so 128K tokens ≈ 500K characters
    SUMMARIZATION_CONTEXT_SIZE = int(os.getenv("SUMMARIZATION_CONTEXT_SIZE", "128000"))

    # Debug mode - enables detailed logging
    DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

    # Speaker diarization toggle
    # Set to False to disable speaker identification (improves performance)
    ENABLE_DIARIZATION = os.getenv("ENABLE_DIARIZATION", "True").lower() in ("true", "1", "yes")

    # Device configuration
    # Options: "cuda" for GPU, "cpu" for CPU
    # Automatically detects CUDA availability
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Whisper model configuration
    # Options: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"
    # Distilled models: "distil-whisper/distil-large-v3" (English-only, 5-6x faster)
    # Larger models = better accuracy but slower and more VRAM
    # turbo (large-v3-turbo): ~6GB VRAM, very good accuracy, multilingual, FAST (~4-6s per audio chunk)
    # large-v3: ~10GB VRAM, most accurate, multilingual, SLOW (~15-20s per audio chunk)
    # medium: ~5GB VRAM, good accuracy, multilingual, FASTER (~5-8s per audio chunk)
    # base: ~1GB VRAM, decent accuracy, multilingual, FASTEST (~2-3s per audio chunk)
    # NOTE: distil-large-v3 is English-only! Use medium/large-v3/turbo for French/multilingual
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "turbo")

    # Whisper chunk length for processing long audio (in seconds)
    # Whisper splits audio longer than this into overlapping chunks
    # INCREASE for better context (but more VRAM usage)
    # DECREASE to reduce memory pressure (but may lose context at boundaries)
    # Recommended: 10-30 seconds
    WHISPER_CHUNK_LENGTH = 30

    # Speaker diarization configuration
    # pyannote.audio model for speaker detection
    # Requires HuggingFace token for access: https://huggingface.co/pyannote/speaker-diarization
    DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

    # Diarization parameters
    # Minimum duration for a speaker turn (in seconds)
    # Lower values = more sensitive to speaker changes
    MIN_SPEAKER_DURATION = 0.3

    # Minimum duration for a speech region (in seconds)
    MIN_DURATION_ON = 0.0

    # Minimum duration for a non-speech region (in seconds)
    MIN_DURATION_OFF = 0.0

    # Clustering threshold for speaker separation (0.0 to 1.0)
    # Lower values = more likely to identify different speakers
    # Higher values = more conservative, fewer speakers detected
    # Default is typically 0.7, try 0.5-0.6 for more speaker separation
    DIARIZATION_CLUSTERING_THRESHOLD = 0.5

    # Embedding batch size for speaker diarization
    # Controls how many audio segments are processed simultaneously for speaker embeddings
    # DECREASE to reduce VRAM spikes (default pyannote is 32, which can cause 14GB+ usage)
    # INCREASE for faster processing if you have VRAM headroom
    # Recommended: 8-16 for 16GB VRAM GPUs
    DIARIZATION_EMBEDDING_BATCH_SIZE = 16

    # Translation model configuration
    # Options:
    #   - "facebook/m2m100_418M" (smaller, faster, ~2GB VRAM, ~1-2s per segment)
    #   - "facebook/m2m100_1.2B" (larger, better quality, ~5GB VRAM, ~3-5s per segment)
    TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "facebook/m2m100_1.2B")

    # Target languages for translation
    # Each entry is a dict with "code" (ISO 639-1) and "name"
    # Main European languages (available for viewers)
    TARGET_LANGUAGES = [
        {"code": "en", "name": "English"},
        {"code": "es", "name": "Spanish"},
        {"code": "fr", "name": "French"},
        {"code": "pt", "name": "Portuguese"},
        {"code": "de", "name": "German"},
        {"code": "it", "name": "Italian"},
    ]

    # Admin languages - subset of TARGET_LANGUAGES to display in admin view
    # System will still translate for all TARGET_LANGUAGES if viewers are watching
    # This just controls which languages appear in the admin grid
    ADMIN_LANGUAGES = [
        {"code": "en", "name": "English"},
        {"code": "de", "name": "German"},
    ]

    # Auto-detect source language
    # If True, uses langdetect to identify the spoken language
    # If False, you need to specify a source language
    SOURCE_LANGUAGE_AUTO_DETECT = True

    # Audio detection thresholds
    # These control when the app decides a sentence has ended

    # Minimum audio duration (seconds) before processing
    # INCREASE to avoid tiny fragments
    # DECREASE to capture shorter sentences
    MIN_AUDIO_LENGTH = 3.0

    # Maximum audio duration (seconds) before forcing processing
    # INCREASE for longer sentences
    # DECREASE if sentences are cut mid-speech
    MAX_AUDIO_LENGTH = 20

    # Volume level considered as silence (0.0 to 1.0)
    # INCREASE if breaking at small pauses
    # DECREASE if not detecting sentence breaks
    SILENCE_THRESHOLD = 0.015

    # Number of consecutive silent chunks to trigger sentence end
    # INCREASE if sentences break too often
    # DECREASE if sentences run together
    SILENCE_CHUNKS = 15

    # Audio processing configuration
    SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
    CHUNK_SIZE = 1024  # Audio buffer chunk size

    # Minimum audio level to process (prevents hallucinations during silence)
    # If average audio level is below this, skip transcription
    MIN_AUDIO_LEVEL = 0.01

    # File paths
    AUDIO_DIR = "audio"  # Directory to save captured audio files
    TRANSCRIPT_FILE = "transcript.txt"  # File to append all transcripts

    @classmethod
    def to_dict(cls):
        """Convert config to dictionary for JSON serialization"""
        return {
            "device": cls.DEVICE,
            "whisper_model": cls.WHISPER_MODEL,
            "translation_model": cls.TRANSLATION_MODEL,
            "target_languages": cls.TARGET_LANGUAGES,
            "source_language_auto_detect": cls.SOURCE_LANGUAGE_AUTO_DETECT,
        }

    @classmethod
    def get_audio_thresholds(cls):
        """Get audio detection thresholds as dict"""
        return {
            "min_audio_length": cls.MIN_AUDIO_LENGTH,
            "max_audio_length": cls.MAX_AUDIO_LENGTH,
            "silence_threshold": cls.SILENCE_THRESHOLD,
            "silence_chunks": cls.SILENCE_CHUNKS,
        }
