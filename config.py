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

    # Debug mode - enables detailed logging
    DEBUG = True

    # Speaker diarization toggle
    # Set to False to disable speaker identification (improves performance)
    ENABLE_DIARIZATION = os.getenv("ENABLE_DIARIZATION", "True").lower() in ("true", "1", "yes")

    # Device configuration
    # Options: "cuda" for GPU, "cpu" for CPU
    # Automatically detects CUDA availability
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Whisper model configuration
    # Options: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
    # Distilled models: "distil-whisper/distil-large-v3" (English-only, 5-6x faster)
    # Larger models = better accuracy but slower and more VRAM
    # large-v3: ~10GB VRAM, most accurate, multilingual, SLOW (~15-20s per audio chunk)
    # medium: ~5GB VRAM, good accuracy, multilingual, FASTER (~5-8s per audio chunk)
    # base: ~1GB VRAM, decent accuracy, multilingual, FASTEST (~2-3s per audio chunk)
    # NOTE: distil-large-v3 is English-only! Use medium/large-v3 for French/multilingual
    WHISPER_MODEL = "medium"

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

    # Translation model configuration
    # Options:
    #   - "facebook/m2m100_418M" (smaller, faster, ~2GB VRAM, ~1-2s per segment)
    #   - "facebook/m2m100_1.2B" (larger, better quality, ~5GB VRAM, ~3-5s per segment)
    TRANSLATION_MODEL = "facebook/m2m100_1.2B"

    # Target languages for translation
    # Each entry is a dict with "code" (ISO 639-1) and "name"
    # Supported codes: en, de, fr, es, it, pt, nl, pl, ru, zh, ja, ko, ar, hi, etc.
    TARGET_LANGUAGES = [
        {"code": "en", "name": "English"},
        {"code": "fr", "name": "French"},
       #{"code": "de", "name": "German"},
        #{"code": "it", "name": "Italian"},
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
    MAX_AUDIO_LENGTH = 30

    # Volume level considered as silence (0.0 to 1.0)
    # INCREASE if breaking at small pauses
    # DECREASE if not detecting sentence breaks
    SILENCE_THRESHOLD = 0.015

    # Number of consecutive silent chunks to trigger sentence end
    # INCREASE if sentences break too often
    # DECREASE if sentences run together
    SILENCE_CHUNKS = 20

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
