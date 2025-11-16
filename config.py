"""
Configuration for Polyglot - Real-time Multi-Language Audio Translator
"""

import torch


class Config:
    """Application configuration"""

    # Device configuration
    # Options: "cuda" for GPU, "cpu" for CPU
    # Automatically detects CUDA availability
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Whisper model configuration
    # Options: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
    # Larger models = better accuracy but slower and more VRAM
    # large-v3 is the most accurate (requires ~10GB VRAM)
    WHISPER_MODEL = "large-v3"

    # Translation model configuration
    # Options:
    #   - "facebook/m2m100_418M" (smaller, faster, ~2GB VRAM)
    #   - "facebook/m2m100_1.2B" (larger, better quality, ~5GB VRAM)
    # With RTX 5080, we can easily use the 1.2B model for better translations
    TRANSLATION_MODEL = "facebook/m2m100_1.2B"

    # Target languages for translation
    # Each entry is a dict with "code" (ISO 639-1) and "name"
    # Supported codes: en, de, fr, es, it, pt, nl, pl, ru, zh, ja, ko, ar, hi, etc.
    TARGET_LANGUAGES = [
        {"code": "en", "name": "English"},
        {"code": "de", "name": "German"},
        {"code": "fr", "name": "French"},
        {"code": "it", "name": "Italian"},
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
    MAX_AUDIO_LENGTH = 100.0

    # Volume level considered as silence (0.0 to 1.0)
    # INCREASE if breaking at small pauses
    # DECREASE if not detecting sentence breaks
    SILENCE_THRESHOLD = 0.01

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
