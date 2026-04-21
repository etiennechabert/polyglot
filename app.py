"""
Polyglot - Real-time Multi-Language Audio Translator
Captures system audio and translates speech into multiple languages simultaneously
"""

import argparse
import collections
import os
import queue
import random
import sys
import threading
import time
import uuid
import warnings
import wave
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path


def log(message, tag=None):
    """Print a log message with timestamp. Tag is optional prefix like [SUMMARY]"""
    ts = datetime.now().strftime('%H:%M:%S')
    if tag:
        print(f"[{ts}] [{tag}] {message}", flush=True)
    else:
        print(f"[{ts}] {message}", flush=True)


# Word list for generating memorable passphrases
PASSPHRASE_WORDS = [
    "apple", "banana", "cherry", "dragon", "eagle", "forest", "garden", "harbor",
    "island", "jungle", "kite", "lemon", "mountain", "night", "ocean", "planet",
    "queen", "river", "sunset", "thunder", "umbrella", "valley", "winter", "yellow",
    "zebra", "anchor", "bridge", "castle", "dolphin", "ember", "falcon", "glacier",
    "horizon", "ivory", "jasmine", "kingdom", "lantern", "meadow", "nebula", "orchid",
    "phoenix", "quartz", "rainbow", "silver", "temple", "universe", "velvet", "whisper",
    "crystal", "breeze", "coral", "dawn", "eclipse", "flame", "golden", "harmony",
    "indigo", "jewel", "karma", "lunar", "marvel", "nimbus", "opal", "prism",
    "quest", "radiant", "sapphire", "twilight", "unity", "venture", "wonder", "zenith"
]


def generate_viewer_password():
    """Generate a 3-word passphrase separated by underscores"""
    words = random.sample(PASSPHRASE_WORDS, 3)
    return "_".join(words)


def load_saved_password():
    """Load previously saved viewer password from file"""
    password_file = Path("viewer_password.txt")
    if password_file.exists():
        try:
            return password_file.read_text().strip()
        except Exception:
            return None
    return None


def save_password(password):
    """Save viewer password to file for reuse"""
    password_file = Path("viewer_password.txt")
    password_file.write_text(password)


def get_existing_transcripts():
    """Get list of existing transcript files"""
    transcripts_dir = Path("transcripts")
    if not transcripts_dir.exists():
        return []
    return sorted(transcripts_dir.glob("*.txt"), key=lambda x: x.stat().st_mtime, reverse=True)


def sanitize_filename(name):
    """Convert meeting name to safe filename"""
    # Replace spaces with underscores, remove special characters
    safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in name)
    safe_name = safe_name.replace(' ', '_')
    return safe_name[:50]  # Limit length


def startup_configuration():
    """Interactive startup configuration for meeting and password"""
    global TRANSCRIPT_FILE, MEETING_NAME

    # Check if running in interactive mode (has a terminal/TTY)
    is_interactive = sys.stdin.isatty() if hasattr(sys.stdin, 'isatty') else False

    print("\n" + "=" * 60)
    print("  POLYGLOT - Startup Configuration")
    print("=" * 60)

    # --- Password Configuration ---
    saved_password = load_saved_password()
    if saved_password:
        print(f"\n[PASSWORD] Previous viewer password found: {saved_password}")
        try:
            while True:
                choice = input("  Reuse this password? (Y/n): ").strip().lower()
                if choice in ('', 'y', 'yes'):
                    Config.VIEWER_PASSWORD = saved_password
                    print(f"  -> Using saved password: {saved_password}")
                    break
                elif choice in ('n', 'no'):
                    Config.VIEWER_PASSWORD = generate_viewer_password()
                    save_password(Config.VIEWER_PASSWORD)
                    print(f"  -> Generated new password: {Config.VIEWER_PASSWORD}")
                    break
                else:
                    print("  Please enter 'y' or 'n'")
        except (EOFError, OSError):
            # Non-interactive mode, use saved password
            Config.VIEWER_PASSWORD = saved_password
            print(f"  -> Auto-using saved password (non-interactive mode): {saved_password}")
    else:
        Config.VIEWER_PASSWORD = generate_viewer_password()
        save_password(Config.VIEWER_PASSWORD)
        print(f"\n[PASSWORD] Generated new viewer password: {Config.VIEWER_PASSWORD}")

    # --- Meeting Name Configuration ---
    if is_interactive:
        print(f"\n[MEETING] Enter a name for this meeting/session")
        try:
            meeting_name = input("  Meeting name (or press Enter for timestamp): ").strip()
        except (EOFError, OSError):
            meeting_name = ""
    else:
        print(f"\n[MEETING] Auto-generating meeting name (non-interactive mode)")
        meeting_name = ""

    if meeting_name:
        MEETING_NAME = meeting_name
        safe_name = sanitize_filename(meeting_name)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        transcript_filename = f"{safe_name}_{timestamp}.txt"
    else:
        MEETING_NAME = f"Meeting {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        transcript_filename = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # --- Transcript File Configuration ---
    existing_transcripts = get_existing_transcripts()

    if existing_transcripts and is_interactive:
        print(f"\n[TRANSCRIPT] Found {len(existing_transcripts)} existing transcript(s):")
        print("  0. Create NEW transcript file")
        for i, tf in enumerate(existing_transcripts[:10], 1):  # Show max 10
            size_kb = tf.stat().st_size / 1024
            mod_time = datetime.fromtimestamp(tf.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            print(f"  {i}. {tf.name} ({size_kb:.1f}KB, {mod_time})")

        while True:
            try:
                choice = input("\n  Select transcript (0 for new, or number to continue): ").strip()
            except (EOFError, OSError):
                choice = '0'

            if choice == '' or choice == '0':
                # Create new transcript
                transcripts_dir = Path("transcripts")
                transcripts_dir.mkdir(exist_ok=True)
                TRANSCRIPT_FILE = transcripts_dir / transcript_filename
                print(f"  -> Creating new transcript: {TRANSCRIPT_FILE.name}")
                break
            elif choice.isdigit() and 1 <= int(choice) <= min(10, len(existing_transcripts)):
                TRANSCRIPT_FILE = existing_transcripts[int(choice) - 1]
                print(f"  -> Continuing transcript: {TRANSCRIPT_FILE.name}")
                # Update meeting name from filename if continuing
                MEETING_NAME = TRANSCRIPT_FILE.stem.replace('_', ' ')
                break
            else:
                print("  Invalid choice, please try again")
    else:
        # No existing transcripts or non-interactive mode, create new
        transcripts_dir = Path("transcripts")
        transcripts_dir.mkdir(exist_ok=True)
        TRANSCRIPT_FILE = transcripts_dir / transcript_filename
        if existing_transcripts and not is_interactive:
            print(f"\n[TRANSCRIPT] Auto-creating new transcript (non-interactive mode): {TRANSCRIPT_FILE.name}")
        else:
            print(f"\n[TRANSCRIPT] Creating new transcript: {TRANSCRIPT_FILE.name}")

    print("\n" + "-" * 60)
    print(f"  Meeting: {MEETING_NAME}")
    print(f"  Password: {Config.VIEWER_PASSWORD}")
    print(f"  Transcript: {TRANSCRIPT_FILE.name}")
    print("-" * 60 + "\n")

import numpy as np
import psutil
import pyaudiowpatch as pyaudio
import resampy
import torch

# CRITICAL FIX: Prevent transformers from trying to use torchcodec
# torchcodec is installed by pyannote.audio but the DLLs are missing
# This causes transformers Whisper pipeline to crash when it tries to use torchcodec
# Solution: Monkey patch transformers.utils.import_utils._torchcodec_available to False
# This makes transformers think torchcodec is not available, so it won't try to use it

from flask import Flask, jsonify, render_template, request, session
from flask_socketio import SocketIO, emit, join_room, leave_room, rooms
from langdetect import LangDetectException, detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Monkey patch transformers to disable torchcodec detection
import transformers.utils.import_utils
transformers.utils.import_utils._torchcodec_available = False

from config import Config

# Suppress CUDA compatibility warning for RTX 5080 (sm_120)
# PyTorch nightly works fine with backward compatibility
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

# Single instance check - prevent multiple app instances
LOCK_FILE = Path("polyglot.lock")

def check_single_instance():
    """Check if another instance of the app is already running"""
    if LOCK_FILE.exists():
        try:
            # Check if the PID in the lock file is still running
            with open(LOCK_FILE, 'r') as f:
                old_pid = int(f.read().strip())

            # Try to check if process exists (Windows-compatible)
            import psutil
            if psutil.pid_exists(old_pid):
                print(f"\n{'='*80}")
                print("ERROR: Another instance of Polyglot is already running!")
                print(f"PID: {old_pid}")
                print("Please close the other instance first, or delete polyglot.lock if it's stale.")
                print(f"{'='*80}\n")
                sys.exit(1)
            else:
                # Stale lock file, remove it
                LOCK_FILE.unlink()
        except (ValueError, FileNotFoundError, ImportError):
            # Invalid lock file or psutil not available, remove it
            LOCK_FILE.unlink()

    # Create lock file with current PID
    with open(LOCK_FILE, 'w') as f:
        f.write(str(os.getpid()))

def cleanup_lock_file():
    """Remove lock file on exit"""
    try:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
    except:
        pass

# Import pyannote.audio for speaker diarization (ALWAYS ENABLED)
from pyannote.audio import Pipeline as DiarizationPipeline

# Configuration from config.py
SAMPLE_RATE = Config.SAMPLE_RATE
CHUNK_SIZE = Config.CHUNK_SIZE

# Transcript and password storage paths
TRANSCRIPTS_DIR = Path("transcripts")
TRANSCRIPTS_DIR.mkdir(exist_ok=True)
PASSWORD_FILE = Path("viewer_password.txt")

# Will be set during startup configuration
TRANSCRIPT_FILE = None
MEETING_NAME = None

# Dynamic audio detection thresholds (can be changed via API)
audio_thresholds = Config.get_audio_thresholds()

# Audio device selection (session-level persistence)
selected_audio_device_index = None  # None means auto-detect

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)  # Secret key for sessions
socketio = SocketIO(app, cors_allowed_origins="*", manage_session=False)

# Global state
audio_queue = queue.Queue()
is_listening = False
is_processing = False  # Lock to prevent overlapping transcriptions
transcription_pipe = None
translation_model = None
translation_tokenizer = None
diarization_pipeline = None  # Speaker diarization pipeline
summarization_pipe = None  # Local LLM for summarization
summarization_paused = False  # Runtime toggle to pause/resume summarization (kill switch)
audio_stream = None
p_audio = None
actual_sample_rate = 48000  # Will be set to the actual device sample rate
num_channels = 2  # Will be set to the actual number of channels

# Viewer and session tracking
# Dictionary to track connected clients: session_id -> {'role': 'admin'/'viewer', 'language': 'en', 'authenticated': bool}
connected_clients = {}
# Track which languages have active viewers: {'en': 2, 'fr': 1, 'de': 0}
active_language_viewers = {lang['code']: 0 for lang in Config.TARGET_LANGUAGES}
# Track admin sessions
admin_sessions = set()
# Track authenticated viewer sessions
viewer_sessions = set()

# Live summarization state
current_summary = {
    "recent_bullets": [],  # Bullet points about recent ~5 min discussion
    "overview_sections": [],  # Sections covering full meeting: [{title: "...", bullets: [...]}]
    "last_updated": None,
    "segments_summarized": 0,
    "time_range": ""  # Time range of summarized content (e.g., "14:30:15 - 14:35:22")
}
# Previous overview for summarization chaining (model builds on this)
previous_overview_text = ""  # Formatted text of last overview for context
summary_lock = threading.Lock()
last_summary_time = time.time()  # Initialize to now so first summary waits for the interval
last_summary_segment_count = 0  # Track how many segments were in last summary
all_meeting_segments = []  # FULL meeting transcript (never cleared)
meeting_start_time = None  # Track when meeting started (first transcription)
summary_pending = False  # Track if a summary generation is waiting

# ── Meet bot state (Phase 4) ──────────────────────────────────────────────────
# Deque of closed speaker segments: (start_ms, end_ms, display_name).
# Populated by the /meet_bot SocketIO namespace; consumed by resolve_speaker_identity().
speaker_timeline = collections.deque(maxlen=500)
# Known participants: display_name -> first_seen_ms
meet_participants = {}
# Open (unended) speaker intervals: display_name -> start_ms
_active_speaker_starts = {}
# Whether the Playwright bot is currently connected
bot_connected = False
# Accumulation buffer for rechunking 320-sample bot frames → CHUNK_SIZE frames
_bot_pcm_buffer = np.array([], dtype=np.float32)
# Wall-clock ms of the most recently received bot audio frame (for Phase 5 time-alignment)
_last_capture_ts_ms = None
# WAV file writer — captures the full raw 16 kHz mono meeting audio for retranscription
_bot_wav_writer = None
_bot_wav_lock = threading.Lock()
# Speaker-switch batching: who the bot says is currently speaking, and whether the most recent
# speaker_start differs from the previous one (triggers process_audio to flush the current batch).
_current_bot_speaker = None
_pending_speaker_switch = False
# Maximum batch length when bot is connected (gives Whisper more context per speaker turn).
BOT_MAX_BATCH_SEC = 60
# Partial (streaming) transcription state. While a speaker keeps talking we
# re-transcribe the growing buffer every audio_thresholds["partial_interval_sec"]
# and emit an is_partial=true WS payload keyed by _current_utterance_id; the
# final flush emits is_partial=false with the same id so the frontend replaces
# the row. The interval is runtime-mutable via /api/thresholds.
_current_utterance_id = None
_last_partial_ts = 0.0


def load_transcript_segments(transcript_path):
    """Load existing transcript file into all_meeting_segments for summarization.

    Parses lines like: [2025-12-01 19:56:33] [0.00s-11.00s] Text here...
    """
    global all_meeting_segments, meeting_start_time
    import re
    from datetime import datetime

    if not transcript_path or not transcript_path.exists():
        return 0

    segments_loaded = 0
    first_unix_time = None

    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse: [2025-12-01 19:56:33] [0.00s-11.00s] Text...
                match = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] \[[\d.]+s-[\d.]+s\] (.+)', line)
                if match:
                    timestamp_str = match.group(1)
                    text = match.group(2)

                    # Parse the timestamp to get unix time
                    try:
                        dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        unix_time = dt.timestamp()
                        if first_unix_time is None:
                            first_unix_time = unix_time
                    except ValueError:
                        unix_time = time.time()

                    # Add segment
                    all_meeting_segments.append({
                        "text": text,
                        "timestamp": timestamp_str,
                        "unix_time": unix_time,
                        "start": 0,
                        "end": 0
                    })
                    segments_loaded += 1

        # Set meeting start time from first segment
        if first_unix_time is not None:
            meeting_start_time = first_unix_time

        print(f"[TRANSCRIPT] Loaded {segments_loaded} segments from existing transcript")

    except Exception as e:
        print(f"[TRANSCRIPT] Error loading transcript: {e}")

    return segments_loaded

# VRAM tracking for admin stats display
model_vram_usage = {
    "whisper": 0,
    "translation": 0,
    "diarization": 0,
    "summarization": 0,
}
gpu_total_memory = 0  # Total GPU memory in GB


def initialize_models():
    """Initialize Whisper, M2M100, Speaker Diarization, and Summarization models

    Model loading order with MODEL_ROTATION enabled:
    1. Load Summarization model on GPU (needs GPU for initial loading)
    2. Immediately move it to CPU to free VRAM
    3. Load transcription models (Whisper, Translation, Diarization) on GPU

    This ensures transcription models have maximum VRAM available during normal operation.
    When summarization is needed, transcription models swap to CPU and summarization moves to GPU.
    """
    global transcription_pipe, translation_model, translation_tokenizer, diarization_pipeline, summarization_pipe
    global model_vram_usage, gpu_total_memory

    # CRITICAL: Ensure CUDA is available - this app requires GPU
    if not torch.cuda.is_available():
        error_msg = (
            "\n" + "=" * 80 + "\n"
            "CRITICAL ERROR: CUDA/GPU is NOT available!\n"
            "This application requires GPU acceleration to run efficiently.\n"
            f"PyTorch version: {torch.__version__}\n"
            f"CUDA available: {torch.cuda.is_available()}\n\n"
            "To fix this:\n"
            "1. Uninstall CPU-only PyTorch:\n"
            "   pip uninstall -y torch torchvision torchaudio\n\n"
            "2. Reinstall with CUDA support:\n"
            "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n\n"
            "3. Verify NVIDIA drivers are installed and GPU is detected\n"
            + "=" * 80 + "\n"
        )
        print(error_msg)
        raise RuntimeError("CUDA is not available - GPU acceleration is required!")

    device = Config.DEVICE
    print("\n" + "=" * 80)
    print("MODEL INITIALIZATION")
    print("=" * 80)
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Total Memory: {gpu_total_memory:.2f} GB")
        print(f"Initial GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB reserved")

    print()

    # Count total models to load
    model_count = 3 if not Config.ENABLE_DIARIZATION else 4
    if Config.ENABLE_SUMMARIZATION:
        model_count += 1
    current_model = 0

    # ==================== PHASE 1: Load Summarization (will be moved to CPU) ====================
    # When MODEL_ROTATION is enabled, we load summarization FIRST so we can move it to CPU
    # before loading the transcription models. This maximizes VRAM for real-time transcription.
    summarization_pipe = None
    if Config.ENABLE_SUMMARIZATION:
        current_model += 1
        print(f"[{current_model}/{model_count}] Loading Summarization Model: {Config.SUMMARIZATION_MODEL}")
        if Config.ENABLE_MODEL_ROTATION:
            print(f"      (Will be moved to CPU after loading - MODEL_ROTATION enabled)")

        # Track VRAM before loading
        pre_summarization_mem = torch.cuda.memory_allocated(0) / 1024**3 if device == "cuda" else 0

        try:
            summarization_pipe = pipeline(
                "text-generation",
                model=Config.SUMMARIZATION_MODEL,
                device=0 if device == "cuda" else -1,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
            )

            if device == "cuda":
                total_mem = torch.cuda.memory_allocated(0) / 1024**3
                summarization_mem = total_mem - pre_summarization_mem
                model_vram_usage["summarization"] = summarization_mem
                print(f"      [OK] Loaded successfully")
                print(f"      GPU Memory: {total_mem:.2f} GB allocated")
                print(f"      Model Size: ~{summarization_mem:.2f} GB")

                # Immediately move to CPU if MODEL_ROTATION is enabled
                if Config.ENABLE_MODEL_ROTATION:
                    print(f"      Moving summarization model to CPU...")
                    summarization_pipe.model.to("cpu")
                    summarization_pipe.device = torch.device("cpu")
                    torch.cuda.empty_cache()
                    print(f"      [OK] Moved to CPU, GPU freed: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB now allocated")
            else:
                print(f"      [OK] Loaded successfully (CPU)")
            print()
        except Exception as e:
            error_msg = (
                "\n" + "=" * 80 + "\n"
                "CRITICAL ERROR: Failed to load summarization model!\n"
                f"Model: {Config.SUMMARIZATION_MODEL}\n"
                f"Error: {e}\n\n"
                "To fix this:\n"
                "1. Check that the model name is correct\n"
                "2. Install any missing dependencies (e.g., pip install tiktoken)\n"
                "3. Note: Phi-3-small requires Triton (Linux only)\n"
                "4. Or set ENABLE_SUMMARIZATION=False in .env to disable summarization\n"
                + "=" * 80 + "\n"
            )
            print(error_msg)
            raise RuntimeError(f"Failed to load summarization model: {e}")

    # ==================== PHASE 2: Load Transcription Models ====================
    # Initialize Whisper with word-level timestamps for precise speaker alignment
    current_model += 1
    print(f"[{current_model}/{model_count}] Loading Whisper Model: {Config.WHISPER_MODEL}")
    # Distil-whisper models use their own namespace, regular models need openai/whisper- prefix
    if "/" in Config.WHISPER_MODEL:
        # Model already has namespace (e.g., distil-whisper/distil-large-v3)
        whisper_model_name = Config.WHISPER_MODEL
    elif Config.WHISPER_MODEL == "turbo":
        # Turbo is a special case: maps to large-v3-turbo
        whisper_model_name = "openai/whisper-large-v3-turbo"
    else:
        # Simple model name needs openai/whisper- prefix (e.g., medium -> openai/whisper-medium)
        whisper_model_name = f"openai/whisper-{Config.WHISPER_MODEL}"
    print(f"      Model: {whisper_model_name}")
    transcription_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model_name,
        device=0 if device == "cuda" else -1,
        chunk_length_s=Config.WHISPER_CHUNK_LENGTH,  # Split long audio into chunks to reduce VRAM usage
        return_timestamps="word",  # Enable word-level timestamps for precise speaker alignment
        generate_kwargs={"task": "transcribe"},  # Transcribe in original language (Whisper auto-detects language)
    )

    if device == "cuda":
        whisper_mem = torch.cuda.memory_allocated(0) / 1024**3
        model_vram_usage["whisper"] = whisper_mem
        print(f"      [OK] Loaded successfully")
        print(f"      GPU Memory: {whisper_mem:.2f} GB allocated")
    else:
        print(f"      [OK] Loaded successfully (CPU)")
    print()

    # Initialize Translation Model (M2M100 or NLLB)
    current_model += 1
    model_name = Config.TRANSLATION_MODEL
    print(f"[{current_model}/{model_count}] Loading Translation Model: {model_name}")

    # Use AutoTokenizer and AutoModel for compatibility with both M2M100 and NLLB
    translation_tokenizer = AutoTokenizer.from_pretrained(model_name)
    translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Debug: Print tokenizer info for NLLB
    if "nllb" in model_name.lower():
        print(f"      NLLB tokenizer type: {type(translation_tokenizer).__name__}")
        # Test language code conversion for our target languages
        test_codes = ['eng_Latn', 'deu_Latn']
        print(f"      Testing language codes:")
        for code in test_codes:
            try:
                token_id = translation_tokenizer.convert_tokens_to_ids(code)
                print(f"        ✓ '{code}' -> token_id={token_id}")
            except Exception as e:
                print(f"        ✗ '{code}' failed: {e}")

    if device == "cuda":
        translation_model = translation_model.cuda()
        translation_mem = torch.cuda.memory_allocated(0) / 1024**3
        model_vram_usage["translation"] = translation_mem - whisper_mem
        print(f"      [OK] Loaded successfully")
        print(f"      GPU Memory: {translation_mem:.2f} GB allocated (total)")
        print(f"      Model Size: ~{translation_mem - whisper_mem:.2f} GB")
    else:
        print(f"      [OK] Loaded successfully (CPU)")
    print()

    # Initialize Speaker Diarization (conditionally)
    current_model += 1
    diarization_pipeline = None
    if Config.ENABLE_DIARIZATION:
        print(f"[{current_model}/{model_count}] Loading Speaker Diarization: {Config.DIARIZATION_MODEL}")
        if not Config.HF_TOKEN:
            error_msg = (
                "\n" + "=" * 80 + "\n"
                "CRITICAL ERROR: HF_TOKEN not set!\n"
                "Speaker diarization requires a HuggingFace token.\n\n"
                "To fix this:\n"
                "1. Copy .env.example to .env\n"
                "2. Get a token at: https://huggingface.co/settings/tokens\n"
                "3. Accept model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "4. Add your token to .env file: HF_TOKEN=your_token_here\n"
                + "=" * 80 + "\n"
            )
            print(error_msg)
            raise RuntimeError("HF_TOKEN is required for speaker diarization!")

        diarization_pipeline = DiarizationPipeline.from_pretrained(
            Config.DIARIZATION_MODEL,
            token=Config.HF_TOKEN
        )

        # Move pipeline to GPU if available
        if device == "cuda":
            diarization_pipeline.to(torch.device("cuda"))
            # Reduce embedding batch size to prevent VRAM spikes during inference
            # Default is 32, which can cause memory to spike from ~8GB to ~16GB
            diarization_pipeline.embedding_batch_size = Config.DIARIZATION_EMBEDDING_BATCH_SIZE
            total_mem = torch.cuda.memory_allocated(0) / 1024**3
            diarization_mem = total_mem - translation_mem
            model_vram_usage["diarization"] = diarization_mem
            print(f"      [OK] Loaded successfully")
            print(f"      GPU Memory: {total_mem:.2f} GB allocated (total)")
            print(f"      Model Size: ~{diarization_mem:.2f} GB")
        else:
            print(f"      [OK] Loaded successfully (CPU)")

        print(f"      Configuration:")
        print(f"      - Clustering threshold: {Config.DIARIZATION_CLUSTERING_THRESHOLD}")
        print(f"      - Min speaker duration: {Config.MIN_SPEAKER_DURATION}s")
        print(f"      - Min duration on: {Config.MIN_DURATION_ON}s")
        print(f"      - Min duration off: {Config.MIN_DURATION_OFF}s")
        print()
    else:
        print(f"[{current_model}/{model_count}] Speaker Diarization: DISABLED")
        print(f"      Speaker identification is turned off (ENABLE_DIARIZATION=False)")
        print(f"      All transcriptions will appear without speaker labels.")
        print()

    # ==================== INITIALIZATION COMPLETE ====================
    print("=" * 80)
    print("ALL MODELS LOADED SUCCESSFULLY!")
    if device == "cuda":
        print(f"Transcription Models VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        if Config.ENABLE_SUMMARIZATION and Config.ENABLE_MODEL_ROTATION:
            print(f"Summarization Model: On CPU (will swap to GPU when needed)")
    print("=" * 80 + "\n")


# Model rotation state tracking
# When model rotation is enabled:
# - Transcription models (Whisper, Translation, Diarization) stay on GPU permanently
# - Summarization model starts on CPU, moves to GPU only during summary generation
# - Transcription must wait while summarization model is on GPU to avoid VRAM overflow
summarization_on_gpu = False  # True when summarization model is loaded on GPU
summarization_started_at = None  # Timestamp when summarization started (for UI feedback)
model_rotation_lock = threading.Lock()  # Protects summarization_on_gpu state changes
transcription_lock = threading.Lock()  # Ensures transcription completes before summarization starts


def load_summarization_to_gpu():
    """Move Summarization model from CPU to GPU for inference.

    Called before generating a summary. Transcription must be paused while
    summarization model is on GPU to avoid VRAM overflow.
    Note: Translation model stays on GPU - it should not be rotated.
    """
    global summarization_pipe, summarization_on_gpu, model_vram_usage

    if not Config.ENABLE_MODEL_ROTATION:
        return True  # Model already on GPU if rotation disabled

    with model_rotation_lock:
        if summarization_on_gpu:
            return True  # Already on GPU

        log("Swapping models for summarization...", "MODEL")
        start_time = time.time()

        try:
            vram_before = torch.cuda.memory_allocated(0) / 1024**3

            # Load summarization to GPU (translation model stays on GPU)
            if summarization_pipe is not None:
                log("  Moving summarization model to GPU...", "MODEL")
                summarization_pipe.model.to("cuda")
                summarization_pipe.device = torch.device("cuda")
                torch.cuda.empty_cache()

            summarization_on_gpu = True
            elapsed = time.time() - start_time
            vram_after = torch.cuda.memory_allocated(0) / 1024**3
            # Update VRAM tracking for the graph
            model_vram_usage["summarization"] = vram_after - vram_before
            log(f"Swap complete in {elapsed:.1f}s, VRAM: {vram_after:.2f} GB", "MODEL")
            return True

        except Exception as e:
            log(f"Error loading summarization to GPU: {e}", "MODEL")
            import traceback
            traceback.print_exc()
            return False


def unload_summarization_to_cpu():
    """Move Summarization model from GPU back to CPU after summary generation.

    Called after generating a summary. Frees GPU memory for transcription models.
    Note: Translation model stays on GPU - it should not be rotated.
    """
    global summarization_pipe, summarization_on_gpu, model_vram_usage

    if not Config.ENABLE_MODEL_ROTATION:
        return True  # Model stays on GPU if rotation disabled

    with model_rotation_lock:
        if not summarization_on_gpu:
            return True  # Already on CPU

        log("Restoring models after summarization...", "MODEL")
        start_time = time.time()

        try:
            # Move summarization back to CPU (translation model stays on GPU)
            if summarization_pipe is not None:
                log("  Moving summarization model to CPU...", "MODEL")
                summarization_pipe.model.to("cpu")
                summarization_pipe.device = torch.device("cpu")

            # Clear CUDA cache to release memory
            torch.cuda.empty_cache()
            model_vram_usage["summarization"] = 0

            summarization_on_gpu = False
            elapsed = time.time() - start_time
            vram_after = torch.cuda.memory_allocated(0) / 1024**3
            log(f"Restore complete in {elapsed:.1f}s, VRAM: {vram_after:.2f} GB", "MODEL")
            return True

        except Exception as e:
            log(f"Error restoring models: {e}", "MODEL")
            import traceback
            traceback.print_exc()
            return False


def audio_callback(in_data, frame_count, time_info, status):
    """Callback for audio stream"""
    if is_listening and in_data:
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        audio_queue.put(audio_data)
    return (in_data, pyaudio.paContinue)


def detect_language(text):
    """Detect language using langdetect library"""
    try:
        detected = detect(text)
        # Map langdetect codes to M2M100 codes if needed
        lang_map = {
            "zh-cn": "zh",
            "zh-tw": "zh",
        }
        return lang_map.get(detected, detected)
    except LangDetectException:
        print("Language detection failed, defaulting to English")
        return "en"  # Default to English if detection fails


def generate_summary(previous_overview, new_transcript, time_range="", minutes_since_start=0):
    """Generate a structured summary using summarization chaining.

    Args:
        previous_overview: The overview from the last summary (for context/continuity)
        new_transcript: New transcript segments since the last summary
        time_range: Time range of the full meeting
        minutes_since_start: How long the meeting has been going
    """
    global current_summary, last_summary_time, all_meeting_segments, summarization_started_at

    if summarization_pipe is None:
        return None

    # Track when summarization started (for UI feedback)
    summarization_started_at = time.time()

    # Model rotation: Wait for any in-progress transcription to finish, then load summarization to GPU
    rotation_enabled = Config.ENABLE_MODEL_ROTATION
    if rotation_enabled:
        # Acquire transcription lock to ensure no transcription is running
        # This blocks until any in-progress transcription completes
        log("Waiting for transcription lock...", "SUMMARY")
        transcription_lock.acquire()
        log("Transcription lock acquired, loading summarization to GPU...", "SUMMARY")
        load_summarization_to_gpu()

    try:
        # Build context section based on whether we have previous overview
        if previous_overview:
            context_section = f"""PREVIOUS MEETING SUMMARY (what was discussed before):
{previous_overview}

---

NEW DISCUSSION (since last summary - analyze this carefully):
{new_transcript}"""
        else:
            # First summary - no previous context
            context_section = f"""MEETING TRANSCRIPT (first summary):
{new_transcript}"""

        # Build the prompt for recent discussion summary only (Llama 3.2 format)
        # Full meeting overview is now handled manually via Claude
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You summarize recent meeting discussion. Output ONLY valid JSON. Never mention speaker names/numbers.<|eot_id|><|start_header_id|>user<|end_header_id|>
Meeting: {minutes_since_start} min total. Time range: {time_range}

{context_section}

Output ONLY this JSON format (array of strings, NOT objects):
{{"recent_details": ["First point as a string", "Second point as a string", "Third point", "Fourth point", "Fifth point"]}}

Rules:
- Exactly 5 short bullet points as plain strings in the array
- Include specific names, numbers, facts mentioned
- Focus only on what was just discussed<|eot_id|><|start_header_id|>assistant<|end_header_id|}}
"""

        # Generate summary (no token limit - let model decide)
        result = summarization_pipe(
            prompt,
            do_sample=False,
            return_full_text=False,
        )

        generated_text = result[0]["generated_text"].strip()
        log(f"Raw LLM output: {generated_text[:300]}...", "SUMMARY")

        # Parse the JSON response
        import json
        import re
        # Try to extract JSON from the response
        json_start = generated_text.find("{")
        json_end = generated_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = generated_text[json_start:json_end]

            # Try to repair truncated/malformed JSON
            try:
                summary_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                log(f"JSON parse error: {e}, attempting repair...", "SUMMARY")

                # Try multiple repair strategies
                repaired = False

                # Strategy 1: Fix unescaped quotes inside strings
                # Replace problematic characters that break JSON
                json_str_clean = json_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

                # Strategy 2: Truncate at the error position and close brackets
                error_pos = e.pos if hasattr(e, 'pos') else len(json_str) // 2

                # Try parsing up to the error, find last complete item
                truncated = json_str_clean[:error_pos]
                # Find last complete string (ends with ", or "])
                last_complete = max(
                    truncated.rfind('",'),
                    truncated.rfind('"]'),
                    truncated.rfind('" ,'),
                    truncated.rfind('" ]')
                )

                if last_complete > 0:
                    truncated = truncated[:last_complete + 2]
                    # Close any open brackets
                    open_brackets = truncated.count('[') - truncated.count(']')
                    open_braces = truncated.count('{') - truncated.count('}')
                    truncated = truncated.rstrip().rstrip(',')
                    truncated += ']' * open_brackets + '}' * open_braces

                    try:
                        summary_data = json.loads(truncated)
                        log(f"Repaired JSON by truncating at pos {last_complete}", "SUMMARY")
                        repaired = True
                    except json.JSONDecodeError:
                        pass

                # Strategy 3: Original bracket-closing approach
                if not repaired:
                    open_brackets = json_str_clean.count('[') - json_str_clean.count(']')
                    open_braces = json_str_clean.count('{') - json_str_clean.count('}')

                    # Remove trailing incomplete strings
                    if json_str_clean.rstrip().endswith('"') or json_str_clean.rstrip().endswith("'"):
                        json_str_clean = re.sub(r',\s*"[^"]*$', '', json_str_clean)
                        json_str_clean = re.sub(r',\s*\'[^\']*$', '', json_str_clean)

                    json_str_clean = json_str_clean.rstrip().rstrip(',')
                    json_str_clean += ']' * open_brackets + '}' * open_braces

                    log(f"Repaired JSON (added {open_brackets} ] and {open_braces} }})", "SUMMARY")
                    summary_data = json.loads(json_str_clean)

            with summary_lock:
                # Update recent details (detailed view of last ~5 min)
                # Support both old field name (recent_bullets) and new (recent_details)
                raw_details = summary_data.get("recent_details", summary_data.get("recent_bullets", []))

                # Handle case where LLM outputs objects instead of strings
                # e.g., [{"point": "...", "fact": "..."}] instead of ["..."]
                current_summary["recent_bullets"] = []
                for item in raw_details:
                    if isinstance(item, str):
                        current_summary["recent_bullets"].append(item)
                    elif isinstance(item, dict):
                        # Extract text from object - try common keys
                        text = item.get("point") or item.get("text") or item.get("fact") or item.get("detail") or str(item)
                        if item.get("fact") and item.get("point"):
                            text = f"{item.get('point')} - {item.get('fact')}"
                        current_summary["recent_bullets"].append(text)

                # Update timeline sections (chronological with time ranges)
                # Support both old field name (overview_sections) and new (timeline)
                timeline = summary_data.get("timeline", summary_data.get("overview_sections", []))
                # Convert timeline format to overview_sections format for UI compatibility
                current_summary["overview_sections"] = []
                for item in timeline:
                    if isinstance(item, dict):
                        # New format: {title, content} -> convert to {title, bullets}
                        if "content" in item:
                            current_summary["overview_sections"].append({
                                "title": item.get("title", ""),
                                "content": item.get("content", "")  # Keep as content for paragraph display
                            })
                        else:
                            # Old format: {title, bullets}
                            current_summary["overview_sections"].append(item)

                current_summary["last_updated"] = time.time()
                current_summary["segments_summarized"] = len(all_meeting_segments)
                current_summary["time_range"] = time_range

            log(f"Generated: {len(current_summary['recent_bullets'])} details, {len(current_summary['overview_sections'])} timeline sections", "SUMMARY")
            return current_summary
        else:
            log(f"Could not parse JSON from response: {generated_text[:300]}", "SUMMARY")
            return None

    except Exception as e:
        log(f"Error generating summary: {e}", "SUMMARY")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # Clear summarization timestamp (UI feedback)
        # Note: summarization_started_at is already declared global at function start
        summarization_started_at = None
        # Model rotation: unload summarization to CPU and release transcription lock
        if rotation_enabled:
            unload_summarization_to_cpu()
            transcription_lock.release()
            log("Transcription lock released", "SUMMARY")


def check_and_generate_summary():
    """Check if it's time to schedule a summary (called after each transcription completes)"""
    global last_summary_time, summary_pending

    if summarization_pipe is None or not Config.ENABLE_SUMMARIZATION:
        return

    if summarization_paused:
        return  # Kill switch is active

    current_time = time.time()

    # Check if it's time for a summary (transcript file must exist)
    if (current_time - last_summary_time >= Config.SUMMARY_INTERVAL_SECONDS
            and TRANSCRIPT_FILE and TRANSCRIPT_FILE.exists()
            and not summary_pending):
        # Mark summary as pending - it will run after current transcription completes
        summary_pending = True
        log("Summary scheduled - will generate after current processing completes", "SUMMARY")


def run_pending_summary():
    """Actually run the summary generation.

    Strategy: Read last 5 minutes of transcript directly from file.
    """
    global last_summary_time, last_summary_segment_count, all_meeting_segments, meeting_start_time, summary_pending, previous_overview_text

    if not summary_pending:
        return

    if summarization_pipe is None or not Config.ENABLE_SUMMARIZATION:
        summary_pending = False
        return

    if summarization_paused:
        summary_pending = False
        return  # Kill switch is active

    # Read last 5 minutes directly from transcript file
    import re
    from datetime import datetime

    current_time = time.time()
    five_minutes_ago = current_time - 300

    recent_lines = []
    first_timestamp_str = None
    last_timestamp_str = None

    if TRANSCRIPT_FILE and TRANSCRIPT_FILE.exists():
        try:
            with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Parse: [2025-12-01 19:56:33] [0.00s-11.00s] Text...
                    match = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] \[[\d.]+s-[\d.]+s\] (.+)', line)
                    if match:
                        timestamp_str = match.group(1)
                        text = match.group(2)

                        # Track first timestamp for time range
                        if first_timestamp_str is None:
                            first_timestamp_str = timestamp_str

                        # Parse timestamp to check if within last 5 min
                        try:
                            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                            unix_time = dt.timestamp()

                            if unix_time >= five_minutes_ago:
                                recent_lines.append(f"[{timestamp_str}] {text}")
                                last_timestamp_str = timestamp_str
                        except ValueError:
                            pass
        except Exception as e:
            log(f"Error reading transcript file: {e}", "SUMMARY")
            summary_pending = False
            return

    if len(recent_lines) == 0:
        log("No transcript content in last 10 minutes", "SUMMARY")
        summary_pending = False
        return

    new_transcript_text = "\n".join(recent_lines)

    # Get time range from what we read
    time_range = f"{first_timestamp_str} - {last_timestamp_str}" if first_timestamp_str and last_timestamp_str else ""

    # Calculate minutes since meeting start (use first timestamp from file)
    minutes_since_start = 0
    if first_timestamp_str:
        try:
            first_dt = datetime.strptime(first_timestamp_str, "%Y-%m-%d %H:%M:%S")
            minutes_since_start = int((current_time - first_dt.timestamp()) / 60)
        except ValueError:
            pass

    # Only summarize if we have content
    if len(new_transcript_text) > 50:
        log(f"Generating summary: {len(recent_lines)} lines in last 10min from transcript file", "SUMMARY")
        log(f"Context: {len(previous_overview_text)} chars prev, {len(new_transcript_text)} chars recent, {minutes_since_start}min total", "SUMMARY")

        # Generate in background thread to not block
        def generate_and_emit():
            global last_summary_time, last_summary_segment_count, summary_pending, previous_overview_text
            try:
                summary = generate_summary(previous_overview_text, new_transcript_text, time_range, minutes_since_start)
                if summary:
                    last_summary_time = time.time()
                    last_summary_segment_count = len(all_meeting_segments)

                    # Store the overview text for next iteration (summarization chaining)
                    overview_lines = []
                    for section in summary.get("overview_sections", []):
                        overview_lines.append(f"## {section.get('title', 'Topic')}")
                        for bullet in section.get("bullets", []):
                            overview_lines.append(f"- {bullet}")
                    previous_overview_text = "\n".join(overview_lines)

                    # Emit original summary to admin room
                    socketio.emit('summary_update', summary, room='admin')

                    # Translate and emit to each language room
                    for lang_code in active_language_viewers:
                        if active_language_viewers[lang_code] > 0:
                            # Translate summary for this language
                            translated_summary = translate_summary(summary, lang_code)
                            socketio.emit('summary_update', translated_summary, room=f'lang_{lang_code}')
            finally:
                summary_pending = False

        summary_thread = threading.Thread(target=generate_and_emit, daemon=True)
        summary_thread.start()
    else:
        summary_pending = False


def translate_summary(summary, target_lang, source_lang="en"):
    """Translate a summary object to target language"""
    if target_lang == source_lang:
        return summary

    try:
        translated_summary = {
            "recent_bullets": [],
            "overview_sections": [],
            "last_updated": summary.get("last_updated"),
            "segments_summarized": summary.get("segments_summarized", 0),
            "time_range": summary.get("time_range", "")  # Don't translate timestamps
        }

        # Translate recent bullets
        for bullet in summary.get("recent_bullets", []):
            translated_bullet = translate_text(bullet, source_lang, target_lang)
            translated_summary["recent_bullets"].append(translated_bullet)

        # Translate overview sections (title + content or bullets)
        for section in summary.get("overview_sections", []):
            translated_section = {
                "title": translate_text(section.get("title", ""), source_lang, target_lang),
            }
            # Handle new paragraph format (content) or old bullet format
            if "content" in section:
                translated_section["content"] = translate_text(section.get("content", ""), source_lang, target_lang)
            elif "bullets" in section:
                translated_section["bullets"] = []
                for bullet in section.get("bullets", []):
                    translated_section["bullets"].append(
                        translate_text(bullet, source_lang, target_lang)
                    )
            translated_summary["overview_sections"].append(translated_section)

        return translated_summary
    except Exception as e:
        print(f"[SUMMARY] Error translating summary to {target_lang}: {e}")
        return summary  # Return original if translation fails


@torch.inference_mode()
def translate_text(text, source_lang, target_lang):
    """Translate text using translation model (M2M100 or NLLB)"""
    if source_lang == target_lang:
        return text

    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    if not text.strip():
        return text

    try:
        # Convert language codes for NLLB models
        src_code = Config.get_translation_lang_code(source_lang)
        tgt_code = Config.get_translation_lang_code(target_lang)

        # Set source language on tokenizer (both NLLB and M2M100 use this approach)
        translation_tokenizer.src_lang = src_code
        encoded = translation_tokenizer(text, return_tensors="pt")

        if Config.DEVICE == "cuda":
            encoded = {k: v.cuda() for k, v in encoded.items()}

        # Get forced BOS token ID for target language
        # For NLLB models, use convert_tokens_to_ids with the language code
        # For M2M100 models, use get_lang_id
        if hasattr(translation_tokenizer, 'get_lang_id'):
            # M2M100 tokenizer
            forced_bos_token_id = translation_tokenizer.get_lang_id(tgt_code)
        else:
            # NLLB tokenizer - convert language code to token ID
            forced_bos_token_id = translation_tokenizer.convert_tokens_to_ids(tgt_code)

        generated_tokens = translation_model.generate(
            **encoded, forced_bos_token_id=forced_bos_token_id, max_length=512
        )

        translated = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        return translated
    except Exception as e:
        print(f"Translation error ({source_lang} -> {target_lang}): {e}")
        return f"[Translation error: {str(e)}]"


@torch.inference_mode()
def perform_speaker_diarization(audio_data, sample_rate):
    """Perform speaker diarization on audio data"""
    if diarization_pipeline is None:
        return None

    try:
        # Ensure audio is properly normalized and in correct format
        # pyannote expects audio in range [-1, 1]
        audio_normalized = audio_data.astype(np.float32)

        # Ensure mono audio
        if audio_normalized.ndim > 1:
            audio_normalized = audio_normalized.mean(axis=1)

        # Create audio dict for pyannote
        # pyannote expects shape (channels, samples)
        audio_tensor = torch.from_numpy(audio_normalized).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension (1, samples)

        waveform_dict = {
            "waveform": audio_tensor,
            "sample_rate": sample_rate
        }

        print(f"[DIARIZATION] Audio shape: {audio_tensor.shape}, duration: {audio_tensor.shape[1] / sample_rate:.2f}s")

        # Run diarization - let the model automatically detect the number of speakers
        diarization_output = diarization_pipeline(waveform_dict)

        # Extract speaker segments from the DiarizeOutput object
        # In pyannote.audio 4.0+, the pipeline returns a DiarizeOutput object
        # with a speaker_diarization attribute that contains the Annotation object
        speaker_segments = []
        unique_speakers = set()

        for turn, _, speaker in diarization_output.speaker_diarization.itertracks(yield_label=True):
            # Filter out very short segments
            duration = turn.end - turn.start
            if duration >= Config.MIN_SPEAKER_DURATION:
                unique_speakers.add(speaker)
                speaker_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })

        print(f"[DIARIZATION] Detected {len(unique_speakers)} unique speakers: {sorted(unique_speakers)}")
        print(f"[DIARIZATION] Found {len(speaker_segments)} speaker segments")

        for seg in speaker_segments:
            print(f"[DIARIZATION]   {seg['speaker']}: {seg['start']:.2f}s - {seg['end']:.2f}s")

        return speaker_segments
    except Exception as e:
        print(f"[ERROR] Diarization error: {e}")
        import traceback
        traceback.print_exc()
        return None


def resolve_speaker_identity(speaker_segments, batch_end_ts_ms, audio_duration_secs):
    """Match pyannote speaker IDs to real names via speaker_timeline overlap.

    Returns {original_pyannote_id: real_name} for speakers with >= 30% time overlap.
    Considers both closed (speaker_timeline) and still-open (_active_speaker_starts)
    intervals, since a speaker who started talking during this batch may not have
    emitted speaker_end yet when the transcription thread kicks off.

    Multi-voice disambiguation: when two or more distinct pyannote IDs in the
    same batch resolve to the SAME caption name (e.g. two people speaking from
    one shared Meet account), we suffix each name with "#1", "#2", etc., ordered
    chronologically by first appearance in the batch. This preserves pyannote's
    split instead of collapsing both voices to a single label.
    """
    if batch_end_ts_ms is None or not speaker_segments:
        return {}

    # Build the full set of speaker intervals to consider.
    intervals = list(speaker_timeline)  # closed: (start_ms, end_ms, name)
    now_ms = int(time.time() * 1000)
    for name, start_ms in _active_speaker_starts.items():
        intervals.append((start_ms, now_ms, name))

    if not intervals:
        return {}

    from collections import defaultdict
    batch_start_ms = batch_end_ts_ms - audio_duration_secs * 1000

    # Group pyannote segments by their original id, preserving first-appearance
    # order (we iterate speaker_segments in chronological order).
    speaker_times = {}
    first_seen = {}
    for seg in speaker_segments:
        orig = seg["speaker"]
        seg_start_ms = batch_start_ms + seg["start"] * 1000
        seg_end_ms = batch_start_ms + seg["end"] * 1000
        if orig not in speaker_times:
            speaker_times[orig] = []
            first_seen[orig] = seg_start_ms
        speaker_times[orig].append((seg_start_ms, seg_end_ms))

    # First pass: majority-vote caption name per pyannote id.
    raw_resolved = {}  # orig_id -> caption_name
    for original_id, time_ranges in speaker_times.items():
        name_overlap = defaultdict(float)
        for seg_start_ms, seg_end_ms in time_ranges:
            for (tl_start, tl_end, name) in intervals:
                overlap = max(0.0, min(seg_end_ms, tl_end) - max(seg_start_ms, tl_start))
                if overlap > 0:
                    name_overlap[name] += overlap
        if not name_overlap:
            continue
        best_name = max(name_overlap, key=name_overlap.get)
        total_ms = sum(end - start for start, end in time_ranges)
        if total_ms > 0 and name_overlap[best_name] / total_ms >= 0.30:
            raw_resolved[original_id] = best_name

    # Second pass: disambiguate collisions. If multiple pyannote IDs map to the
    # same caption name, assign each a "#N" suffix in chronological order.
    by_name = defaultdict(list)  # caption_name -> [orig_id...] (first-appearance order)
    for orig_id in sorted(raw_resolved, key=lambda oid: first_seen[oid]):
        by_name[raw_resolved[orig_id]].append(orig_id)

    resolved = {}
    for name, ids in by_name.items():
        if len(ids) == 1:
            resolved[ids[0]] = name
        else:
            for i, orig_id in enumerate(ids, start=1):
                resolved[orig_id] = f"{name} Speaker #{i}"

    return resolved


@torch.inference_mode()
def partial_transcribe_and_emit(audio_data, utterance_id, speaker_hint, batch_end_ts_ms):
    """Streaming-style partial transcription.

    Runs Whisper + translations on a growing buffer snapshot while a speaker
    is still talking, then emits a WS payload with is_partial=true. Frontends
    key segments by utterance_id and replace the row when the final pass
    arrives with the same id. Skips diarization (uses speaker_hint directly),
    skips transcript file write, skips summarization accumulation.
    """
    # Share the same GPU lock as the final path so Whisper calls serialise.
    if Config.ENABLE_MODEL_ROTATION:
        if not transcription_lock.acquire(blocking=False):
            # Final pass is running — skip this tick; next one will retry.
            return
    else:
        transcription_lock.acquire()

    try:
        audio_duration = len(audio_data) / SAMPLE_RATE
        if np.abs(audio_data).mean() < Config.MIN_AUDIO_LEVEL:
            return

        result = transcription_pipe(audio_data)
        full_transcript = (result["text"] if isinstance(result, dict) else str(result)).strip()
        del result
        if Config.DEVICE == "cuda":
            torch.cuda.empty_cache()
        if not full_transcript:
            return

        try:
            source_lang = detect(full_transcript)
        except LangDetectException:
            source_lang = "en"

        speaker = speaker_hint or "Speaker 1"
        segment = {"text": full_transcript, "speaker": speaker, "start": 0.0, "end": audio_duration}
        # Anchor the batch on wall-clock so the viewer can compute the true
        # moment each sentence was spoken (seg.start is seconds from batch start).
        batch_anchor_ms = batch_end_ts_ms if batch_end_ts_ms else int(time.time() * 1000)
        ws_payload = {
            "transcript": full_transcript,
            "source_language": source_lang,
            "translations": {},
            "timestamp": time.time(),
            "segments": [segment],
            "utterance_id": utterance_id,
            "is_partial": True,
            "batch_end_ts_ms": batch_anchor_ms,
            "audio_duration_secs": audio_duration,
        }
        socketio.emit("new_translation", ws_payload, room="admin")
        for lang_code, n in active_language_viewers.items():
            if n > 0:
                socketio.emit("new_translation", ws_payload, room=f"lang_{lang_code}")

        # Translate per language and emit a second partial update with translated segments.
        languages_to_translate = [
            lang for lang in Config.TARGET_LANGUAGES
            if active_language_viewers.get(lang["code"], 0) > 0 and lang["code"] != source_lang
        ]
        translated_segments_by_lang = {lang["code"]: [] for lang in languages_to_translate}
        for lang_info in languages_to_translate:
            translated = translate_text(full_transcript, source_lang, lang_info["code"])
            translated_segments_by_lang[lang_info["code"]].append({
                "text": translated, "speaker": speaker, "start": 0.0, "end": audio_duration,
            })
        if source_lang in [lang["code"] for lang in Config.TARGET_LANGUAGES] and \
                active_language_viewers.get(source_lang, 0) > 0:
            translated_segments_by_lang[source_lang] = [dict(segment)]

        ws_payload_final = {
            "transcript": full_transcript,
            "source_language": source_lang,
            "translated_segments": translated_segments_by_lang,
            "timestamp": ws_payload["timestamp"],
            "segments": [segment],
            "utterance_id": utterance_id,
            "is_partial": True,
            "is_update": True,
            "batch_end_ts_ms": batch_anchor_ms,
            "audio_duration_secs": audio_duration,
        }
        socketio.emit("new_translation", ws_payload_final, room="admin")
        for lang_code in translated_segments_by_lang:
            if active_language_viewers.get(lang_code, 0) > 0:
                socketio.emit("new_translation", ws_payload_final, room=f"lang_{lang_code}")

        log(f"Partial {utterance_id[:6]} ({audio_duration:.1f}s): {full_transcript[:60]}{'...' if len(full_transcript) > 60 else ''}", "PARTIAL")
    except Exception as e:
        print(f"[PARTIAL] error: {e}")
    finally:
        try:
            transcription_lock.release()
        except RuntimeError:
            pass


@torch.inference_mode()
def transcribe_and_translate(audio_data, audio_duration, batch_end_ts_ms=None, utterance_id=None):
    """Background thread for transcription and translation with speaker diarization"""
    global is_processing, all_meeting_segments

    # Acquire transcription lock - this ensures we don't run while summarization is on GPU
    # If summarization is waiting for the lock, we'll wait here until it's done
    if Config.ENABLE_MODEL_ROTATION:
        if summarization_on_gpu:
            log("Waiting for summarization to complete...", "TRANSCRIBE")
        transcription_lock.acquire()

    try:
        # Timestamp for logging and transcript file
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Perform speaker diarization first (if enabled)
        speaker_segments = None
        if Config.ENABLE_DIARIZATION:
            log("Performing speaker diarization...", "DIARIZE")
            speaker_segments = perform_speaker_diarization(audio_data, SAMPLE_RATE)

        # Transcribe with timestamps
        result = transcription_pipe(audio_data)

        # Extract full transcript and chunks IMMEDIATELY to avoid keeping reference to result
        if isinstance(result, dict) and "text" in result:
            full_transcript = result["text"].strip()
            # Deep copy the chunks data to avoid keeping reference to result object
            chunks = []
            for chunk in result.get("chunks", []):
                if isinstance(chunk, dict):
                    # Extract only the data we need, not the full chunk object
                    chunks.append({
                        "text": str(chunk.get("text", "")),
                        "timestamp": tuple(chunk.get("timestamp", (0, 0)))
                    })
        else:
            full_transcript = str(result).strip()
            chunks = []

        # Explicitly delete the result object to release Whisper's internal tensors
        del result

        # Clear CUDA cache to release unused memory
        if Config.DEVICE == "cuda":
            torch.cuda.empty_cache()

        if not full_transcript:
            is_processing = False
            return

        # Normalize all-caps text to sentence case
        # If the text is mostly uppercase (more than 70% caps), convert to sentence case
        def normalize_caps(text):
            if not text:
                return text
            # Count uppercase vs total letters
            letters = [c for c in text if c.isalpha()]
            if not letters:
                return text
            uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters)

            # If more than 70% uppercase, it's likely all-caps - convert to sentence case
            if uppercase_ratio > 0.7:
                # Use capitalize() for first letter, lower() for rest
                return text.capitalize()
            return text

        full_transcript = normalize_caps(full_transcript)

        # Detect source language from full transcript
        try:
            source_lang = detect(full_transcript)
        except LangDetectException:
            source_lang = "en"

        # Process segments with speaker labels using word-level timestamps
        segments_with_speakers = []

        if speaker_segments and chunks:
            # First, renumber speakers chronologically (first speaker to appear becomes SPEAKER_01)
            sorted_speakers = sorted(speaker_segments, key=lambda x: x["start"])
            speaker_mapping = {}
            speaker_counter = 1

            for seg in sorted_speakers:
                original_id = seg["speaker"]
                if original_id not in speaker_mapping:
                    speaker_mapping[original_id] = f"SPEAKER_{speaker_counter:02d}"
                    speaker_counter += 1

            # Phase 5: resolve pyannote IDs → real names from speaker_timeline.
            resolved_names = resolve_speaker_identity(speaker_segments, batch_end_ts_ms, audio_duration)
            if resolved_names:
                for orig_id, real_name in resolved_names.items():
                    speaker_xx = speaker_mapping.get(orig_id, orig_id)
                    log(f"Resolved {speaker_xx} → {real_name}", "BOT")
                    socketio.emit("rename_speaker", {"speaker_id": speaker_xx, "name": real_name})

            # Extract all words with timestamps from chunks
            all_words = []
            for chunk in chunks:
                # With return_timestamps="word", chunks contain word-level timestamps
                if isinstance(chunk, dict) and "timestamp" in chunk:
                    word_timestamp = chunk.get("timestamp", [0, 0])
                    word_text = normalize_caps(chunk.get("text", "").strip())
                    if word_text and word_timestamp:
                        word_start = word_timestamp[0] if word_timestamp[0] is not None else 0
                        word_end = word_timestamp[1] if word_timestamp[1] is not None else word_start
                        all_words.append({
                            "text": word_text,
                            "start": word_start,
                            "end": word_end
                        })


            # Assign each word to a speaker segment based on overlap
            words_with_speakers = []
            for word in all_words:
                # Find the speaker segment with maximum overlap
                best_speaker = "Unknown"
                best_segment_idx = -1
                max_overlap = 0

                for idx, seg in enumerate(speaker_segments):
                    overlap_start = max(word["start"], seg["start"])
                    overlap_end = min(word["end"], seg["end"])
                    overlap = max(0, overlap_end - overlap_start)

                    if overlap > max_overlap:
                        max_overlap = overlap
                        orig = seg["speaker"]
                        # Prefer resolved real name; fall back to renumbered SPEAKER_XX.
                        best_speaker = resolved_names.get(orig, speaker_mapping.get(orig, orig))
                        best_segment_idx = idx

                words_with_speakers.append({
                    "text": word["text"],
                    "speaker": best_speaker,
                    "segment_idx": best_segment_idx,  # Track which diarization segment this belongs to
                    "start": word["start"],
                    "end": word["end"]
                })

            # Group words by their diarization segment (not just by speaker!)
            # This ensures we get separate messages for each speaker turn
            if words_with_speakers:
                current_segment = {
                    "speaker": words_with_speakers[0]["speaker"],
                    "segment_idx": words_with_speakers[0]["segment_idx"],
                    "words": [words_with_speakers[0]["text"]],
                    "start": words_with_speakers[0]["start"],
                    "end": words_with_speakers[0]["end"]
                }

                for word in words_with_speakers[1:]:
                    # Only group if same speaker AND same diarization segment
                    if (word["speaker"] == current_segment["speaker"] and
                        word["segment_idx"] == current_segment["segment_idx"]):
                        # Same segment - add word to current segment
                        current_segment["words"].append(word["text"])
                        current_segment["end"] = word["end"]
                    else:
                        # Different segment - save current segment and start new one
                        segments_with_speakers.append({
                            "text": " ".join(current_segment["words"]),
                            "speaker": current_segment["speaker"],
                            "start": current_segment["start"],
                            "end": current_segment["end"]
                        })

                        current_segment = {
                            "speaker": word["speaker"],
                            "segment_idx": word["segment_idx"],
                            "words": [word["text"]],
                            "start": word["start"],
                            "end": word["end"]
                        }

                # Don't forget the last segment
                segments_with_speakers.append({
                    "text": " ".join(current_segment["words"]),
                    "speaker": current_segment["speaker"],
                    "start": current_segment["start"],
                    "end": current_segment["end"]
                })

            # Filter out very short "Unknown" segments (noise from words between speaker turns)
            segments_with_speakers = [
                seg for seg in segments_with_speakers
                if not (seg["speaker"] == "Unknown" and len(seg["text"].strip()) < 10)
            ]

            # Merge consecutive segments from the same speaker (for cleaner output)
            # This happens when the same speaker has multiple diarization segments in a row
            merged_segments = []
            if segments_with_speakers:
                current_merged = segments_with_speakers[0].copy()

                for seg in segments_with_speakers[1:]:
                    # If same speaker and segments are close together (within 2 seconds gap)
                    time_gap = seg["start"] - current_merged["end"]
                    if seg["speaker"] == current_merged["speaker"] and time_gap < 2.0:
                        # Merge: extend text and end time
                        current_merged["text"] += " " + seg["text"]
                        current_merged["end"] = seg["end"]
                    else:
                        # Different speaker or large gap - save current and start new
                        merged_segments.append(current_merged)
                        current_merged = seg.copy()

                # Don't forget the last merged segment
                merged_segments.append(current_merged)
                segments_with_speakers = merged_segments

            # Log each segment with speaker info
            for seg in segments_with_speakers:
                print(f"[{timestamp}] [{seg['start']:.2f}s-{seg['end']:.2f}s] {seg['speaker']}: {seg['text']}")
        else:
            # No diarization or chunks - use full transcript
            segments_with_speakers.append({
                "text": full_transcript,
                "speaker": "Speaker 1",
                "start": 0,
                "end": audio_duration
            })
            print(f"[{timestamp}] [{audio_duration:.2f}s] Speaker 1: {full_transcript}")

        # Write to transcript file
        with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
            for seg in segments_with_speakers:
                f.write(f"[{timestamp}] [{seg['start']:.2f}s-{seg['end']:.2f}s] {seg['text']}\n")

        # Accumulate segments for summarization (add timestamp for reference)
        current_unix_time = time.time()
        for seg in segments_with_speakers:
            seg_with_time = seg.copy()
            seg_with_time["timestamp"] = timestamp  # Add human-readable timestamp
            seg_with_time["unix_time"] = current_unix_time  # Add Unix timestamp for filtering
            all_meeting_segments.append(seg_with_time)

        # Set meeting start time on first transcription
        global meeting_start_time
        if meeting_start_time is None:
            meeting_start_time = time.time()
            print(f"[SUMMARY] Meeting started at {timestamp}")

        # Keep all_meeting_segments reasonable (last 500 segments max - roughly 1-2 hours)
        # With summarization chaining, we only need new segments since last summary
        if len(all_meeting_segments) > 500:
            all_meeting_segments[:] = all_meeting_segments[-500:]

        # Check if it's time to generate a summary
        check_and_generate_summary()

        # Wall-clock anchor for the batch so the viewer can compute each
        # segment's true spoken-at time from batch_end_ts_ms + seg.start.
        final_batch_anchor_ms = batch_end_ts_ms if batch_end_ts_ms else int(time.time() * 1000)

        # IMMEDIATELY send transcription to UI (before translations)
        # This makes the UI feel much more responsive
        ws_payload_initial = {
            "transcript": full_transcript,
            "source_language": source_lang,
            "translations": {},  # Empty for now
            "timestamp": time.time(),
            "segments": segments_with_speakers,
            "utterance_id": utterance_id,
            "is_partial": False,
            "is_initial": True,  # Flag to indicate translations are coming
            "batch_end_ts_ms": final_batch_anchor_ms,
            "audio_duration_secs": audio_duration,
        }

        if Config.DEBUG:
            print(f"\n[WS EMIT 1/2] Sending source transcript ({len(segments_with_speakers)} segments):")
            for i, seg in enumerate(segments_with_speakers):
                print(f"  [{i+1}] [{seg['start']:.2f}s-{seg['end']:.2f}s] {seg['speaker']}: {seg['text'][:80]}{'...' if len(seg['text']) > 80 else ''}")

            # DEBUG: Print actual JSON payload to see what's being sent
            import json
            print(f"\n[WS DEBUG] Full payload being sent:")
            print(json.dumps(ws_payload_initial, indent=2, ensure_ascii=False))

        # Send initial message to admin room and all active language rooms
        socketio.emit("new_translation", ws_payload_initial, room='admin')
        for lang_code in active_language_viewers:
            if active_language_viewers[lang_code] > 0:
                socketio.emit("new_translation", ws_payload_initial, room=f'lang_{lang_code}')

        # Only translate for languages that have active viewers
        # We no longer translate all languages just because admin is present
        # Admin can see translations for languages that have viewers
        # Also skip translating to the source language (no need for FR -> FR)
        languages_to_translate = [
            lang for lang in Config.TARGET_LANGUAGES
            if active_language_viewers.get(lang["code"], 0) > 0 and lang["code"] != source_lang
        ]

        print(f"[TRANSLATION] Translating for {len(languages_to_translate)} languages: {[l['code'] for l in languages_to_translate]}")
        if source_lang in [lang["code"] for lang in Config.TARGET_LANGUAGES]:
            print(f"[TRANSLATION] Skipping {source_lang} (source language, no translation needed)")

        # Now translate each segment individually for each language
        def translate_segment_for_language(lang_info, segment):
            target_lang = lang_info["code"]
            translated_text = translate_text(segment["text"], source_lang, target_lang)
            return target_lang, {
                "text": translated_text,
                "speaker": segment["speaker"],
                "start": segment["start"],
                "end": segment["end"]
            }

        # Translate all segments for active languages in parallel
        translated_segments_by_lang = {lang["code"]: [] for lang in languages_to_translate}

        if languages_to_translate:
            with ThreadPoolExecutor(max_workers=len(languages_to_translate) * len(segments_with_speakers)) as executor:
                # Create all translation tasks
                futures = []
                for lang_info in languages_to_translate:
                    for segment in segments_with_speakers:
                        future = executor.submit(translate_segment_for_language, lang_info, segment)
                        futures.append((future, lang_info["code"]))

                # Collect results
                for future, lang_code in futures:
                    _, translated_segment = future.result()
                    translated_segments_by_lang[lang_code].append(translated_segment)

        # Add source language segments (no translation needed, just copy from segments_with_speakers)
        if source_lang in [lang["code"] for lang in Config.TARGET_LANGUAGES]:
            if active_language_viewers.get(source_lang, 0) > 0:
                translated_segments_by_lang[source_lang] = [
                    {
                        "text": seg["text"],
                        "speaker": seg["speaker"],
                        "start": seg["start"],
                        "end": seg["end"]
                    }
                    for seg in segments_with_speakers
                ]

        # Send translations update with per-segment translations
        ws_payload_final = {
            "transcript": full_transcript,
            "source_language": source_lang,
            "translated_segments": translated_segments_by_lang,  # Per-segment translations
            "timestamp": ws_payload_initial["timestamp"],  # Use same timestamp
            "segments": segments_with_speakers,  # Original source segments
            "utterance_id": utterance_id,
            "is_partial": False,
            "is_update": True,  # Flag to indicate this is a translation update
            "batch_end_ts_ms": final_batch_anchor_ms,
            "audio_duration_secs": audio_duration,
        }

        if Config.DEBUG:
            print(f"[WS EMIT 2/2] Sending translated segments for: {list(translated_segments_by_lang.keys())}")
            for lang, segs in translated_segments_by_lang.items():
                print(f"  {lang}: {len(segs)} segments")
            if Config.DEVICE == "cuda":
                print(f"[GPU MEMORY] After Translations: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB reserved")

        # Send translations to admin room and language-specific rooms
        socketio.emit("new_translation", ws_payload_final, room='admin')
        for lang_code in translated_segments_by_lang.keys():
            if active_language_viewers.get(lang_code, 0) > 0:
                socketio.emit("new_translation", ws_payload_final, room=f'lang_{lang_code}')
    except Exception as e:
        print(f"[ERROR] Processing error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        is_processing = False
        # Release transcription lock if model rotation is enabled
        if Config.ENABLE_MODEL_ROTATION:
            transcription_lock.release()
        # Now that transcription + translation is complete, run any pending summary
        # This ensures model rotation doesn't interrupt active transcription
        run_pending_summary()


def process_audio():
    """Process audio chunks and emit transcriptions/translations"""
    global is_processing

    buffer = []
    silence_counter = 0

    # Wait for actual_sample_rate to be set by audio stream initialization
    while actual_sample_rate == 48000 and not is_listening:
        time.sleep(0.1)

    print("[AUDIO] Started process_audio thread", flush=True)
    print(f"[AUDIO] Using actual_sample_rate: {actual_sample_rate}Hz")

    while is_listening:
        try:
            # Skip if already processing
            if is_processing:
                # Safety check: if buffer is 3x max size, force release the lock
                # This prevents getting stuck if background thread crashes
                max_chunks = int(actual_sample_rate * audio_thresholds["max_audio_length"] / CHUNK_SIZE)
                if len(buffer) > max_chunks * 3:
                    print(f"[AUDIO] SAFETY: Releasing stuck processing lock (buffer: {len(buffer)} > {max_chunks * 3})")
                    is_processing = False
                    buffer = []
                    silence_counter = 0
                    continue

                # Continue collecting audio into buffer even while processing
                # This prevents losing audio between sentences
                try:
                    chunk = audio_queue.get(timeout=0.1)

                    # Calculate audio level for UI (admin only)
                    audio_level = np.abs(chunk).mean()
                    socketio.emit("audio_level", {"level": float(audio_level * 100)}, room='admin')

                    # Add to buffer for next sentence
                    buffer.append(chunk)

                    # Mint a fresh utterance_id for the next batch while the
                    # previous final is still running. Without this, once we
                    # exit is_processing the buffer already has many chunks and
                    # the mint condition never fires → partials never trigger.
                    global _current_utterance_id, _last_partial_ts
                    if bot_connected and _current_utterance_id is None:
                        _current_utterance_id = uuid.uuid4().hex[:12]
                        _last_partial_ts = time.time()

                    # Update silence counter
                    is_silent = audio_level < audio_thresholds["silence_threshold"]
                    if is_silent:
                        silence_counter += 1
                    else:
                        silence_counter = 0

                    # Emit debug data showing we're still collecting audio (admin only)
                    min_chunks = int(actual_sample_rate * audio_thresholds["min_audio_length"] / CHUNK_SIZE)

                    socketio.emit(
                        "debug_data",
                        {
                            "audio_level": float(audio_level),
                            "buffer_chunks": len(buffer),
                            "silence_counter": silence_counter,
                            "is_processing": True,
                            "min_chunks": min_chunks,
                            "max_chunks": max_chunks,
                            "silence_threshold": audio_thresholds["silence_threshold"],
                            "silence_chunks_req": audio_thresholds["silence_chunks"],
                        },
                        room='admin'
                    )
                except queue.Empty:
                    pass
                continue

            # Get audio chunk
            chunk = audio_queue.get(timeout=1)

            # Calculate audio level (admin only)
            audio_level = np.abs(chunk).mean()
            socketio.emit("audio_level", {"level": float(audio_level * 100)}, room='admin')

            # Calculate chunk limits dynamically based on current thresholds
            min_chunks = int(actual_sample_rate * audio_thresholds["min_audio_length"] / CHUNK_SIZE)
            max_chunks = int(actual_sample_rate * audio_thresholds["max_audio_length"] / CHUNK_SIZE)

            # Emit debug data for UI (admin only)
            socketio.emit(
                "debug_data",
                {
                    "audio_level": float(audio_level),
                    "buffer_chunks": len(buffer),
                    "silence_counter": silence_counter,
                    "is_processing": is_processing,
                    "min_chunks": min_chunks,
                    "max_chunks": max_chunks,
                    "silence_threshold": audio_thresholds["silence_threshold"],
                    "silence_chunks_req": audio_thresholds["silence_chunks"],
                },
                room='admin'
            )

            # Check if current chunk is silent
            is_silent = audio_level < audio_thresholds["silence_threshold"]

            if is_silent:
                silence_counter += 1
            else:
                silence_counter = 0  # Reset on any sound

            buffer.append(chunk)

            # Bot mode: only flush on speaker switch or 60 s cap. No silence
            # detection — a speaker's natural pauses emit speaker_end/start
            # toggles and we explicitly do NOT flush on those. Mic-only mode
            # falls back to the original level-based silence heuristic.
            global _pending_speaker_switch
            bot_mode = bot_connected
            if bot_mode:
                max_chunks = int(actual_sample_rate * BOT_MAX_BATCH_SEC / CHUNK_SIZE)

            speaker_switched = bot_mode and _pending_speaker_switch and len(buffer) >= min_chunks
            if speaker_switched:
                _pending_speaker_switch = False

            # Mint a fresh utterance_id if one isn't set (buffer just started
            # fresh). The is_processing=True branch may have done this already
            # if it collected chunks during a previous final — in that case
            # this is a no-op.
            if bot_mode and _current_utterance_id is None:
                _current_utterance_id = uuid.uuid4().hex[:12]
                _last_partial_ts = time.time()

            # Fire a partial transcription tick every partial_interval_sec
            # while the buffer keeps growing. Skips when a final is already
            # running — they'd contend on the GPU lock and a full flush is
            # imminent anyway. The interval is runtime-mutable.
            partial_interval = float(audio_thresholds.get("partial_interval_sec", 5))
            if (
                bot_mode
                and _current_utterance_id is not None
                and not is_processing
                and len(buffer) >= min_chunks
                and (time.time() - _last_partial_ts) >= partial_interval
            ):
                _last_partial_ts = time.time()
                audio_snapshot = np.concatenate(list(buffer), axis=0).flatten().astype(np.float32)
                if actual_sample_rate != SAMPLE_RATE:
                    audio_snapshot = resampy.resample(audio_snapshot, actual_sample_rate, SAMPLE_RATE)
                threading.Thread(
                    target=partial_transcribe_and_emit,
                    args=(audio_snapshot, _current_utterance_id, _current_bot_speaker, _last_capture_ts_ms),
                    daemon=True,
                ).start()

            # Level-based silence detection — only used when bot isn't driving batching.
            silence_detected = (
                not bot_mode
                and len(buffer) >= min_chunks
                and silence_counter >= audio_thresholds["silence_chunks"]
            )
            max_length_reached = len(buffer) >= max_chunks
            should_process = silence_detected or max_length_reached or speaker_switched

            if should_process:
                is_processing = True  # Set lock

                audio_data = np.concatenate(buffer, axis=0)
                buffer = []
                silence_counter = 0

                # Clear the queue BEFORE processing to avoid duplicate audio
                while not audio_queue.empty():
                    try:
                        audio_queue.get_nowait()
                    except queue.Empty:
                        break

                # Convert to float32
                audio_float = audio_data.flatten().astype(np.float32)

                # Reshape to (samples, channels) if stereo
                if num_channels == 2:
                    audio_float = audio_float.reshape(-1, 2)
                    # Convert stereo to mono by averaging channels
                    audio_mono = audio_float.mean(axis=1)
                else:
                    audio_mono = audio_float

                # Resample from actual_sample_rate to 16000Hz for Whisper
                if actual_sample_rate != SAMPLE_RATE:
                    audio_resampled = resampy.resample(audio_mono, actual_sample_rate, SAMPLE_RATE)
                else:
                    audio_resampled = audio_mono

                # Check average audio level to detect if it's mostly silence
                avg_audio_level = np.abs(audio_resampled).mean()

                # Skip transcription if audio is too quiet (likely silence/hallucination)
                if avg_audio_level < Config.MIN_AUDIO_LEVEL:
                    print(f"[AUDIO] Skipping transcription - audio too quiet ({avg_audio_level:.4f} < {Config.MIN_AUDIO_LEVEL})")
                    is_processing = False
                    continue

                # Calculate audio duration
                audio_duration = len(audio_resampled) / SAMPLE_RATE

                # Snapshot the wall-clock anchor for Phase 5 speaker resolution.
                batch_end_ts = _last_capture_ts_ms

                # Capture the utterance_id for the final emit, then clear so the
                # next batch mints a fresh one on its first chunk.
                final_utt_id = _current_utterance_id
                _current_utterance_id = None

                # Launch background thread for transcription and translation
                # This keeps the main loop responsive for WebSocket updates
                processing_thread = threading.Thread(
                    target=transcribe_and_translate, args=(audio_resampled, audio_duration, batch_end_ts, final_utt_id), daemon=True
                )
                processing_thread.start()

                # Emit processing started event for UI flash effect (admin only)
                socketio.emit("processing_started", {"timestamp": time.time()}, room='admin')

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[ERROR] Processing error: {e}")
            import traceback

            traceback.print_exc()
            is_processing = False  # Release lock on error
            continue


def _no_cache(resp):
    """Attach no-cache headers so the browser always fetches fresh HTML/CSS/JS.
    Applied to the viewer and admin pages during active development."""
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/")
def index():
    """Redirect to viewer page by default"""
    return _no_cache(app.make_response(render_template("viewer.html")))


@app.route("/viewer")
def viewer():
    """Serve the viewer page"""
    return _no_cache(app.make_response(render_template("viewer.html")))


@app.route("/admin")
def admin():
    """Serve the admin page"""
    return _no_cache(app.make_response(render_template("admin.html")))


@app.route("/api/config", methods=["GET"])
def get_config():
    """Get current configuration"""
    return jsonify(Config.to_dict())


@app.route("/api/config", methods=["POST"])
def update_config():
    """Update configuration - Not implemented for Config class"""
    # Config is now in config.py, changes require editing that file
    return jsonify({"status": "error", "message": "Config is now in config.py - edit that file to make changes"})


@app.route("/api/thresholds", methods=["GET"])
def get_thresholds():
    """Get current audio detection thresholds"""
    return jsonify(audio_thresholds)


@app.route("/api/thresholds", methods=["POST"])
def update_thresholds():
    """Update audio detection thresholds"""
    from flask import request

    data = request.json
    audio_thresholds.update(data)
    print(f"[CONFIG] Updated thresholds: {audio_thresholds}")
    return jsonify({"status": "success", "thresholds": audio_thresholds})


@app.route("/api/audio-devices", methods=["GET"])
def get_audio_devices():
    """Get list of available audio input devices"""
    try:
        p_audio = pyaudio.PyAudio()
        devices = []

        for i in range(p_audio.get_device_count()):
            try:
                dev_info = p_audio.get_device_info_by_index(i)

                # Only include devices with input channels
                if dev_info.get("maxInputChannels", 0) > 0:
                    is_loopback = dev_info.get("isLoopbackDevice", False)
                    is_default = (i == p_audio.get_default_input_device_info()["index"])

                    devices.append({
                        "index": i,
                        "name": dev_info["name"],
                        "channels": int(dev_info["maxInputChannels"]),
                        "sample_rate": int(dev_info["defaultSampleRate"]),
                        "is_loopback": is_loopback,
                        "is_default": is_default
                    })
            except Exception as e:
                # Skip devices that can't be queried
                continue

        p_audio.terminate()

        # Sort: loopback devices first, then by name
        devices.sort(key=lambda d: (not d["is_loopback"], d["name"]))

        return jsonify({
            "status": "success",
            "devices": devices,
            "selected_index": selected_audio_device_index
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/audio-device", methods=["POST"])
def set_audio_device():
    """Set the audio device to use for listening"""
    global selected_audio_device_index

    try:
        data = request.json
        device_index = data.get("device_index")

        # None means auto-detect
        if device_index is None:
            selected_audio_device_index = None
            print("[CONFIG] Audio device selection: auto-detect")
            return jsonify({"status": "success", "message": "Auto-detect enabled"})

        # Validate device exists and has input channels
        p_audio = pyaudio.PyAudio()
        try:
            dev_info = p_audio.get_device_info_by_index(device_index)
            if dev_info.get("maxInputChannels", 0) <= 0:
                p_audio.terminate()
                return jsonify({"status": "error", "message": "Device has no input channels"}), 400

            selected_audio_device_index = device_index
            print(f"[CONFIG] Audio device selection updated: index={device_index} ({dev_info['name']})")
            p_audio.terminate()

            return jsonify({
                "status": "success",
                "message": "Device selected. Will be used on next 'Start Listening'.",
                "device_index": device_index
            })

        except Exception as e:
            p_audio.terminate()
            return jsonify({"status": "error", "message": f"Invalid device index: {str(e)}"}), 400

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/admin/auth", methods=["POST"])
def admin_auth():
    """Authenticate admin user"""
    data = request.json
    password = data.get("password", "")

    if password == Config.ADMIN_PASSWORD:
        return jsonify({"status": "success", "authenticated": True})
    else:
        return jsonify({"status": "error", "authenticated": False, "message": "Invalid password"}), 401


@app.route("/api/viewer/auth", methods=["POST"])
def viewer_auth():
    """Authenticate viewer user with passphrase"""
    data = request.json
    password = data.get("password", "")

    if password == Config.VIEWER_PASSWORD:
        return jsonify({"status": "success", "authenticated": True})
    else:
        return jsonify({"status": "error", "authenticated": False, "message": "Invalid passphrase"}), 401


@app.route("/api/viewer/password", methods=["GET"])
def get_viewer_password():
    """Get the current viewer password (admin only - requires admin password in header)"""
    auth_header = request.headers.get("Authorization", "")
    if auth_header == Config.ADMIN_PASSWORD:
        return jsonify({"password": Config.VIEWER_PASSWORD})
    else:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401


@app.route("/api/languages", methods=["GET"])
def get_languages():
    """Get available languages for viewers"""
    return jsonify(Config.TARGET_LANGUAGES)


@app.route("/api/admin/languages", methods=["GET"])
def get_admin_languages():
    """Get languages to display in admin view"""
    return jsonify(Config.ADMIN_LANGUAGES)


def get_system_stats():
    """Get system resource statistics including VRAM usage"""
    stats = {
        "language_viewers": active_language_viewers,
        "total_viewers": sum(active_language_viewers.values()),
        "total_admins": len(admin_sessions),
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_used_gb": psutil.virtual_memory().used / 1024**3,
        "ram_total_gb": psutil.virtual_memory().total / 1024**3,
    }

    # Add GPU stats if available
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated(0) / 1024**3
        vram_reserved = torch.cuda.memory_reserved(0) / 1024**3
        stats["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "vram_total_gb": gpu_total_memory,
            "vram_allocated_gb": vram_allocated,
            "vram_reserved_gb": vram_reserved,
            "model_vram": model_vram_usage,
            # Dynamic VRAM = total allocated - sum of static model usage
            "vram_dynamic_gb": max(0, vram_allocated - sum(model_vram_usage.values())),
            # Model names for display
            "model_names": {
                "whisper": Config.WHISPER_MODEL,
                "translation": Config.TRANSLATION_MODEL,
                "diarization": Config.DIARIZATION_MODEL if Config.ENABLE_DIARIZATION else "disabled",
                "summarization": Config.SUMMARIZATION_MODEL if Config.ENABLE_SUMMARIZATION else "disabled",
            },
            # Model rotation status
            "summarization_on_gpu": summarization_on_gpu,
            "summarization_started_at": summarization_started_at,  # Timestamp for UI elapsed time
            "model_rotation_enabled": Config.ENABLE_MODEL_ROTATION,
        }
    else:
        stats["gpu"] = None

    return stats


@app.route("/api/admin/stats", methods=["GET"])
def get_admin_stats():
    """Get admin statistics including system resources"""
    return jsonify(get_system_stats())


# Background stats emitter control
stats_emitter_running = False


def start_stats_emitter():
    """Start background thread that emits system stats to admin clients every 2 seconds"""
    global stats_emitter_running

    if stats_emitter_running:
        return

    stats_emitter_running = True

    def emit_stats_loop():
        while stats_emitter_running:
            try:
                stats = get_system_stats()
                # Emit full stats to admin sessions
                if len(admin_sessions) > 0:
                    socketio.emit('system_stats', stats, room='admin')
                # Emit lightweight summarization status to all viewers
                # This enables the "Generating Summary..." banner
                if stats.get("gpu") and stats["gpu"].get("summarization_started_at"):
                    socketio.emit('system_stats', {
                        "gpu": {
                            "summarization_started_at": stats["gpu"]["summarization_started_at"]
                        }
                    })
                elif sum(active_language_viewers.values()) > 0:
                    # Clear banner for viewers when summarization done
                    socketio.emit('system_stats', {"gpu": {"summarization_started_at": None}})
            except Exception as e:
                print(f"[STATS] Error emitting stats: {e}")

            # Sleep for 500ms between updates (2 updates/sec)
            time.sleep(0.5)

    stats_thread = threading.Thread(target=emit_stats_loop, daemon=True)
    stats_thread.start()
    print("[STATS] Background stats emitter started")


@app.route("/api/admin/transcript", methods=["GET"])
def download_transcript():
    """Download the full transcript"""
    from flask import send_file
    if TRANSCRIPT_FILE.exists():
        return send_file(TRANSCRIPT_FILE, as_attachment=True, download_name=TRANSCRIPT_FILE.name)
    else:
        return jsonify({"status": "error", "message": "No transcript file found"}), 404


@socketio.on("connect")
def handle_connect():
    """Handle client connection"""
    # Initialize client session
    connected_clients[request.sid] = {
        'role': None,
        'language': None,
        'authenticated': False
    }


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection"""

    # Clean up client session
    if request.sid in connected_clients:
        client = connected_clients[request.sid]

        # Update viewer counts if was a viewer
        if client['role'] == 'viewer' and client['language']:
            lang = client['language']
            if lang in active_language_viewers:
                active_language_viewers[lang] = max(0, active_language_viewers[lang] - 1)
                print(f"[VIEWER] Viewer left {lang} room. Active viewers for {lang}: {active_language_viewers[lang]}")
                # Broadcast updated stats to admins
                socketio.emit('viewer_stats', active_language_viewers, room='admin')

        # Remove from admin sessions if was admin
        if client['role'] == 'admin' and request.sid in admin_sessions:
            admin_sessions.discard(request.sid)

        del connected_clients[request.sid]


@socketio.on("admin_authenticate")
def handle_admin_authenticate(data):
    """Authenticate admin user via WebSocket"""
    password = data.get("password", "")

    if password == Config.ADMIN_PASSWORD:
        connected_clients[request.sid]['role'] = 'admin'
        connected_clients[request.sid]['authenticated'] = True
        admin_sessions.add(request.sid)
        join_room('admin')
        print(f"[ADMIN] Admin authenticated: {request.sid}")
        emit('admin_authenticated', {'success': True, 'viewer_password': Config.VIEWER_PASSWORD})
        # Send current stats
        emit('viewer_stats', active_language_viewers)
        # Send current listening status
        emit('status', {'listening': is_listening})
        # Replay the current Meet-bot state so a late-arriving admin tab
        # knows whether a bot is already connected and which meeting URL.
        emit('bot_status', {'connected': bot_connected, 'url': _meet_bot_url})
        if bot_connected and meet_participants:
            emit('meet_roster', {
                'participants': list(meet_participants.keys()),
                'bot_connected': True,
            })
    else:
        emit('admin_authenticated', {'success': False, 'message': 'Invalid password'})


@socketio.on("viewer_authenticate")
def handle_viewer_authenticate(data):
    """Authenticate viewer user via WebSocket"""
    password = data.get("password", "")

    if password == Config.VIEWER_PASSWORD:
        connected_clients[request.sid]['authenticated'] = True
        viewer_sessions.add(request.sid)
        print(f"[VIEWER] Viewer authenticated: {request.sid}")
        emit('viewer_authenticated', {'success': True})
    else:
        emit('viewer_authenticated', {'success': False, 'message': 'Invalid passphrase'})


@socketio.on("get_languages")
def handle_get_languages():
    """Send available languages to client"""
    emit('available_languages', Config.TARGET_LANGUAGES)


@socketio.on("get_status")
def handle_get_status():
    """Send current listening status to client"""
    emit('status', {'listening': is_listening})


@socketio.on("join_language_room")
def handle_join_language_room(data):
    """Handle viewer joining a language-specific room"""
    language = data.get("language")

    # Check if viewer is authenticated
    if request.sid not in connected_clients or not connected_clients[request.sid].get('authenticated'):
        emit('error', {'message': 'Authentication required. Please enter the passphrase.'})
        emit('joined_room', {'language': language, 'success': False, 'reason': 'not_authenticated'})
        return

    if language and language in active_language_viewers:
        # Update client info
        connected_clients[request.sid]['role'] = 'viewer'
        connected_clients[request.sid]['language'] = language

        # Join the language room
        join_room(f'lang_{language}')

        # Update viewer count
        active_language_viewers[language] += 1

        print(f"[VIEWER] Viewer joined {language} room. Active viewers for {language}: {active_language_viewers[language]}")

        # Broadcast updated stats to admins
        socketio.emit('viewer_stats', active_language_viewers, room='admin')

        # Send current summary to the new viewer
        with summary_lock:
            if current_summary["last_updated"]:
                emit('summary_update', current_summary)

        emit('joined_room', {'language': language, 'success': True})


@socketio.on("leave_language_room")
def handle_leave_language_room(data):
    """Handle viewer leaving a language-specific room"""
    language = data.get("language")

    if language and request.sid in connected_clients:
        client = connected_clients[request.sid]

        # Leave the room
        leave_room(f'lang_{language}')

        # Update viewer count
        if language in active_language_viewers:
            active_language_viewers[language] = max(0, active_language_viewers[language] - 1)

        print(f"[VIEWER] Viewer left {language} room. Active viewers for {language}: {active_language_viewers[language]}")

        # Update client info
        client['language'] = None

        # Broadcast updated stats to admins
        socketio.emit('viewer_stats', active_language_viewers, room='admin')

        emit('left_room', {'language': language, 'success': True})


def start_listening_internal():
    """Internal function to start listening - no authentication check"""
    global is_listening, audio_stream, p_audio, actual_sample_rate, num_channels

    if not is_listening:
        is_listening = True

        # Start the processing thread first (same for both audio sources).
        audio_thread = threading.Thread(target=process_audio, daemon=True)
        audio_thread.start()

        # Meet-bot path: audio arrives over SocketIO at 16 kHz mono; skip PyAudio entirely.
        if Config.AUDIO_SOURCE == "meet_bot":
            actual_sample_rate = Config.SAMPLE_RATE  # 16000
            num_channels = 1
            print("[AUDIO] Source: Meet bot (waiting for bot to connect and stream audio)")
            socketio.emit("status", {"listening": True})
            return

        # Initialize PyAudio
        p_audio = pyaudio.PyAudio()

        # Look for loopback device
        loopback_device = None

        # Check if device manually selected
        if selected_audio_device_index is not None:
            try:
                dev_info = p_audio.get_device_info_by_index(selected_audio_device_index)
                if dev_info.get("maxInputChannels", 0) > 0:
                    loopback_device = selected_audio_device_index
                    print(f"Using manually selected device: {dev_info['name']}")
                else:
                    print(f"[WARNING] Selected device has no input channels, falling back to auto-detect")
                    socketio.emit("admin_error", {"message": "Selected audio device is not available. Using auto-detect."}, room="admin")
            except Exception as e:
                print(f"[WARNING] Selected device not available: {e}. Falling back to auto-detect")
                socketio.emit("admin_error", {"message": "Selected audio device is not available. Using auto-detect."}, room="admin")

        # Auto-detect loopback device if not manually selected or if manual selection failed
        if loopback_device is None:
            for i in range(p_audio.get_device_count()):
                dev_info = p_audio.get_device_info_by_index(i)
                if dev_info.get("isLoopbackDevice") and "Speakers" in dev_info["name"]:
                    loopback_device = i
                    print(f"Using loopback device: {dev_info['name']}")
                    break

        if loopback_device is None:
            print("No loopback device found! Using default microphone.")
            loopback_device = p_audio.get_default_input_device_info()["index"]

        # Get loopback device info
        loopback_info = p_audio.get_device_info_by_index(loopback_device)
        actual_sample_rate = int(loopback_info["defaultSampleRate"])
        num_channels = int(loopback_info["maxInputChannels"])
        print(f"Audio config: {num_channels} channels @ {actual_sample_rate}Hz")

        # Start recording from loopback
        audio_stream = p_audio.open(
            format=pyaudio.paInt16,
            channels=int(loopback_info["maxInputChannels"]),
            rate=int(loopback_info["defaultSampleRate"]),
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            input_device_index=loopback_device,
            stream_callback=audio_callback,
        )
        audio_stream.start_stream()

        socketio.emit("status", {"listening": True})  # Broadcast to all clients
        print("Started listening to system audio...")


@socketio.on("start_listening")
def handle_start_listening():
    """Start listening for audio (admin only)"""
    print(f"[START] Received start_listening from {request.sid}")

    # Check if user is authenticated admin
    if request.sid not in connected_clients:
        print(f"[START] ERROR: {request.sid} not in connected_clients")
        emit('error', {'message': 'Unauthorized. Admin access required.'})
        return

    client_role = connected_clients[request.sid].get('role')
    print(f"[START] Client {request.sid} has role: {client_role}")

    if client_role != 'admin':
        print(f"[START] ERROR: Client role is '{client_role}', not 'admin'")
        emit('error', {'message': 'Unauthorized. Admin access required.'})
        return

    print(f"[START] Authorization passed, calling start_listening_internal()")
    start_listening_internal()


def stop_listening_internal():
    """Internal function to stop listening - no authentication check"""
    global is_listening, audio_stream, p_audio

    is_listening = False

    if audio_stream:
        audio_stream.stop_stream()
        audio_stream.close()
        audio_stream = None

    if p_audio:
        p_audio.terminate()
        p_audio = None

    socketio.emit("status", {"listening": False})  # Broadcast to all clients
    print("Stopped listening...")


@socketio.on("stop_listening")
def handle_stop_listening():
    """Stop listening for audio (admin only)"""
    print(f"[STOP] Received stop_listening from {request.sid}")

    # Check if user is authenticated admin
    if request.sid not in connected_clients:
        print(f"[STOP] ERROR: {request.sid} not in connected_clients")
        emit('error', {'message': 'Unauthorized. Admin access required.'})
        return

    client_role = connected_clients[request.sid].get('role')
    print(f"[STOP] Client {request.sid} has role: {client_role}")

    if client_role != 'admin':
        print(f"[STOP] ERROR: Client role is '{client_role}', not 'admin'")
        emit('error', {'message': 'Unauthorized. Admin access required.'})
        return

    print(f"[STOP] Authorization passed, calling stop_listening_internal()")
    stop_listening_internal()


@socketio.on("trigger_summary")
def handle_trigger_summary():
    """Manually trigger a summary generation (admin only)"""
    global last_summary_time, all_meeting_segments, summary_pending, last_summary_segment_count, previous_overview_text

    # Check if user is authenticated admin
    if request.sid not in connected_clients or connected_clients[request.sid]['role'] != 'admin':
        emit('error', {'message': 'Unauthorized. Admin access required.'})
        return

    if summarization_pipe is None or not Config.ENABLE_SUMMARIZATION:
        emit('summary_error', {'message': 'Summarization is not enabled'})
        return

    if summarization_paused:
        emit('summary_error', {'message': 'Summarization is paused. Resume it first.'})
        return

    if len(all_meeting_segments) == 0:
        emit('summary_error', {'message': 'No transcript content to summarize yet'})
        return

    if summary_pending:
        emit('summary_error', {'message': 'Summary already pending, please wait'})
        return

    # Get NEW segments since last summary (for recent transcript)
    new_segments = all_meeting_segments[last_summary_segment_count:]

    # Build transcript of NEW segments only (since last summary)
    new_transcript_lines = []
    for seg in new_segments:
        text = seg.get("text", "")
        ts = seg.get("timestamp", "")
        new_transcript_lines.append(f"[{ts}] {text}")

    new_transcript_text = "\n".join(new_transcript_lines)

    # Get time range of summarized content
    first_timestamp = all_meeting_segments[0].get("timestamp", "") if all_meeting_segments else ""
    last_timestamp = all_meeting_segments[-1].get("timestamp", "") if all_meeting_segments else ""
    time_range = f"{first_timestamp} - {last_timestamp}" if first_timestamp and last_timestamp else ""

    # Calculate minutes since meeting start
    current_time = time.time()
    minutes_since_start = 0
    if meeting_start_time is not None:
        minutes_since_start = int((current_time - meeting_start_time) / 60)

    if len(new_transcript_text) < 50 and last_summary_segment_count > 0:
        emit('summary_error', {'message': 'Not enough new content to summarize (need at least 50 new characters)'})
        return

    # If currently processing transcription, schedule for after completion
    if is_processing:
        print(f"[SUMMARY] Manual trigger - scheduling after current transcription completes...")
        emit('summary_generating', {'message': 'Waiting for transcription to complete...'})
        summary_pending = True
        return

    print(f"[SUMMARY] Manual trigger - generating summary for {len(all_meeting_segments)} total segments, {len(new_segments)} new ({time_range})...")
    emit('summary_generating', {'message': 'Generating...'})

    # Generate in background thread to not block
    def generate_and_emit():
        global last_summary_time, last_summary_segment_count, previous_overview_text
        summary = generate_summary(previous_overview_text, new_transcript_text, time_range, minutes_since_start)
        if summary:
            last_summary_time = time.time()
            last_summary_segment_count = len(all_meeting_segments)

            # Store the overview text for next iteration (summarization chaining)
            overview_lines = []
            for section in summary.get("overview_sections", []):
                overview_lines.append(f"## {section.get('title', 'Topic')}")
                for bullet in section.get("bullets", []):
                    overview_lines.append(f"- {bullet}")
            previous_overview_text = "\n".join(overview_lines)

            # Emit original summary to admin room
            socketio.emit('summary_update', summary, room='admin')

            # Translate and emit to each language room
            for lang_code in active_language_viewers:
                if active_language_viewers[lang_code] > 0:
                    translated_summary = translate_summary(summary, lang_code)
                    socketio.emit('summary_update', translated_summary, room=f'lang_{lang_code}')
        else:
            socketio.emit('summary_error', {'message': 'Failed to generate summary'}, room='admin')

    summary_thread = threading.Thread(target=generate_and_emit, daemon=True)
    summary_thread.start()


@socketio.on("toggle_summarization")
def handle_toggle_summarization():
    """Toggle summarization on/off (admin only) - kill switch for performance issues"""
    global summarization_paused

    # Check if user is authenticated admin
    if request.sid not in connected_clients or connected_clients[request.sid]['role'] != 'admin':
        emit('error', {'message': 'Unauthorized. Admin access required.'})
        return

    if summarization_pipe is None or not Config.ENABLE_SUMMARIZATION:
        emit('summarization_status', {'enabled': False, 'reason': 'Summarization not configured'})
        return

    # Toggle the state
    summarization_paused = not summarization_paused
    status = "paused" if summarization_paused else "resumed"
    print(f"[SUMMARY] Summarization {status} by admin")

    # Emit to all admin clients
    socketio.emit('summarization_status', {
        'enabled': not summarization_paused,
        'paused': summarization_paused
    }, room='admin')


@socketio.on("broadcast_manual_summary")
def handle_broadcast_manual_summary(data):
    """Broadcast a manually pasted summary (from Claude) to all viewers, translated to their language"""
    global current_summary

    # Check if user is authenticated admin
    if request.sid not in connected_clients or connected_clients[request.sid]['role'] != 'admin':
        emit('error', {'message': 'Unauthorized. Admin access required.'})
        return

    summary_markdown = data.get('summary_markdown', '')
    source_lang = data.get('source_lang', 'en')  # Default to English
    if not summary_markdown:
        emit('error', {'message': 'No summary provided'})
        return

    # Store original in current_summary for new viewers
    with summary_lock:
        current_summary["manual_overview"] = summary_markdown
        current_summary["manual_overview_source_lang"] = source_lang
        current_summary["last_updated"] = time.time()

    log(f"Broadcasting manual summary ({len(summary_markdown)} chars) to viewers", "SUMMARY")

    # Cache translations to avoid re-translating for same language
    translations_cache = {source_lang: summary_markdown}

    # Send to each viewer in their language
    for sid, client in connected_clients.items():
        if client.get('role') == 'viewer':
            target_lang = client.get('language', source_lang)

            # Get or create translation
            if target_lang not in translations_cache:
                log(f"Translating manual summary to {target_lang}", "SUMMARY")
                translations_cache[target_lang] = translate_text(summary_markdown, source_lang, target_lang)

            socketio.emit('manual_summary_update', {
                'summary_markdown': translations_cache[target_lang],
                'timestamp': time.time()
            }, room=sid)

    # Confirm to admin
    emit('manual_summary_broadcast', {'success': True, 'languages_sent': list(translations_cache.keys())})


# ── Meet bot SocketIO namespace (Phase 4) ────────────────────────────────────
#
# The Playwright bot connects here as a socket.io-client at /meet_bot.
# It streams two event types:
#   audio_frame  — binary PCM16 payload + JSON meta {capture_ts_ms, sample_rate, channels}
#   speaker_event — JSON {type, name, wall_clock_ms} for speaker_start/end/roster_update

def _open_bot_wav():
    """Open a WAV file to record the full raw meeting audio from the bot."""
    global _bot_wav_writer
    with _bot_wav_lock:
        if _bot_wav_writer is not None or not TRANSCRIPT_FILE:
            return
        wav_path = TRANSCRIPT_FILE.with_suffix(".wav")
        try:
            _bot_wav_writer = wave.open(str(wav_path), "wb")
            _bot_wav_writer.setnchannels(1)
            _bot_wav_writer.setsampwidth(2)  # int16
            _bot_wav_writer.setframerate(Config.SAMPLE_RATE)
            log(f"Recording meeting audio → {wav_path.name}", "BOT")
        except Exception as e:
            log(f"Failed to open WAV: {e}", "BOT")
            _bot_wav_writer = None


def _close_bot_wav():
    global _bot_wav_writer
    with _bot_wav_lock:
        if _bot_wav_writer is None:
            return
        try:
            _bot_wav_writer.close()
            log("Meeting audio file closed", "BOT")
        except Exception as e:
            log(f"Error closing WAV: {e}", "BOT")
        _bot_wav_writer = None


if Config.MEET_BOT_ENABLED:

    @socketio.on("connect", namespace="/meet_bot")
    def bot_connect():
        global bot_connected
        bot_connected = True
        log("Meet bot connected", "BOT")
        _open_bot_wav()
        socketio.emit("bot_status", {"connected": True, "url": _meet_bot_url}, room="admin")

    @socketio.on("bot_info", namespace="/meet_bot")
    def bot_info(info):
        """Receive self-report from the bot (meeting URL) and refresh admin status."""
        global _meet_bot_url
        url = (info or {}).get("url")
        if url:
            _meet_bot_url = url
            socketio.emit("bot_status", {"connected": True, "url": url}, room="admin")

    @socketio.on("disconnect", namespace="/meet_bot")
    def bot_disconnect():
        global bot_connected, _active_speaker_starts
        bot_connected = False
        # Close any open speaker intervals so timeline stays consistent.
        now_ms = int(time.time() * 1000)
        for name, start in list(_active_speaker_starts.items()):
            speaker_timeline.append((start, now_ms, name))
        _active_speaker_starts.clear()
        _close_bot_wav()
        log("Meet bot disconnected", "BOT")
        socketio.emit("bot_status", {"connected": False, "url": None}, room="admin")

    @socketio.on("audio_frame", namespace="/meet_bot")
    def bot_audio_frame(meta, data):
        global _bot_pcm_buffer, _last_capture_ts_ms
        if not is_listening:
            return
        # Persist raw int16 PCM for offline retranscription.
        # writeframes (not writeframesraw) patches the header on every write so
        # the file is valid even if the server is killed without clean shutdown.
        if _bot_wav_writer is not None:
            try:
                with _bot_wav_lock:
                    if _bot_wav_writer is not None:
                        _bot_wav_writer.writeframes(data)
            except Exception:
                pass
        # data arrives as bytes (Int16 PCM, 16 kHz mono, 320 samples = 20 ms).
        frame = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        _bot_pcm_buffer = np.concatenate([_bot_pcm_buffer, frame])
        # Rechunk to CHUNK_SIZE (1024) so process_audio's duration math is correct.
        while len(_bot_pcm_buffer) >= CHUNK_SIZE:
            audio_queue.put(_bot_pcm_buffer[:CHUNK_SIZE].copy())
            _bot_pcm_buffer = _bot_pcm_buffer[CHUNK_SIZE:]
        _last_capture_ts_ms = meta.get("capture_ts_ms")

    @socketio.on("speaker_event", namespace="/meet_bot")
    def bot_speaker_event(ev):
        global meet_participants, _active_speaker_starts, _current_bot_speaker, _pending_speaker_switch
        ev_type = ev.get("type")
        name = ev.get("name")
        ts_ms = ev.get("wall_clock_ms", int(time.time() * 1000))

        if ev_type == "roster_update":
            for p in ev.get("participants", []):
                if p and p not in meet_participants:
                    meet_participants[p] = ts_ms
            socketio.emit("meet_roster", {
                "participants": list(meet_participants.keys()),
                "bot_connected": True,
            }, room="admin")
            return

        if not name:
            return

        if ev_type == "speaker_start":
            _active_speaker_starts[name] = ts_ms
            # Flag a pending switch if the speaker changed — process_audio uses this to flush.
            if _current_bot_speaker is not None and _current_bot_speaker != name:
                _pending_speaker_switch = True
            _current_bot_speaker = name
            log(f"Speaking: {name}", "BOT")
            _broadcast_active_speakers()

        elif ev_type == "speaker_end":
            start = _active_speaker_starts.pop(name, ts_ms - 1000)
            speaker_timeline.append((start, ts_ms, name))
            log(f"Segment: {name} {ts_ms - start} ms", "BOT")
            # Don't flush here — a single speaker's short pauses emit
            # speaker_end/speaker_start toggles, so flushing on end would
            # chop their turn into tiny fragments. We only flush when a
            # DIFFERENT speaker starts (speaker_switched) or the 60 s cap
            # is reached. _current_bot_speaker stays as the last name so
            # the same speaker resuming does not trigger a switch.
            _broadcast_active_speakers()


def _broadcast_active_speakers():
    """Push the current set of speaking participants to admin + all viewers."""
    names = list(_active_speaker_starts.keys())
    payload = {"speakers": names, "wall_clock_ms": int(time.time() * 1000)}
    socketio.emit("active_speakers", payload, room="admin")
    for lang_code, count in active_language_viewers.items():
        if count > 0:
            socketio.emit("active_speakers", payload, room=f"lang_{lang_code}")


# ── Admin-triggered bot spawning ─────────────────────────────────────────────
# Tracks the currently running Meet bot subprocess so we can start/stop it
# from the admin panel. Only one bot instance is supported at a time.
_meet_bot_process = None
_meet_bot_url = None
_meet_bot_lock = threading.Lock()


def _bot_script_path():
    return Path(__file__).parent / "meet-bot" / "index.js"


@socketio.on("start_meet_bot")
def handle_start_meet_bot(data):
    """Spawn the Meet bot pointing at the given URL. Admin-only."""
    global _meet_bot_process, _meet_bot_url

    url = (data or {}).get("url", "").strip()
    if not url.startswith("http"):
        emit("meet_bot_control_result", {"ok": False, "error": "URL must start with http(s)://"})
        return

    with _meet_bot_lock:
        # If a bot is already running, refuse to start another.
        if _meet_bot_process is not None and _meet_bot_process.poll() is None:
            emit("meet_bot_control_result", {"ok": False, "error": "Bot already running — stop it first"})
            return

        script = _bot_script_path()
        if not script.exists():
            emit("meet_bot_control_result", {"ok": False, "error": f"Bot script not found at {script}"})
            return

        import subprocess
        try:
            _meet_bot_process = subprocess.Popen(
                ["node", str(script),
                 "--url", url,
                 "--polyglot-url", "http://localhost:5000",
                 "--headful"],
                cwd=str(script.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            _meet_bot_url = url
            log(f"Spawned Meet bot (pid={_meet_bot_process.pid}) → {url}", "BOT")
            emit("meet_bot_control_result", {"ok": True, "pid": _meet_bot_process.pid, "url": url})
        except Exception as e:
            log(f"Failed to spawn bot: {e}", "BOT")
            emit("meet_bot_control_result", {"ok": False, "error": str(e)})


@socketio.on("stop_meet_bot")
def handle_stop_meet_bot():
    """Terminate the running Meet bot subprocess."""
    global _meet_bot_process, _meet_bot_url
    with _meet_bot_lock:
        if _meet_bot_process is None or _meet_bot_process.poll() is not None:
            emit("meet_bot_control_result", {"ok": False, "error": "No bot running"})
            _meet_bot_process = None
            _meet_bot_url = None
            return
        try:
            _meet_bot_process.terminate()
            try:
                _meet_bot_process.wait(timeout=5)
            except Exception:
                _meet_bot_process.kill()
            log(f"Stopped Meet bot subprocess", "BOT")
            _meet_bot_process = None
            _meet_bot_url = None
            emit("meet_bot_control_result", {"ok": True})
        except Exception as e:
            emit("meet_bot_control_result", {"ok": False, "error": str(e)})


if __name__ == "__main__":
    # Check for single instance before doing anything else
    check_single_instance()

    # Register cleanup function to remove lock file on exit
    import atexit
    atexit.register(cleanup_lock_file)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Polyglot - Real-time Audio Translator")
    parser.add_argument(
        "--auto-listen",
        action="store_true",
        help="Automatically start listening when the app launches"
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Automatically open browser when the app starts"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on (default: 5000)"
    )
    args = parser.parse_args()

    # Interactive startup configuration (password, meeting name, transcript)
    startup_configuration()

    print("Loading configuration...")
    print("Initializing models (this may take a minute)...")
    initialize_models()

    # Start background stats emitter for admin dashboard
    start_stats_emitter()

    print("\n" + "=" * 60)
    print("Polyglot - Real-time Audio Translator")
    print("=" * 60)
    print(f"\nMeeting: {MEETING_NAME}")
    print(f"Transcript: {TRANSCRIPT_FILE.name}")
    print(f"\nOpen your browser to: http://localhost:{args.port}")
    print(f"Device: {Config.DEVICE}")
    print(f"Whisper Model: {Config.WHISPER_MODEL}")
    print(f"Translation Model: {Config.TRANSLATION_MODEL}")
    print(f"Target languages: {', '.join([lang['name'] for lang in Config.TARGET_LANGUAGES])}")
    print(f"\n{'='*60}")
    print(f"VIEWER PASSPHRASE: {Config.VIEWER_PASSWORD}")
    print(f"{'='*60}")
    print("Share this passphrase with viewers to grant them access.")

    if args.auto_listen:
        print("\n[AUTO-LISTEN] Will automatically start listening after server starts")

    if args.open_browser:
        print(f"[AUTO-OPEN] Opening browser to http://localhost:{args.port}")
        # Open browser after a short delay to let server start
        def open_browser_delayed():
            time.sleep(2)  # Wait for server to start
            webbrowser.open(f"http://localhost:{args.port}")

        browser_thread = threading.Thread(target=open_browser_delayed, daemon=True)
        browser_thread.start()

    print("\nPress Ctrl+C to stop\n")

    # If auto-listen is enabled, start listening after server starts
    if args.auto_listen:
        def auto_start_listening():
            time.sleep(3)  # Wait for server to fully initialize
            print("[AUTO-LISTEN] Starting audio capture...")
            start_listening_internal()

        listen_thread = threading.Thread(target=auto_start_listening, daemon=True)
        listen_thread.start()

    socketio.run(app, debug=False, host="0.0.0.0", port=args.port)
