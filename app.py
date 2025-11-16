"""
Polyglot - Real-time Multi-Language Audio Translator
Captures system audio and translates speech into multiple languages simultaneously
"""

import argparse
import os
import queue
import sys
import threading
import time
import warnings
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio
import resampy
import torch

# CRITICAL FIX: Prevent transformers from trying to use torchcodec
# torchcodec is installed by pyannote.audio but the DLLs are missing
# This causes transformers Whisper pipeline to crash when it tries to use torchcodec
# Solution: Monkey patch transformers.utils.import_utils._torchcodec_available to False
# This makes transformers think torchcodec is not available, so it won't try to use it

from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO
from langdetect import LangDetectException, detect
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, pipeline

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

# Create timestamped transcript file
TRANSCRIPTS_DIR = Path("transcripts")
TRANSCRIPTS_DIR.mkdir(exist_ok=True)
TRANSCRIPT_FILE = TRANSCRIPTS_DIR / f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Dynamic audio detection thresholds (can be changed via API)
audio_thresholds = Config.get_audio_thresholds()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
audio_queue = queue.Queue()
is_listening = False
is_processing = False  # Lock to prevent overlapping transcriptions
transcription_pipe = None
translation_model = None
translation_tokenizer = None
diarization_pipeline = None  # Speaker diarization pipeline
audio_stream = None
p_audio = None
actual_sample_rate = 48000  # Will be set to the actual device sample rate
num_channels = 2  # Will be set to the actual number of channels


def initialize_models():
    """Initialize Whisper, M2M100, and Speaker Diarization models"""
    global transcription_pipe, translation_model, translation_tokenizer, diarization_pipeline

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
        print(f"Initial GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB reserved")

    print()

    # Initialize Whisper with word-level timestamps for precise speaker alignment
    print(f"[1/3] Loading Whisper Model: {Config.WHISPER_MODEL}")
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
        print(f"      [OK] Loaded successfully")
        print(f"      GPU Memory: {whisper_mem:.2f} GB allocated")
    else:
        print(f"      [OK] Loaded successfully (CPU)")
    print()

    # Initialize M2M100
    model_name = Config.TRANSLATION_MODEL
    print(f"[2/3] Loading Translation Model: {model_name}")
    translation_model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    translation_tokenizer = M2M100Tokenizer.from_pretrained(model_name)

    if device == "cuda":
        translation_model = translation_model.cuda()
        translation_mem = torch.cuda.memory_allocated(0) / 1024**3
        print(f"      [OK] Loaded successfully")
        print(f"      GPU Memory: {translation_mem:.2f} GB allocated (total)")
        print(f"      Model Size: ~{translation_mem - whisper_mem:.2f} GB")
    else:
        print(f"      [OK] Loaded successfully (CPU)")
    print()

    # Initialize Speaker Diarization (conditionally)
    diarization_pipeline = None
    if Config.ENABLE_DIARIZATION:
        print(f"[3/3] Loading Speaker Diarization: {Config.DIARIZATION_MODEL}")
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
            total_mem = torch.cuda.memory_allocated(0) / 1024**3
            diarization_mem = total_mem - translation_mem
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
        print(f"[3/3] Speaker Diarization: DISABLED")
        print(f"      Speaker identification is turned off (ENABLE_DIARIZATION=False)")
        print(f"      All transcriptions will appear without speaker labels.")
        print()

    print("=" * 80)
    print("ALL MODELS LOADED SUCCESSFULLY!")
    if device == "cuda":
        print(f"Total GPU Memory Used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print("=" * 80 + "\n")


def audio_callback(in_data, frame_count, time_info, status):
    """Callback for audio stream"""
    if Config.DEBUG and status:
        print(f"[DEBUG] audio_callback status: {status}")

    if is_listening and in_data:
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        if Config.DEBUG and len(audio_data) > 0:
            level = np.abs(audio_data).mean()
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


def translate_text(text, source_lang, target_lang):
    """Translate text using M2M100"""
    if source_lang == target_lang:
        return text

    try:
        translation_tokenizer.src_lang = source_lang
        encoded = translation_tokenizer(text, return_tensors="pt")

        if Config.DEVICE == "cuda":
            encoded = {k: v.cuda() for k, v in encoded.items()}

        generated_tokens = translation_model.generate(
            **encoded, forced_bos_token_id=translation_tokenizer.get_lang_id(target_lang), max_length=512
        )

        translated = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        return translated
    except Exception as e:
        print(f"Translation error ({source_lang} -> {target_lang}): {e}")
        return f"[Translation error: {str(e)}]"


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


def transcribe_and_translate(audio_data, audio_duration):
    """Background thread for transcription and translation with speaker diarization"""
    global is_processing

    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if Config.DEBUG:
            print(f"\n[DEBUG] {timestamp} transcribe_and_translate called")
            print(f"[DEBUG] Audio data shape: {audio_data.shape}")
            print(f"[DEBUG] Audio duration: {audio_duration:.2f}s")
            print(f"[DEBUG] Audio min: {audio_data.min():.4f}, max: {audio_data.max():.4f}, mean: {audio_data.mean():.4f}")
            if Config.DEVICE == "cuda":
                print(f"[GPU MEMORY] Start: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB reserved")

        # Perform speaker diarization first (if enabled)
        speaker_segments = None
        if Config.ENABLE_DIARIZATION:
            print(f"[{timestamp}] Performing speaker diarization...")
            speaker_segments = perform_speaker_diarization(audio_data, SAMPLE_RATE)
        else:
            print(f"[{timestamp}] Speaker diarization disabled - skipping")

        if Config.DEBUG:
            print(f"[DEBUG] Speaker segments: {speaker_segments}")

        # Transcribe with timestamps
        if Config.DEBUG:
            print(f"[DEBUG] Starting Whisper transcription...")
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

        if Config.DEBUG:
            print(f"[DEBUG] Transcription complete: {len(full_transcript)} chars, {len(chunks)} chunks")
            if Config.DEVICE == "cuda":
                print(f"[GPU MEMORY] After Whisper: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB reserved")

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

            print(f"[DEBUG] Speaker mapping: {speaker_mapping}")

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

            if Config.DEBUG:
                print(f"[DEBUG] Extracted {len(all_words)} words from Whisper")

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
                        best_speaker = speaker_mapping.get(seg["speaker"], seg["speaker"])
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

        # IMMEDIATELY send transcription to UI (before translations)
        # This makes the UI feel much more responsive
        ws_payload_initial = {
            "transcript": full_transcript,
            "source_language": source_lang,
            "translations": {},  # Empty for now
            "timestamp": time.time(),
            "segments": segments_with_speakers,
            "is_initial": True,  # Flag to indicate translations are coming
        }

        if Config.DEBUG:
            print(f"\n[WS EMIT 1/2] Sending source transcript ({len(segments_with_speakers)} segments):")
            for i, seg in enumerate(segments_with_speakers):
                print(f"  [{i+1}] [{seg['start']:.2f}s-{seg['end']:.2f}s] {seg['speaker']}: {seg['text'][:80]}{'...' if len(seg['text']) > 80 else ''}")

            # DEBUG: Print actual JSON payload to see what's being sent
            import json
            print(f"\n[WS DEBUG] Full payload being sent:")
            print(json.dumps(ws_payload_initial, indent=2, ensure_ascii=False))

        socketio.emit("new_translation", ws_payload_initial)

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

        # Translate all segments for all languages in parallel
        translated_segments_by_lang = {lang["code"]: [] for lang in Config.TARGET_LANGUAGES}

        with ThreadPoolExecutor(max_workers=len(Config.TARGET_LANGUAGES) * len(segments_with_speakers)) as executor:
            # Create all translation tasks
            futures = []
            for lang_info in Config.TARGET_LANGUAGES:
                for segment in segments_with_speakers:
                    future = executor.submit(translate_segment_for_language, lang_info, segment)
                    futures.append((future, lang_info["code"]))

            # Collect results
            for future, lang_code in futures:
                _, translated_segment = future.result()
                translated_segments_by_lang[lang_code].append(translated_segment)

        # Send translations update with per-segment translations
        ws_payload_final = {
            "transcript": full_transcript,
            "source_language": source_lang,
            "translated_segments": translated_segments_by_lang,  # Per-segment translations
            "timestamp": ws_payload_initial["timestamp"],  # Use same timestamp
            "segments": segments_with_speakers,  # Original source segments
            "is_update": True,  # Flag to indicate this is a translation update
        }

        if Config.DEBUG:
            print(f"[WS EMIT 2/2] Sending translated segments for: {list(translated_segments_by_lang.keys())}")
            for lang, segs in translated_segments_by_lang.items():
                print(f"  {lang}: {len(segs)} segments")
            if Config.DEVICE == "cuda":
                print(f"[GPU MEMORY] After Translations: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB reserved")

        socketio.emit("new_translation", ws_payload_final)
    except Exception as e:
        print(f"[ERROR] Processing error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        is_processing = False


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

                    # Calculate audio level for UI
                    audio_level = np.abs(chunk).mean()
                    socketio.emit("audio_level", {"level": float(audio_level * 100)})

                    # Add to buffer for next sentence
                    buffer.append(chunk)

                    # Update silence counter
                    is_silent = audio_level < audio_thresholds["silence_threshold"]
                    if is_silent:
                        silence_counter += 1
                    else:
                        silence_counter = 0

                    # Emit debug data showing we're still collecting audio
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
                    )
                except queue.Empty:
                    pass
                continue

            # Get audio chunk
            chunk = audio_queue.get(timeout=1)

            # Calculate audio level
            audio_level = np.abs(chunk).mean()
            socketio.emit("audio_level", {"level": float(audio_level * 100)})

            # Calculate chunk limits dynamically based on current thresholds
            min_chunks = int(actual_sample_rate * audio_thresholds["min_audio_length"] / CHUNK_SIZE)
            max_chunks = int(actual_sample_rate * audio_thresholds["max_audio_length"] / CHUNK_SIZE)

            # Emit debug data for UI
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
            )

            # Check if current chunk is silent
            is_silent = audio_level < audio_thresholds["silence_threshold"]

            if is_silent:
                silence_counter += 1
            else:
                silence_counter = 0  # Reset on any sound

            buffer.append(chunk)

            # Process when we detect end of sentence (silence after minimum audio) OR max buffer reached
            silence_detected = len(buffer) >= min_chunks and silence_counter >= audio_thresholds["silence_chunks"]
            max_length_reached = len(buffer) >= max_chunks
            should_process = silence_detected or max_length_reached

            if should_process:
                if Config.DEBUG:
                    print(f"[DEBUG] Processing trigger - silence_detected: {silence_detected}, max_length: {max_length_reached}")
                    print(f"[DEBUG] Buffer size: {len(buffer)} chunks, silence_counter: {silence_counter}")

                is_processing = True  # Set lock

                audio_data = np.concatenate(buffer, axis=0)
                buffer = []
                silence_counter = 0

                if Config.DEBUG:
                    print(f"[DEBUG] Concatenated audio shape: {audio_data.shape}")

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

                if Config.DEBUG:
                    print(f"[DEBUG] Mono audio shape: {audio_mono.shape}, channels: {num_channels}")

                # Resample from actual_sample_rate to 16000Hz for Whisper
                if actual_sample_rate != SAMPLE_RATE:
                    audio_resampled = resampy.resample(audio_mono, actual_sample_rate, SAMPLE_RATE)
                    if Config.DEBUG:
                        print(f"[DEBUG] Resampled {actual_sample_rate}Hz -> {SAMPLE_RATE}Hz: {audio_resampled.shape}")
                else:
                    audio_resampled = audio_mono

                # Check average audio level to detect if it's mostly silence
                avg_audio_level = np.abs(audio_resampled).mean()

                if Config.DEBUG:
                    print(f"[DEBUG] Average audio level: {avg_audio_level:.4f}, threshold: {Config.MIN_AUDIO_LEVEL}")

                # Skip transcription if audio is too quiet (likely silence/hallucination)
                if avg_audio_level < Config.MIN_AUDIO_LEVEL:
                    print(f"[AUDIO] Skipping transcription - audio too quiet ({avg_audio_level:.4f} < {Config.MIN_AUDIO_LEVEL})")
                    is_processing = False
                    continue

                # Calculate audio duration
                audio_duration = len(audio_resampled) / SAMPLE_RATE

                # Launch background thread for transcription and translation
                # This keeps the main loop responsive for WebSocket updates
                processing_thread = threading.Thread(
                    target=transcribe_and_translate, args=(audio_resampled, audio_duration), daemon=True
                )
                processing_thread.start()

                # Emit processing started event for UI flash effect
                socketio.emit("processing_started", {"timestamp": time.time()})

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[ERROR] Processing error: {e}")
            import traceback

            traceback.print_exc()
            is_processing = False  # Release lock on error
            continue


@app.route("/")
def index():
    """Serve the main page"""
    return render_template("index.html")


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


@socketio.on("connect")
def handle_connect():
    """Handle client connection"""
    print("[DEBUG] Client connected to SocketIO!")


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection"""
    print("[DEBUG] Client disconnected from SocketIO!")


@socketio.on("start_listening")
def handle_start_listening():
    """Start listening for audio"""
    global is_listening, audio_stream, p_audio, actual_sample_rate, num_channels

    print("[DEBUG] handle_start_listening called!")

    if not is_listening:
        is_listening = True

        # Start audio stream
        audio_thread = threading.Thread(target=process_audio, daemon=True)
        audio_thread.start()

        # Initialize PyAudio
        p_audio = pyaudio.PyAudio()

        # Look for loopback device
        loopback_device = None
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

        socketio.emit("status", {"listening": True})
        print("Started listening to system audio...")


@socketio.on("stop_listening")
def handle_stop_listening():
    """Stop listening for audio"""
    global is_listening, audio_stream, p_audio
    is_listening = False

    if audio_stream:
        audio_stream.stop_stream()
        audio_stream.close()
        audio_stream = None

    if p_audio:
        p_audio.terminate()
        p_audio = None

    socketio.emit("status", {"listening": False})
    print("Stopped listening...")


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnect - stop listening and clean up"""
    print("Client disconnected, stopping audio capture...")
    handle_stop_listening()


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

    print("Loading configuration...")
    print("Initializing models (this may take a minute)...")
    initialize_models()

    print("\n" + "=" * 60)
    print("Polyglot - Real-time Audio Translator")
    print("=" * 60)
    print(f"\nOpen your browser to: http://localhost:{args.port}")
    print(f"Device: {Config.DEVICE}")
    print(f"Whisper Model: {Config.WHISPER_MODEL}")
    print(f"Translation Model: {Config.TRANSLATION_MODEL}")
    print(f"Target languages: {', '.join([lang['name'] for lang in Config.TARGET_LANGUAGES])}")

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
            handle_start_listening()

        listen_thread = threading.Thread(target=auto_start_listening, daemon=True)
        listen_thread.start()

    socketio.run(app, debug=False, host="0.0.0.0", port=args.port)
