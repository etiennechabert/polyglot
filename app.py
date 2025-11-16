"""
Polyglot - Real-time Multi-Language Audio Translator
Captures system audio and translates speech into multiple languages simultaneously
"""

import queue
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio
import resampy
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO
from langdetect import LangDetectException, detect
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, pipeline

from config import Config

# Suppress CUDA compatibility warning for RTX 5080 (sm_120)
# PyTorch nightly works fine with backward compatibility
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

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
audio_stream = None
p_audio = None
actual_sample_rate = 48000  # Will be set to the actual device sample rate
num_channels = 2  # Will be set to the actual number of channels


def initialize_models():
    """Initialize Whisper and M2M100 models"""
    global transcription_pipe, translation_model, translation_tokenizer

    # CRITICAL: Ensure CUDA is available - this app requires GPU
    import torch
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
    print(f"Initializing models on {device}...")

    # Initialize Whisper
    print("Loading Whisper...")
    transcription_pipe = pipeline(
        "automatic-speech-recognition",
        model=f"openai/whisper-{Config.WHISPER_MODEL}",
        device=0 if device == "cuda" else -1,
        chunk_length_s=30,
        return_timestamps=False,
    )

    # Initialize M2M100
    model_name = Config.TRANSLATION_MODEL
    print(f"Loading M2M100 ({model_name})...")
    translation_model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    translation_tokenizer = M2M100Tokenizer.from_pretrained(model_name)

    if device == "cuda":
        translation_model = translation_model.cuda()

    print("Models loaded successfully!")


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


def transcribe_and_translate(audio_data, audio_duration):
    """Background thread for transcription and translation"""
    global is_processing

    try:
        # Transcribe using resampled audio (auto-detect language)
        result = transcription_pipe(audio_data)
        transcript = result["text"].strip()

        if transcript:
            # Append to transcript file with timestamp and duration
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] [{audio_duration:.2f}s] {transcript}\n")

            # Detect source language
            try:
                source_lang = detect(transcript)
            except LangDetectException:
                source_lang = "en"

            def translate_to_language(lang_info):
                target_lang = lang_info["code"]
                translated = translate_text(transcript, source_lang, target_lang)
                return target_lang, translated

            # Use ThreadPoolExecutor to parallelize translations
            translations = {}
            with ThreadPoolExecutor(max_workers=len(Config.TARGET_LANGUAGES)) as executor:
                results = executor.map(translate_to_language, Config.TARGET_LANGUAGES)
                for target_lang, translated in results:
                    translations[target_lang] = translated

            # Emit to frontend
            socketio.emit(
                "new_translation",
                {
                    "transcript": transcript,
                    "source_language": source_lang,
                    "translations": translations,
                    "timestamp": time.time(),
                },
            )
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
                    print(f"[AUDIO] Skipping transcription - audio too quiet")
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


@socketio.on("start_listening")
def handle_start_listening():
    """Start listening for audio"""
    global is_listening, audio_stream, p_audio, actual_sample_rate, num_channels

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
    print("Loading configuration...")
    print("Initializing models (this may take a minute)...")
    initialize_models()

    print("\n" + "=" * 60)
    print("Polyglot - Real-time Audio Translator")
    print("=" * 60)
    print("\nOpen your browser to: http://localhost:5000")
    print(f"Device: {Config.DEVICE}")
    print(f"Whisper Model: {Config.WHISPER_MODEL}")
    print(f"Translation Model: {Config.TRANSLATION_MODEL}")
    print(f"Target languages: {', '.join([lang['name'] for lang in Config.TARGET_LANGUAGES])}")
    print("\nPress Ctrl+C to stop\n")

    socketio.run(app, debug=False, host="0.0.0.0", port=5000)
