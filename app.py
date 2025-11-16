"""
Polyglot - Real-time Multi-Language Audio Translator
Captures system audio and translates speech into multiple languages simultaneously
"""

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio
import resampy
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO
from langdetect import LangDetectException, detect
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, pipeline

from config import Config

# Configuration from config.py
TRANSCRIPT_FILE = Path(Config.TRANSCRIPT_FILE)
SAMPLE_RATE = Config.SAMPLE_RATE
CHUNK_SIZE = Config.CHUNK_SIZE

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
                # Drain queue during processing to avoid buildup
                drained = 0
                try:
                    audio_queue.get_nowait()
                    drained += 1
                except queue.Empty:
                    pass
                if drained > 0:
                    print(f"[AUDIO] Drained {drained} chunks while processing")
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

            # Log audio level every 50 chunks for debugging
            if len(buffer) % 50 == 0:
                print(
                    f"[AUDIO] Level: {audio_level:.4f}, Buffer: {len(buffer)} chunks, Silence: {silence_counter}",
                    flush=True,
                )

            # Check if current chunk is silent
            is_silent = audio_level < audio_thresholds["silence_threshold"]

            if is_silent:
                silence_counter += 1
            else:
                if silence_counter > 0:
                    print(
                        f"[AUDIO] Sound detected (level: {audio_level:.4f}), "
                        f"resetting silence counter (was {silence_counter})",
                        flush=True,
                    )
                silence_counter = 0  # Reset on any sound

            buffer.append(chunk)

            # Process when we detect end of sentence (silence after minimum audio) OR max buffer reached
            silence_detected = len(buffer) >= min_chunks and silence_counter >= audio_thresholds["silence_chunks"]
            max_length_reached = len(buffer) >= max_chunks
            should_process = silence_detected or max_length_reached

            if should_process:
                # Determine which condition triggered processing
                trigger_reason = []
                if silence_detected:
                    trigger_reason.append(
                        f"SILENCE_DETECTED (buffer: {len(buffer)}/{min_chunks} chunks, "
                        f"silence: {silence_counter}/{audio_thresholds['silence_chunks']} chunks)"
                    )
                if max_length_reached:
                    trigger_reason.append(f"MAX_LENGTH_REACHED ({len(buffer)}/{max_chunks} chunks)")

                print("[AUDIO] ===== STARTING PROCESSING =====")
                print(f"[AUDIO] Trigger: {' AND '.join(trigger_reason)}")
                print(f"[AUDIO] Buffer size: {len(buffer)} chunks, Silence counter: {silence_counter}")
                print(f"[AUDIO] Queue size before flush: {audio_queue.qsize()}")

                is_processing = True  # Set lock

                audio_data = np.concatenate(buffer, axis=0)
                buffer = []
                silence_counter = 0

                # Clear the queue BEFORE processing to avoid duplicate audio
                flushed_count = 0
                while not audio_queue.empty():
                    try:
                        audio_queue.get_nowait()
                        flushed_count += 1
                    except queue.Empty:
                        break
                print(f"[AUDIO] Flushed {flushed_count} chunks from queue")

                # Convert to float32
                audio_float = audio_data.flatten().astype(np.float32)

                # Reshape to (samples, channels) if stereo
                if num_channels == 2:
                    audio_float = audio_float.reshape(-1, 2)
                    # Convert stereo to mono by averaging channels
                    audio_mono = audio_float.mean(axis=1)
                else:
                    audio_mono = audio_float

                print(f"[AUDIO] Original: {len(audio_mono)} samples @ {actual_sample_rate}Hz")

                # Resample from actual_sample_rate to 16000Hz for Whisper
                if actual_sample_rate != SAMPLE_RATE:
                    audio_resampled = resampy.resample(audio_mono, actual_sample_rate, SAMPLE_RATE)
                    print(f"[AUDIO] Resampled: {len(audio_resampled)} samples @ {SAMPLE_RATE}Hz")
                else:
                    audio_resampled = audio_mono

                print(f"[AUDIO] Duration: {len(audio_resampled) / SAMPLE_RATE:.2f} seconds")

                # Check average audio level to detect if it's mostly silence
                avg_audio_level = np.abs(audio_resampled).mean()
                print(f"[AUDIO] Average level: {avg_audio_level:.4f}")

                # Skip transcription if audio is too quiet (likely silence/hallucination)
                if avg_audio_level < Config.MIN_AUDIO_LEVEL:
                    print(
                        f"[AUDIO] Audio too quiet ({avg_audio_level:.4f} < {Config.MIN_AUDIO_LEVEL}), "
                        f"skipping transcription to avoid hallucination"
                    )
                    is_processing = False
                    continue

                # Transcribe using resampled audio (auto-detect language)
                print("[TRANSCRIBE] Starting Whisper transcription...")
                start_time = time.time()
                result = transcription_pipe(audio_resampled)
                transcribe_time = time.time() - start_time
                print(f"[TRANSCRIBE] Completed in {transcribe_time:.2f} seconds")

                transcript = result["text"].strip()

                if transcript:
                    print(f"[TRANSCRIBE] Result: {transcript[:100]}...")

                    # Append to transcript file
                    with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
                        f.write(f"{transcript}\n")
                    print(f"[FILE] Appended to {TRANSCRIPT_FILE}")

                    # Detect source language
                    try:
                        source_lang = detect(transcript)
                        print(f"[LANGUAGE] Detected: {source_lang}")
                    except LangDetectException:
                        source_lang = "en"
                        print(f"[LANGUAGE] Detection failed, defaulting to: {source_lang}")

                    # Prepare translations in parallel
                    print(f"[TRANSLATE] Starting parallel translation to {len(Config.TARGET_LANGUAGES)} languages...")
                    trans_start = time.time()

                    def translate_to_language(lang_info):
                        target_lang = lang_info["code"]
                        print(f"[TRANSLATE] {source_lang} -> {target_lang}...")
                        start = time.time()
                        translated = translate_text(transcript, source_lang, target_lang)
                        elapsed = time.time() - start
                        print(f"[TRANSLATE] {source_lang} -> {target_lang} completed in {elapsed:.2f}s")
                        return target_lang, translated

                    # Use ThreadPoolExecutor to parallelize translations
                    translations = {}
                    with ThreadPoolExecutor(max_workers=len(Config.TARGET_LANGUAGES)) as executor:
                        results = executor.map(translate_to_language, Config.TARGET_LANGUAGES)
                        for target_lang, translated in results:
                            translations[target_lang] = translated

                    trans_time = time.time() - trans_start
                    print(f"[TRANSLATE] All translations completed in {trans_time:.2f}s")

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

                    total_time = time.time() - start_time
                    print(f"[AUDIO] ===== PROCESSING COMPLETE (Total: {total_time:.2f}s) =====")
                else:
                    print("[TRANSCRIBE] Empty transcript, skipping")

                is_processing = False  # Release lock
                print(f"[AUDIO] Released processing lock, queue size: {audio_queue.qsize()}")

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
    print("Polyglot 🌍 - Real-time Audio Translator")
    print("=" * 60)
    print("\nOpen your browser to: http://localhost:5000")
    print(f"Device: {Config.DEVICE}")
    print(f"Whisper Model: {Config.WHISPER_MODEL}")
    print(f"Translation Model: {Config.TRANSLATION_MODEL}")
    print(f"Target languages: {', '.join([lang['name'] for lang in Config.TARGET_LANGUAGES])}")
    print("\nPress Ctrl+C to stop\n")

    socketio.run(app, debug=False, host="0.0.0.0", port=5000)
