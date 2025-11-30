#!/usr/bin/env python3
"""
Benchmark script for model swapping between VRAM and system RAM.

This script measures the actual time it takes to:
1. Load models to GPU
2. Move models from GPU to CPU (RAM)
3. Move models from CPU back to GPU
4. Generate inference with summarization model

Run this on the target machine (5080 + 64GB RAM) to get real numbers.

Usage:
    python scripts/benchmark_model_swap.py
"""

import gc
import time
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)


@dataclass
class BenchmarkResult:
    operation: str
    duration_seconds: float
    model_name: str
    direction: str = ""  # "to_cpu" or "to_cuda"

    def __str__(self):
        if self.direction:
            return f"{self.operation} ({self.direction}): {self.duration_seconds:.2f}s - {self.model_name}"
        return f"{self.operation}: {self.duration_seconds:.2f}s - {self.model_name}"


def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "free_gb": total - reserved,
        }
    return None


def print_gpu_status(label=""):
    """Print current GPU memory status"""
    info = get_gpu_memory_info()
    if info:
        print(f"  GPU Memory {label}: {info['allocated_gb']:.2f}GB allocated, "
              f"{info['free_gb']:.2f}GB free of {info['total_gb']:.2f}GB total")


def clear_gpu_memory():
    """Clear GPU cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def benchmark_model_load(model_name: str, model_class, device: str = "cuda") -> tuple:
    """Benchmark loading a model directly to device"""
    print(f"\n{'='*60}")
    print(f"Loading {model_name} to {device}...")
    print_gpu_status("before")

    start = time.perf_counter()

    if model_class == "causal":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            model = model.to("cpu")
    elif model_class == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        model = model.to(device)

    if device == "cuda":
        torch.cuda.synchronize()

    duration = time.perf_counter() - start

    print_gpu_status("after")
    print(f"Load time: {duration:.2f}s")

    return model, BenchmarkResult("load", duration, model_name)


def benchmark_model_move(model, model_name: str, target_device: str) -> BenchmarkResult:
    """Benchmark moving a model between devices"""
    direction = f"to_{target_device}"
    print(f"\nMoving {model_name} to {target_device}...")
    print_gpu_status("before")

    start = time.perf_counter()

    model.to(target_device)

    if target_device == "cuda":
        torch.cuda.synchronize()
    else:
        clear_gpu_memory()

    duration = time.perf_counter() - start

    print_gpu_status("after")
    print(f"Move time: {duration:.2f}s")

    return BenchmarkResult("move", duration, model_name, direction)


def benchmark_summarization_inference(model, tokenizer, model_name: str) -> BenchmarkResult:
    """Benchmark a summarization inference"""
    # Sample meeting transcript
    transcript = """
    Speaker 1: Welcome everyone to today's meeting. We need to discuss the Q4 roadmap.
    Speaker 2: Thanks. I think we should prioritize the mobile app redesign.
    Speaker 1: Good point. What's the estimated timeline for that?
    Speaker 2: Our team estimates about 6 weeks for the core features.
    Speaker 3: We also need to consider the API changes required.
    Speaker 1: Right, let's make sure backend and frontend are aligned.
    Speaker 2: I'll schedule a sync meeting with both teams.
    Speaker 3: Should we also discuss the budget allocation?
    Speaker 1: Yes, that's next on the agenda. We have 50K allocated for Q4.
    """

    prompt = f"""<|im_start|>system
You are a meeting summarizer. Provide a brief summary of the key points.
<|im_end|>
<|im_start|>user
Summarize this meeting transcript:

{transcript}
<|im_end|>
<|im_start|>assistant
"""

    print(f"\nRunning inference on {model_name}...")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    if model.device.type == "cuda":
        torch.cuda.synchronize()

    duration = time.perf_counter() - start

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Inference time: {duration:.2f}s")
    print(f"Generated {outputs.shape[1] - inputs['input_ids'].shape[1]} tokens")

    return BenchmarkResult("inference", duration, model_name)


def benchmark_full_swap_cycle(
    transcription_models: list,
    summarizer_name: str,
):
    """
    Benchmark a full swap cycle:
    1. Move transcription models to CPU
    2. Load summarizer to GPU
    3. Run inference
    4. Move summarizer to CPU
    5. Move transcription models back to GPU
    """
    results = []

    print("\n" + "="*60)
    print("FULL SWAP CYCLE BENCHMARK")
    print("="*60)

    # Step 1: Move transcription models to CPU
    print("\n--- Step 1: Move transcription models to CPU ---")
    step1_start = time.perf_counter()
    for name, model in transcription_models:
        result = benchmark_model_move(model, name, "cpu")
        results.append(result)
    step1_duration = time.perf_counter() - step1_start
    print(f"\nStep 1 total: {step1_duration:.2f}s")

    clear_gpu_memory()
    print_gpu_status("after clearing cache")

    # Step 2: Load summarizer to GPU
    print("\n--- Step 2: Load summarizer to GPU ---")
    step2_start = time.perf_counter()

    summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_name, trust_remote_code=True)
    summarizer_model, load_result = benchmark_model_load(summarizer_name, "causal", "cuda")
    results.append(load_result)

    step2_duration = time.perf_counter() - step2_start
    print(f"\nStep 2 total: {step2_duration:.2f}s")

    # Step 3: Run inference
    print("\n--- Step 3: Run summarization inference ---")
    inference_result = benchmark_summarization_inference(
        summarizer_model, summarizer_tokenizer, summarizer_name
    )
    results.append(inference_result)

    # Step 4: Move summarizer to CPU
    print("\n--- Step 4: Move summarizer to CPU ---")
    step4_start = time.perf_counter()
    result = benchmark_model_move(summarizer_model, summarizer_name, "cpu")
    results.append(result)
    clear_gpu_memory()
    step4_duration = time.perf_counter() - step4_start
    print(f"\nStep 4 total: {step4_duration:.2f}s")

    # Step 5: Move transcription models back to GPU
    print("\n--- Step 5: Move transcription models back to GPU ---")
    step5_start = time.perf_counter()
    for name, model in transcription_models:
        result = benchmark_model_move(model, name, "cuda")
        results.append(result)
    step5_duration = time.perf_counter() - step5_start
    print(f"\nStep 5 total: {step5_duration:.2f}s")

    # Summary
    total_duration = step1_duration + step2_duration + inference_result.duration_seconds + step4_duration + step5_duration

    print("\n" + "="*60)
    print("SWAP CYCLE SUMMARY")
    print("="*60)
    print(f"Step 1 (transcription → CPU): {step1_duration:.2f}s")
    print(f"Step 2 (load summarizer):     {step2_duration:.2f}s")
    print(f"Step 3 (inference):           {inference_result.duration_seconds:.2f}s")
    print(f"Step 4 (summarizer → CPU):    {step4_duration:.2f}s")
    print(f"Step 5 (transcription → GPU): {step5_duration:.2f}s")
    print(f"{'─'*40}")
    print(f"TOTAL CYCLE TIME:             {total_duration:.2f}s")

    return results, total_duration


def main():
    print("="*60)
    print("MODEL SWAP BENCHMARK")
    print("="*60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        return

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print_gpu_status("initial")

    # Models to test
    # Using smaller models for the benchmark - adjust as needed
    TRANSLATION_MODEL = "facebook/m2m100_418M"  # Smaller version for testing
    # TRANSLATION_MODEL = "facebook/m2m100_1.2B"  # Full version

    SUMMARIZER_MODELS = [
        "Qwen/Qwen2-1.5B-Instruct",  # Current small model
        # "Qwen/Qwen2.5-7B-Instruct",  # Larger model - uncomment to test
        # "mistralai/Mistral-7B-Instruct-v0.3",  # Alternative 7B
    ]

    print("\n" + "="*60)
    print("PHASE 1: Load translation model to GPU (simulating normal operation)")
    print("="*60)

    # Load translation model (simulating the transcription stack)
    translation_model, _ = benchmark_model_load(TRANSLATION_MODEL, "seq2seq", "cuda")

    transcription_models = [
        (TRANSLATION_MODEL, translation_model),
    ]

    # Note: In real app, would also load Whisper and Pyannote here
    # Skipping for benchmark to keep it faster

    print("\n" + "="*60)
    print("PHASE 2: Test swap cycles with different summarizer models")
    print("="*60)

    for summarizer_name in SUMMARIZER_MODELS:
        print(f"\n{'#'*60}")
        print(f"Testing with: {summarizer_name}")
        print(f"{'#'*60}")

        try:
            results, total_time = benchmark_full_swap_cycle(
                transcription_models,
                summarizer_name,
            )

            print(f"\n>>> {summarizer_name}: {total_time:.2f}s total cycle time")

        except Exception as e:
            print(f"ERROR testing {summarizer_name}: {e}")
            import traceback
            traceback.print_exc()

        # Clean up between tests
        clear_gpu_memory()

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
