# Inference Optimization PR

## Context

The app uses PyTorch for all ML models (Whisper, M2M100, pyannote, summarization) but runs with default "training mode" settings. Since we only do inference, we should optimize for that.

## Problem

PyTorch defaults assume you might want to train:
- Gradient tracking enabled (wastes VRAM storing computation graph)
- No inference optimizations applied
- Memory not aggressively freed

## Solution: Use `@torch.inference_mode()`

`inference_mode` is stricter than `no_grad()` and provides better optimizations:
- Disables gradient tracking entirely
- Enables additional inference optimizations
- Slightly faster than `no_grad()`

### Implementation

Decorate all inference entry points:

```python
@torch.inference_mode()
def transcribe_and_translate(audio_data, audio_duration):
    """Background thread for transcription and translation with speaker diarization"""
    # All model calls inside are automatically optimized
    ...

@torch.inference_mode()
def perform_speaker_diarization(audio_data, sample_rate):
    """Perform speaker diarization on audio data"""
    ...

@torch.inference_mode()
def generate_summary(transcript_text):
    """Generate a structured summary using the local LLM"""
    ...

@torch.inference_mode()
def translate_text(text, source_lang, target_lang):
    """Translate text using M2M100"""
    ...
```

### Functions to update in `app.py`

1. `transcribe_and_translate()` - main processing function
2. `perform_speaker_diarization()` - diarization inference
3. `generate_summary()` - summarization inference
4. `translate_text()` - translation inference
5. `translate_summary()` - calls translate_text, may not need decorator if translate_text has it

### Additional optimizations to consider

1. **Explicit `torch.cuda.empty_cache()`** after large operations
2. **Verify all models use `torch.float16`** consistently
3. **Consider `torch.backends.cudnn.benchmark = True`** for consistent input sizes

## Testing

1. Verify VRAM usage is lower/more stable
2. Verify inference speed is same or better
3. Verify output quality unchanged
4. Run extended session to check for memory leaks

## Branch name suggestion

`inference-mode-optimization`

## Dependencies

None - this is an independent optimization that can be merged before or after other PRs.
