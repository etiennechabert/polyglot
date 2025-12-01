# Model Swapping for Summarization

## Context

The app runs on an RTX 5080 (16GB VRAM) with 64GB system RAM. The current approach loads all models simultaneously, which limits the summarization model to ~1.5B parameters.

## Current VRAM Usage (All Models Loaded)

| Model | VRAM |
|-------|------|
| Whisper turbo | ~6GB |
| M2M100 1.2B | ~5GB |
| Pyannote diarization | ~3GB |
| Qwen2-1.5B (summarizer) | ~1.5GB |
| **Total** | **~15.5GB** |

This leaves no headroom and limits summary quality.

## Proposed Solution: Dynamic Model Swapping

Instead of keeping all models in VRAM, leverage the 64GB system RAM:

1. **At startup**: Load ALL models into system RAM
2. **Normal operation**: Transcription stack (Whisper + M2M100 + Pyannote) in VRAM, summarizer warm in RAM
3. **Summary time**: Swap transcription stack to RAM, load summarizer to VRAM
4. **After summary**: Swap back, process any queued audio

### Benefits

- Can use much larger summarization models (7B, 14B+)
- Better summary quality
- No audio loss (just queued for later processing)

### Estimated Swap Times

PCIe 4.0 x16 bandwidth: ~15-20 GB/s realistic

| Operation | Time |
|-----------|------|
| Move transcription stack to RAM | ~1-2s |
| Move summarizer to VRAM | ~1s |
| Generate summary | ~5-10s |
| Move summarizer to RAM | ~1s |
| Move transcription stack to VRAM | ~1-2s |
| **Total cycle** | **~10-15s** |

This is acceptable for 5-10 minute summary intervals.

## Recommended Summarization Models

With full 16GB available during summarization:

| Model | Parameters | VRAM (fp16) | Quality |
|-------|------------|-------------|---------|
| Qwen2.5-7B-Instruct | 7B | ~14GB | Very good |
| Mistral-7B-Instruct-v0.3 | 7B | ~14GB | Very good |
| Llama-3.1-8B-Instruct | 8B | ~16GB | Excellent |
| Phi-3-medium-4k-instruct | 14B | ~28GB* | Excellent |

*Would require quantization (Q4/Q8) to fit in 16GB

## Implementation Notes

### PyTorch Model Moving

```python
# Move to CPU (stays warm in RAM)
model.to('cpu')
torch.cuda.empty_cache()  # Release VRAM

# Move to GPU
model.to('cuda')
```

### Key Considerations

1. **Audio buffering**: During swap, audio continues to be captured and queued
2. **Backlog processing**: After swap back, process queued audio segments
3. **CUDA cache**: Must call `torch.cuda.empty_cache()` after moving models off GPU
4. **Warmup**: First inference after moving to GPU may be slightly slower

### Architecture Changes Needed

1. Modify `initialize_models()` to support loading to CPU
2. Create `swap_to_summarization_mode()` function
3. Create `swap_to_transcription_mode()` function
4. Modify audio processing loop to handle queuing during swap
5. Add configuration for summary interval and model choice

## Testing Required

1. Benchmark actual swap times on 5080
2. Measure summary generation time with larger models
3. Test audio queue handling during swap
4. Verify no memory leaks over extended operation

## Open Questions

- [ ] Optimal summary interval (1min, 5min, 10min)?
- [ ] Which 7B model produces best meeting summaries?
- [ ] Should we quantize to fit even larger models?
- [ ] Notify viewers during swap or stay silent?
