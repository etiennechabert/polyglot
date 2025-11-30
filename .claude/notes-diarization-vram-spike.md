# Diarization VRAM Spike Issue

## Observed Problem

When running with pyannote diarization enabled, VRAM consumption spikes from ~8GB to ~16GB at the end of each audio buffer processing. This spike happens during the diarization inference step.

## Suspected Causes

### 1. Default `embedding_batch_size` is too high

The pyannote diarization pipeline defaults to `embedding_batch_size=32`, meaning it processes 32 audio segments simultaneously for speaker embeddings. This roughly doubles VRAM during inference.

**Evidence**: GitHub issue [#1580](https://github.com/pyannote/pyannote-audio/issues/1580) reports speaker-diarization-3.1 using 14GB instead of 6GB compared to previous versions. Maintainers recommend reducing `embedding_batch_size`.

### 2. Missing `torch.no_grad()` context

The diarization call at line 551 in `app.py` is not wrapped in `torch.no_grad()`. PyTorch may be storing intermediate tensors for gradient computation even though we only need inference.

## Proposed Fix

### Fix 1: Reduce embedding batch size

In `app.py`, after line 274 where the pipeline is moved to GPU:

```python
diarization_pipeline.to(torch.device("cuda"))
diarization_pipeline.embedding_batch_size = 16  # ADD THIS - reduces from default 32
```

#### Testing Strategy

Reduce gradually and monitor VRAM spikes:

| Batch Size | Expected VRAM | Expected Speed | Test Order |
|------------|---------------|----------------|------------|
| 32 (default) | ~16GB spike | Fastest | Baseline |
| 16 | ~12GB spike | Fast | Try first |
| 8 | ~10GB spike | Moderate | If 16 still spikes |
| 1 | ~8-9GB stable | Slowest | Last resort |

**Quality is unaffected** - same embeddings are computed, just fewer in parallel.

#### Speed vs Memory Tradeoff

GPUs are optimized for parallel processing. Smaller batches = more sequential work = underutilization.

For a ~30 second audio chunk, rough estimates:
- Batch 32: ~0.5-1s diarization time
- Batch 16: ~1-1.5s
- Batch 8: ~1.5-2s
- Batch 1: ~2-4s

Since we process short chunks (not hour-long files), the speed difference is seconds, not minutes. Acceptable tradeoff for stable VRAM.

### Fix 2: Wrap inference in no_grad

In `app.py`, around line 551:

```python
# Current code:
diarization_output = diarization_pipeline(waveform_dict)

# Should be:
with torch.no_grad():
    diarization_output = diarization_pipeline(waveform_dict)
```

## Testing Required

1. Monitor VRAM with `nvidia-smi -l 1` during audio processing
2. Test with `embedding_batch_size` values: 1, 8, 16, 32 (default)
3. Measure impact on diarization speed
4. Verify speaker detection quality is not affected

## Expected Outcome

With `embedding_batch_size=1` and `torch.no_grad()`, VRAM should stay relatively stable around 8-11GB instead of spiking to 16GB.

## References

- [speaker-diarization-3.1 high memory usage · Issue #1580](https://github.com/pyannote/pyannote-audio/issues/1580)
- [CUDA runs out of memory · Issue #684](https://github.com/pyannote/pyannote-audio/issues/684)
- [Diarization fails on long audio · Issue #1897](https://github.com/pyannote/pyannote-audio/issues/1897)
