// Phase 2: in-browser audio capture for the Meet bot.
//
// Two-part design:
//   1. RTC_INIT_SCRIPT — must be registered via context.addInitScript() BEFORE
//      page.goto() so it runs before Meet initialises its RTCPeerConnections.
//      It patches RTCPeerConnection to funnel every remote audio track into a
//      single shared MediaStream (window.__pgStream).
//
//   2. setupAudioCapture(page, onChunk) — called after the bot has joined.
//      Injects an AudioWorklet that downsamples all audio in __pgStream to
//      16 kHz mono PCM16, buffers into 20 ms frames, and sends each frame
//      back to Node via an exposed function.

// ── 1. RTC patch (init script) ───────────────────────────────────────────────

export const RTC_INIT_SCRIPT = `(function () {
  window.__pgStream = new MediaStream();
  const _Orig = window.RTCPeerConnection;
  class _Patched extends _Orig {
    constructor(...a) {
      super(...a);
      this.addEventListener('track', (ev) => {
        if (ev.track.kind !== 'audio') return;
        if (!window.__pgStream.getTrackById(ev.track.id))
          window.__pgStream.addTrack(ev.track);
      });
    }
  }
  window.RTCPeerConnection = _Patched;
})();`;

// ── 2. AudioWorklet processor source ─────────────────────────────────────────
//
// Nearest-neighbour resampler: maintains a fractional index across process()
// calls so downsampling is consistent across block boundaries.
// Buffers output until 320 samples (20 ms @ 16 kHz) are ready, then posts
// { pcm: ArrayBuffer, ts: number } to the main thread.

const WORKLET_SRC = `
class PgResampler extends AudioWorkletProcessor {
  constructor() { super(); this._idx = 0; this._buf = []; }

  process(inputs) {
    const ch = inputs[0]?.[0];
    if (!ch) return true;

    const ratio = sampleRate / 16000; // e.g. 3.0 for 48 kHz input
    while (this._idx < ch.length) {
      const s = ch[Math.floor(this._idx)];
      this._buf.push(Math.round(Math.max(-1, Math.min(1, s)) * 32767));
      this._idx += ratio;
    }
    this._idx -= ch.length; // carry fractional offset to next block

    while (this._buf.length >= 320) {
      const arr = new Int16Array(this._buf.splice(0, 320));
      this.port.postMessage({ pcm: arr.buffer, ts: Date.now() }, [arr.buffer]);
    }
    return true;
  }
}
registerProcessor('pg-resampler', PgResampler);
`;

// ── 3. setupAudioCapture ──────────────────────────────────────────────────────
//
// onChunk(pcm: Buffer, captureTs: number) is called for each 20 ms PCM16 frame.
// captureTs is wall-clock ms at the moment the worklet produced the frame —
// used later by resolve_speaker_identity() for time-alignment.

export async function setupAudioCapture(page, onChunk) {
  // Bridge from browser → Node. exposeFunction is safe to call post-navigate.
  await page.exposeFunction('__pgChunk', (b64, ts) => {
    onChunk(Buffer.from(b64, 'base64'), ts);
  });

  await page.evaluate(async (src) => {
    // Inject worklet via blob URL (no local server needed).
    const url = URL.createObjectURL(new Blob([src], { type: 'application/javascript' }));
    const ctx = new AudioContext();
    await ctx.resume(); // bypass autoplay suspension — bot has no user gesture
    await ctx.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);

    const node = new AudioWorkletNode(ctx, 'pg-resampler');

    // Worklet → Node bridge: encode PCM16 ArrayBuffer as base64 string so it
    // can cross the Playwright IPC boundary (exposeFunction only handles JSON).
    node.port.onmessage = ({ data: { pcm, ts } }) => {
      const u8 = new Uint8Array(pcm);
      let s = '';
      for (let i = 0; i < u8.length; i++) s += String.fromCharCode(u8[i]);
      window.__pgChunk(btoa(s), ts);
    };

    function connectTrack(track) {
      // Each track gets its own MediaStreamSource; sharing a single source
      // across tracks doesn't work — each source reads one stream.
      ctx.createMediaStreamSource(new MediaStream([track])).connect(node);
    }

    // Connect tracks already in the shared stream (joined mid-call or after
    // participants were already speaking).
    window.__pgStream.getAudioTracks().forEach(connectTrack);

    // Connect tracks added after this point (people join late, etc.).
    window.__pgStream.addEventListener('addtrack', (e) => {
      if (e.track.kind === 'audio') connectTrack(e.track);
    });

    // Fallback: some Meet versions route audio through <audio> elements instead
    // of exposing it via RTCPeerConnection track events. Tap those too.
    function connectEl(el) {
      if (el._pg) return;
      el._pg = true;
      try {
        const src = ctx.createMediaElementSource(el);
        src.connect(node);
        src.connect(ctx.destination); // keep the original playback alive
      } catch (_) { /* element may already be claimed */ }
    }
    document.querySelectorAll('audio').forEach(connectEl);
    new MutationObserver(() => document.querySelectorAll('audio').forEach(connectEl))
      .observe(document.documentElement, { childList: true, subtree: true });

  }, WORKLET_SRC);
}
