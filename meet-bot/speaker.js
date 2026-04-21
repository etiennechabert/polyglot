// Phase 3: active-speaker detection + roster scraping.
//
// Primary signal: Meet's live captions. When captions are enabled, each
// caption block carries the speaker's name as the first text node and the
// spoken words as the rest. This is far more reliable than DOM-class
// heuristics (which change every Meet rollout and rotate as ambient pulse).
//
// Fallback signals: the legacy data-is-speaking attribute and aria-label
// text — still present on rare pre-join / breakout-style UIs.

export async function setupSpeakerDetection(page, onEvent) {
  await page.exposeFunction("__pgSpeakerEvent", (json) => onEvent(JSON.parse(json)));

  await page.evaluate(() => {

    // ── Helpers ────────────────────────────────────────────────────────────

    function isUIAction(s) {
      return /^(?:pin|unpin|mute|unmute|remove|reframe|spotlight|present|share|more options|turn on|turn off|stop|start)/i.test(s.trim());
    }

    function nameFromTile(el) {
      const root = el.closest("[data-participant-id]") || el;

      const nameEl = root.querySelector("[data-self-name]");
      if (nameEl?.dataset.selfName) {
        const n = nameEl.dataset.selfName.trim();
        if (n && !isUIAction(n)) return n;
      }
      if (nameEl?.textContent) {
        const n = nameEl.textContent.trim();
        if (n && !isUIAction(n)) return n;
      }

      const rootLabel = root.getAttribute("aria-label") || "";
      if (rootLabel) {
        const m = rootLabel.match(/^(.+?)(?:'s\s+(?:video|screen|camera|tile|presentation)|(?:\s*\(you\)))/i);
        if (m?.[1] && !isUIAction(m[1])) return m[1].trim();
        if (rootLabel.length < 60 && !isUIAction(rootLabel) && !/\b(?:from|your|main|screen|presentation)\b/i.test(rootLabel))
          return rootLabel.trim();
      }

      for (const sel of ["span[jsname='r8qRAd']", "div[jsname='Cpqoke']", "div[data-tooltip]"]) {
        const candidate = root.querySelector(sel);
        const text = (candidate?.textContent || candidate?.dataset?.tooltip || "").trim();
        if (text && text.length < 60 && !isUIAction(text)) return text;
      }

      return null;
    }

    function emit(ev) {
      window.__pgSpeakerEvent(JSON.stringify(ev));
    }

    // ── Captions-based speaker detection ───────────────────────────────────
    //
    // We look for a container whose aria-label mentions "caption" /
    // "transcription" / "untertitel" etc. Meet renders each caption block as
    // a group with the speaker name as the first text child and the spoken
    // words as siblings. When a caption block updates (new words appended),
    // that speaker is currently active.

    const CAPTION_REGION_RE = /caption|transcript|untertitel|sous-titre|subtitulo|subtitle/i;

    function findCaptionContainer() {
      // 1. Strong preference: role=region with aria-label exactly "Captions"
      //    (or localized equivalent matching the captions-word regex).
      const regions = [...document.querySelectorAll('[role="region"]')];
      for (const el of regions) {
        const lbl = el.getAttribute("aria-label") || "";
        if (CAPTION_REGION_RE.test(lbl)) return el;
      }
      // 2. Fallback: ANY element whose aria-label is exactly "Captions" etc.
      for (const el of document.querySelectorAll('[aria-label]')) {
        const lbl = (el.getAttribute("aria-label") || "").trim();
        // Exact match on the WORD captions (not combobox "Caption type").
        if (/^(captions?|transcript|untertitel|sous-titres)$/i.test(lbl)) return el;
      }
      // 3. Known jsnames.
      for (const sel of ['div[jsname="dsyhDe"]', 'div[jsname="YSxPC"]', 'div[jsname="r5nxDd"]']) {
        const el = document.querySelector(sel);
        if (el) return el;
      }
      return null;
    }

    // Track active speakers: name → { lastUpdateMs, startMs }
    // A speaker is "active" if their caption block updated in the last 1.5 s.
    const activeSpeakers = new Map();
    const SPEAKER_TIMEOUT_MS = 1500;

    // Find the enclosing caption block for any DOM node. A caption block is
    // the wrapper that contains one speaker's current utterance. Meet uses a
    // class like `nMcdL` on this wrapper; if that rotates we fall back to
    // structural heuristics (a div whose direct children include both a
    // short "name" span and a larger "text" div).
    function findCaptionBlock(node) {
      let cur = node.nodeType === 1 ? node : node.parentElement;
      const container = findCaptionContainer();
      while (cur && cur !== container && cur !== document.body) {
        if (cur.matches && cur.matches('[class*="nMcdL"]')) return cur;
        cur = cur.parentElement;
      }
      // Structural fallback: direct child of container with children that look
      // like a (name, text) pair.
      if (container) {
        let p = node.nodeType === 1 ? node : node.parentElement;
        while (p && p.parentElement !== container) p = p.parentElement;
        if (p && p.parentElement === container) return p;
      }
      return null;
    }

    function extractBlockSpeaker(block) {
      // Try the known name span first.
      const nameEl = block.querySelector('span.NWpY1d, [class*="NWpY1d"]');
      if (nameEl) {
        const name = (nameEl.textContent || "").trim();
        if (name && name.length < 60 && !isUIAction(name) && /^[\p{L}]/u.test(name))
          return name;
      }
      // Structural fallback: find a short-text descendant at the top of the
      // block whose text is distinct from the long "spoken text" sibling.
      const texts = [];
      for (const el of block.querySelectorAll('*')) {
        const t = (el.textContent || "").trim();
        if (!t || t.length > 60) continue;
        if (!/^[\p{L}][\p{L}\s.'-]+$/u.test(t)) continue;
        if (isUIAction(t)) continue;
        texts.push(t);
      }
      return texts[0] || null;
    }

    function onCaptionMutation(mutations) {
      const now = Date.now();
      const processedBlocks = new Set();

      // Per-mutation block lookup — catches characterData updates.
      for (const m of mutations || []) {
        const block = findCaptionBlock(m.target);
        if (!block || processedBlocks.has(block)) continue;
        processedBlocks.add(block);
        markSpeakerActive(extractBlockSpeaker(block), now);
      }

      // Whole-container re-scan — covers cases where Meet replaces entire
      // caption blocks rather than appending characterData to existing ones.
      // We use a data attribute to track each block's last-seen text length
      // and treat any growth as active speech.
      const container = findCaptionContainer();
      if (!container) return;
      for (const block of container.querySelectorAll('[class*="nMcdL"]')) {
        if (processedBlocks.has(block)) continue;
        const textEl = block.querySelector('[class*="ygicle"], [class*="VbkSUe"]');
        const len = textEl ? (textEl.textContent || "").length : 0;
        const prev = parseInt(block.getAttribute("data-pg-len") || "-1", 10);
        if (len !== prev) {
          block.setAttribute("data-pg-len", String(len));
          markSpeakerActive(extractBlockSpeaker(block), now);
        }
      }
    }

    function markSpeakerActive(name, now) {
      if (!name) return;
      const existing = activeSpeakers.get(name);
      if (!existing) {
        activeSpeakers.set(name, { lastUpdateMs: now, startMs: now });
        emit({ type: "speaker_start", name, wall_clock_ms: now });
      } else {
        existing.lastUpdateMs = now;
      }
    }

    // Sweeper: close intervals for speakers whose captions haven't updated.
    function sweepInactive() {
      const now = Date.now();
      for (const [name, info] of activeSpeakers) {
        if (now - info.lastUpdateMs > SPEAKER_TIMEOUT_MS) {
          emit({ type: "speaker_end", name, wall_clock_ms: now });
          activeSpeakers.delete(name);
        }
      }
    }

    // ── Legacy DOM signals (fallback) ──────────────────────────────────────

    let legacyLastSpeaker = null;
    function checkLegacySpeaker() {
      // Strategy 1: data-is-speaking="true"
      const s1 = document.querySelector('[data-is-speaking="true"]');
      let speaker = s1 ? nameFromTile(s1) : null;

      if (!speaker) {
        // Strategy 2: aria-label "X is speaking"
        for (const el of document.querySelectorAll("[aria-label]")) {
          const lbl = el.getAttribute("aria-label");
          const m = lbl.match(/^(.+?)\s+is speaking/i);
          if (m?.[1] && !isUIAction(m[1])) { speaker = m[1].trim(); break; }
        }
      }

      if (speaker === legacyLastSpeaker) return;
      const now = Date.now();
      if (legacyLastSpeaker) emit({ type: "speaker_end", name: legacyLastSpeaker, wall_clock_ms: now });
      if (speaker) emit({ type: "speaker_start", name: speaker, wall_clock_ms: now });
      legacyLastSpeaker = speaker;
    }

    // ── Roster scraping ────────────────────────────────────────────────────

    function scrapeRoster() {
      const names = new Set();
      for (const tile of document.querySelectorAll("[data-participant-id]")) {
        const n = nameFromTile(tile);
        if (n && n.length < 60 && !/^\(you\)$|^you$/i.test(n) && !isUIAction(n))
          names.add(n);
      }
      return [...names];
    }

    let lastRosterKey = "";
    function checkRoster() {
      const roster = scrapeRoster();
      const key = roster.slice().sort().join("|");
      if (key === lastRosterKey) return;
      lastRosterKey = key;
      emit({ type: "roster_update", participants: roster, wall_clock_ms: Date.now() });
    }

    // ── Observers / scheduler ──────────────────────────────────────────────

    // Caption observer: attach once the container appears; re-attach if Meet
    // rerenders it.
    let captionObserver = null;
    let boundContainer = null;
    function ensureCaptionObserver() {
      const container = findCaptionContainer();
      if (!container || container === boundContainer) return;
      if (captionObserver) captionObserver.disconnect();
      captionObserver = new MutationObserver(onCaptionMutation);
      captionObserver.observe(container, { childList: true, subtree: true, characterData: true });
      boundContainer = container;
      // Immediate scan in case captions already exist.
      onCaptionMutation();
    }

    // Roster observer — tiles come and go with gallery pagination.
    const rosterObserver = new MutationObserver(() => checkRoster());
    rosterObserver.observe(document.body, {
      childList: true, subtree: true, attributes: true,
      attributeFilter: ["data-self-name", "aria-label"],
    });

    // Periodic work: ensure observers attached, sweep inactive speakers, run
    // legacy fallback, refresh roster.
    setInterval(() => {
      ensureCaptionObserver();
      sweepInactive();
      checkLegacySpeaker();
      checkRoster();
    }, 500);

    setTimeout(() => {
      ensureCaptionObserver();
      checkRoster();
    }, 2000);

  });
}
