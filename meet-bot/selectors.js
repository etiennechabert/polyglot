// Google Meet DOM selectors.
//
// Meet's CSS classes are obfuscated and rotate. This file centralizes every
// selector the bot relies on so a UI change is a one-file fix. Prefer stable
// anchors (aria-label, role, visible text) over class names.

export const SELECTORS = {
  // ── Active-speaker / roster detection (Phase 3) ──────────────────────
  //
  // Meet's classes are obfuscated; everything here anchors on aria-label,
  // role, or data-* attributes that have been stable across rollouts.
  //
  // Tile container — wraps one participant's video + name + mic ring.
  // data-participant-id is the most stable anchor we have.
  participantTile: '[data-participant-id]',

  // Speaking indicators — Meet has used several over time; we try all.
  // Strategy 1: explicit boolean attribute (newer Meet)
  speakingAttr: '[data-is-speaking="true"]',
  // Strategy 2: aria-label on the tile or mic button says "… is speaking"
  speakingAriaLabel: '[aria-label*="is speaking" i]',
  // Strategy 3: the audio-level bars inside a tile animate when speaking.
  // Class is obfuscated, but the element always carries [data-is-muted="false"]
  // and its closest tile ancestor is the active speaker. Fallback only.
  audioLevelBar: '[data-is-muted="false"]',

  // Name extraction — checked in order inside a tile element.
  tileNameSelectors: [
    '[data-self-name]',             // newer Meet
    '[jsname="r8qRAd"]',            // one known jsname for name label
    'div[class][data-tooltip]',     // tooltip often holds display name
  ],

  // People panel — open it to get full roster.
  peopleButton:
    'button[aria-label*="people" i], button[aria-label*="everyone" i], button[aria-label*="participants" i]',
  // Each row in the People panel roster.
  rosterItem: '[data-participant-id] span[jsname], [role="listitem"] span',

  // Pre-join screen (anonymous guest path) -------------------------------
  // The "Your name" input shown to signed-out users on the Meet landing page
  // before joining. Meet has used multiple implementations; try in order.
  nameInput: [
    'input[aria-label="Your name"]',
    'input[placeholder="Your name"]',
    'input[jsname][type="text"]',
  ],

  // The button that submits the pre-join form. Its label depends on meeting
  // config: "Ask to join" when the host hasn't admitted you, "Join now" when
  // you're the host or pre-admitted. Match on visible text via Playwright's
  // getByRole('button', { name: ... }) at call sites — selector here is a
  // fallback for the ARIA role.
  joinButtonNames: ["Ask to join", "Join now", "Join"],

  // Pre-join sometimes prompts to turn off mic/cam — these buttons toggle
  // them. Anchored on aria-label which Meet has kept stable for years.
  micToggle: 'div[role="button"][aria-label*="microphone" i]',
  camToggle: 'div[role="button"][aria-label*="camera" i]',

  // In-call indicators ---------------------------------------------------
  // Presence of the leave-call button is the most reliable "we are in the
  // meeting" signal. Its aria-label is "Leave call".
  leaveCallButton: 'button[aria-label="Leave call"]',

  // Lobby / denial detection. When the host denies entry, Meet shows a
  // message containing this text.
  deniedText: /You can't join this call|no one responded|denied/i,

  // Bot / access blocked detection. Google redirects here or shows this text
  // when the meeting blocks automated access or the link is invalid.
  blockedUrls: ["workspace.google.com/products/meet", "accounts.google.com/v3/signin/rejected"],
  blockedText: /you can't join this video call|this meeting is locked|you're not allowed|not available/i,
};

// Helper: return the first selector from a list that matches something on
// the page. Used for resilient element lookup when Meet ships A/B variants.
export async function firstMatching(page, selectorList, timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    for (const sel of selectorList) {
      const el = await page.$(sel);
      if (el) return { selector: sel, element: el };
    }
    await page.waitForTimeout(250);
  }
  return null;
}
