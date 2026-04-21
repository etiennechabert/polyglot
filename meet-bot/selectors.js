// Google Meet DOM selectors.
//
// Meet's CSS classes are obfuscated and rotate. This file centralizes every
// selector the bot relies on so a UI change is a one-file fix. Prefer stable
// anchors (aria-label, role, visible text) over class names.

export const SELECTORS = {
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
