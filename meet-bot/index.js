// Polyglot Meet Bot — phase 1: anonymous guest join + lobby wait.
//
// Usage:
//   node index.js --url https://meet.google.com/xxx-yyyy-zzz [--name "Polyglot Bot"] [--headful]
//
// Exits 0 on clean leave, non-zero on join failure / denial / crash. Audio
// capture and DOM speaker-event scraping are not implemented yet — this
// phase validates the riskiest piece (can we actually get into a meeting?)
// before wiring anything to Polyglot.

import { chromium } from "playwright";
import { SELECTORS, firstMatching } from "./selectors.js";

function parseArgs(argv) {
  const args = { url: null, name: "Polyglot Bot", headful: false };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--url") args.url = argv[++i];
    else if (a === "--name") args.name = argv[++i];
    else if (a === "--headful") args.headful = true;
  }
  if (!args.url) {
    console.error("Usage: node index.js --url <meet-link> [--name <display>] [--headful]");
    process.exit(2);
  }
  return args;
}

function log(msg, ...rest) {
  const ts = new Date().toISOString();
  console.log(`[${ts}] ${msg}`, ...rest);
}

async function joinMeeting({ url, name, headful }) {
  log(`Launching Chromium (headful=${headful})`);

  const browser = await chromium.launch({
    headless: !headful,
    args: [
      // Auto-grant mic/cam permission prompts so Meet's pre-join screen
      // doesn't block. The bot mutes both immediately after join.
      "--use-fake-ui-for-media-stream",
      // Some distros need this when running Chromium without a full desktop.
      "--no-sandbox",
      "--disable-dev-shm-usage",
    ],
  });

  // Fresh context = incognito-equivalent. No cookies, no persistent profile,
  // no Google sign-in. Meet treats us as an anonymous guest.
  const context = await browser.newContext({
    permissions: ["microphone", "camera"],
  });
  const page = await context.newPage();

  log(`Navigating to ${url}`);
  await page.goto(url, { waitUntil: "domcontentloaded" });

  // Fill the "Your name" field. Meet has had multiple implementations of
  // this input across A/B rollouts, so we try each known selector.
  log("Waiting for name input");
  const match = await firstMatching(page, SELECTORS.nameInput, 20000);
  if (!match) {
    throw new Error(
      "Could not find the 'Your name' field. This meeting may require a signed-in Google account."
    );
  }
  log(`Filling name field (${match.selector}) with "${name}"`);
  await match.element.fill(name);

  // Make sure mic + camera are off *before* joining so we don't blast audio
  // into the meeting or show a black-tile camera. Meet's pre-join toggles
  // default to on; we flip them if aria-label indicates they're currently on.
  for (const [label, sel] of [
    ["microphone", SELECTORS.micToggle],
    ["camera", SELECTORS.camToggle],
  ]) {
    const btn = await page.$(sel);
    if (btn) {
      const aria = (await btn.getAttribute("aria-label")) || "";
      // Meet writes "Turn off <device>" when currently on, "Turn on <device>"
      // when currently off. We want them off pre-join.
      if (/turn off/i.test(aria)) {
        log(`Muting ${label} (was on)`);
        await btn.click();
      }
    }
  }

  // Click the join button. Label varies: "Ask to join" (normal guest),
  // "Join now" (pre-admitted). Try each known label in order.
  let clicked = false;
  for (const label of SELECTORS.joinButtonNames) {
    const btn = page.getByRole("button", { name: label });
    if (await btn.count()) {
      log(`Clicking "${label}"`);
      await btn.first().click();
      clicked = true;
      break;
    }
  }
  if (!clicked) throw new Error("Could not find a Join / Ask-to-join button.");

  // Wait for one of three terminal states:
  //   1. Leave-call button appears -> we're in the meeting.
  //   2. "Denied" text appears -> host rejected us.
  //   3. Timeout -> still in lobby, host never admitted.
  log("Waiting for host to admit from lobby (up to 2 min)…");
  const inMeeting = await Promise.race([
    page
      .waitForSelector(SELECTORS.leaveCallButton, { timeout: 120000 })
      .then(() => "joined")
      .catch(() => null),
    page
      .waitForFunction(
        (pattern) => new RegExp(pattern.source, pattern.flags).test(document.body.innerText),
        { source: SELECTORS.deniedText.source, flags: SELECTORS.deniedText.flags },
        { timeout: 120000 }
      )
      .then(() => "denied")
      .catch(() => null),
  ]);

  if (inMeeting === "joined") {
    log("JOINED — bot is in the meeting.");
    // Keep the page alive so you can see it in Meet. Phase 2 will add
    // audio capture + DOM scraping here. For now, stay connected until the
    // leave button disappears (meeting ended / we were removed) or SIGINT.
    await page
      .waitForSelector(SELECTORS.leaveCallButton, { state: "detached", timeout: 0 })
      .catch(() => {});
    log("Leave-call button gone — meeting ended or bot was removed.");
    return 0;
  }

  if (inMeeting === "denied") {
    log("DENIED — host rejected the join request.");
    return 3;
  }

  log("TIMED OUT in lobby (2 min) — host did not admit.");
  return 4;
}

(async () => {
  const args = parseArgs(process.argv);
  let code = 1;
  try {
    code = await joinMeeting(args);
  } catch (err) {
    log("ERROR:", err.message);
    code = 1;
  }
  process.exit(code);
})();
