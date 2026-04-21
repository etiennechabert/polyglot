// Polyglot Meet Bot — phases 1–3: join + audio capture + speaker detection.
//
// Usage:
//   node index.js --url https://meet.google.com/xxx-yyyy-zzz \
//                 [--name "Polyglot Bot"] \
//                 [--headful] \
//                 [--polyglot-url http://localhost:5001] \
//                 [--profile-dir <path>]   (default: ~/.polyglot-bot-profile)
//
// Exit codes: 0 clean leave, 1 crash, 2 bad args, 3 denied by host, 4 lobby timeout,
//             6 blocked (bot detected / meeting locked / link invalid).
//
// Persistent profile: the bot reuses a Chrome profile across runs so Google
// sign-in cookies survive. On first run the user logs in manually in the
// headful window; subsequent runs are already authenticated.

import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import { chromium } from "playwright";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
import { SELECTORS, firstMatching } from "./selectors.js";
import { RTC_INIT_SCRIPT, setupAudioCapture } from "./audio.js";
import { setupSpeakerDetection } from "./speaker.js";

// Dedicated bot profile — separate from the user's real Chrome so there's no
// instance conflict. Stored inside meet-bot/ so it's self-contained.
const DEFAULT_PROFILE = path.join(__dirname, "chrome-profile");

function parseArgs(argv) {
  const args = {
    url: null,
    name: "Polyglot Bot",
    headful: false,
    polyglotUrl: null,
    profileDir: DEFAULT_PROFILE,
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--url") args.url = argv[++i];
    else if (a === "--name") args.name = argv[++i];
    else if (a === "--headful") args.headful = true;
    else if (a === "--polyglot-url") args.polyglotUrl = argv[++i];
    else if (a === "--profile-dir") args.profileDir = argv[++i];
  }
  if (!args.url) {
    console.error(
      "Usage: node index.js --url <meet-link> [--name <display>] [--headful] [--polyglot-url <url>] [--profile-dir <path>]"
    );
    process.exit(2);
  }
  return args;
}

function log(msg, ...rest) {
  const ts = new Date().toISOString();
  console.log(`[${ts}] ${msg}`, ...rest);
}

// Connect to Polyglot's /meet_bot SocketIO namespace.
// Returns { sendAudio, sendEvent } or null if not configured / unavailable.
async function connectPolyglot(polyglotUrl, meetUrl) {
  if (!polyglotUrl) return null;
  try {
    const { io } = await import("socket.io-client");
    // Socket.IO reconnects automatically; give the initial handshake 15 s so
    // Polyglot has time to finish any lazy namespace registration.
    const socket = io(`${polyglotUrl}/meet_bot`, {
      transports: ["websocket"],
      reconnection: true,
      reconnectionDelay: 500,
      reconnectionDelayMax: 2000,
    });
    await new Promise((resolve, reject) => {
      socket.once("connect", resolve);
      setTimeout(() => reject(new Error("connect timeout")), 15000);
    });
    log(`Connected to Polyglot at ${polyglotUrl}/meet_bot`);
    // Self-report the meeting URL so the admin panel can show which call the
    // bot is attached to, even when the bot was started from the CLI.
    socket.emit("bot_info", { url: meetUrl });
    return {
      sendAudio: (pcm, captureTs) =>
        socket.emit("audio_frame", { capture_ts_ms: captureTs, sample_rate: 16000, channels: 1 }, pcm),
      sendEvent: (ev) =>
        socket.emit("speaker_event", ev),
    };
  } catch (err) {
    log(`WARN: Could not connect to Polyglot (${err.message}). Audio will not be forwarded.`);
    return null;
  }
}

async function joinMeeting({ url, name, headful, polyglotUrl, profileDir }) {
  log(`Launching Chrome (headful=${headful}, profile=${profileDir})`);

  // launchPersistentContext keeps cookies/localStorage across runs — the user
  // signs in once and subsequent runs are already authenticated.
  const context = await chromium.launchPersistentContext(profileDir, {
    headless: !headful,
    channel: "chrome",
    args: [
      "--use-fake-ui-for-media-stream",
      "--disable-blink-features=AutomationControlled",
      "--autoplay-policy=no-user-gesture-required",
      "--no-sandbox",
      "--disable-dev-shm-usage",
    ],
    permissions: ["microphone", "camera"],
  });

  // Stealth: unset navigator.webdriver before any page JS runs.
  await context.addInitScript(() => {
    Object.defineProperty(navigator, "webdriver", { get: () => undefined });
  });

  // Phase 2: patch RTCPeerConnection before Meet's JS initialises WebRTC.
  await context.addInitScript(RTC_INIT_SCRIPT);

  const page = await context.newPage();

  log(`Navigating to ${url}`);
  await page.goto(url, { waitUntil: "domcontentloaded" });

  // Pre-join loop — handles:
  //   • Anonymous guest flow (name field → join button)
  //   • Signed-in flow (join button directly, no name field)
  //   • Google sign-in redirect (wait indefinitely for user to log in)
  let clicked = false;
  while (!clicked) {
    await page.waitForTimeout(1200); // let Meet's JS settle

    const currentUrl = page.url();
    log(`Page: ${currentUrl.slice(0, 80)}`);

    // Block / bot-detection page — Google redirected us away from Meet entirely.
    if (SELECTORS.blockedUrls.some((u) => currentUrl.includes(u))) {
      log("BLOCKED — Google rejected access (bot detected, meeting locked, or link invalid).");
      return 6;
    }
    // "You can't join this video call" text shown inline on meet.google.com.
    const pageText = await page.evaluate(() => document.body.innerText).catch(() => "");
    if (SELECTORS.blockedText.test(pageText)) {
      log(`BLOCKED — page says: "${pageText.slice(0, 120)}"`);
      return 6;
    }

    // Sign-in wall — wait for the user to complete login in the headful window.
    if (currentUrl.includes("accounts.google.com")) {
      log("Google sign-in required — log in in the browser window. Bot will resume automatically.");
      await page.waitForURL((u) => !u.href.includes("accounts.google.com"), { timeout: 0 });
      log("Back on Meet — retrying pre-join flow.");
      await page.waitForLoadState("domcontentloaded");
      continue;
    }

    // Name field — present for anonymous guests only.
    const match = await firstMatching(page, SELECTORS.nameInput, 3000);
    if (match) {
      log(`Filling name field with "${name}"`);
      try {
        await page.locator(match.selector).first().fill(name, { timeout: 4000 });
      } catch (_) {
        continue; // navigation happened mid-fill — loop will detect it
      }
    } else {
      log("No name field — signed-in account.");
    }

    // Mute mic + camera before entering.
    for (const [label, sel] of [
      ["microphone", SELECTORS.micToggle],
      ["camera", SELECTORS.camToggle],
    ]) {
      try {
        const btn = await page.$(sel);
        if (btn) {
          const aria = (await btn.getAttribute("aria-label")) || "";
          if (/turn off/i.test(aria)) {
            log(`Muting ${label}`);
            await btn.click();
          }
        }
      } catch (_) {}
    }

    // Click join button.
    for (const label of SELECTORS.joinButtonNames) {
      const btn = page.getByRole("button", { name: label });
      if (await btn.count()) {
        log(`Clicking "${label}"`);
        try {
          await btn.first().click({ timeout: 5000 });
          clicked = true;
        } catch (e) {
          log(`  click failed: ${e.message.split("\n")[0]}`);
          // Try forcing through any overlay.
          try {
            await btn.first().click({ force: true, timeout: 3000 });
            clicked = true;
            log(`  forced click succeeded`);
          } catch (e2) {
            log(`  forced click also failed: ${e2.message.split("\n")[0]}`);
          }
        }
        break;
      }
    }

    if (!clicked) {
      log("Join button not visible yet — retrying in 2 s…");
      await page.waitForTimeout(2000);
    }
  }

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

    // Enable Meet's live captions — primary source of reliable speaker-identity.
    // Try in order: click the toolbar button, then the keyboard shortcut.
    try {
      await page.waitForTimeout(2000);
      // First, try to find the captions button by aria-label and click it.
      const captionsBtn = page.locator(
        'button[aria-label*="caption" i], button[aria-label*="subtitle" i]'
      ).first();
      if (await captionsBtn.count()) {
        const label = await captionsBtn.getAttribute("aria-label");
        // Only click if label indicates captions are OFF (turn on...).
        if (/turn on|show/i.test(label || "")) {
          await captionsBtn.click({ timeout: 3000 });
          log(`Enabled captions via button: "${label}"`);
        } else {
          log(`Captions already on: "${label}"`);
        }
      } else {
        // Fallback: keyboard shortcut. Click main area first to ensure focus.
        await page.click("body").catch(() => {});
        await page.keyboard.press("c");
        log("Enabled captions via keyboard shortcut.");
      }
    } catch (e) {
      log(`WARN: Could not enable captions: ${e.message}`);
    }

    // Phase 2: start audio capture.
    const polyglot = await connectPolyglot(polyglotUrl, url);
    let chunkCount = 0;
    log("Setting up audio capture…");
    await setupAudioCapture(page, (pcm, captureTs) => {
      chunkCount++;
      if (chunkCount % 50 === 0) {
        log(`Audio: ${chunkCount * 20} ms captured, last ts=${captureTs}`);
      }
      polyglot?.sendAudio(pcm, captureTs);
    });
    log("Audio capture active (16 kHz mono PCM16, 20 ms frames).");

    // Phase 3: speaker events and roster.
    await setupSpeakerDetection(page, (ev) => {
      if (ev.type === "roster_update") {
        log(`Roster: ${ev.participants.join(", ") || "(empty)"}`);
      } else {
        log(`${ev.type === "speaker_start" ? "  Speaking" : "Silent   "}  ${ev.name}`);
      }
      polyglot?.sendEvent(ev);
    });
    log("Speaker detection active.");

    await page
      .waitForSelector(SELECTORS.leaveCallButton, { state: "detached", timeout: 0 })
      .catch(() => {});
    log(`Meeting ended. Total audio captured: ${chunkCount * 20} ms`);
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
