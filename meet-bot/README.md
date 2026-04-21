# Polyglot Meet Bot

Headless Chromium bot that joins a Google Meet as an anonymous guest. Future phases will stream meeting audio and active-speaker names back to the Polyglot server; this initial phase validates only the join-and-get-admitted flow.

## Setup

```bash
cd meet-bot
npm install
npx playwright install chromium
```

Node 20+ required.

## Run

```bash
# Typical use — fully headless:
node index.js --url "https://meet.google.com/xxx-yyyy-zzz"

# Watch what the bot sees (debug Meet UI issues):
node index.js --url "https://meet.google.com/xxx-yyyy-zzz" --headful

# Override the displayed name (default "Polyglot Bot"):
node index.js --url "..." --name "Transcription Bot"
```

## What it does (phase 1)

1. Launches a fresh, cookieless Chromium — no Google sign-in.
2. Opens the Meet URL, waits for the pre-join screen.
3. Fills the "Your name" field, mutes mic + camera, clicks **Ask to join**.
4. Waits up to 2 minutes for the host to admit it.
5. Once admitted, stays connected until the meeting ends or it's removed.

Exit codes:

| Code | Meaning |
|------|---------|
| 0    | Joined successfully, then meeting ended / bot removed cleanly |
| 1    | Crash / unexpected error (see stderr) |
| 2    | Bad CLI arguments |
| 3    | Host explicitly denied the join request |
| 4    | Timed out in the lobby (host never admitted) |

## Testing

The easy test: open Meet in a normal browser tab, start a meeting as host, run the bot with `--headful --url <link>`, and admit it from the participants panel when it shows up as "Polyglot Bot". You should see the bot's Chromium window join the call.

## What's NOT here yet

- Audio capture (tab audio → 16 kHz PCM16 → Polyglot WebSocket)
- DOM scraping of active-speaker name and participant roster
- WebSocket connection to the Polyglot backend
- Control channel (join/leave commands from Polyglot's admin UI)

Those land in subsequent phases once we've validated the bot can reliably get into meetings.

## Selectors

All Meet DOM selectors live in `selectors.js`. When Meet ships a UI change and the bot breaks, that's the file to update — nothing else should need touching.
