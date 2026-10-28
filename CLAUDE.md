# CLAUDE.md — Real-Time AI Voice Changer

## How to work with me (read first)

- **Ask before acting.** If anything is ambiguous, missing, or has more than one reasonable approach, ask me questions BEFORE writing code or running commands. Batch questions into one message.
- **If you have no questions, just go.** Don't ask for permission on obvious steps.
- **Be concise and direct.** No filler, no long recaps. After finishing a step, give 1-2 sentences on what changed and what's next.
- **Give me complete, pasteable files**, not snippets or diffs, whenever you create or change code.
- **Work in phases** (see Roadmap). Finish and verify one phase before starting the next. Tell me how to test each phase.
- **Don't install system software or change macOS settings silently.** Tell me the exact command first (e.g. `brew install blackhole-2ch`) and why.
- **Never print, log, or commit my API key.**
- **Verify APIs against the current ElevenLabs docs** before coding against them (model names, endpoints, parameters change). If docs and this file disagree, docs win; tell me.

## Project goal

A tool that takes my live microphone, speaks my words back in my custom ElevenLabs voice, and outputs that audio to a **virtual microphone** so apps (Discord, Zoom, games, OBS) hear the AI voice as if it were my mic.

Same-language only (no translation). Don't add translation unless I ask.

## Decisions already made

| Item | Choice |
|---|---|
| OS | macOS |
| Language | Python 3.11+ |
| Virtual mic | BlackHole 2ch (`brew install blackhole-2ch`) |
| Pipelines | **Both, switchable at runtime/config:** (A) streaming STT → TTS, (B) ElevenLabs speech-to-speech (Voice Changer) |
| Interface | **CLI first**, PyQt6 GUI later (Phase 5) |
| Voice | My existing custom voice in ElevenLabs (ID goes in `.env`) |
| Translation | Not included |

## Pipelines

**Pipeline A: Streaming STT → TTS (target: lowest latency)**
1. Capture mic audio in small frames.
2. Stream to ElevenLabs realtime speech-to-text (Scribe v2 Realtime, ~150ms, built-in VAD turn detection).
3. Feed text (partial/committed) into the ElevenLabs TTS **websocket** (`stream-input`) using **Flash v2.5** (~75ms inference).
4. Play returned audio chunks to the virtual mic as they arrive. Don't wait for the full sentence.

**Pipeline B: Speech-to-speech (Voice Changer)**
1. Capture mic audio; use VAD to detect start/end of a phrase.
2. Send the phrase to the speech-to-speech **stream** endpoint with my voice ID (English STS model, highest sensible latency optimization level).
3. Play returned audio chunks to the virtual mic.
4. Keeps my emotion and timing but is chunked, so latency is higher. That's expected.

Both pipelines implement the same interface so they can be swapped without touching audio I/O code.

## Audio routing (macOS)

- Input: my real mic (selectable by device name/index).
- Output: **BlackHole 2ch** (this is what Discord/Zoom/etc. select as their microphone).
- Optional monitor: also play the converted audio to my headphones (toggle). **Headphones required**, otherwise speakers feed back into the mic.
- Tell me how to grant mic permission to Terminal/VS Code if capture fails.
- Provide a `--list-devices` command that prints available input/output devices.

## Tech stack

- `elevenlabs` SDK + `websockets` (check current docs for realtime STT and TTS websocket usage)
- `sounddevice` + `numpy` for capture/playback
- VAD: `webrtcvad` (or `webrtcvad-wheels` if the build fails on Mac), with a simple RMS-threshold fallback
- `python-dotenv` for config
- `asyncio` for the streaming pipeline
- PyQt6 for the GUI (Phase 5 only)

## Project structure

```
voice-changer/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example          # ELEVENLABS_API_KEY=, ELEVENLABS_VOICE_ID=
├── .gitignore            # must include .env, .venv, __pycache__
├── main.py               # CLI entry point
├── config.py             # loads .env + CLI args, model IDs, audio settings
├── devices.py            # list/select audio devices
├── audio_io.py           # mic capture, virtual-mic playback, monitor output
├── vad.py                # voice activity detection
├── latency.py            # timing/metrics helper
├── pipelines/
│   ├── base.py           # Pipeline interface (start, stop, feed_audio, on_audio_out)
│   ├── stt_tts.py        # Pipeline A
│   └── sts.py            # Pipeline B
├── gui/                  # Phase 5 (PyQt6), later
└── tests/                # small tests for VAD, config, chunking
```

## Roadmap

**Phase 0: Setup**
- Create project folder, `.venv`, `requirements.txt`, `.gitignore`, `.env.example`, `README.md`.
- Tell me exactly how to install BlackHole and set up `.env`.
- Done when: `python main.py --list-devices` shows my mic and BlackHole.

**Phase 1: Audio loopback**
- Mic → BlackHole passthrough with no AI, plus optional headphone monitor.
- Done when: I can select BlackHole as the mic in a call app and hear myself.

**Phase 2: Pipeline B (speech-to-speech)**
- Simplest AI version first: VAD → phrase → Voice Changer stream → BlackHole.
- Done when: I speak a sentence and the app hears it in my ElevenLabs voice.

**Phase 3: Pipeline A (streaming STT → TTS)**
- Realtime STT → TTS websocket (Flash v2.5) → BlackHole, with streaming playback.
- Done when: speech starts playing before I finish a long sentence.

**Phase 4: Switching, latency, and polish**
- `--mode stt-tts | sts` flag (and config default).
- Log latency per utterance (end of speech → first audio out).
- Handle: dropped websocket (auto-reconnect), API errors, empty/noisy input, clean shutdown with Ctrl+C.
- Tune chunk sizes, VAD sensitivity, and output format for speed.

**Phase 5: PyQt6 GUI**
- Window with: Start/Stop, mode toggle, input/output device dropdowns, monitor toggle, live latency readout, voice settings (stability/similarity).
- GUI wraps the same pipeline classes; no pipeline logic duplicated in the GUI.

## Coding rules

- Keep it simple and readable; comment the non-obvious parts (audio formats, sample rates, async flow).
- Type hints on function signatures.
- One sample rate/format end to end where possible; convert in one place only (`audio_io.py`).
- No global mutable state; pass config in.
- Handle errors with clear messages (missing key, wrong device, no BlackHole, API quota hit).
- ElevenLabs credits burn fast: add a `--dry-run` or short-test option, and warn me before long sessions.

## Ethics / terms

Only use voices I own or have permission to use, in line with ElevenLabs' terms. Don't build features meant to impersonate real people without consent.

## When a phase is done

1. Say what was built and how to run/test it (exact commands).
2. List anything I need to do manually.
3. Ask whether to move to the next phase, or continue if the roadmap makes it obvious.
