# Real-Time AI Voice Changer

Speaks your live mic audio back in your custom ElevenLabs voice and routes it to a virtual
microphone (BlackHole 2ch), so Discord, Zoom, OBS and games hear the AI voice as your mic.

Status: **Phase 5 (GUI)**. All modes work from the CLI and the GUI: `stt-tts` (speech-to-text -> text-to-speech, the default), `sts` (speech-to-speech) and `passthrough` (no AI).

## Requirements

- macOS, Python 3.11+
- [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole) virtual audio driver
- An ElevenLabs account with a voice you own or have permission to use

## Setup

```bash
# 1. Install BlackHole (one-time; may need a logout/reboot before it shows up)
brew install blackhole-2ch

# 2. Create the virtualenv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure your key and voice
cp .env.example .env
# edit .env: ELEVENLABS_API_KEY=...  ELEVENLABS_VOICE_ID=...
```

Find your voice ID in ElevenLabs under **Voices**: open your voice, then use **Copy voice ID**.
Create an API key under **Developers > API Keys**. `.env` is gitignored, so never commit it.

## GUI

```bash
python -m gui                       # window with mode, devices, monitor, voice settings, credit cap, live latency
python -m gui --timing balanced     # options without a control in the window are taken from the command line
```

Pick a mode and devices, press **Start**, then choose **BlackHole 2ch** as the mic in your call app.
Settings are applied when you press Start and remembered for next time. Speed only applies to
speech-to-text -> text-to-speech; dry run only to speech-to-speech.

**Voices:** paste a voice ID (ElevenLabs > Voices > your voice > Copy voice ID) into the Voice box and
press Add. It's checked with ElevenLabs (free) and saved under its name. Pick any saved voice from
the dropdown at any time. While running, the switch is live: your next phrase uses the new voice.
Your `.env` voice is always in the list. Only use voices you own or have permission to use.

## CLI

```bash
python main.py --list-devices                          # show microphones and outputs (BlackHole should appear)
python main.py --mode passthrough                      # default mic -> BlackHole, 48 kHz, no AI, no credits
python main.py --mode passthrough --input "BroBro"     # pick a mic by name (or index)
python main.py --mode passthrough --monitor "<headphones>"  # also hear yourself (headphones only!)
python main.py                                         # default: stt-tts (or VOICE_CHANGER_MODE in .env)
python main.py --mode sts --dry-run                    # VAD test: each phrase echoed back, no credits
python main.py --mode sts                              # your ElevenLabs voice (stops after 60 s of speech sent)
python main.py --mode sts --max-speech 300             # raise the credit cap (0 = no cap)
python main.py --mode sts --output-format pcm_24000    # lower output quality (default pcm_48000)
python main.py --mode sts --denoise                    # remove background noise (adds latency)
python main.py --mode sts --no-word-check              # also convert hmm/uh/sighs (skips the speech-to-text check)
python main.py --mode stt-tts                          # transcribe -> your voice reads each phrase after you pause (cleanest, no mumbles)
python main.py --mode stt-tts --timing balanced        # start mid-sentence instead (Flash model, ~4 s behind you, choppier)
python main.py --mode stt-tts --tts-model eleven_v4    # try another voice model (eleven_v4_turbo default, eleven_v4, eleven_v3, eleven_flash_v2_5)
python main.py --mode stt-tts --delivery "low, steady voice, laid-back"   # change how the voice sounds ("" = off)
python main.py --mode stt-tts --speed 0.95             # slightly slower voice (0.7-1.2)
python main.py --stability 0.4 --similarity 0.95       # voice settings 0-1, both AI modes (sts default: the voice's own)
python main.py --pause 0.35                            # silence that ends a phrase (0.3-1.5 s, default 0.5): faster, may split sentences
python main.py --latency-log latency.csv               # append each phrase's latency breakdown to a CSV
python main.py --help                                  # all options
```

In `sts` mode, speak a sentence and pause: after ~0.5 s of silence the phrase is sent to
ElevenLabs and comes back in your voice. The terminal shows each phrase, the latency, and the
running total of speech sent.

Non-word sounds are filtered out in two steps: very short or quiet sounds (clicks, breaths) are
dropped locally for free, and every remaining phrase is transcribed with ElevenLabs speech-to-text
at the same time as it's converted. If it has no real words ("hmm", "uh", a sigh), it's thrown away
instead of played. The terminal shows `ignored sound: ...` for each one. The word check needs the
Speech to Text permission on your API key.

Then in Discord/Zoom/etc., choose **BlackHole 2ch** as the microphone. Ctrl+C stops it.

### Which AI mode?

| | `--mode sts` (speech-to-speech) | `--mode stt-tts` (speech-to-text -> text-to-speech) |
|---|---|---|
| Sounds like | your delivery, emotion and timing in your AI voice | clean, newly generated speech from the words you said |
| Mumbles / fillers | can come through | removed (only the words are spoken) |
| When it plays | after each pause (~1.2 s after you stop) | after each pause (~1 s after you stop); `--timing balanced` starts mid-sentence |
| Cost | voice-changer credits per second of speech | cheap realtime transcription + TTS credits per character |

### Latency

Each phrase prints a line like `latency: 1240 ms (end of speech -> first audio out: pause 500 + voice 740)`:

- **pause**: the silence it waits for to know you've finished (`--pause` / the GUI slider)
- **queue**: waiting behind an earlier phrase (hidden when ~0)
- **voice**: ElevenLabs generating the first audio
- **word check** (sts only): extra wait for the "is this a real word?" check

The total counts from when you actually stopped talking, so it includes the pause.

A temporary ElevenLabs error (server error, rate limit, network blip) retries the phrase once. A
phrase that already started playing isn't retried, so you never hear it twice.

## Microphone permission

If capture fails or the device list is empty, go to **System Settings > Privacy & Security >
Microphone**, enable your terminal app (Terminal, iTerm, or VS Code), then fully quit and reopen it.

## Tests

```bash
pytest
```

## Notes

- Use headphones when monitoring, or your speakers will feed back into the mic.
- ElevenLabs credits go fast. Use `--dry-run` while testing. `sts` mode stops after 60 s of speech per run by default.
- The API key needs the Speech to Speech permission; Voices: Read and User: Read enable the free startup check.
- `webrtcvad-wheels` is used instead of `webrtcvad`, because the original fails to import on Python 3.12.
