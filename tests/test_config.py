from __future__ import annotations

import pytest

import config
from config import Config, ConfigError, load_config, require_credentials


def test_frame_constants_are_consistent():
    assert config.FRAME_SAMPLES == 320
    assert config.FRAME_BYTES == 640


def test_defaults_with_empty_env(env_file):
    cfg = load_config([], env_path=env_file())
    assert cfg == Config(
        api_key=None, voice_id=None, mode="stt-tts", input_device=None,
        output_device="BlackHole 2ch", monitor_device=None,
        vad_aggressiveness=3, dry_run=False, list_devices=False,
    )


def test_reads_values_from_env_file(env_file):
    path = env_file("ELEVENLABS_API_KEY=test-key\nELEVENLABS_VOICE_ID=test-voice\nVOICE_CHANGER_MODE=stt-tts\n")
    cfg = load_config([], env_path=path)
    assert (cfg.api_key, cfg.voice_id, cfg.mode) == ("test-key", "test-voice", "stt-tts")


def test_existing_env_var_wins_over_file(env_file, clean_env):
    clean_env.setenv("ELEVENLABS_API_KEY", "shell-key")
    path = env_file("ELEVENLABS_API_KEY=file-key\n")
    assert load_config([], env_path=path).api_key == "shell-key"


def test_cli_args(env_file):
    cfg = load_config(
        ["--list-devices", "--mode", "stt-tts", "--input", "3", "--output", "Speakers",
         "--monitor", "Headphones", "--vad", "0", "--dry-run"],
        env_path=env_file("VOICE_CHANGER_MODE=sts\n"),
    )
    assert cfg.list_devices and cfg.dry_run
    assert cfg.mode == "stt-tts"  # CLI beats env
    assert (cfg.input_device, cfg.output_device, cfg.monitor_device) == ("3", "Speakers", "Headphones")
    assert cfg.vad_aggressiveness == 0


def test_empty_values_become_none(env_file):
    cfg = load_config([], env_path=env_file("ELEVENLABS_API_KEY=\nELEVENLABS_VOICE_ID=\n"))
    assert cfg.api_key is None and cfg.voice_id is None


def test_invalid_mode_from_env(env_file):
    with pytest.raises(ConfigError, match="Invalid mode 'bogus'"):
        load_config([], env_path=env_file("VOICE_CHANGER_MODE=bogus\n"))


@pytest.mark.parametrize("argv", [["--mode", "bogus"], ["--vad", "4"], ["--nope"]])
def test_invalid_cli_args_exit(env_file, argv):
    with pytest.raises(SystemExit):
        load_config(argv, env_path=env_file())


@pytest.mark.parametrize(
    "key, voice, missing",
    [(None, None, ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"]),
     ("test-key", None, ["ELEVENLABS_VOICE_ID"]),
     (None, "test-voice", ["ELEVENLABS_API_KEY"])],
)
def test_require_credentials_missing(env_file, key, voice, missing):
    cfg = load_config([], env_path=env_file())
    cfg = Config(**{**cfg.__dict__, "api_key": key, "voice_id": voice})
    with pytest.raises(ConfigError) as info:
        require_credentials(cfg)
    for name in missing:
        assert name in str(info.value)
    assert "test-key" not in str(info.value)


def test_require_credentials_ok(env_file):
    path = env_file("ELEVENLABS_API_KEY=test-key\nELEVENLABS_VOICE_ID=test-voice\n")
    require_credentials(load_config([], env_path=path))



# --- Phase 2 (speech-to-speech) options ---

def test_sts_defaults(env_file):
    cfg = load_config([], env_path=env_file())
    assert cfg.output_format == config.DEFAULT_OUTPUT_FORMAT == "pcm_48000"
    assert cfg.output_rate == 48_000
    assert cfg.max_speech_s == config.DEFAULT_MAX_SPEECH_S == 60.0
    assert cfg.denoise is False


@pytest.mark.parametrize("fmt, rate", [
    ("pcm_16000", 16_000), ("pcm_22050", 22_050), ("pcm_24000", 24_000), ("pcm_44100", 44_100),
    ("pcm_48000", 48_000),
])
def test_output_format_choices_and_rate(env_file, fmt, rate):
    cfg = load_config(["--output-format", fmt], env_path=env_file())
    assert cfg.output_format == fmt
    assert cfg.output_rate == rate


def test_output_formats_are_all_raw_pcm():
    assert config.DEFAULT_OUTPUT_FORMAT in config.OUTPUT_FORMATS
    assert all(f.startswith("pcm_") for f in config.OUTPUT_FORMATS)


@pytest.mark.parametrize("fmt", ["mp3_44100_128", "pcm_8000", "ulaw_8000", "48000"])
def test_invalid_output_format_exits(env_file, fmt):
    with pytest.raises(SystemExit):
        load_config(["--output-format", fmt], env_path=env_file())


@pytest.mark.parametrize("arg, expected", [("0", 0.0), ("2.5", 2.5), ("300", 300.0)])
def test_max_speech(env_file, arg, expected):
    assert load_config(["--max-speech", arg], env_path=env_file()).max_speech_s == expected


@pytest.mark.parametrize("arg", ["-1", "-0.1"])
def test_negative_max_speech_is_config_error(env_file, arg):
    with pytest.raises(ConfigError, match="--max-speech must be >= 0"):
        load_config([f"--max-speech={arg}"], env_path=env_file())


def test_non_numeric_max_speech_exits(env_file):
    with pytest.raises(SystemExit):
        load_config(["--max-speech", "lots"], env_path=env_file())


def test_denoise_flag(env_file):
    assert load_config(["--denoise"], env_path=env_file()).denoise is True


def test_config_fields_have_phase2_defaults():
    cfg = Config(api_key=None, voice_id=None, mode="sts", input_device=None, output_device="BlackHole 2ch",
                 monitor_device=None, vad_aggressiveness=2, dry_run=False, list_devices=False)
    assert (cfg.output_format, cfg.max_speech_s, cfg.denoise) == ("pcm_48000", 60.0, False)


def test_stability_and_similarity_flags(env_file):
    cfg = load_config(["--stability", "0.3", "--similarity", "1"], env_path=env_file())
    assert (cfg.stability, cfg.similarity) == (0.3, 1.0)
    settings = config.voice_settings(cfg)
    assert (settings["stability"], settings["similarity_boost"]) == (0.3, 1.0)


def test_voice_settings_defaults_when_not_overridden(env_file):
    cfg = load_config(["--speed", "0.9"], env_path=env_file())
    assert (cfg.stability, cfg.similarity) == (None, None)
    assert config.voice_settings(cfg) == {**config.TTS_VOICE_SETTINGS, "speed": 0.9}


@pytest.mark.parametrize("flag, value", [("--stability", "-0.1"), ("--similarity", "1.5")])
def test_voice_setting_out_of_range(env_file, flag, value):
    with pytest.raises(ConfigError, match=f"{flag} must be between 0 and 1"):
        load_config([flag, value], env_path=env_file())
