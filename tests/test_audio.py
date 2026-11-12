import numpy as np
import pytest
import soundfile as sf

from isolate.core.audio import Audio, AudioLoadError, load, save

RATE = 44100


def tone(channels: int, frames: int = RATE // 10, rate: int = RATE) -> np.ndarray:
    t = np.arange(frames) / rate
    mono = 0.5 * np.sin(2 * np.pi * 440 * t)
    return np.stack([mono * (1 - 0.1 * c) for c in range(channels)]).astype(np.float32)


@pytest.mark.parametrize(
    ("suffix", "fmt", "subtype", "tolerance"),
    [
        ("wav", "WAV", "FLOAT", 0),
        ("aiff", "AIFF", "PCM_24", 1e-6),
        ("flac", "FLAC", "PCM_24", 1e-6),
        # MP3 is lossy and adds encoder delay, so only the shape is checked.
        ("mp3", "MP3", "MPEG_LAYER_III", None),
    ],
)
@pytest.mark.parametrize("channels", [1, 2])
def test_load_formats(tmp_path, suffix, fmt, subtype, tolerance, channels):
    samples = tone(channels)
    path = tmp_path / f"in.{suffix}"
    sf.write(path, samples.T, RATE, format=fmt, subtype=subtype)

    audio = load(path)

    assert audio.sample_rate == RATE
    assert audio.channels == channels
    assert audio.samples.dtype == np.float32
    assert audio.samples.flags.c_contiguous
    if tolerance is not None:
        assert audio.frames == samples.shape[1]
        np.testing.assert_allclose(audio.samples, samples, atol=tolerance)


def test_load_keeps_sample_rate(tmp_path):
    path = tmp_path / "in.wav"
    sf.write(path, tone(2, rate=48000).T, 48000, subtype="FLOAT")
    assert load(path).sample_rate == 48000


@pytest.mark.parametrize(
    ("contents", "reason"),
    [
        (b"", "Error opening"),
        (b"not audio at all", "Error opening"),
        (b"RIFF\x00\x00\x00\x00WAVEfmt garbage", "Error opening"),
    ],
    ids=["empty-file", "text-file", "corrupt-wav"],
)
def test_load_rejects_unreadable_files(tmp_path, contents, reason):
    path = tmp_path / "bad.wav"
    path.write_bytes(contents)
    with pytest.raises(AudioLoadError, match=rf"bad\.wav: .*{reason}"):
        load(path)


def test_load_rejects_missing_file(tmp_path):
    with pytest.raises(AudioLoadError, match="file not found"):
        load(tmp_path / "nope.wav")


def test_load_rejects_zero_frames(tmp_path):
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros((0, 2), dtype=np.float32), RATE)
    with pytest.raises(AudioLoadError, match="no audio"):
        load(path)


def test_load_rejects_more_than_two_channels(tmp_path):
    path = tmp_path / "surround.wav"
    sf.write(path, tone(6).T, RATE)
    with pytest.raises(AudioLoadError, match="6 channels"):
        load(path)


@pytest.mark.parametrize(
    ("fmt", "bit_depth", "subtype"),
    [("wav", None, "FLOAT"), ("wav", "24", "PCM_24"), ("flac", None, "PCM_24")],
)
def test_save_round_trip(tmp_path, fmt, bit_depth, subtype):
    audio = Audio(tone(2), RATE)
    path = tmp_path / f"out.{fmt}"

    save(path, audio, fmt=fmt, bit_depth=bit_depth)

    assert sf.info(path).subtype == subtype
    np.testing.assert_allclose(
        load(path).samples, audio.samples, atol=0 if subtype == "FLOAT" else 1e-6
    )


def test_save_rejects_float_flac(tmp_path):
    with pytest.raises(ValueError, match="FLAC can't be written with bit depth 32f"):
        save(tmp_path / "out.flac", Audio(tone(2), RATE), fmt="flac", bit_depth="32f")
