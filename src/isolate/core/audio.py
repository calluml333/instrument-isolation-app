"""Reading and writing audio files (SPEC R1, R5)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import soundfile as sf

Float32Array = npt.NDArray[np.float32]
OutputFormat = Literal["wav", "flac"]
BitDepth = Literal["24", "32f"]

MAX_CHANNELS = 2

# FLAC has no floating-point mode, so it is always written as 24-bit PCM.
_SUBTYPES: dict[tuple[OutputFormat, BitDepth], str] = {
    ("wav", "32f"): "FLOAT",
    ("wav", "24"): "PCM_24",
    ("flac", "24"): "PCM_24",
}


class AudioLoadError(Exception):
    """An input file can't be used: missing, unsupported, corrupt or empty."""


@dataclass(frozen=True)
class Audio:
    """Float32 samples shaped `(channels, frames)`, plus their sample rate."""

    samples: Float32Array
    sample_rate: int

    @property
    def channels(self) -> int:
        return int(self.samples.shape[0])

    @property
    def frames(self) -> int:
        return int(self.samples.shape[1])


def load(path: str | Path) -> Audio:
    """Read WAV, AIFF, FLAC or MP3 (mono or stereo, any sample rate)."""
    path = Path(path)
    try:
        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except (sf.LibsndfileError, OSError) as exc:
        reason = "file not found" if not path.exists() else str(exc)
        raise AudioLoadError(f"Can't read {path}: {reason}") from exc
    frames, channels = data.shape
    if frames == 0:
        raise AudioLoadError(f"Can't read {path}: it contains no audio")
    if channels > MAX_CHANNELS:
        raise AudioLoadError(
            f"Can't read {path}: it has {channels} channels; "
            "only mono and stereo are supported"
        )
    return Audio(np.ascontiguousarray(data.T), int(sample_rate))


def default_bit_depth(fmt: OutputFormat) -> BitDepth:
    """32-bit float for WAV (it can't clip); FLAC only supports 24-bit PCM."""
    return "32f" if fmt == "wav" else "24"


def save(
    path: str | Path,
    audio: Audio,
    *,
    fmt: OutputFormat = "wav",
    bit_depth: BitDepth | None = None,
) -> None:
    """Write `audio` as WAV (32-bit float or 24-bit PCM) or FLAC (24-bit)."""
    bit_depth = bit_depth or default_bit_depth(fmt)
    subtype = _SUBTYPES.get((fmt, bit_depth))
    if subtype is None:
        raise ValueError(f"{fmt.upper()} can't be written with bit depth {bit_depth}")
    sf.write(
        Path(path),
        audio.samples.T,
        audio.sample_rate,
        subtype=subtype,
        format=fmt.upper(),
    )
