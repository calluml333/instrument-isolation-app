"""The separation engine (SPEC R1-R4, R7, R10).

`separate()` turns a whole song into stems with any `Backend`: it resamples,
standardises if the backend asks for it, cuts the song into overlapping
segments, cross-fades the results back together, optionally averages random
time shifts, and finally makes the stems sum exactly to the input.
"""

import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import soxr

from isolate.core.audio import Audio, Float32Array
from isolate.models.base import Backend

Progress = Callable[[float], None]

MAX_SHIFT_SECONDS = 0.5
RESIDUAL_STEM = "other"


@dataclass(frozen=True)
class Options:
    """How to run a separation. See `PRESETS` for the named combinations."""

    overlap: float = 0.25
    shifts: int = 0
    residual: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.overlap < 1:
            raise ValueError(f"overlap must be in [0, 1), got {self.overlap}")
        if self.shifts < 0:
            raise ValueError(f"shifts must be >= 0, got {self.shifts}")


# Provisional, for htdemucs_6s; revisited with measurements in Phase 3.
PRESETS: dict[str, Options] = {
    "fast": Options(overlap=0.25, shifts=0),
    "best": Options(overlap=0.25, shifts=2),
}


class Cancelled(Exception):
    """Raised by `separate()` when its `CancelToken` is cancelled."""


class CancelToken:
    """Thread-safe flag a caller sets to stop a separation between chunks."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


def separate(
    audio: Audio,
    backend: Backend,
    *,
    options: Options = PRESETS["best"],
    stems: Sequence[str] | None = None,
    progress: Progress | None = None,
    cancel: CancelToken | None = None,
) -> dict[str, Audio]:
    """Separate `audio` into stems matching its length, rate and channels."""
    wanted = tuple(stems) if stems is not None else backend.stems
    unknown = sorted(set(wanted) - set(backend.stems))
    if unknown:
        raise ValueError(
            f"{backend.name} has no stem(s) {', '.join(unknown)}; "
            f"it has {', '.join(backend.stems)}"
        )

    mix = _resample(_to_stereo(audio.samples), audio.sample_rate, backend.sample_rate)
    offset, scale = _standardise(mix) if backend.normalize_input else (0.0, 1.0)
    mix = (mix - offset) / scale

    runner = _ChunkRunner(backend, options, mix.shape[1], progress, cancel)
    separated = runner.run(mix)

    result: dict[str, Float32Array] = {}
    for stem, samples in separated.items():
        samples = samples * scale + offset
        samples = _resample(samples, backend.sample_rate, audio.sample_rate)
        samples = _fit_length(samples, audio.frames)
        if audio.channels == 1:
            samples = samples.mean(axis=0, keepdims=True)
        result[stem] = samples.astype(np.float32)

    if options.residual:
        total = np.sum(list(result.values()), axis=0)
        result[RESIDUAL_STEM] += audio.samples - total

    return {stem: Audio(result[stem], audio.sample_rate) for stem in wanted}


class _ChunkRunner:
    """Overlap-adds one or more passes (one per shift) and reports progress."""

    def __init__(
        self,
        backend: Backend,
        options: Options,
        frames: int,
        progress: Progress | None,
        cancel: CancelToken | None,
    ) -> None:
        self.backend = backend
        self.options = options
        self.progress = progress
        self.cancel = cancel
        self.segment = max(1, int(backend.segment_seconds * backend.sample_rate))
        self.stride = max(1, int(self.segment * (1 - options.overlap)))
        self.window = _triangular_window(self.segment)
        self.max_shift = int(MAX_SHIFT_SECONDS * backend.sample_rate)

        if options.shifts == 0:
            self.total_chunks = self._chunk_count(frames)
        else:
            per_pass = self._chunk_count(frames + self.max_shift)
            self.total_chunks = options.shifts * per_pass
        self.done = 0

    def run(self, mix: Float32Array) -> dict[str, Float32Array]:
        if self.options.shifts == 0:
            return self._overlap_add(mix)

        frames = mix.shape[1]
        padded = np.pad(mix, ((0, 0), (self.max_shift, self.max_shift)))
        rng = np.random.default_rng(self.options.seed)
        total: dict[str, Float32Array] = {}
        for _ in range(self.options.shifts):
            shift = int(rng.integers(0, self.max_shift)) if self.max_shift else 0
            shifted = padded[:, shift : shift + frames + self.max_shift]
            start = self.max_shift - shift
            for stem, samples in self._overlap_add(shifted).items():
                part = samples[:, start : start + frames]
                total[stem] = total[stem] + part if stem in total else part.copy()
        return {
            stem: (samples / self.options.shifts).astype(np.float32, copy=False)
            for stem, samples in total.items()
        }

    def _chunk_count(self, frames: int) -> int:
        if frames <= self.segment:
            return 1
        return math.ceil((frames - self.segment) / self.stride) + 1

    def _overlap_add(self, mix: Float32Array) -> dict[str, Float32Array]:
        frames = mix.shape[1]
        out = {stem: np.zeros_like(mix) for stem in self.backend.stems}
        weights = np.zeros(frames, dtype=np.float32)

        for index in range(self._chunk_count(frames)):
            if self.cancel is not None and self.cancel.cancelled:
                raise Cancelled
            start = index * self.stride
            chunk = mix[:, start : start + self.segment]
            length = chunk.shape[1]
            window = self.window[:length]

            separated = self.backend.separate(np.ascontiguousarray(chunk))
            for stem in self.backend.stems:
                out[stem][:, start : start + length] += separated[stem] * window
            weights[start : start + length] += window

            self.done += 1
            if self.progress is not None:
                self.progress(self.done / self.total_chunks)

        return {
            stem: (samples / weights).astype(np.float32, copy=False)
            for stem, samples in out.items()
        }


def _triangular_window(length: int) -> Float32Array:
    """Peaks at 1 in the middle and stays above 0 at both ends."""
    half = length // 2
    ramp = np.concatenate([np.arange(1, half + 1), np.arange(length - half, 0, -1)])
    window: Float32Array = (ramp / float(ramp.max())).astype(np.float32)
    return window


def _to_stereo(samples: Float32Array) -> Float32Array:
    if samples.shape[0] == 1:
        return np.repeat(samples, 2, axis=0).astype(np.float32, copy=False)
    return samples


def _standardise(mix: Float32Array) -> tuple[float, float]:
    """Mean and std of the mono mix, over the whole track (as Demucs does)."""
    reference = mix.mean(axis=0)
    std = float(reference.std())
    return float(reference.mean()), std if std > 1e-8 else 1.0


def _resample(samples: Float32Array, from_rate: int, to_rate: int) -> Float32Array:
    if from_rate == to_rate:
        return samples
    resampled: Float32Array = soxr.resample(
        samples.T, from_rate, to_rate, quality="VHQ"
    )
    return np.ascontiguousarray(resampled.T).astype(np.float32, copy=False)


def _fit_length(samples: Float32Array, frames: int) -> Float32Array:
    """Trim or zero-pad so resampling round trips keep the exact length."""
    if samples.shape[1] >= frames:
        return samples[:, :frames]
    return np.pad(samples, ((0, 0), (0, frames - samples.shape[1])))
