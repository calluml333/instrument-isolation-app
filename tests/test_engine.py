from dataclasses import dataclass

import numpy as np
import pytest

from isolate.core.audio import Audio, Float32Array
from isolate.core.engine import PRESETS, Cancelled, CancelToken, Options, separate
from tests.fakes import STEMS, IdentityBackend, LossyBackend

RATE = 44100
NO_RESIDUAL = Options(residual=False)


def noise(channels: int, frames: int, seed: int = 0) -> Float32Array:
    rng = np.random.default_rng(seed)
    return (0.25 * rng.standard_normal((channels, frames))).astype(np.float32)


def tone(channels: int, seconds: float, rate: int) -> Float32Array:
    t = np.arange(int(seconds * rate)) / rate
    mono = 0.5 * np.sin(2 * np.pi * 1000 * t)
    return np.stack([mono] * channels).astype(np.float32)


@dataclass
class PositionalBackend(IdentityBackend):
    """Output depends on where the chunk boundaries fall, so shifts matter."""

    name: str = "positional"

    def separate(self, chunk: Float32Array) -> dict[str, Float32Array]:
        out = super().separate(chunk)
        ramp = np.linspace(0.5, 1.0, chunk.shape[1], dtype=np.float32)
        out["other"] = out["other"] * ramp
        return out


@pytest.mark.parametrize("segments", [None, 1.0, 3.5], ids=["1-sample", "1", "3.5"])
def test_overlap_add_identity_round_trip(segments):
    backend = IdentityBackend()
    segment = int(backend.segment_seconds * RATE)
    frames = 1 if segments is None else int(segments * segment)
    mix = noise(2, frames)

    stems = separate(Audio(mix, RATE), backend, options=NO_RESIDUAL)

    assert set(stems) == set(STEMS)
    np.testing.assert_allclose(stems["other"].samples, mix, atol=1e-6)
    for stem in set(STEMS) - {"other"}:
        assert not stems[stem].samples.any()


def test_chunks_never_exceed_one_segment():
    backend = IdentityBackend()
    segment = int(backend.segment_seconds * RATE)

    separate(Audio(noise(2, 5 * segment + 7), RATE), backend, options=NO_RESIDUAL)

    assert max(backend.calls) == segment
    # 25% overlap: a stride of 0.75 segments, stopping once the end is covered.
    assert len(backend.calls) == 7


@pytest.mark.parametrize("rate", [22050, 44100, 48000, 96000])
@pytest.mark.parametrize("channels", [1, 2])
def test_resampling_and_channels_match_input(rate, channels):
    mix = tone(channels, 0.5, rate)

    stems = separate(Audio(mix, rate), IdentityBackend(), options=NO_RESIDUAL)

    for audio in stems.values():
        assert audio.sample_rate == rate
        assert audio.samples.shape == mix.shape
        assert audio.samples.dtype == np.float32
    error = stems["other"].samples - mix
    if rate == RATE:
        assert np.abs(error).max() < 1e-6
    else:
        rms = np.sqrt(np.mean(error**2))
        assert 20 * np.log10(rms) < -60


def test_mono_input_gives_mono_stems_that_sum_to_it():
    mix = noise(1, 10_000)

    stems = separate(Audio(mix, RATE), LossyBackend(), options=Options())

    total = sum(audio.samples for audio in stems.values())
    assert all(audio.channels == 1 for audio in stems.values())
    np.testing.assert_allclose(total, mix, atol=1e-5)


def test_shifts_identity_round_trip():
    mix = noise(2, 30_000)

    stems = separate(
        Audio(mix, RATE), IdentityBackend(), options=Options(shifts=3, residual=False)
    )

    np.testing.assert_allclose(stems["other"].samples, mix, atol=1e-6)


def test_shifts_deterministic():
    mix = Audio(noise(2, 30_000), RATE)

    def run(seed: int) -> Float32Array:
        options = Options(shifts=2, residual=False, seed=seed)
        return separate(mix, PositionalBackend(), options=options)["other"].samples

    np.testing.assert_array_equal(run(0), run(0))
    assert not np.array_equal(run(0), run(1))


def test_shifts_average_several_passes():
    backend = IdentityBackend()
    frames = 30_000

    separate(Audio(noise(2, frames), RATE), backend, options=Options(shifts=2))

    single = IdentityBackend()
    separate(Audio(noise(2, frames + int(0.5 * RATE)), RATE), single, options=Options())
    assert len(backend.calls) == 2 * len(single.calls)


def test_presets():
    assert PRESETS["fast"] == Options(overlap=0.25, shifts=0, residual=True)
    assert PRESETS["best"] == Options(overlap=0.25, shifts=2, residual=True)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"overlap": 1.0}, "overlap"),
        ({"overlap": -0.1}, "overlap"),
        ({"shifts": -1}, "shifts"),
    ],
)
def test_options_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Options(**kwargs)


def test_progress_rises_to_exactly_one():
    backend = IdentityBackend()
    seen: list[float] = []

    separate(
        Audio(noise(2, 20_000), RATE),
        backend,
        options=Options(shifts=2),
        progress=seen.append,
    )

    assert len(seen) == len(backend.calls)
    assert seen == sorted(seen)
    assert seen[-1] == 1.0


def test_cancel_stops_between_chunks():
    backend = IdentityBackend()
    token = CancelToken()

    def cancel_after_two(fraction: float) -> None:
        if len(backend.calls) == 2:
            token.cancel()

    with pytest.raises(Cancelled):
        separate(
            Audio(noise(2, 50_000), RATE),
            backend,
            progress=cancel_after_two,
            cancel=token,
        )
    assert len(backend.calls) == 2


def test_residual_projection_sums_to_input():
    mix = noise(2, 20_000)

    stems = separate(Audio(mix, RATE), LossyBackend(), options=Options())

    total = sum(audio.samples for audio in stems.values())
    np.testing.assert_allclose(total, mix, atol=1e-5)


def test_residual_off():
    mix = noise(2, 20_000)

    stems = separate(Audio(mix, RATE), LossyBackend(), options=NO_RESIDUAL)

    total = sum(audio.samples for audio in stems.values())
    np.testing.assert_allclose(total, 0.9 * mix, atol=1e-5)


def test_residual_goes_to_other_even_when_other_is_not_returned():
    mix = Audio(noise(2, 20_000), RATE)

    everything = separate(mix, LossyBackend(), options=Options())
    some = separate(mix, LossyBackend(), options=Options(), stems=["drums", "bass"])

    assert list(some) == ["drums", "bass"]
    np.testing.assert_array_equal(some["drums"].samples, everything["drums"].samples)


def test_unknown_stems_are_rejected():
    with pytest.raises(ValueError, match=r"no stem.*kazoo"):
        separate(Audio(noise(2, 100), RATE), IdentityBackend(), stems=["kazoo"])


def test_normalised_backends_see_standardised_input():
    @dataclass
    class Recording(IdentityBackend):
        normalize_input: bool = True
        seen: list[Float32Array] | None = None

        def separate(self, chunk: Float32Array) -> dict[str, Float32Array]:
            self.seen = [*(self.seen or []), chunk]
            return super().separate(chunk)

    backend = Recording(segment_seconds=1.0)
    mix = 3.0 * noise(2, 20_000) + 0.5

    stems = separate(Audio(mix, RATE), backend, options=NO_RESIDUAL)

    assert backend.seen
    seen = backend.seen[0].mean(axis=0)
    assert abs(seen.mean()) < 1e-3
    assert abs(seen.std() - 1) < 1e-2
    np.testing.assert_allclose(stems["other"].samples, mix, atol=1e-5)


def test_normalising_silence_does_not_divide_by_zero():
    backend = IdentityBackend(normalize_input=True)
    silence = np.zeros((2, 5_000), dtype=np.float32)

    stems = separate(Audio(silence, RATE), backend)

    assert all(np.isfinite(audio.samples).all() for audio in stems.values())


@pytest.mark.parametrize("frames", [3, 5, 8])
def test_fit_length_trims_or_pads(frames):
    from isolate.core.engine import _fit_length

    samples = np.ones((2, 5), dtype=np.float32)

    fitted = _fit_length(samples, frames)

    assert fitted.shape == (2, frames)
    assert fitted[:, :5].all() if frames >= 5 else fitted.all()
    assert not fitted[:, 5:].any()
