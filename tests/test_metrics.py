import math

import numpy as np
import pytest

from isolate.core.audio import Audio
from isolate.eval.metrics import is_silent, sdr, si_sdr

RATE = 44100


def noise(frames=3 * RATE, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((2, frames)).astype(np.float32)


def with_snr(signal, snr_db, seed=1):
    extra = noise(signal.shape[1], seed)
    extra *= np.sqrt(np.mean(signal**2) / np.mean(extra**2) / 10 ** (snr_db / 10))
    return signal + extra


def test_si_sdr_of_a_signal_with_itself():
    ref = noise()
    assert si_sdr(ref, ref) >= 100


def test_si_sdr_is_scale_invariant():
    ref = noise()
    assert si_sdr(ref, 0.5 * ref) >= 100
    est = with_snr(ref, 10)
    assert si_sdr(ref, 0.5 * est) == pytest.approx(si_sdr(ref, est), abs=1e-6)


@pytest.mark.parametrize("snr", [0, 10, 20])
def test_si_sdr_matches_a_known_snr(snr):
    ref = noise()
    assert si_sdr(ref, with_snr(ref, snr)) == pytest.approx(snr, abs=0.1)


def test_si_sdr_of_an_exact_estimate_is_infinite():
    ref = np.ones((2, 10), dtype=np.float32)
    assert si_sdr(ref, ref) == math.inf


@pytest.mark.parametrize("snr", [10, 20])
def test_sdr_matches_a_known_snr(snr):
    refs = {"a": Audio(noise(seed=0), RATE), "b": Audio(noise(seed=5), RATE)}
    ests = {
        name: Audio(with_snr(a.samples, snr, seed=9), RATE) for name, a in refs.items()
    }

    scores = sdr(refs, ests)

    assert set(scores) == {"a", "b"}
    for value in scores.values():
        assert value == pytest.approx(snr, abs=0.3)


def test_is_silent():
    assert is_silent(Audio(np.zeros((2, 100), dtype=np.float32), RATE))
    assert not is_silent(Audio(noise(100), RATE))


def test_sdr_of_a_stem_does_not_depend_on_the_others():
    refs = {"a": Audio(noise(seed=0), RATE), "b": Audio(noise(seed=5), RATE)}
    ests = {
        name: Audio(with_snr(ref.samples, 12, seed=9), RATE)
        for name, ref in refs.items()
    }

    together = sdr(refs, ests)
    alone = sdr({"a": refs["a"]}, {"a": ests["a"]})

    assert alone["a"] == pytest.approx(together["a"], abs=1e-6)


def test_sdr_handles_silent_references_and_estimates():
    silent = Audio(np.zeros((2, 3 * RATE), dtype=np.float32), RATE)
    refs = {
        "a": Audio(noise(seed=0), RATE),
        "b": silent,
        "c": Audio(noise(seed=2), RATE),
    }
    ests = {
        "a": Audio(with_snr(refs["a"].samples, 10), RATE),
        "b": refs["a"],
        "c": silent,
    }

    scores = sdr(refs, ests)

    assert scores["a"] == pytest.approx(10, abs=0.3)
    assert scores["b"] is None
    assert scores["c"] == -math.inf
