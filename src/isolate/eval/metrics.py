"""Separation quality metrics: SDR (BSSEval v4) and SI-SDR (SPEC §8)."""

import math

import numpy as np

from isolate.core.audio import Audio, Float32Array
from isolate.eval.deps import require

# A reference quieter than this (-120 dBFS RMS) counts as silent, and that
# track is left out of the stem's median: SDR against silence is meaningless.
SILENCE_RMS = 1e-6


def is_silent(audio: Audio) -> bool:
    return float(np.sqrt(np.mean(audio.samples.astype(np.float64) ** 2))) < SILENCE_RMS


def si_sdr(reference: Float32Array, estimate: Float32Array) -> float:
    """Scale-invariant SDR in dB over the whole clip (all channels together)."""
    ref = reference.astype(np.float64).ravel()
    est = estimate.astype(np.float64).ravel()
    alpha = float(np.dot(est, ref) / np.dot(ref, ref))
    target = alpha * ref
    noise = est - target
    noise_energy = float(np.dot(noise, noise))
    if noise_energy == 0:
        return math.inf
    return 10 * math.log10(float(np.dot(target, target)) / noise_energy)


def sdr(references: dict[str, Audio], estimates: dict[str, Audio]) -> dict[str, float]:
    """BSSEval v4 SDR per stem: 1 s windows, median over windows (museval)."""
    museval = require("museval")
    stems = list(references)
    rate = references[stems[0]].sample_rate
    refs = np.stack([references[stem].samples.T for stem in stems])
    ests = np.stack([estimates[stem].samples.T for stem in stems])
    scores, _isr, _sir, _sar = museval.evaluate(refs, ests, win=rate, hop=rate)
    return {
        stem: float(np.nanmedian(scores[index])) for index, stem in enumerate(stems)
    }
