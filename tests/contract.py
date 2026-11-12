"""The behaviour every `Backend` must have (SPEC M1, P1-R1)."""

import numpy as np

from isolate.models.base import Backend


def check_backend(backend: Backend, *, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    segment = int(backend.segment_seconds * backend.sample_rate)

    assert backend.name
    assert backend.stems
    assert len(set(backend.stems)) == len(backend.stems)
    assert "other" in backend.stems
    assert backend.sample_rate > 0
    assert segment > 0

    # A full segment, a short tail and a single sample must all work.
    for frames in (segment, segment // 3, 1):
        chunk = (0.1 * rng.standard_normal((2, frames))).astype(np.float32)
        out = backend.separate(chunk)

        assert set(out) == set(backend.stems)
        for stem, samples in out.items():
            assert samples.shape == (2, frames), stem
            assert samples.dtype == np.float32, stem
            assert np.isfinite(samples).all(), stem
