import numpy as np
import pytest

from isolate.core.audio import Audio
from isolate.core.engine import PRESETS, separate
from isolate.models.demucs import DemucsBackend, load_demucs
from isolate.models.loader import load_backend
from tests.contract import check_backend
from tests.tiny_demucs import save_tiny_htdemucs

RATE = 44100


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> DemucsBackend:
    path = save_tiny_htdemucs(tmp_path_factory.mktemp("tiny") / "tiny.safetensors")
    return load_demucs("tiny", [path], "cpu")


def test_tiny_demucs_meets_the_contract(tiny):
    assert tiny.normalize_input
    check_backend(tiny)


def test_tiny_demucs_through_the_engine(tiny):
    rng = np.random.default_rng(0)
    mix = (0.1 * rng.standard_normal((2, int(2.5 * RATE)))).astype(np.float32)

    stems = separate(Audio(mix, RATE), tiny, options=PRESETS["fast"])

    assert list(stems) == list(tiny.stems)
    total = sum(audio.samples for audio in stems.values())
    np.testing.assert_allclose(total, mix, atol=1e-5)


@pytest.fixture(scope="module")
def htdemucs_6s() -> DemucsBackend:
    backend = load_backend("htdemucs_6s", "cpu")
    assert isinstance(backend, DemucsBackend)
    return backend


@pytest.mark.weights
def test_htdemucs_6s_meets_the_contract(htdemucs_6s):
    assert htdemucs_6s.stems == ("drums", "bass", "other", "vocals", "guitar", "piano")
    assert htdemucs_6s.sample_rate == RATE
    check_backend(htdemucs_6s)


@pytest.mark.weights
def test_htdemucs_6s_separates_fixture(htdemucs_6s):
    # A synthetic 10 s "song": a bass line, a hi-hat-like noise burst and a lead.
    t = np.arange(10 * RATE) / RATE
    bass = 0.3 * np.sin(2 * np.pi * 55 * t)
    lead = 0.2 * np.sin(2 * np.pi * 880 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)
    rng = np.random.default_rng(0)
    hats = 0.05 * rng.standard_normal(t.size) * ((t * 4) % 1 < 0.05)
    mix = np.stack([bass + lead + hats, bass + 0.8 * lead + hats]).astype(np.float32)

    stems = separate(Audio(mix, RATE), htdemucs_6s, options=PRESETS["fast"])

    assert set(stems) == set(htdemucs_6s.stems)
    for audio in stems.values():
        assert audio.samples.shape == mix.shape
        assert np.isfinite(audio.samples).all()
    total = sum(audio.samples for audio in stems.values())
    np.testing.assert_allclose(total, mix, atol=1e-4)
    # The bass line should land mostly in the bass stem.
    energy = {stem: float(np.mean(audio.samples**2)) for stem, audio in stems.items()}
    assert energy["bass"] == max(energy.values())
