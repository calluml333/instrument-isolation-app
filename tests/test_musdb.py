import numpy as np
import pytest

from isolate.eval.deps import EvalDependencyError, require
from isolate.eval.musdb import SOURCES, UnknownTrackError, dataset_dir, iter_tracks
from tests.fake_musdb import RATE, write_fake_musdb


@pytest.fixture
def fake_root(tmp_path):
    return write_fake_musdb(tmp_path / "musdb", tracks=3)


def test_reads_tracks_with_all_references(fake_root):
    tracks = list(iter_tracks(fake_root, is_wav=True))

    assert [t.name for t in tracks] == [f"Artist {i} - Song {i}" for i in range(3)]
    for track in tracks:
        assert track.mixture.sample_rate == RATE
        assert track.mixture.samples.shape == (2, 2 * RATE)
        assert track.mixture.samples.dtype == np.float32
        assert set(track.references) == set(SOURCES)
        total = sum(ref.samples for ref in track.references.values())
        # musdb decodes WAVs to 16-bit through FFmpeg, hence the tolerance.
        np.testing.assert_allclose(total, track.mixture.samples, atol=2e-4)


def test_limit_and_names(fake_root):
    assert len(list(iter_tracks(fake_root, is_wav=True, limit=2))) == 2
    named = list(iter_tracks(fake_root, is_wav=True, names=["Artist 2 - Song 2"]))
    assert [t.name for t in named] == ["Artist 2 - Song 2"]


def test_unknown_track_name(fake_root):
    with pytest.raises(UnknownTrackError, match="No MUSDB test track"):
        list(iter_tracks(fake_root, is_wav=True, names=["Nobody - Nothing"]))


def test_dataset_lives_in_the_isolate_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ISOLATE_CACHE_DIR", str(tmp_path))
    assert dataset_dir() == tmp_path / "datasets" / "musdb18-7"


def test_missing_dependency_explains_the_fix():
    with pytest.raises(EvalDependencyError, match="poetry install --with eval"):
        require("isolate_no_such_module")


def test_missing_ffmpeg_explains_the_fix(monkeypatch):
    import importlib

    def no_ffmpeg(name):
        raise RuntimeError("ffmpeg or ffprobe could not be found!")

    monkeypatch.setattr(importlib, "import_module", no_ffmpeg)
    with pytest.raises(EvalDependencyError, match="FFmpeg"):
        require("musdb")
