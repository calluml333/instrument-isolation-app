import dataclasses
import io

import pytest

from isolate.models import weights
from isolate.models.weights import (
    ChecksumError,
    DownloadError,
    cache_root,
    ensure_weights,
    is_downloaded,
    model_dir,
)


def test_cache_root_defaults_to_user_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("ISOLATE_CACHE_DIR")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cache_root() == tmp_path / ".cache" / "isolate"


def test_first_use_downloads_and_verifies(fake_model):
    spec = fake_model()
    seen: list[tuple[int, int]] = []

    directory = ensure_weights(
        spec, progress=lambda done, total: seen.append((done, total))
    )

    assert directory == model_dir(spec)
    assert (directory / "weights.bin").read_bytes() == b"fake weights"
    assert is_downloaded(spec)
    assert seen[-1] == (12, 12)
    assert not list(directory.glob("*.part"))


def test_cached_weights_are_not_downloaded_again(fake_model, monkeypatch):
    spec = fake_model()
    ensure_weights(spec)
    calls: list[str] = []
    monkeypatch.setattr(weights, "_open", lambda url: calls.append(url))

    ensure_weights(spec)

    assert calls == []


def test_corrupt_cached_file_is_replaced(fake_model):
    spec = fake_model()
    path = ensure_weights(spec) / "weights.bin"
    path.write_bytes(b"tampered!!!!")

    ensure_weights(spec)

    assert path.read_bytes() == b"fake weights"


def test_checksum_mismatch_deletes_the_download(fake_model):
    spec = fake_model()
    bad = dataclasses.replace(spec.files[0], sha256="0" * 64)
    spec = dataclasses.replace(spec, files=(bad,))

    with pytest.raises(ChecksumError, match=r"weights\.bin .* failed its checksum"):
        ensure_weights(spec)

    assert not list(model_dir(spec).iterdir())
    assert not is_downloaded(spec)


def test_interrupted_download_leaves_no_file(fake_model, monkeypatch):
    spec = fake_model()

    class Broken(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            raise ConnectionResetError("connection dropped")

    monkeypatch.setattr(weights, "_open", lambda url: Broken())

    with pytest.raises(ConnectionResetError):
        ensure_weights(spec)

    assert not list(model_dir(spec).iterdir())


def test_unreachable_url_is_retried_then_reported(fake_model, tmp_path):
    spec = fake_model()
    missing = dataclasses.replace(spec.files[0], url=(tmp_path / "gone.bin").as_uri())

    with pytest.raises(DownloadError, match=r"Couldn't download .*gone\.bin"):
        ensure_weights(dataclasses.replace(spec, files=(missing,)))


def test_wrong_size_is_not_downloaded(fake_model):
    spec = fake_model()
    path = ensure_weights(spec) / "weights.bin"
    path.write_bytes(b"short")

    assert not is_downloaded(spec)
