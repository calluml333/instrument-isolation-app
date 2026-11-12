"""Downloading, verifying and caching model weights (SPEC R8)."""

import hashlib
import os
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, cast

from isolate import __version__
from isolate.models.registry import ModelSpec, WeightFile

CACHE_ENV = "ISOLATE_CACHE_DIR"
_BLOCK = 1 << 20

# Called with (bytes so far, total bytes) while downloading.
DownloadProgress = Callable[[int, int], None]


class ChecksumError(Exception):
    """A downloaded file's SHA-256 didn't match the registry."""


class DownloadError(Exception):
    """A weights file couldn't be fetched."""


def cache_root() -> Path:
    return Path(os.environ.get(CACHE_ENV) or Path.home() / ".cache" / "isolate")


def model_dir(spec: ModelSpec) -> Path:
    return cache_root() / "models" / spec.name


def is_downloaded(spec: ModelSpec) -> bool:
    """Cheap check (sizes only) for listings; `ensure_weights` verifies hashes."""
    directory = model_dir(spec)
    return all(
        (directory / file.filename).is_file()
        and (directory / file.filename).stat().st_size == file.size
        for file in spec.files
    )


def ensure_weights(spec: ModelSpec, progress: DownloadProgress | None = None) -> Path:
    """Return the directory holding `spec`'s verified weights, fetching any missing."""
    directory = model_dir(spec)
    for file in spec.files:
        path = directory / file.filename
        if path.is_file() and _sha256(path) == file.sha256:
            continue
        path.unlink(missing_ok=True)
        _download(file, path, progress)
    return directory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def _open(url: str) -> BinaryIO:
    request = urllib.request.Request(
        url, headers={"User-Agent": f"isolate/{__version__}"}
    )
    for attempt in range(2):
        try:
            return cast(BinaryIO, urllib.request.urlopen(request, timeout=60))
        except urllib.error.URLError as exc:
            if attempt == 1:
                raise DownloadError(f"Couldn't download {url}: {exc.reason}") from exc
    raise AssertionError("unreachable")


def _download(file: WeightFile, path: Path, progress: DownloadProgress | None) -> None:
    """Stream to a temp file beside `path`, verify, then rename into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".part")
    temp = Path(temp_name)
    try:
        digest = hashlib.sha256()
        received = 0
        with os.fdopen(handle, "wb") as out, _open(file.url) as response:
            while block := response.read(_BLOCK):
                out.write(block)
                digest.update(block)
                received += len(block)
                if progress is not None:
                    progress(received, file.size)
        if digest.hexdigest() != file.sha256:
            raise ChecksumError(
                f"{file.filename} from {file.url} failed its checksum "
                f"(expected {file.sha256}, got {digest.hexdigest()})"
            )
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
