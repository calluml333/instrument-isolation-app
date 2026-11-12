import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from isolate.models import registry
from isolate.models.registry import ModelSpec, WeightFile
from tests.fakes import STEMS, IdentityBackend


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-weights",
        action="store_true",
        help="also run tests that download and use the real model weights",
    )


# Tests that need the real weights (and, for regression, the MUSDB18 sample
# set): skipped unless --run-weights is given.
ONLINE_MARKERS = ("weights", "regression")


def _is_online(keywords: pytest.Item | pytest.FixtureRequest) -> bool:
    return any(marker in keywords.keywords for marker in ONLINE_MARKERS)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    run_weights = config.getoption("--run-weights")
    skip = pytest.mark.skip(reason="needs real model weights; use --run-weights")
    for item in items:
        if not _is_online(item):
            continue
        # Online tests may download; everything else stays offline.
        item.add_marker(pytest.mark.enable_socket if run_weights else skip)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path_factory, monkeypatch, request):
    """Every test gets its own cache, except online tests (CI caches theirs)."""
    if not _is_online(request):
        monkeypatch.setenv("ISOLATE_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))


def weight_file(path: Path, contents: bytes) -> WeightFile:
    """A registry entry served from a local file (file:// needs no sockets)."""
    path.write_bytes(contents)
    return WeightFile(
        filename=path.name,
        url=path.as_uri(),
        sha256=hashlib.sha256(contents).hexdigest(),
        size=len(contents),
    )


@pytest.fixture
def fake_model(tmp_path, monkeypatch) -> Callable[..., ModelSpec]:
    """Register a model backed by `IdentityBackend` and local weight files."""

    def register(name: str = "fake", **backend_kwargs: object) -> ModelSpec:
        source = tmp_path / "source" / name
        source.mkdir(parents=True)
        spec = ModelSpec(
            name=name,
            description="Test stand-in",
            stems=STEMS,
            licence="Test only",
            licence_url="https://example.com/licence",
            files=(weight_file(source / "weights.bin", b"fake weights"),),
            load=lambda _dir, _device: IdentityBackend(**backend_kwargs),  # type: ignore[arg-type]
        )
        monkeypatch.setitem(registry.MODELS, name, spec)
        return spec

    return register
