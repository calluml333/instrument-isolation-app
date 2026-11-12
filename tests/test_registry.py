import re

import pytest

from isolate.models.registry import DEFAULT_MODEL, MODELS, UnknownModelError, get_model


def test_default_model_is_registered():
    spec = get_model(DEFAULT_MODEL)
    assert spec.stems == ("drums", "bass", "other", "vocals", "guitar", "piano")


@pytest.mark.parametrize("spec", MODELS.values(), ids=lambda spec: spec.name)
def test_entries_are_complete(spec):
    assert spec.description
    assert spec.licence
    assert spec.licence_url.startswith("https://")
    assert spec.files
    for file in spec.files:
        # Weights are pinned: an immutable commit in the URL and a full SHA-256.
        assert re.fullmatch(
            r"https://huggingface\.co/.+/resolve/[0-9a-f]{40}/.+", file.url
        )
        assert file.url.endswith(file.filename)
        assert re.fullmatch(r"[0-9a-f]{64}", file.sha256)
        assert file.size > 0


def test_unknown_model_lists_the_valid_names():
    with pytest.raises(
        UnknownModelError, match="Unknown model 'nope'; available: htdemucs_6s"
    ):
        get_model("nope")


def test_load_backend_fetches_weights_then_builds(fake_model):
    from isolate.models.loader import load_backend
    from isolate.models.weights import is_downloaded

    spec = fake_model()

    backend = load_backend("fake", "cpu")

    assert backend.name == "identity"
    assert is_downloaded(spec)
