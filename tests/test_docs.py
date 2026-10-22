from pathlib import Path

import pytest

MODELS_DOC = Path(__file__).resolve().parents[1] / "docs" / "models.md"


@pytest.mark.parametrize("model", ["htdemucs_6s", "htdemucs", "htdemucs_ft"])
def test_models_doc_covers_phase_models(model):
    assert f"`{model}`" in MODELS_DOC.read_text()
