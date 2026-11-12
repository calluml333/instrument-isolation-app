"""The models `isolate` knows about, and where their weights come from (SPEC M4)."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from isolate.models.base import Backend

DEFAULT_MODEL = "htdemucs_6s"


@dataclass(frozen=True)
class WeightFile:
    filename: str
    url: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    stems: tuple[str, ...]
    licence: str
    licence_url: str
    files: tuple[WeightFile, ...]
    # Builds the backend from the directory holding `files`, on a torch device.
    load: Callable[[Path, str], Backend]


class UnknownModelError(ValueError):
    pass


def _load_htdemucs_6s(weights_dir: Path, device: str) -> Backend:
    from isolate.models.demucs import load_demucs

    return load_demucs("htdemucs_6s", [weights_dir / "5c90dfd2.safetensors"], device)


_HF = "https://huggingface.co"

MODELS: dict[str, ModelSpec] = {
    "htdemucs_6s": ModelSpec(
        name="htdemucs_6s",
        description="Hybrid Transformer Demucs, 6 stems (Meta, demucs v4)",
        stems=("drums", "bass", "other", "vocals", "guitar", "piano"),
        licence="Code MIT; weights research use only",
        licence_url="https://github.com/facebookresearch/demucs/issues/327",
        files=(
            WeightFile(
                filename="5c90dfd2.safetensors",
                url=f"{_HF}/adefossez/HTDemucs-6s/resolve/"
                "3c5ee475be622df764938de97e4281a7b07ffa58/5c90dfd2.safetensors",
                sha256="d2a1745f0744721f6b8ca5bf469b67c651ea5ed1b52998cab033b2158609d411",
                size=54_885_744,
            ),
        ),
        load=_load_htdemucs_6s,
    ),
}


def get_model(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError:
        raise UnknownModelError(
            f"Unknown model {name!r}; available: {', '.join(sorted(MODELS))}"
        ) from None
