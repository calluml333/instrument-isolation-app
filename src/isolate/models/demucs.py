"""Demucs (HTDemucs) backends, loaded from safetensors weights."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from demucs.apply import BagOfModels, apply_model
from demucs.hf import load_safetensors_model

from isolate.core.audio import Float32Array


class DemucsBackend:
    """Runs a Demucs bag of models on one chunk; the engine does the rest."""

    normalize_input = True

    def __init__(self, name: str, model: BagOfModels, device: str) -> None:
        self.name = name
        self.stems: tuple[str, ...] = tuple(model.sources)
        self.sample_rate = int(model.samplerate)
        self.segment_seconds = float(model.max_allowed_segment)
        self._device = device
        self._model = model.to(device).eval()

    def separate(self, chunk: Float32Array) -> dict[str, Float32Array]:
        mix = torch.from_numpy(chunk).unsqueeze(0)
        with torch.inference_mode():
            # The engine already chunks and shifts, so demucs mustn't.
            out = apply_model(
                self._model,
                mix,
                shifts=0,
                split=False,
                progress=False,
                device=self._device,
            )
        sources = out[0].cpu().numpy()
        return {
            stem: np.ascontiguousarray(sources[index], dtype=np.float32)
            for index, stem in enumerate(self.stems)
        }


def load_demucs(
    name: str,
    paths: Sequence[Path],
    device: str,
    weights: list[list[float]] | None = None,
) -> DemucsBackend:
    """Build a backend from one safetensors file per model in the bag."""
    models = [load_safetensors_model(path) for path in paths]
    return DemucsBackend(name, BagOfModels(models, weights), device)
