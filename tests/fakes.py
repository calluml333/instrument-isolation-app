"""Stand-in backends, fast enough to run the engine in every test."""

from dataclasses import dataclass, field

import numpy as np

from isolate.core.audio import Float32Array

STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")


@dataclass
class IdentityBackend:
    """Returns the input as `other` and silence for every other stem."""

    name: str = "identity"
    stems: tuple[str, ...] = STEMS
    sample_rate: int = 44100
    segment_seconds: float = 0.1
    normalize_input: bool = False
    calls: list[int] = field(default_factory=list)

    def separate(self, chunk: Float32Array) -> dict[str, Float32Array]:
        self.calls.append(chunk.shape[1])
        out = {stem: np.zeros_like(chunk) for stem in self.stems}
        out["other"] = chunk.copy()
        return out


@dataclass
class LossyBackend(IdentityBackend):
    """Splits the input evenly across the stems but drops 10% of it."""

    name: str = "lossy"

    def separate(self, chunk: Float32Array) -> dict[str, Float32Array]:
        self.calls.append(chunk.shape[1])
        share = (0.9 * chunk / len(self.stems)).astype(np.float32)
        return {stem: share.copy() for stem in self.stems}
