"""The interface every separation model sits behind (SPEC M1)."""

from typing import Protocol

from isolate.core.audio import Float32Array


class Backend(Protocol):
    """A separation model that works on one segment at a time.

    The engine (`isolate.core.engine`) handles resampling, chunking, shifts and
    residual projection, so a backend only maps a single stereo chunk at
    `sample_rate` to one array per stem.
    """

    @property
    def name(self) -> str: ...

    @property
    def stems(self) -> tuple[str, ...]:
        """Stem names, in the order the model produces them."""
        ...

    @property
    def sample_rate(self) -> int: ...

    @property
    def segment_seconds(self) -> float:
        """The longest chunk `separate` accepts."""
        ...

    @property
    def normalize_input(self) -> bool:
        """Whether the engine should standardise the whole mix first.

        Demucs models expect the track scaled to zero mean and unit variance
        (computed over the whole track, not per chunk), and their outputs
        scaled back.
        """
        ...

    def separate(self, chunk: Float32Array) -> dict[str, Float32Array]:
        """Separate a float32 `(2, n)` chunk, `n` at most one segment.

        Returns a float32 `(2, n)` array for every name in `stems`.
        """
        ...
