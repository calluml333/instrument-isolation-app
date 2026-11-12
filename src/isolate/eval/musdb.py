"""The MUSDB18 7-second sample set: 50 test tracks with 4 reference stems (SPEC §8).

Used for evaluation only, never redistributed (non-commercial research licence).
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from isolate.core.audio import Audio
from isolate.eval.deps import require
from isolate.models.weights import cache_root

SOURCES = ("drums", "bass", "other", "vocals")


@dataclass(frozen=True)
class Track:
    name: str
    mixture: Audio
    references: dict[str, Audio]


def dataset_dir() -> Path:
    return cache_root() / "datasets" / "musdb18-7"


def iter_tracks(
    root: Path | None = None,
    *,
    names: Sequence[str] | None = None,
    limit: int | None = None,
    is_wav: bool = False,
) -> Iterator[Track]:
    """Yield test-split tracks one at a time (they're loaded lazily).

    With no `root`, the sample set is downloaded (147 MB) into the cache on
    first use. `is_wav` reads a folder of WAV stems instead (used by tests).
    """
    musdb = require("musdb")
    download = root is None
    db = musdb.DB(
        root=str(root or dataset_dir()),
        download=download,
        subsets="test",
        is_wav=is_wav,
    )
    tracks = list(db.tracks)
    if names is not None:
        by_name = {track.name: track for track in tracks}
        missing = sorted(set(names) - set(by_name))
        if missing:
            raise ValueError(f"No MUSDB test track(s) named {', '.join(missing)}")
        tracks = [by_name[name] for name in names]
    for track in tracks[:limit]:
        rate = int(track.rate)
        yield Track(
            name=track.name,
            mixture=Audio(_channels_first(track.audio), rate),
            references={
                source: Audio(_channels_first(track.targets[source].audio), rate)
                for source in SOURCES
            },
        )


def _channels_first(samples: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
    return np.ascontiguousarray(samples.T, dtype=np.float32)
