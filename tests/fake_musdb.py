"""A tiny MUSDB-style dataset on disk, in the WAV layout musdb reads offline."""

from pathlib import Path

import numpy as np
import soundfile as sf

from isolate.eval.musdb import SOURCES

RATE = 44100


def write_fake_musdb(root: Path, tracks: int = 2, seconds: float = 2.0) -> Path:
    rng = np.random.default_rng(0)
    frames = int(seconds * RATE)
    for index in range(tracks):
        folder = root / "test" / f"Artist {index} - Song {index}"
        folder.mkdir(parents=True)
        stems = {
            source: (0.1 * rng.standard_normal((frames, 2))).astype(np.float32)
            for source in SOURCES
        }
        for source, samples in stems.items():
            sf.write(folder / f"{source}.wav", samples, RATE, subtype="FLOAT")
        sf.write(folder / "mixture.wav", sum(stems.values()), RATE, subtype="FLOAT")
    return root
