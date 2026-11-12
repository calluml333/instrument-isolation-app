"""Speed and memory benchmark on a synthetic song (SPEC N1, N3).

Run it as its own process (`isolate bench`), once per measurement: peak
memory is that process's high-water mark, so it covers loading the model,
holding the song and separating it.
"""

import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from isolate.core.audio import Audio
from isolate.core.device import Device
from isolate.core.engine import PRESETS, separate
from isolate.models.loader import load_backend

RATE = 44100
_CGROUP_LIMIT = Path("/sys/fs/cgroup/memory.max")
_MEMINFO = Path("/proc/meminfo")


@dataclass(frozen=True)
class BenchResult:
    model: str
    preset: str
    duration_seconds: float
    load_seconds: float
    separate_seconds: float
    peak_rss_mb: float
    cpus: int
    torch_threads: int
    memory_limit_mb: float | None

    @property
    def real_time_factor(self) -> float:
        """Separation time divided by song length (below 1 is faster than real time)."""
        return self.separate_seconds / self.duration_seconds

    def to_markdown(self) -> str:
        limit = (
            "unknown"
            if self.memory_limit_mb is None
            else f"{self.memory_limit_mb:.0f} MB"
        )
        rows = [
            ("Model", f"`{self.model}`"),
            ("Preset", f"`{self.preset}`"),
            (
                "Song length",
                f"{self.duration_seconds:.0f} s (synthetic, stereo, 44.1 kHz)",
            ),
            ("Model load", f"{self.load_seconds:.1f} s"),
            ("Separation", f"{self.separate_seconds:.1f} s"),
            ("Real-time factor", f"{self.real_time_factor:.3f}"),
            ("Peak memory (RSS)", f"{self.peak_rss_mb:.0f} MB"),
            ("CPUs / torch threads", f"{self.cpus} / {self.torch_threads}"),
            ("Memory limit", limit),
        ]
        return "\n".join(["| | |", "|---|---|", *(f"| {k} | {v} |" for k, v in rows)])


def synthetic_song(seconds: float, rate: int = RATE, seed: int = 0) -> Audio:
    """Bass, a gated lead and noise bursts: enough to look like music to a model.

    Separation speed doesn't depend on the content, only the length.
    """
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    rng = np.random.default_rng(seed)
    bass = 0.3 * np.sin(2 * np.pi * 55 * t)
    lead = 0.2 * np.sin(2 * np.pi * 440 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)
    hats = 0.05 * rng.standard_normal(t.size, dtype=np.float32) * ((t * 4) % 1 < 0.05)
    left = bass + lead + hats
    return Audio(np.stack([left, bass + 0.8 * lead + hats]).astype(np.float32), rate)


def memory_limit_mb() -> float | None:
    """The container's memory limit (cgroup v2), else the machine's total RAM."""
    try:
        limit = _CGROUP_LIMIT.read_text().strip()
        if limit != "max":
            return int(limit) / 2**20
    except OSError:
        pass
    try:
        for line in _MEMINFO.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return peak / 2**20 if sys.platform == "darwin" else peak / 1024


def run_bench(
    model: str, preset: str, seconds: float, device: Device = "auto"
) -> BenchResult:
    started = time.perf_counter()
    backend = load_backend(model, device)
    loaded = time.perf_counter()

    song = synthetic_song(seconds)
    separating = time.perf_counter()
    separate(song, backend, options=PRESETS[preset])
    finished = time.perf_counter()

    return BenchResult(
        model=model,
        preset=preset,
        duration_seconds=seconds,
        load_seconds=loaded - started,
        separate_seconds=finished - separating,
        peak_rss_mb=peak_rss_mb(),
        cpus=os.cpu_count() or 1,
        torch_threads=torch.get_num_threads(),
        memory_limit_mb=memory_limit_mb(),
    )
