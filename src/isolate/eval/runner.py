"""Running a model over the evaluation set and reporting the scores (SPEC §8)."""

import json
import math
import platform
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from isolate.core.audio import Audio
from isolate.core.engine import Options, separate
from isolate.eval.metrics import is_silent, sdr, si_sdr
from isolate.eval.musdb import SOURCES, Track
from isolate.models.base import Backend

# Distribution name -> label in reports.
_PACKAGES = {
    "instrument-isolation-app": "isolate",
    "torch": "torch",
    "demucs": "demucs",
    "numpy": "numpy",
    "museval": "museval",
}


@dataclass(frozen=True)
class TrackResult:
    name: str
    # None where the reference stem is silent, so the score means nothing.
    sdr: dict[str, float | None]
    si_sdr: dict[str, float | None]
    seconds: float


@dataclass(frozen=True)
class StemSummary:
    sdr: float | None
    si_sdr: float | None
    scored: int
    excluded: int


@dataclass
class EvalReport:
    model: str
    preset: str
    options: Options
    tracks: list[TrackResult] = field(default_factory=list)
    wall_seconds: float = 0.0
    date: str = field(default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%d"))
    versions: dict[str, str] = field(default_factory=lambda: _versions())

    def summary(self) -> dict[str, StemSummary]:
        result = {}
        for stem in SOURCES:
            sdrs = _finite(track.sdr.get(stem) for track in self.tracks)
            si_sdrs = _finite(track.si_sdr.get(stem) for track in self.tracks)
            result[stem] = StemSummary(
                sdr=_median(sdrs),
                si_sdr=_median(si_sdrs),
                scored=len(sdrs),
                excluded=len(self.tracks) - len(sdrs),
            )
        return result

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "preset": self.preset,
            "options": asdict(self.options),
            "date": self.date,
            "wall_seconds": round(self.wall_seconds, 1),
            "versions": self.versions,
            "summary": {stem: asdict(s) for stem, s in self.summary().items()},
            "tracks": [
                {
                    **asdict(track),
                    "sdr": _clean(track.sdr),
                    "si_sdr": _clean(track.si_sdr),
                }
                for track in self.tracks
            ],
        }

    def to_markdown(self) -> str:
        opts = self.options
        versions = ", ".join(f"{name} {v}" for name, v in self.versions.items())
        lines = [
            f"# Evaluation: `{self.model}`, preset `{self.preset}`",
            "",
            "| | |",
            "|---|---|",
            f"| **Model** | `{self.model}` |",
            f"| **Preset** | `{self.preset}` (overlap {opts.overlap}, "
            f"shifts {opts.shifts}, seed {opts.seed}) |",
            f"| **Dataset** | MUSDB18 7-second samples, test split "
            f"({len(self.tracks)} tracks) |",
            f"| **Date** | {self.date} |",
            f"| **Wall time** | {self.wall_seconds:.0f} s |",
            f"| **Versions** | {versions} |",
            "",
            "| Stem | SDR (dB, median) | SI-SDR (dB, median) | Tracks scored |",
            "|---|---|---|---|",
        ]
        for stem, s in self.summary().items():
            excluded = f" ({s.excluded} silent, excluded)" if s.excluded else ""
            lines.append(
                f"| `{stem}` | {_fmt(s.sdr)} | {_fmt(s.si_sdr)} "
                f"| {s.scored}{excluded} |"
            )
        lines += [
            "",
            "SDR is BSSEval v4 (museval, 1 s windows); SI-SDR is over the whole clip. "
            "Both are medians over tracks. The model's `guitar` and `piano` are added "
            "to `other` before scoring, because MUSDB18 has no separate guitar or "
            "piano reference. These scores are only comparable with other runs on "
            "this sample set, not with published MUSDB18-HQ results (SPEC §8).",
            "",
        ]
        return "\n".join(lines)

    def write(self, path: Path) -> tuple[Path, Path]:
        """Write the Markdown report to `path` and the JSON scores beside it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown())
        data = path.with_suffix(".json")
        data.write_text(json.dumps(self.to_json(), indent=2) + "\n")
        return path, data


def to_musdb_stems(stems: dict[str, Audio]) -> dict[str, Audio]:
    """Map a model's stems onto MUSDB's four: unmatched ones join `other`."""
    named = {source: stems[source] for source in SOURCES if source != "other"}
    rest = [audio for stem, audio in stems.items() if stem not in named]
    first = rest[0]
    other = Audio(sum((a.samples for a in rest[1:]), first.samples), first.sample_rate)
    return {**named, "other": other}


def evaluate(
    tracks: Iterable[Track],
    backend: Backend,
    *,
    model: str,
    preset: str,
    options: Options,
    on_track: Callable[[TrackResult], None] | None = None,
) -> EvalReport:
    report = EvalReport(model=model, preset=preset, options=options)
    started = time.perf_counter()
    for track in tracks:
        track_started = time.perf_counter()
        estimates = to_musdb_stems(separate(track.mixture, backend, options=options))
        sdrs: dict[str, float | None] = dict(sdr(track.references, estimates))
        si_sdrs: dict[str, float | None] = {}
        for stem, reference in track.references.items():
            if is_silent(reference):
                sdrs[stem] = None
                si_sdrs[stem] = None
            else:
                si_sdrs[stem] = si_sdr(reference.samples, estimates[stem].samples)
        result = TrackResult(
            name=track.name,
            sdr=sdrs,
            si_sdr=si_sdrs,
            seconds=round(time.perf_counter() - track_started, 2),
        )
        report.tracks.append(result)
        if on_track is not None:
            on_track(result)
    report.wall_seconds = time.perf_counter() - started
    return report


def _versions() -> dict[str, str]:
    found = {"python": platform.python_version()}
    for package, label in _PACKAGES.items():
        try:
            found[label] = version(package)
        except PackageNotFoundError:
            continue
    return found


def _finite(values: Iterable[float | None]) -> list[float]:
    return [v for v in values if v is not None and math.isfinite(v)]


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _clean(scores: dict[str, float | None]) -> dict[str, float | None]:
    """JSON has no infinities: store non-finite scores as null."""
    return {
        stem: round(v, 3) if v is not None and math.isfinite(v) else None
        for stem, v in scores.items()
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"
