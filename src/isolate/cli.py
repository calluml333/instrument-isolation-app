"""The `isolate` command line: `separate` and `models` (SPEC §3 W1, R9)."""

import argparse
import shutil
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, get_args

from tqdm import tqdm

from isolate import __version__
from isolate.core.audio import (
    Audio,
    AudioLoadError,
    BitDepth,
    OutputFormat,
    load,
    save,
)
from isolate.core.device import DEVICES, DeviceUnavailableError
from isolate.core.engine import PRESETS, Options, Progress, separate
from isolate.eval.deps import EvalDependencyError
from isolate.eval.musdb import UnknownTrackError
from isolate.models.loader import load_backend
from isolate.models.registry import DEFAULT_MODEL, MODELS, UnknownModelError, get_model
from isolate.models.weights import (
    ChecksumError,
    DownloadError,
    DownloadProgress,
    ensure_weights,
    is_downloaded,
)

EXIT_ERROR = 1
EXIT_INTERRUPTED = 130


class UnknownStemError(ValueError):
    """`--stems` named a stem the chosen model doesn't produce."""


# Failures that are the user's input or environment, shown without a traceback.
_EXPECTED_ERRORS = (
    AudioLoadError,
    ChecksumError,
    DeviceUnavailableError,
    DownloadError,
    EvalDependencyError,
    UnknownModelError,
    UnknownStemError,
    UnknownTrackError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isolate",
        description="Split a stereo mix into instrument stems.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    sep = commands.add_parser(
        "separate",
        help="separate audio files into stems",
        description="Separate audio files into one file per stem.",
    )
    sep.add_argument("inputs", nargs="+", type=Path, metavar="INPUT")
    sep.add_argument(
        "--out",
        type=Path,
        default=Path("stems"),
        help="output folder; stems go in OUT/<input name>/ (default: stems)",
    )
    sep.add_argument(
        "--stems",
        type=lambda value: [stem.strip() for stem in value.split(",") if stem.strip()],
        help="comma-separated stems to write (default: all the model's stems)",
    )
    sep.add_argument("--model", default=DEFAULT_MODEL, help="default: %(default)s")
    sep.add_argument("--preset", choices=sorted(PRESETS), default="best")
    sep.add_argument("--device", choices=DEVICES, default="auto")
    sep.add_argument("--format", choices=get_args(OutputFormat), default="wav")
    sep.add_argument(
        "--bit-depth",
        choices=("24", "32f"),
        help="default: 32f for WAV, 24 for FLAC",
    )
    sep.add_argument(
        "--no-residual",
        dest="residual",
        action="store_false",
        help="don't add the leftover (mix minus stems) to 'other'",
    )

    models = commands.add_parser("models", help="list or download models")
    model_commands = models.add_subparsers(
        dest="models_command", metavar="ACTION", required=True
    )
    model_commands.add_parser("list", help="list the available models")
    download = model_commands.add_parser("download", help="fetch a model's weights")
    download.add_argument("name", metavar="NAME")

    ev = commands.add_parser(
        "eval",
        help="score a model on the MUSDB18 sample set",
        description="Separate the MUSDB18 7-second test split and score it "
        "(SDR and SI-SDR, median per stem).",
    )
    ev.add_argument("--model", default=DEFAULT_MODEL, help="default: %(default)s")
    ev.add_argument("--preset", choices=sorted(PRESETS), default="fast")
    ev.add_argument("--device", choices=DEVICES, default="auto")
    ev.add_argument("--tracks", type=int, metavar="N", help="only the first N tracks")
    ev.add_argument(
        "--track",
        action="append",
        dest="names",
        metavar="NAME",
        help="only this track (repeatable)",
    )
    ev.add_argument(
        "--dataset",
        type=Path,
        metavar="DIR",
        help="a MUSDB-layout folder (default: download the sample set to the cache)",
    )
    ev.add_argument(
        "--wav", action="store_true", help="DIR holds WAV stems, not .stem.mp4 files"
    )
    ev.add_argument(
        "--out",
        type=Path,
        metavar="FILE",
        help="write the Markdown report here and per-track JSON scores beside it",
    )

    be = commands.add_parser(
        "bench",
        help="time a separation and measure its peak memory",
        description="Separate a synthetic song and report speed and peak memory. "
        "Run once per measurement: peak memory covers this whole process.",
    )
    be.add_argument("--model", default=DEFAULT_MODEL, help="default: %(default)s")
    be.add_argument("--preset", choices=sorted(PRESETS), default="fast")
    be.add_argument("--device", choices=DEVICES, default="auto")
    be.add_argument(
        "--duration",
        type=float,
        default=180.0,
        metavar="SECONDS",
        help="song length (default: %(default)s, a 3-minute song)",
    )
    be.add_argument("--out", type=Path, metavar="FILE", help="also write JSON here")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "separate" and args.format == "flac" and args.bit_depth == "32f":
        parser.error("FLAC has no 32-bit float mode; use --bit-depth 24")

    try:
        if args.command == "separate":
            return _separate(args)
        if args.command == "eval":
            return _eval(args)
        if args.command == "bench":
            return _bench(args)
        if args.models_command == "list":
            return _models_list()
        return _models_download(args.name)
    except _EXPECTED_ERRORS as exc:
        print(f"isolate: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("isolate: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED


def _separate(args: argparse.Namespace) -> int:
    spec = get_model(args.model)
    if args.stems is not None:
        unknown = sorted(set(args.stems) - set(spec.stems))
        if unknown or not args.stems:
            raise UnknownStemError(
                f"{spec.name} has no stem(s) {', '.join(unknown) or '(none given)'}; "
                f"it has {', '.join(spec.stems)}"
            )
    # Read every input first, so a bad file fails before the model loads.
    songs = [(path, load(path)) for path in args.inputs]

    backend = load_backend(spec.name, args.device, _download_bar(spec.name))
    for path, audio in songs:
        with tqdm(total=1.0, desc=path.name, unit="", file=sys.stderr) as bar:
            stems = separate(
                audio,
                backend,
                options=_options(args.preset, args.residual),
                stems=args.stems,
                progress=_advance(bar),
            )
        target = _write_stems(args.out, path.stem, stems, args.format, args.bit_depth)
        print(target)
    return 0


def _advance(bar: "tqdm[Any]") -> Progress:
    """Adapt the engine's fraction-done callback to a 0-1 progress bar."""

    def update(fraction: float) -> None:
        bar.update(fraction - bar.n)

    return update


def _options(preset: str, residual: bool) -> Options:
    return replace(PRESETS[preset], residual=residual)


def _write_stems(
    out: Path,
    song: str,
    stems: dict[str, Audio],
    fmt: OutputFormat,
    bit_depth: BitDepth | None,
) -> Path:
    """Write into a hidden folder, then move into place, so an interrupted
    run never leaves a half-written set of stems behind."""
    target = out / song
    partial = out / f".{song}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    try:
        for stem, audio in stems.items():
            save(partial / f"{stem}.{fmt}", audio, fmt=fmt, bit_depth=bit_depth)
        target.mkdir(parents=True, exist_ok=True)
        for file in partial.iterdir():
            file.replace(target / file.name)
    finally:
        shutil.rmtree(partial, ignore_errors=True)
    return target


class _DownloadBar:
    """A byte progress bar that only appears once a download actually starts."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.bar: tqdm[Any] | None = None

    def __call__(self, done: int, total: int) -> None:
        bar = self.bar
        if bar is None:
            bar = self.bar = tqdm(
                total=total,
                desc=f"Downloading {self.name}",
                unit="B",
                unit_scale=True,
                file=sys.stderr,
            )
        bar.update(done - bar.n)
        if done >= total:
            bar.close()


def _download_bar(name: str) -> DownloadProgress:
    return _DownloadBar(name)


def _models_list() -> int:
    rows = [("NAME", "STEMS", "LICENCE", "DOWNLOADED")]
    for spec in MODELS.values():
        rows.append(
            (
                spec.name,
                ",".join(spec.stems),
                spec.licence,
                "yes" if is_downloaded(spec) else "no",
            )
        )
    widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
    for row in rows:
        print(
            "  ".join(
                cell.ljust(width) for cell, width in zip(row, widths, strict=True)
            ).rstrip()
        )
    return 0


def _models_download(name: str) -> int:
    spec = get_model(name)
    print(ensure_weights(spec, _download_bar(name)))
    return 0


def _eval(args: argparse.Namespace) -> int:
    from isolate.eval.musdb import iter_tracks
    from isolate.eval.runner import TrackResult, evaluate

    tracks = iter_tracks(
        args.dataset, names=args.names, limit=args.tracks, is_wav=args.wav
    )
    backend = load_backend(args.model, args.device, _download_bar(args.model))

    def show(result: TrackResult) -> None:
        scores = "  ".join(
            f"{stem} {value:5.2f}" if value is not None else f"{stem}   n/a"
            for stem, value in result.sdr.items()
        )
        print(f"{result.name[:40]:40}  SDR  {scores}", file=sys.stderr)

    report = evaluate(
        tracks,
        backend,
        model=args.model,
        preset=args.preset,
        options=PRESETS[args.preset],
        on_track=show,
    )
    if not report.tracks:
        raise UnknownTrackError("No tracks were evaluated")
    if args.out is not None:
        for path in report.write(args.out):
            print(path, file=sys.stderr)
    print(report.to_markdown())
    return 0


def _bench(args: argparse.Namespace) -> int:
    import json
    from dataclasses import asdict

    from isolate.eval.bench import run_bench

    # Fetch the weights first, so download time isn't counted as load time.
    ensure_weights(get_model(args.model), _download_bar(args.model))
    result = run_bench(args.model, args.preset, args.duration, args.device)
    if args.out is not None:
        data = {**asdict(result), "real_time_factor": result.real_time_factor}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, indent=2) + "\n")
    print(result.to_markdown())
    return 0
