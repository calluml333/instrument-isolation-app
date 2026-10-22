"""The `isolate` command line: separate, app, eval and models (from Phase 1)."""

import argparse
from collections.abc import Sequence

from isolate import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isolate",
        description="Split a stereo mix into instrument stems.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    # No subcommands until Phase 1, so a bare `isolate` just shows usage.
    parser.print_help()
    return 0
