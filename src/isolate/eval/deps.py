"""Importing the optional evaluation dependencies (the Poetry `eval` group)."""

from types import ModuleType


class EvalDependencyError(ImportError):
    """musdb or museval isn't installed, or FFmpeg is missing."""


def require(module: str) -> ModuleType:
    """Import `module`, explaining how to fix it if that fails.

    musdb and museval import stempeg, which raises RuntimeError at import time
    when FFmpeg isn't on PATH.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise EvalDependencyError(
            f"Evaluation needs {module}: run `poetry install --with eval`"
        ) from exc
    except RuntimeError as exc:
        raise EvalDependencyError(f"Evaluation needs FFmpeg on PATH ({exc})") from exc
