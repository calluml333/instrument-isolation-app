"""Turning a model name into a ready-to-run backend."""

from isolate.core.device import Device, resolve_device
from isolate.models.base import Backend
from isolate.models.registry import get_model
from isolate.models.weights import DownloadProgress, ensure_weights


def load_backend(
    name: str,
    device: Device = "auto",
    progress: DownloadProgress | None = None,
) -> Backend:
    """Fetch (if needed) and verify `name`'s weights, then build its backend."""
    spec = get_model(name)
    resolved = resolve_device(device)
    return spec.load(ensure_weights(spec, progress), resolved)
