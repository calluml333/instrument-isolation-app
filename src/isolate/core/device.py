"""Choosing the torch device a model runs on (SPEC R6)."""

from typing import Literal, get_args

import torch

Device = Literal["auto", "cpu", "mps", "cuda"]
DEVICES: tuple[Device, ...] = get_args(Device)

# The order `auto` tries them in.
_PREFERENCE = ("mps", "cuda", "cpu")


class DeviceUnavailableError(Exception):
    """A device was asked for explicitly but this machine doesn't have it."""


def available_devices() -> dict[str, bool]:
    return {
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
        "cpu": True,
    }


def resolve_device(requested: Device = "auto") -> str:
    """Turn `auto` into the best available device, or check a forced one."""
    available = available_devices()
    if requested == "auto":
        return next(device for device in _PREFERENCE if available[device])
    if requested not in available:
        raise ValueError(f"Unknown device {requested!r}; expected one of {DEVICES}")
    if not available[requested]:
        hint = " (Docker on macOS can't use the GPU)" if requested == "mps" else ""
        raise DeviceUnavailableError(f"Device {requested!r} isn't available{hint}")
    return requested
