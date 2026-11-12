import pytest
import torch

from isolate.core.device import DeviceUnavailableError, resolve_device


@pytest.fixture
def machine(monkeypatch):
    def configure(*, mps: bool, cuda: bool) -> None:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)

    return configure


@pytest.mark.parametrize(
    ("mps", "cuda", "expected"),
    [(True, True, "mps"), (False, True, "cuda"), (False, False, "cpu")],
)
def test_auto_prefers_mps_then_cuda_then_cpu(machine, mps, cuda, expected):
    machine(mps=mps, cuda=cuda)
    assert resolve_device("auto") == expected


def test_auto_is_cpu_in_the_container():
    assert resolve_device() == "cpu"


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_forced_device_when_available(machine, device):
    machine(mps=True, cuda=True)
    assert resolve_device(device) == device


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_forced_device_when_unavailable(machine, device):
    machine(mps=False, cuda=False)
    with pytest.raises(DeviceUnavailableError, match=device):
        resolve_device(device)


def test_unknown_device():
    with pytest.raises(ValueError, match="Unknown device 'tpu'"):
        resolve_device("tpu")  # type: ignore[arg-type]
