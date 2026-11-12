"""A tiny, randomly initialised HTDemucs saved as safetensors, for offline tests."""

import json
from pathlib import Path

from demucs.htdemucs import HTDemucs
from safetensors.torch import save_file

from tests.fakes import STEMS


def save_tiny_htdemucs(path: Path, segment: int = 1) -> Path:
    model = HTDemucs(
        sources=list(STEMS),
        channels=16,
        depth=2,
        t_layers=1,
        t_heads=1,
        bottom_channels=0,
        segment=segment,
        samplerate=44100,
    )
    args, kwargs = model._init_args_kwargs
    state = {key: value.contiguous() for key, value in model.state_dict().items()}
    metadata = {
        "klass": f"{HTDemucs.__module__}.{HTDemucs.__name__}",
        "args": json.dumps(list(args)),
        "kwargs": json.dumps(kwargs),
    }
    save_file(state, str(path), metadata=metadata)
    return path
