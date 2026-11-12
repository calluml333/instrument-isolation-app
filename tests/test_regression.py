"""Fails if htdemucs_6s quality drops on a few fixed MUSDB18 tracks (SPEC §9).

Needs the real weights, the MUSDB18 sample set and the eval group, so it only
runs with --run-weights (the `eval` CI workflow runs it).
"""

import json
from pathlib import Path

import pytest

from isolate.core.engine import PRESETS
from isolate.eval.musdb import iter_tracks
from isolate.eval.runner import evaluate
from isolate.models.loader import load_backend

BASELINE = Path(__file__).parent / "data" / "regression-baseline.json"


@pytest.mark.regression
def test_sdr_has_not_regressed():
    baseline = json.loads(BASELINE.read_text())
    report = evaluate(
        iter_tracks(names=baseline["tracks"]),
        load_backend(baseline["model"], "cpu"),
        model=baseline["model"],
        preset=baseline["preset"],
        options=PRESETS[baseline["preset"]],
    )

    summary = report.summary()
    drops = {
        stem: (expected, summary[stem].sdr)
        for stem, expected in baseline["sdr"].items()
        if summary[stem].sdr is None
        or summary[stem].sdr < expected - baseline["tolerance_db"]
    }
    assert not drops, (
        f"Median SDR fell more than {baseline['tolerance_db']} dB: {drops}"
    )
