import json

import numpy as np
import pytest

from isolate.cli import main
from isolate.core.audio import Audio
from isolate.core.engine import PRESETS
from isolate.eval.musdb import SOURCES, Track, iter_tracks
from isolate.eval.runner import EvalReport, TrackResult, evaluate, to_musdb_stems
from tests.fake_musdb import RATE, write_fake_musdb
from tests.fakes import LossyBackend


@pytest.fixture
def fake_root(tmp_path):
    return write_fake_musdb(tmp_path / "musdb", tracks=2)


def ones(value: float) -> Audio:
    return Audio(np.full((2, 4), value, dtype=np.float32), RATE)


def test_extra_model_stems_are_added_to_other():
    stems = {
        s: ones(i)
        for i, s in enumerate(("drums", "bass", "other", "vocals", "guitar", "piano"))
    }

    mapped = to_musdb_stems(stems)

    assert list(mapped) == ["drums", "bass", "vocals", "other"]
    assert mapped["drums"] is stems["drums"]
    np.testing.assert_array_equal(mapped["other"].samples, ones(2 + 4 + 5).samples)


def test_evaluate_scores_every_track_and_stem(fake_root):
    seen: list[TrackResult] = []

    report = evaluate(
        iter_tracks(fake_root, is_wav=True),
        LossyBackend(),
        model="lossy",
        preset="fast",
        options=PRESETS["fast"],
        on_track=seen.append,
    )

    assert [t.name for t in report.tracks] == [t.name for t in seen]
    assert len(report.tracks) == 2
    summary = report.summary()
    assert set(summary) == set(SOURCES)
    for stem in SOURCES:
        assert summary[stem].scored == 2
        assert summary[stem].excluded == 0
        assert summary[stem].sdr is not None
        assert summary[stem].si_sdr is not None
    assert report.wall_seconds > 0


def test_silent_references_are_excluded():
    rng = np.random.default_rng(0)
    refs = {
        s: Audio((0.1 * rng.standard_normal((2, RATE))).astype(np.float32), RATE)
        for s in SOURCES
    }
    refs["bass"] = Audio(np.zeros((2, RATE), dtype=np.float32), RATE)
    mixture = Audio(np.sum([a.samples for a in refs.values()], axis=0), RATE)

    report = evaluate(
        [Track("No bass", mixture, refs)],
        LossyBackend(),
        model="lossy",
        preset="fast",
        options=PRESETS["fast"],
    )

    assert report.tracks[0].sdr["bass"] is None
    assert report.summary()["bass"].excluded == 1
    assert "1 silent, excluded" in report.to_markdown()


def test_report_files(tmp_path):
    report = EvalReport(model="m", preset="fast", options=PRESETS["fast"])
    report.tracks = [
        TrackResult("A", {s: 5.0 for s in SOURCES}, {s: 4.0 for s in SOURCES}, 1.0),
        TrackResult(
            "B",
            {**{s: 7.0 for s in SOURCES}, "drums": float("-inf")},
            {s: 6.0 for s in SOURCES},
            1.0,
        ),
    ]

    md, data = report.write(tmp_path / "results" / "report.md")

    text = md.read_text()
    assert "# Evaluation: `m`, preset `fast`" in text
    assert "| `bass` | 6.00 | 5.00 | 2 |" in text
    # A non-finite score is left out of the median and stored as null.
    assert "| `drums` | 5.00 | 5.00 | 1 (1 silent, excluded) |" in text
    scores = json.loads(data.read_text())
    assert data.name == "report.json"
    assert scores["tracks"][1]["sdr"]["drums"] is None
    assert scores["summary"]["bass"]["sdr"] == 6.0
    assert "isolate" in scores["versions"]


def test_empty_report_has_no_medians():
    summary = EvalReport(model="m", preset="fast", options=PRESETS["fast"]).summary()
    assert summary["vocals"].sdr is None
    assert (
        "| `vocals` | n/a | n/a | 0 |"
        in EvalReport(model="m", preset="fast", options=PRESETS["fast"]).to_markdown()
    )


def test_eval_command(fake_root, fake_model, tmp_path, capsys):
    fake_model()
    out = tmp_path / "eval.md"

    code = main(
        [
            "eval",
            "--model",
            "fake",
            "--dataset",
            str(fake_root),
            "--wav",
            "--tracks",
            "1",
            "--out",
            str(out),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "| `drums` |" in captured.out
    assert "Artist 0 - Song 0" in captured.err
    assert out.exists()
    assert out.with_suffix(".json").exists()


def test_eval_command_unknown_track(fake_root, fake_model, capsys):
    fake_model()

    code = main(
        [
            "eval",
            "--model",
            "fake",
            "--dataset",
            str(fake_root),
            "--wav",
            "--track",
            "Nobody - Nothing",
        ]
    )

    assert code == 1
    assert "No MUSDB test track" in capsys.readouterr().err


def test_eval_command_no_tracks(fake_root, fake_model, capsys):
    fake_model()

    code = main(
        [
            "eval",
            "--model",
            "fake",
            "--dataset",
            str(fake_root),
            "--wav",
            "--tracks",
            "0",
        ]
    )

    assert code == 1
    assert "No tracks were evaluated" in capsys.readouterr().err
