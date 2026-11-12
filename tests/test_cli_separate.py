import numpy as np
import pytest
import soundfile as sf

from isolate import cli
from isolate.cli import main
from isolate.core import audio
from tests.fakes import STEMS

RATE = 48000


@pytest.fixture
def song(tmp_path):
    rng = np.random.default_rng(0)
    samples = (0.2 * rng.standard_normal((RATE // 2, 2))).astype(np.float32)
    path = tmp_path / "My Song.wav"
    sf.write(path, samples, RATE, subtype="FLOAT")
    return path


def test_writes_every_stem_matching_the_input(song, tmp_path, fake_model, capsys):
    fake_model()
    out = tmp_path / "out"

    code = main(["separate", str(song), "--model", "fake", "--out", str(out)])

    assert code == 0
    folder = out / "My Song"
    assert capsys.readouterr().out.strip() == str(folder)
    assert sorted(p.name for p in folder.iterdir()) == sorted(f"{s}.wav" for s in STEMS)
    total = 0
    for stem in STEMS:
        info = sf.info(folder / f"{stem}.wav")
        assert (info.samplerate, info.channels, info.subtype) == (RATE, 2, "FLOAT")
        assert info.frames == RATE // 2
        total = total + sf.read(folder / f"{stem}.wav", dtype="float32")[0]
    # Residual projection is on by default.
    np.testing.assert_allclose(total, sf.read(song, dtype="float32")[0], atol=1e-5)
    assert not list(out.glob(".*"))


def test_selected_stems_format_and_bit_depth(song, tmp_path, fake_model):
    fake_model()
    out = tmp_path / "out"

    code = main(
        [
            "separate",
            str(song),
            "--model",
            "fake",
            "--out",
            str(out),
            "--stems",
            "drums, other",
            "--format",
            "flac",
        ]
    )

    assert code == 0
    files = sorted((out / "My Song").iterdir())
    assert [f.name for f in files] == ["drums.flac", "other.flac"]
    assert sf.info(files[0]).subtype == "PCM_24"


def test_24_bit_wav(song, tmp_path, fake_model):
    fake_model()
    main(
        [
            "separate",
            str(song),
            "--model",
            "fake",
            "--out",
            str(tmp_path),
            "--bit-depth",
            "24",
            "--preset",
            "fast",
            "--no-residual",
        ]
    )
    assert sf.info(tmp_path / "My Song" / "other.wav").subtype == "PCM_24"


def test_several_inputs(song, tmp_path, fake_model):
    fake_model()
    second = tmp_path / "second.flac"
    sf.write(second, np.zeros((1000, 1), dtype=np.float32), 22050)

    code = main(
        [
            "separate",
            str(song),
            str(second),
            "--model",
            "fake",
            "--out",
            str(tmp_path / "out"),
            "--preset",
            "fast",
        ]
    )

    assert code == 0
    info = sf.info(tmp_path / "out" / "second" / "bass.wav")
    assert (info.samplerate, info.channels, info.frames) == (22050, 1, 1000)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--model", "nope"], "Unknown model 'nope'"),
        (["--model", "fake", "--stems", "kazoo"], "has no stem(s) kazoo"),
        (["--model", "fake", "--stems", ","], "(none given)"),
    ],
)
def test_readable_errors_exit_1(song, fake_model, capsys, extra, message):
    fake_model()

    code = main(["separate", str(song), *extra])

    err = capsys.readouterr().err
    assert code == 1
    assert message in err
    assert "Traceback" not in err


def test_unreadable_input_fails_before_loading_the_model(
    tmp_path, fake_model, capsys, monkeypatch
):
    fake_model()
    bad = tmp_path / "bad.wav"
    bad.write_text("not audio")
    monkeypatch.setattr(cli, "load_backend", lambda *a: pytest.fail("model loaded"))

    assert main(["separate", str(bad), "--model", "fake"]) == 1
    assert "bad.wav" in capsys.readouterr().err


@pytest.mark.parametrize(
    "args",
    [
        ["separate"],
        ["separate", "x.wav", "--preset", "slow"],
        ["separate", "x.wav", "--format", "flac", "--bit-depth", "32f"],
    ],
)
def test_usage_errors_exit_2(args, capsys):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_interrupt_exits_130_and_leaves_no_partial_output(
    song, tmp_path, fake_model, monkeypatch, capsys
):
    fake_model()
    out = tmp_path / "out"
    real_save = audio.save
    calls = 0

    def save_then_interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt
        real_save(*args, **kwargs)

    monkeypatch.setattr("isolate.cli.save", save_then_interrupt)

    code = main(["separate", str(song), "--model", "fake", "--out", str(out)])

    assert code == 130
    assert "interrupted" in capsys.readouterr().err
    assert list(out.iterdir()) == []


@pytest.mark.weights
def test_end_to_end_with_htdemucs_6s(song, tmp_path):
    code = main(["separate", str(song), "--out", str(tmp_path), "--preset", "fast"])

    assert code == 0
    assert len(list((tmp_path / "My Song").glob("*.wav"))) == 6
