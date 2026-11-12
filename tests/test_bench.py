import numpy as np
import pytest

from isolate.cli import main
from isolate.eval import bench
from isolate.eval.bench import memory_limit_mb, run_bench, synthetic_song


def test_synthetic_song():
    song = synthetic_song(2.0)
    assert song.samples.shape == (2, 88200)
    assert song.samples.dtype == np.float32
    assert 0 < np.abs(song.samples).max() < 1
    np.testing.assert_array_equal(song.samples, synthetic_song(2.0).samples)


def test_run_bench_with_a_stand_in(fake_model):
    fake_model()

    result = run_bench("fake", "fast", 5.0, device="cpu")

    assert result.duration_seconds == 5.0
    assert result.separate_seconds > 0
    assert result.load_seconds >= 0
    assert result.real_time_factor == pytest.approx(result.separate_seconds / 5.0)
    assert result.peak_rss_mb > 0
    assert result.cpus >= 1
    text = result.to_markdown()
    assert "| Real-time factor |" in text
    assert "| Song length | 5 s" in text


@pytest.mark.parametrize(
    ("cgroup", "meminfo", "expected"),
    [
        ("8589934592\n", None, 8192.0),
        ("max\n", "MemTotal:       16384000 kB\n", 16000.0),
        (None, "MemTotal:       2048000 kB\n", 2000.0),
        (None, None, None),
    ],
)
def test_memory_limit(tmp_path, monkeypatch, cgroup, meminfo, expected):
    for name, contents in (("_CGROUP_LIMIT", cgroup), ("_MEMINFO", meminfo)):
        path = tmp_path / name
        if contents is not None:
            path.write_text(contents)
        monkeypatch.setattr(bench, name, path)

    assert memory_limit_mb() == expected


def test_bench_command(fake_model, tmp_path, capsys):
    fake_model()
    out = tmp_path / "bench.json"

    code = main(["bench", "--model", "fake", "--duration", "3", "--out", str(out)])

    assert code == 0
    assert "| Separation |" in capsys.readouterr().out
    assert '"real_time_factor"' in out.read_text()
