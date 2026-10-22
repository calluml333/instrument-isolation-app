import shutil
import subprocess
import sys

import pytest

from isolate import __version__
from isolate.cli import main


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out == f"isolate {__version__}\n"


def test_no_arguments_prints_usage(capsys):
    assert main([]) == 0
    assert capsys.readouterr().out.startswith("usage: isolate")


def test_unknown_option_exits_2(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--no-such-option"])
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize(
    "command",
    [
        pytest.param([shutil.which("isolate") or "isolate"], id="console-script"),
        pytest.param([sys.executable, "-m", "isolate"], id="python-m"),
    ],
)
def test_installed_command_prints_version(command):
    result = subprocess.run(
        [*command, "--version"], capture_output=True, text=True, check=True
    )
    assert result.stdout == f"isolate {__version__}\n"
