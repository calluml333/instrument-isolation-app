import importlib
from importlib.metadata import metadata, version

import pytest

import isolate

DIST = "instrument-isolation-app"


@pytest.mark.parametrize(
    "module",
    ["isolate.core", "isolate.models", "isolate.eval", "isolate.app", "isolate.cli"],
)
def test_module_imports(module):
    imported = importlib.import_module(module)
    assert imported.__doc__


def test_version_matches_metadata():
    assert isolate.__version__ == version(DIST)


def test_requires_python():
    assert metadata(DIST)["Requires-Python"] == ">=3.12,<3.13"
