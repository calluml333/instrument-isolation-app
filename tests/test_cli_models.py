from isolate.cli import main
from isolate.models.weights import is_downloaded


def test_list_shows_models_and_download_state(fake_model, capsys):
    spec = fake_model()

    assert main(["models", "list"]) == 0
    before = capsys.readouterr().out.splitlines()
    main(["models", "download", "fake"])
    capsys.readouterr()
    main(["models", "list"])
    after = capsys.readouterr().out.splitlines()

    assert before[0].split() == ["NAME", "STEMS", "LICENCE", "DOWNLOADED"]
    assert any(line.startswith("htdemucs_6s") for line in before)
    fake_before = next(line for line in before if line.startswith("fake"))
    fake_after = next(line for line in after if line.startswith("fake"))
    assert ",".join(spec.stems) in fake_before
    assert fake_before.endswith("no")
    assert fake_after.endswith("yes")


def test_download_fetches_and_prints_the_folder(fake_model, capsys):
    spec = fake_model()

    assert main(["models", "download", "fake"]) == 0

    assert capsys.readouterr().out.strip().endswith("models/fake")
    assert is_downloaded(spec)


def test_download_unknown_model(capsys):
    assert main(["models", "download", "nope"]) == 1
    assert "Unknown model 'nope'" in capsys.readouterr().err
