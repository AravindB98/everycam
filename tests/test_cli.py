import os

import pytest

from everycam.cli import main


def test_version_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_demo_command(tmp_path):
    rc = main(
        ["demo", "--episodes", "2", "--frames", "20", "--epochs", "40",
         "--out", str(tmp_path / "d")]
    )
    assert rc == 0
    assert os.path.exists(str(tmp_path / "d" / "dataset" / "meta" / "info.json"))


def test_info_command(demo_dataset, capsys):
    rc = main(["info", demo_dataset])
    assert rc == 0
    assert "total_episodes" in capsys.readouterr().out
