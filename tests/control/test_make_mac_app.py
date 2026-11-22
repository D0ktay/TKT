"""TKT.app kabuğu: dosya yapısı ve başlatıcı (uygulama çalıştırılmaz)."""
import os
import plistlib

import pytest

from tkt.tools import make_mac_app as m


def test_build_writes_bundle(tmp_path):
    repo = tmp_path / "repo with space"
    python = repo / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    app = m.build(tmp_path / "Apps", repo=repo)
    assert app == tmp_path / "Apps/TKT.app"
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    assert info["CFBundleExecutable"] == "TKT" and info["LSUIElement"] is True
    assert info["NSCameraUsageDescription"] and info["NSAppleEventsUsageDescription"]
    launcher = app / "Contents/MacOS/TKT"
    assert os.access(launcher, os.X_OK)
    text = launcher.read_text(encoding="utf-8")
    assert f'cd "{repo}"' in text and f'exec "{python}" -u -m tkt.app.auto_mode' in text
    assert "Library/Logs/TKT.log" in text
    assert m.build(tmp_path / "Apps", repo=repo) == app        # üstüne yazılabilir


def test_build_needs_the_venv(tmp_path):
    with pytest.raises(FileNotFoundError):
        m.build(tmp_path / "Apps", repo=tmp_path / "bos")
