import json

import pytest

pytest.importorskip("AppKit")          # yalnız macOS: araç Quartz/AppKit kullanır
pytest.importorskip("ApplicationServices")

from tkt.tools import auto_mode as am  # noqa: E402


@pytest.fixture
def args(tmp_path, monkeypatch):
    monkeypatch.setattr(am, "ARGS", am.ARGS.__class__(
        a_vision="config/vision.toml", c_vision="c.toml", control="config/control.toml",
        surface=str(tmp_path / "surface.json"), start="A"))
    return tmp_path


def test_profile_a_without_surface_uses_thumb_and_no_surface(args):
    vcfg, cfg, mirror, assist, surface = am.profile("A")
    assert vcfg == "config/vision.toml"
    assert cfg["general"]["click_method"] == "thumb"
    assert surface is None and not mirror and not assist
    assert cfg["surface_pointer"]["enabled"] is False


def test_profile_a_with_surface_enables_surface_pointer(args):
    (args / "surface.json").write_text(json.dumps({
        "version": 1, "corners_img": [[0.2, 0.25], [0.9, 0.27], [0.85, 0.5], [0.3, 0.55]],
        "aspect": 1.5, "image_aspect": 4 / 3}))
    _vcfg, cfg, _mirror, _assist, surface = am.profile("A")
    assert surface is not None
    assert cfg["surface_pointer"]["enabled"] is True


def test_profile_c_uses_index_pinch_mirrored_with_assist(args):
    vcfg, cfg, mirror, assist, surface = am.profile("C")
    assert vcfg == "c.toml"
    assert cfg["general"]["click_method"] == "index_pinch"
    assert mirror and assist and surface is None
