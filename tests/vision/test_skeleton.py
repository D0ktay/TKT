import importlib
import inspect
import tomllib
from pathlib import Path

import pytest

MODULES = [
    "tkt.vision.camera",
    "tkt.vision.tracker",
    "tkt.vision.hand_calib",
    "tkt.vision.features",
    "tkt.vision.lift",
    "tkt.vision.live_source",
    "tkt.vision.debug_view",
    "tkt.tools.record",
]

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_vision_does_not_import_control():
    for name in MODULES:
        src = inspect.getsource(importlib.import_module(name))
        assert "tkt.control" not in src and "tkt.app" not in src, name


def test_live_source_signatures():
    from tkt.vision import live_source

    assert list(inspect.signature(live_source.open_live_source).parameters) == ["config_path"]
    assert list(inspect.signature(live_source.LiveSource.__init__).parameters) == [
        "self", "camera", "tracker", "calib", "hand", "lift"]


def test_default_calibration_covers_fingers():
    from tkt.core.types import FINGERS
    from tkt.vision.hand_calib import DEFAULT

    assert set(DEFAULT.open_ref) == set(FINGERS) == set(DEFAULT.fist_ref)
    assert DEFAULT.tap_ref is None


def test_record_cli_options():
    from tkt.tools.record import build_parser

    a = build_parser().parse_args(
        ["--out", "x.jsonl", "--raw", "--video", "--prompt", "50", "--kind", "tap2"])
    assert a.raw and a.video and a.prompt == 50 and a.kind == "tap2"


def test_vision_config_complete():
    cfg = tomllib.loads((ROOT / "config" / "vision.toml").read_text(encoding="utf-8"))
    assert set(cfg) == {"camera", "tracker", "hand", "lift"}
    assert cfg["camera"]["fps"] == 60
    assert set(cfg["lift"]) == {"w_angle", "w_length", "baseline_s", "curl_rate_suppress"}
