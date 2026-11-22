import ast
import sys
import tomllib
from importlib import import_module
from importlib.util import resolve_name
from pathlib import Path

import pytest
from tests.control.conftest import make_frame

from tkt.core.types import FINGERS, Finger, HandFrame

CONTROL_MODULES = (
    "tkt.control.sources",
    "tkt.control.filters",
    "tkt.control.pointer",
    "tkt.control.scroll",
    "tkt.control.tap",
    "tkt.control.gestures",
    "tkt.control.os_input.base",
    "tkt.control.os_input.dry_backend",
    "tkt.control.os_input.pynput_backend",
)
CLI_MODULES = ("tkt.app.main", "tkt.tools.replay", "tkt.tools.eval_taps")
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", (*CONTROL_MODULES, "tkt.app.config", *CLI_MODULES))
def test_module_imports(name, monkeypatch):
    monkeypatch.setitem(sys.modules, "pynput", None)
    import_module(name)


@pytest.mark.parametrize("name", CONTROL_MODULES)
def test_control_does_not_import_vision(name):
    module = import_module(name)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            package = resolve_name("." * node.level + (node.module or ""), module.__package__)
            imports.add(package)
            imports.update(f"{package}.{alias.name}" for alias in node.names)

    assert not any(name == "tkt.vision" or name.startswith("tkt.vision.") for name in imports)


def test_make_frame_defaults():
    frame = make_frame()
    assert isinstance(frame, HandFrame)
    assert frame.present
    assert tuple(frame.fingers) == FINGERS
    assert all(isinstance(finger, Finger) for finger in frame.fingers.values())
    assert all(0.0 <= finger.curl <= 1.0 for finger in frame.fingers.values())


def test_make_frame_accepts_overrides():
    frame = make_frame(
        t=2.5,
        frame_id=42,
        curls={"index": 0.3},
        lift={"index": 0.8},
        thumb_out=0.2,
        palm=(0.4, 0.5),
        index_pos=(0.3, 0.4),
        positions={"middle": (0.6, 0.4)},
        surfaces={"index": (0.2, 0.3)},
        hand_size=0.25,
        confidence=0.9,
    )
    assert frame == HandFrame(
        t=2.5,
        frame_id=42,
        present=True,
        fingers={
            "index": Finger(img=(0.3, 0.4), surface=(0.2, 0.3), curl=0.3, lift=0.8),
            "middle": Finger(img=(0.6, 0.4), surface=None, curl=0.0, lift=None),
            "ring": Finger(img=(0.5, 0.4), surface=None, curl=0.0, lift=None),
            "pinky": Finger(img=(0.5, 0.4), surface=None, curl=0.0, lift=None),
        },
        thumb_out=0.2,
        palm=(0.4, 0.5),
        hand_size=0.25,
        confidence=0.9,
    )


def test_make_frame_without_hand_has_no_fingers():
    frame = make_frame(present=False, curls={"index": 0.3})
    assert not frame.present
    assert frame.fingers == {}


def test_make_frame_has_no_pinch_by_default():
    assert make_frame().pinch is None


def test_make_frame_fills_missing_pinch_fingers_as_open():
    frame = make_frame(pinch={"index": 0.0})
    assert frame.pinch == {"index": 0.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0}


def test_make_frame_without_hand_has_no_pinch():
    assert make_frame(present=False, pinch={"index": 0.0}).pinch is None


def test_make_frame_accepts_shared_lift():
    frame = make_frame(lift=0.7)
    assert all(finger.lift == 0.7 for finger in frame.fingers.values())


def test_control_config_sections():
    with (ROOT / "config" / "control.toml").open("rb") as file:
        config = tomllib.load(file)
    assert set(config) == {
        "general", "mode", "thumb", "pointer", "scroll", "tap", "swipe", "pinch", "index_pinch",
        "pointer_front", "surface_pointer",
    }
    assert config["surface_pointer"]["enabled"] is False   # deney: varsayılan kapalı


@pytest.mark.parametrize("name", CLI_MODULES)
def test_cli_help(name, capsys):
    with pytest.raises(SystemExit) as exc:
        import_module(name).main(["--help"])
    assert exc.value.code == 0
    assert f"python -m {name}" in capsys.readouterr().out


def test_app_cli_options():
    from tkt.app.main import build_parser

    args = build_parser().parse_args(
        ["--source", "replay", "--file", "sample.jsonl", "--backend", "dry", "--fast", "--loop"]
    )
    assert args.source == "replay"
    assert args.file == Path("sample.jsonl")
    assert args.backend == "dry"
    assert args.fast and args.loop
