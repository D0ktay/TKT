import json
from pathlib import Path

import pytest

from tkt.control.sources import SyntheticSource
from tkt.core.io import Prompt, prompts_path, write_prompts, write_recording
from tkt.tools import eval_taps

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "control.toml"


@pytest.fixture
def recordings(tmp_path, monkeypatch):
    prompted = tmp_path / "taps.jsonl"
    quiet = tmp_path / "quiet.jsonl"
    source = (
        SyntheticSource(fps=20)
        .open(1, 0.5)
        .tap(ms=200)
        .rest(1.3)
        .open(2, 0.2)
        .tap(("index", "middle"), ms=200)
        .rest(2)
    )
    write_recording(prompted, source.info, source.frames())
    write_prompts(
        prompts_path(prompted),
        [Prompt(2.1, "tap2"), Prompt(0.45, "tap1"), Prompt(0.50, "tap1")],
    )
    source = SyntheticSource(fps=20).rest(4)
    write_recording(quiet, source.info, source.frames())
    configs = []

    class FakeEngine:
        def __init__(self, cfg, backend, info):
            configs.append((cfg["general"]["click_method"], cfg["tap"]["lift_up"],
                            cfg["tap"]["min_ms"]))
            self.backend = backend

        def process(self, frame):
            assert frame.t == frame.frame_id / 20
            if frame.frame_id in (9, 13):
                self.backend.move_rel(10, 0)
            if frame.frame_id == 10:
                self.backend.move_rel(4, 3)
            if frame.frame_id == 13:
                self.backend.move_rel(-4, -1)
            button = {13: "left", 47: "right", 75: "right"}.get(frame.frame_id)
            if button:
                self.backend.button(button, True)
                self.backend.button(button, False)
            if frame.frame_id == 13:
                # Aynı karede tıktan sonraki hareket kaymaya dahil edilmez.
                self.backend.move_rel(20, 0)

    monkeypatch.setattr(eval_taps, "GestureEngine", FakeEngine)
    return prompted, quiet, configs


def test_json_replays_multiple_recordings(recordings, capsys):
    prompted, quiet, configs = recordings
    eval_taps.main([
        str(prompted), str(quiet), "--config", str(_CONFIG),
        "--click-method", "both", "--set", "tap.lift_up=0.6", "--json",
    ])
    report = json.loads(capsys.readouterr().out)
    assert configs == [("both", 0.6, 50)] * 2
    assert report["capture"] == {
        "tap1": {"caught": 1, "total": 2, "rate": 0.5},
        "tap2": {"caught": 1, "total": 1, "rate": 1.0},
    }
    taps, unprompted = report["recordings"]
    assert taps["misses"] == [{"kind": "tap1", "t": 0.5, "frame_id": 10}]
    assert [item["drift_px"] for item in taps["captures"]] == [pytest.approx(104**0.5), 0]
    assert report["latency_ms"] == pytest.approx({"median": 225, "p95": 250})
    assert [taps["false_clicks"][key] for key in ("left", "right")] == [0, 1]
    assert [unprompted["false_clicks"][key] for key in ("left", "right")] == [1, 2]
    false = report["false_clicks"]
    assert false["duration_s"] == pytest.approx(4.35 + 3.95)
    assert false["outside_windows_s"] == pytest.approx(false["duration_s"] - 3.05)
    assert false["per_minute"] == pytest.approx(4 * 60 / false["outside_windows_s"])
    assert report["gate"] == {
        "passed": False,
        "checks": {"tap1": False, "tap2": True, "false_clicks": False, "drift": False},
    }


def test_text_with_explicit_prompts_and_grid(recordings, tmp_path, capsys):
    prompted, _, configs = recordings
    sidecar = tmp_path / "custom.json"
    write_prompts(sidecar, [Prompt(0.45, "tap1"), Prompt(10, "tap2")])
    args = [str(prompted), "--config", str(_CONFIG), "--prompts", str(sidecar)]
    eval_taps.main(args)
    output = capsys.readouterr().out
    assert "tap1 1/1" in output
    assert "Kaçan tap2 t=10.000000 frame_id=None" in output
    assert "Kapı KALDI" in output

    eval_taps.main([*args, "--grid", "lift_up=0.4,0.6", "min_ms=40,50"])
    output = capsys.readouterr().out.splitlines()
    assert len(output) == 4
    assert output[0].startswith("tap.lift_up=0.4 tap.min_ms=40 | ")
    assert configs == [("tap", 0.5, 50), ("tap", 0.4, 40), ("tap", 0.4, 50),
                       ("tap", 0.6, 40), ("tap", 0.6, 50)]


def test_real_engine_passes_gate_on_clean_synthetic_taps(tmp_path, capsys):
    recording = tmp_path / "taps.jsonl"
    source = (
        SyntheticSource()
        .open(1, 0.5)
        .tap()
        .rest(1.35)
        .open(2, 0.5)
        .tap(("index", "middle"))
        .rest(1.5)
    )
    write_recording(recording, source.info, source.frames())
    write_prompts(prompts_path(recording), [Prompt(0.4, "tap1"), Prompt(2.4, "tap2")])

    eval_taps.main([str(recording), "--config", str(_CONFIG), "--json"])

    report = json.loads(capsys.readouterr().out)
    assert report["gate"]["passed"]
    assert report["drift_px"] == {"count": 2, "median": 0, "max": 0}
