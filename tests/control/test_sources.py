from itertools import islice
from threading import Event
from unittest.mock import Mock

import pytest
from tests.control.conftest import make_frame

from tkt.control.sources import ReplaySource, SyntheticSource
from tkt.core.io import SchemaError, write_recording
from tkt.core.types import SCHEMA_VERSION, SourceInfo


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / "recording.jsonl"
    info = SourceInfo(SCHEMA_VERSION, 1280, 720, 20, "right")
    frames = [make_frame(t=10, frame_id=5), make_frame(t=10.25, frame_id=10, present=False)]
    write_recording(path, info, frames)
    return path, info, frames


def test_replay_round_trip_and_fast(recording, monkeypatch):
    path, info, frames = recording
    wait = Mock(side_effect=AssertionError("Fast playback waited"))
    monkeypatch.setattr(Event, "wait", wait)
    source = ReplaySource(path, fast=True)
    assert source.info == info
    assert list(source.frames()) == frames


def test_replay_paces_and_loops(recording, monkeypatch):
    path, _, _ = recording
    monkeypatch.setattr("tkt.control.sources.monotonic", lambda: 100.0)
    wait = Mock(return_value=False)
    monkeypatch.setattr(Event, "wait", wait)
    source = ReplaySource(path, loop=True)
    with_frames = source.frames()
    frames = list(islice(with_frames, 4))
    assert [f.t for f in frames] == pytest.approx([10, 10.25, 10.30, 10.55])
    assert [f.frame_id for f in frames] == [5, 10, 11, 16]
    assert [call.args[0] for call in wait.call_args_list] == pytest.approx([0, 0.25, 0.30, 0.55])
    source.close()
    assert list(with_frames) == []


def test_replay_rejects_wrong_schema(recording):
    path, _, _ = recording
    path.write_text('{"type":"header","schema":999}\n', encoding="utf-8")
    with pytest.raises(SchemaError, match="999"):
        ReplaySource(path)


def test_synthetic_script():
    source = SyntheticSource(fps=20)
    assert source.rest(0.1).open(2, 0.1).move(0.2, -0.1, 0.1) is source
    frames = list(
        source.thumb(True)
        .tap(("index", "middle"), ms=200)
        .curl_all()
        .vanish(0.1)
        .rest(0.1)
        .frames()
    )
    assert (source.info.width, source.info.height) == (1280, 720)
    assert all(f.curl == 1 for f in frames[0].fingers.values())
    assert [f.curl for f in frames[2].fingers.values()] == [0, 0, 1, 1]
    assert frames[4].palm == pytest.approx((0.6, 0.55))
    assert frames[5].palm == pytest.approx((0.7, 0.5))
    assert frames[6].thumb_out == 0
    assert [f.fingers["index"].lift for f in frames[8:12]] == [0.5, 1, 0.5, 0]
    assert frames[9].fingers["middle"].lift == 1
    assert all(f.curl == 1 for f in frames[12].fingers.values())
    assert all(not f.present and f.fingers == {} for f in frames[16:18])
    assert frames[-1].present and frames[-1].thumb_out == 1


def test_synthetic_noise_drops_and_disabled_lift():
    def script():
        return (
            SyntheticSource(noise_px=2, drop_rate=0.4, seed=7, with_lift=False)
            .open(1)
            .tap()
            .move(0.1, 0, 1)
        )

    frames = list(script().frames())
    assert frames == list(script().frames())
    assert any(b.frame_id - a.frame_id > 1 for a, b in zip(frames, frames[1:], strict=False))
    assert all(f.t == pytest.approx(f.frame_id / 60) for f in frames)
    assert all(finger.lift is None for f in frames for finger in f.fingers.values())
    assert (
        SyntheticSource().open(1).frames().__next__().fingers["index"].img
        != frames[0].fingers["index"].img
    )


def test_synthetic_realtime(monkeypatch):
    monkeypatch.setattr("tkt.control.sources.monotonic", lambda: 100.0)
    wait = Mock(return_value=False)
    monkeypatch.setattr(Event, "wait", wait)
    frames = list(SyntheticSource(fps=20, realtime=True).rest(0.1).frames())
    assert [f.t for f in frames] == pytest.approx([100, 100.05])
    assert [call.args[0] for call in wait.call_args_list] == pytest.approx([0, 0.05])
