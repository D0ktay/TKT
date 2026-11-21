import json
import threading
from dataclasses import replace

import numpy as np
import pytest

from tkt.core.io import SchemaError, info_to_header, iter_frames, read_header, read_recording
from tkt.core.types import FINGERS, SCHEMA_VERSION, Finger, HandFrame, SourceInfo
from tkt.tools import record as rec
from tkt.tools.record import (
    RawSample,
    RawWriter,
    iter_raw,
    raw_header,
    read_raw,
    record,
    sibling_paths,
    write_raw,
)
from tkt.vision.tracker import RawHand

INFO = SourceInfo(SCHEMA_VERSION, 512, 640, 60.0, "right", "")


def hand(seed):
    rng = np.random.default_rng(seed)
    return RawHand(rng.random((21, 2)), rng.random((21, 3)) / 10, 0.93)


def frame(i, present=True):
    if not present:
        return HandFrame(10.0 + i / 60, i, False, {}, 0.0, (0.0, 0.0), 0.0, 0.0)
    fingers = {f: Finger((0.5, 0.4), None, 0.2, None) for f in FINGERS}
    return HandFrame(10.0 + i / 60, i, True, fingers, 0.7, (0.5, 0.6), 0.2, 0.9)


class FakeSource:
    def __init__(self, n=10, absent=()):
        self.info = INFO
        self.items = []
        for i in range(1, n + 1):
            present = i not in absent
            img = np.full((8, 6, 3), i, np.uint8)
            self.items.append((frame(i, present), hand(i) if present else None, img))

    def frames_raw(self):
        yield from self.items


# ------------------------------------------------------------------ ham biçim


def test_raw_roundtrip(tmp_path):
    p = tmp_path / "a.raw.jsonl"
    samples = [RawSample(1.5, 1, hand(1)), RawSample(1.52, 2, None), RawSample(1.54, 3, hand(3))]
    assert write_raw(p, INFO, samples) == 3
    info, back = read_raw(p)
    assert info == INFO
    assert [(s.t, s.frame_id, s.present) for s in back] == [(1.5, 1, True), (1.52, 2, False),
                                                              (1.54, 3, True)]
    np.testing.assert_allclose(back[0].hand.image_pts, hand(1).image_pts, atol=1e-5)
    np.testing.assert_allclose(back[0].hand.world_pts, hand(1).world_pts, atol=1e-6)
    assert back[0].hand.score == pytest.approx(0.93)
    assert back[0].hand.image_pts.shape == (21, 2) and back[0].hand.world_pts.shape == (21, 3)
    assert back[1].hand is None


def test_raw_file_layout(tmp_path):
    p = tmp_path / "a.raw.jsonl"
    write_raw(p, replace(INFO, note="n"), [RawSample(1.0, 7, hand(1)), RawSample(1.1, 8, None)])
    lines = p.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert header == {**info_to_header(replace(INFO, note="n")), "raw": True}
    row = json.loads(lines[1])
    assert set(row) == {"t", "frame_id", "present", "image_pts", "world_pts", "score"}
    assert len(row["image_pts"]) == 21 and len(row["image_pts"][0]) == 2
    assert len(row["world_pts"]) == 21 and len(row["world_pts"][0]) == 3
    assert json.loads(lines[2])["present"] is False


def test_raw_header_matches_info_to_header():
    assert raw_header(INFO) == {**info_to_header(INFO), "raw": True}


def test_raw_reader_rejects_non_raw_and_bad_schema(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps(info_to_header(INFO)) + "\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="raw"):
        read_raw(p)
    p.write_text(json.dumps({**raw_header(INFO), "schema": 99}) + "\n", encoding="utf-8")
    with pytest.raises(SchemaError):
        read_raw(p)


def test_raw_reader_reports_bad_line(tmp_path):
    p = tmp_path / "x.raw.jsonl"
    p.write_text(json.dumps(raw_header(INFO)) + "\n" + '{"t": 1}\n', encoding="utf-8")
    with pytest.raises(Exception, match=":2"):
        list(iter_raw(p))


def test_raw_writer_context_closes(tmp_path):
    with RawWriter(tmp_path / "w.raw.jsonl", INFO) as w:
        w.write(1.0, 1, None)
    assert w._fh is None


# ------------------------------------------------------------------ record


def test_sibling_paths(tmp_path):
    out, raw, video = sibling_paths(tmp_path / "ad.jsonl")
    assert (out.name, raw.name, video.name) == ("ad.jsonl", "ad.raw.jsonl", "ad.mp4")


def test_record_writes_core_io_readable_file(tmp_path):
    out = tmp_path / "sub" / "ad.jsonl"
    n = record(FakeSource(10, absent={4}), out, note="deneme", raw=True)
    assert n == 10
    info, frames = read_recording(out)
    assert info.note == "deneme" and info.width == 512 and info.hand == "right"
    assert [f.frame_id for f in frames] == list(range(1, 11))
    assert not frames[3].present and frames[3].fingers == {}
    assert frames[0].present and set(frames[0].fingers) == set(FINGERS)
    rinfo, raws = read_raw(tmp_path / "sub" / "ad.raw.jsonl")
    assert rinfo == info
    assert [s.frame_id for s in raws] == list(range(1, 11))
    assert [s.present for s in raws] == [f.present for f in frames]
    assert [s.t for s in raws] == [pytest.approx(f.t, abs=1e-6) for f in frames]
    assert not (tmp_path / "sub" / "ad.mp4").exists()


def test_record_without_raw_writes_no_raw_file(tmp_path):
    record(FakeSource(3), tmp_path / "a.jsonl")
    assert (tmp_path / "a.jsonl").exists() and not (tmp_path / "a.raw.jsonl").exists()
    assert read_header(tmp_path / "a.jsonl").note == ""


def test_record_duration_stops_by_frame_time(tmp_path):
    # kareler 1/60 sn arayla; 0,09 sn -> 7 kare (t farkı >= 0,09 olan ilk kareyle durur)
    n = record(FakeSource(60), tmp_path / "a.jsonl", duration=0.09)
    assert n == 7
    assert len(list(iter_frames(tmp_path / "a.jsonl"))) == 7


def test_record_stop_event(tmp_path):
    stop = threading.Event()
    src = FakeSource(10)

    def gen():
        for i, item in enumerate(src.items):
            if i == 3:
                stop.set()
            yield item

    src.frames_raw = gen
    assert record(src, tmp_path / "a.jsonl", stop=stop) == 4


def test_record_keyboard_interrupt_closes_cleanly(tmp_path):
    src = FakeSource(10)

    def gen():
        yield src.items[0]
        yield src.items[1]
        raise KeyboardInterrupt

    src.frames_raw = gen
    assert record(src, tmp_path / "a.jsonl", raw=True) == 2
    assert len(list(iter_frames(tmp_path / "a.jsonl"))) == 2
    assert len(read_raw(tmp_path / "a.raw.jsonl")[1]) == 2


def test_record_video_written(tmp_path):
    record(FakeSource(5), tmp_path / "a.jsonl", video=True)
    video = tmp_path / "a.mp4"
    assert video.exists() and video.stat().st_size > 0


def test_cli_rejects_prompt_and_from_raw():
    with pytest.raises(NotImplementedError, match="prompt"):
        rec.main(["--out", "x.jsonl", "--prompt", "5"])
    with pytest.raises(NotImplementedError, match="from-raw"):
        rec.main(["--out", "x.jsonl", "--from-raw", "a.raw.jsonl"])


def test_cli_main_records_with_live_source(tmp_path, monkeypatch, capsys):
    import tkt.vision.live_source as ls

    closed = []

    class Src(FakeSource):
        def close(self):
            closed.append(True)

    monkeypatch.setattr(ls, "open_live_source", lambda cfg: Src(4))
    out = tmp_path / "c.jsonl"
    rec.main(["--out", str(out), "--note", "x", "--raw", "--duration", "5"])
    assert closed == [True]
    assert read_header(out).note == "x"
    assert "4 kare" in capsys.readouterr().out
