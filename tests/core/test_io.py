import json
import operator
from dataclasses import replace

import pytest

from tkt.core.io import (
    Prompt,
    RecordFormatError,
    RecordingWriter,
    SchemaError,
    frame_to_dict,
    info_to_header,
    iter_frames,
    prompts_path,
    read_header,
    read_prompts,
    read_recording,
    write_prompts,
    write_recording,
)
from tkt.core.source import HandFrameSource
from tkt.core.types import FINGERS, SCHEMA_VERSION, Finger, HandFrame, SourceInfo

INFO = SourceInfo(schema=SCHEMA_VERSION, width=1280, height=720, fps=60.0, hand="right",
                  note="aci25 tap50")


def _frame(i: int, present: bool = True, lift: float | None = None) -> HandFrame:
    if not present:
        return HandFrame(t=10.0 + i / 60, frame_id=i, present=False, fingers={},
                         thumb_out=0.0, palm=(0.0, 0.0), hand_size=0.0, confidence=0.0)
    fingers = {
        name: Finger(img=(0.4 + 0.05 * k, 0.43 - 0.001 * i), surface=None,
                     curl=0.1 * k, lift=lift)
        for k, name in enumerate(FINGERS)
    }
    return HandFrame(t=10.0 + i / 60, frame_id=i, present=True, fingers=fingers,
                     thumb_out=0.91, palm=(0.50, 0.62), hand_size=0.21, confidence=0.97)


def _assert_same(a: HandFrame, b: HandFrame) -> None:
    assert a.frame_id == b.frame_id
    assert a.present == b.present
    assert a.t == pytest.approx(b.t, abs=1e-6)
    assert set(a.fingers) == set(b.fingers)
    for name in a.fingers:
        fa, fb = a.fingers[name], b.fingers[name]
        assert fa.img == pytest.approx(fb.img, abs=1e-5)
        assert fa.curl == pytest.approx(fb.curl, abs=1e-5)
        assert (fa.lift is None) == (fb.lift is None)
        if fa.lift is not None:
            assert fa.lift == pytest.approx(fb.lift, abs=1e-5)
        assert fa.surface == fb.surface
    if a.present:
        assert a.thumb_out == pytest.approx(b.thumb_out, abs=1e-5)
        assert a.palm == pytest.approx(b.palm, abs=1e-5)
        assert a.hand_size == pytest.approx(b.hand_size, abs=1e-5)
        assert a.confidence == pytest.approx(b.confidence, abs=1e-5)


def test_write_read_roundtrip(tmp_path):
    frames = [_frame(i) for i in range(5)] + [_frame(5, present=False), _frame(6, lift=0.42)]
    path = tmp_path / "rec.jsonl"
    assert write_recording(path, INFO, frames) == len(frames)

    info, got = read_recording(path)
    assert info == INFO
    assert len(got) == len(frames)
    for a, b in zip(frames, got, strict=True):
        _assert_same(a, b)
    assert got[5].fingers == {}
    assert all(isinstance(v, tuple) for v in (got[0].palm, got[0].fingers["index"].img))


@pytest.mark.parametrize(("field", "value", "compare", "threshold"), [
    ("curl", 0.349999, operator.lt, 0.35),
    ("curl", 0.550001, operator.gt, 0.55),
    ("lift", 0.25 - 1e-7, operator.lt, 0.25),
])
def test_roundtrip_preserves_threshold_decision(tmp_path, field, value, compare, threshold):
    frame = _frame(0)
    frame.fingers["index"] = replace(frame.fingers["index"], **{field: value})
    path = tmp_path / "esik.jsonl"
    write_recording(path, INFO, [frame])

    replayed, = iter_frames(path)
    replayed_value = getattr(replayed.fingers["index"], field)
    assert compare(replayed_value, threshold) == compare(value, threshold)
    assert replayed_value == value


def test_header_line_matches_contract(tmp_path):
    path = tmp_path / "rec.jsonl"
    write_recording(path, INFO, [_frame(0)])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"type": "header", "schema": 1, "width": 1280,
                                    "height": 720, "fps": 60.0, "hand": "right",
                                    "note": "aci25 tap50"}
    row = json.loads(lines[1])
    assert set(row) == {"t", "frame_id", "present", "fingers", "thumb_out", "palm",
                        "hand_size", "confidence"}
    assert set(row["fingers"]["index"]) == {"img", "surface", "curl", "lift"}


def test_reads_recording_with_integer_fps_and_optional_finger_fields(tmp_path):
    path = tmp_path / "ornek.jsonl"
    path.write_text(
        '{"type":"header","schema":1,"width":1280,"height":720,"fps":60,"hand":"right",'
        '"note":"aci25 tap50"}\n'
        '{"t":12.3456,"frame_id":741,"present":true,"fingers":{"index":{"img":[0.512,0.431],'
        '"surface":null,"curl":0.08,"lift":null},"middle":{"img":[0.55,0.43],"curl":0.1},'
        '"ring":{"img":[0.58,0.44],"curl":0.2},"pinky":{"img":[0.62,0.45],"curl":0.3}},'
        '"thumb_out":0.91,"palm":[0.50,0.62],'
        '"hand_size":0.21,"confidence":0.97}\n'
        "\n",
        encoding="utf-8",
    )
    info, frames = read_recording(path)
    assert info.fps == 60.0 and info.note == "aci25 tap50"
    assert len(frames) == 1
    f = frames[0]
    assert f.frame_id == 741 and f.present
    assert f.fingers["index"].img == (0.512, 0.431)
    assert f.fingers["index"].lift is None
    assert f.fingers["middle"].surface is None
    assert f.fingers["middle"].lift is None


def test_writer_context_manager_streams(tmp_path):
    path = tmp_path / "sub" / "rec.jsonl"
    with RecordingWriter(path, INFO, flush_every=2) as w:
        for i in range(3):
            w.write(_frame(i))
    assert w.count == 3
    assert [f.frame_id for f in iter_frames(path)] == [0, 1, 2]


def test_wrong_schema_rejected(tmp_path):
    path = tmp_path / "eski.jsonl"
    path.write_text('{"type":"header","schema":0,"width":1,"height":1,"fps":60,"hand":"right"}\n',
                    encoding="utf-8")
    with pytest.raises(SchemaError, match="şema sürümü"):
        read_header(path)
    with pytest.raises(SchemaError):
        list(iter_frames(path))


@pytest.mark.parametrize(("field", "value"), [
    ("schema", True),
    ("schema", 1.0),
    ("width", 0),
    ("width", True),
    ("width", 1.5),
    ("height", -1),
    ("height", "720"),
    ("fps", 0),
    ("fps", -1),
    ("fps", True),
    ("fps", float("nan")),
    ("fps", float("inf")),
    ("hand", "unknown"),
])
def test_invalid_header_field_rejected(tmp_path, field, value):
    path = tmp_path / "baslik.jsonl"
    path.write_text(json.dumps({**info_to_header(INFO), field: value}) + "\n", encoding="utf-8")
    with pytest.raises(SchemaError) as exc:
        read_header(path)
    assert str(path) in str(exc.value)


def test_normal_readers_reject_raw_header(tmp_path):
    path = tmp_path / "ham.raw.jsonl"
    path.write_text(json.dumps({**info_to_header(INFO), "raw": True}) + "\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="ham kayıt; --from-raw ile dönüştürün"):
        read_header(path)
    with pytest.raises(SchemaError, match="ham kayıt; --from-raw ile dönüştürün"):
        list(iter_frames(path))


def test_missing_header_rejected(tmp_path):
    path = tmp_path / "bas.jsonl"
    path.write_text('{"t":1.0,"frame_id":0,"present":false,"fingers":{}}\n', encoding="utf-8")
    with pytest.raises(SchemaError, match="başlık"):
        read_header(path)


def test_empty_file_rejected(tmp_path):
    path = tmp_path / "bos.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(SchemaError):
        read_header(path)


def test_writer_refuses_other_schema(tmp_path):
    info = SourceInfo(schema=SCHEMA_VERSION + 1, width=1, height=1, fps=60, hand="right")
    with pytest.raises(SchemaError):
        RecordingWriter(tmp_path / "x.jsonl", info)


def test_bad_frame_line_reports_line_number(tmp_path):
    path = tmp_path / "bozuk.jsonl"
    write_recording(path, INFO, [_frame(0)])
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"t": 1.0}\n')
    with pytest.raises(RecordFormatError, match=":3:"):
        list(iter_frames(path))


@pytest.mark.parametrize(("keys", "value"), [
    (("t",), float("nan")),
    (("frame_id",), float("inf")),
    (("thumb_out",), float("-inf")),
    (("palm",), [float("inf"), 0.5]),
    (("hand_size",), float("nan")),
    (("confidence",), float("nan")),
    (("confidence",), -0.01),
    (("confidence",), 1.01),
    (("fingers",), ["index"]),
    (("fingers", "index"), None),
    (("fingers", "index", "img"), [0.5, float("inf")]),
    (("fingers", "index", "surface"), [float("nan"), 0.5]),
    (("fingers", "index", "curl"), float("inf")),
    (("fingers", "index", "lift"), float("nan")),
])
def test_invalid_frame_field_reports_file_and_line(tmp_path, keys, value):
    path = tmp_path / "bozuk.jsonl"
    write_recording(path, INFO, [_frame(0)])
    row = frame_to_dict(_frame(1))
    target = row
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n" + json.dumps(row) + "\n")

    with pytest.raises(RecordFormatError) as exc:
        list(iter_frames(path))
    assert str(exc.value).startswith(f"{path}:4:")


def test_present_frame_requires_all_fingers(tmp_path):
    frame = _frame(0)
    del frame.fingers["pinky"]
    path = tmp_path / "eksik.jsonl"
    path.write_text(
        json.dumps(info_to_header(INFO)) + "\n" + json.dumps(frame_to_dict(frame)) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RecordFormatError) as exc:
        list(iter_frames(path))
    assert str(exc.value).startswith(f"{path}:2:")
    assert "pinky" in str(exc.value)


def test_reads_absent_frame_with_omitted_fields(tmp_path):
    path = tmp_path / "el-yok.jsonl"
    path.write_text(
        json.dumps(info_to_header(INFO)) + '\n{"t":10,"frame_id":0,"present":false}\n',
        encoding="utf-8",
    )
    assert list(iter_frames(path)) == [_frame(0, present=False)]


def test_unknown_finger_rejected(tmp_path):
    path = tmp_path / "parmak.jsonl"
    write_recording(path, INFO, [])
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"t":1,"frame_id":0,"present":true,"fingers":{"thumb":{"img":[0,0],'
                 '"surface":null,"curl":0,"lift":null}},"thumb_out":0,"palm":[0,0],'
                 '"hand_size":0.2,"confidence":1}\n')
    with pytest.raises(RecordFormatError):
        list(iter_frames(path))


def test_prompts_roundtrip(tmp_path):
    rec = tmp_path / "20250610_tap50_aci25_a.jsonl"
    side = prompts_path(rec)
    assert side.name == "20250610_tap50_aci25_a.prompts.json"
    assert prompts_path(tmp_path / "ad.raw.jsonl").name == "ad.prompts.json"

    prompts = [Prompt(t=12.30, kind="tap1"), Prompt(t=14.31, kind="tap2")]
    write_prompts(side, prompts)
    assert read_prompts(side) == prompts
    assert json.loads(side.read_text(encoding="utf-8")) == [
        {"t": 12.30, "kind": "tap1"}, {"t": 14.31, "kind": "tap2"}]


def test_prompts_bad_kind(tmp_path):
    with pytest.raises(ValueError):
        write_prompts(tmp_path / "x.prompts.json", [Prompt(t=1.0, kind="tap3")])
    p = tmp_path / "y.prompts.json"
    p.write_text('[{"t": 1.0, "kind": "tap3"}]', encoding="utf-8")
    with pytest.raises(RecordFormatError):
        read_prompts(p)


def test_source_protocol_runtime_check():
    class Fake:
        info = INFO

        def frames(self):
            yield _frame(0)

        def close(self):
            pass

    assert isinstance(Fake(), HandFrameSource)
    assert not isinstance(object(), HandFrameSource)
