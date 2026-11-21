import json
import threading
from dataclasses import replace

import numpy as np
import pytest

from tkt.core.io import (
    Prompt,
    SchemaError,
    info_to_header,
    iter_frames,
    prompts_path,
    read_header,
    read_prompts,
    read_recording,
    write_prompts,
)
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


# ------------------------------------------------------------------ --prompt


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class ClockedSource:
    """Her kare öncesi sahte saati `step` ilerleten sonsuz sahte kaynak."""

    def __init__(self, clock, step=0.5, absent=()):
        self.info = INFO
        self._clock = clock
        self._step = step
        self._absent = set(absent)

    def frames_raw(self):
        i = 0
        while True:
            i += 1
            self._clock.now += self._step
            present = i not in self._absent
            img = np.zeros((4, 4, 3), np.uint8)
            yield frame(i, present), hand(i) if present else None, img


def test_record_prompt_writes_beep_times_and_stops_after_tail(tmp_path):
    clock = FakeClock()
    beeps = []
    out = tmp_path / "p.jsonl"
    src = ClockedSource(clock)
    n = record(src, out, raw=True, prompt=3, kind="tap2", clock=clock,
               beep=lambda: beeps.append(clock.now))
    prompts = read_prompts(prompts_path(out))
    assert len(prompts) == 3 and all(p.kind == "tap2" for p in prompts)
    # ilk kare 100.5; ilk bip +3 sn, sonrakiler 2 sn arayla; bip anıyla aynı saat
    assert [p.t for p in prompts] == pytest.approx([103.5, 105.5, 107.5])
    assert beeps == pytest.approx([p.t for p in prompts])
    # son bipten (107.5) 2 sn sonra (109.5) durur: kare 19
    assert n == 19
    assert len(list(iter_frames(out))) == 19
    assert prompts_path(out) == tmp_path / "p.prompts.json"
    assert len(read_raw(tmp_path / "p.raw.jsonl")[1]) == 19


def test_record_prompt_duration_overrides_auto_stop(tmp_path):
    clock = FakeClock()
    out = tmp_path / "d.jsonl"
    # kareler 1/60 sn arayla (frame.t); 0,09 sn sonra durur, bip hiç gelmez
    n = record(ClockedSource(clock, step=0.01), out, duration=0.09, prompt=5, clock=clock,
               beep=lambda: None)
    assert n == 7
    assert read_prompts(prompts_path(out)) == []


def test_record_prompt_written_on_interrupt(tmp_path):
    clock = FakeClock()
    out = tmp_path / "k.jsonl"
    src = ClockedSource(clock)

    def gen():
        for i, item in enumerate(ClockedSource.frames_raw(src)):
            if i == 8:
                raise KeyboardInterrupt
            yield item

    src.frames_raw = gen
    record(src, out, prompt=4, clock=clock, beep=lambda: None)
    assert len(read_prompts(prompts_path(out))) == 1


def test_record_prompt_validates_args(tmp_path):
    with pytest.raises(ValueError):
        record(FakeSource(2), tmp_path / "a.jsonl", prompt=0)
    with pytest.raises(ValueError):
        record(FakeSource(2), tmp_path / "a.jsonl", prompt=2, kind="tap3")


def test_play_beep_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(rec, "_beep_mac", lambda: calls.append("mac"))
    monkeypatch.setattr(rec, "_beep_windows", lambda: calls.append("win"))
    monkeypatch.setattr(rec, "_beep_terminal", lambda: calls.append("term"))
    for platform in ("darwin", "win32", "linux"):
        monkeypatch.setattr(rec.sys, "platform", platform)
        rec.play_beep()
    assert calls == ["mac", "win", "term"]


def test_beep_mac_uses_nonblocking_popen(monkeypatch):
    popens = []

    class P:
        def __init__(self, args, **kw):
            popens.append(args)

        def poll(self):
            return None

    monkeypatch.setattr(rec.subprocess, "Popen", P)
    monkeypatch.setattr(rec, "_mac_procs", [])
    rec._beep_mac()
    assert popens == [["afplay", "/System/Library/Sounds/Tink.aiff"]]


def test_cli_prompt_passes_args(tmp_path, monkeypatch):
    import tkt.vision.live_source as ls

    seen = {}

    class Src(FakeSource):
        def close(self):
            pass

    def fake_record(source, out, note, duration, raw, video, stop, **kw):
        seen.update(kw)
        return 0

    monkeypatch.setattr(ls, "open_live_source", lambda cfg: Src(1))
    monkeypatch.setattr(rec, "record", fake_record)
    rec.main(["--out", str(tmp_path / "x.jsonl"), "--prompt", "5", "--kind", "tap2"])
    assert seen == {"prompt": 5, "kind": "tap2"}


# ------------------------------------------------------------------ --from-raw


def write_config(tmp_path, calib_path):
    cfg = tmp_path / "vision.toml"
    cfg.write_text(
        f'[hand]\nside = "right"\ncalibration = "{calib_path.as_posix()}"\n'
        "[lift]\nw_angle = 1.0\n", encoding="utf-8")
    return cfg


def make_raw(tmp_path, n=12, absent=(4, 5)):
    samples = [RawSample(10.0 + i / 60, i, None if i in absent else hand(i))
               for i in range(1, n + 1)]
    raw = tmp_path / "ad.raw.jsonl"
    write_raw(raw, replace(INFO, note="ham"), samples)
    return raw, samples


def test_from_raw_rebuilds_frames_readable_by_core_io(tmp_path, capsys):
    raw, samples = make_raw(tmp_path)
    cfg = write_config(tmp_path, tmp_path / "yok.json")   # dosya yok -> DEFAULT
    out = tmp_path / "ad.jsonl"
    rec.main(["--from-raw", str(raw), "--out", str(out), "--config", str(cfg)])
    info, frames = read_recording(out)
    assert info.note == "ham" and info.width == 512
    assert [f.frame_id for f in frames] == [s.frame_id for s in samples]
    assert [f.present for f in frames] == [s.present for s in samples]
    assert not frames[3].present and frames[3].fingers == {}
    assert all(f.fingers["index"].lift is None for f in frames if f.present)
    assert "12 kare" in capsys.readouterr().out

    # canlıdaki zincirle (prev_hand_size dahil) aynı sonuç
    prev = None
    _, stored = read_raw(raw)   # ham kayıt yuvarlanır; zincir yuvarlanmış değerlerle kurulur
    for s, f in zip(stored, frames, strict=True):
        ref = rec_build(s, prev)
        assert f.confidence == pytest.approx(ref.confidence, abs=1e-5)
        assert f.hand_size == pytest.approx(ref.hand_size, abs=1e-5)
        prev = ref.hand_size if ref.present else None


def rec_build(s, prev):
    from tkt.vision import features, hand_calib
    return features.build_frame(s.hand, s.t, s.frame_id, hand_calib.DEFAULT, INFO,
                                prev_hand_size=prev)


def test_from_raw_copies_prompts_sidecar(tmp_path):
    raw, _ = make_raw(tmp_path)
    write_prompts(prompts_path(raw), [Prompt(10.05, "tap1"), Prompt(10.1, "tap1")])
    cfg = write_config(tmp_path, tmp_path / "yok.json")
    out = tmp_path / "yeni" / "b.jsonl"
    rec.rebuild_from_raw(raw, out, cfg)
    assert read_prompts(prompts_path(out)) == read_prompts(prompts_path(raw))
    # bip yan dosyası yoksa oluşturulmaz
    raw2 = tmp_path / "z" / "c.raw.jsonl"
    write_raw(raw2, INFO, [RawSample(1.0, 1, None)])
    rec.rebuild_from_raw(raw2, tmp_path / "z" / "d.jsonl", cfg)
    assert not prompts_path(tmp_path / "z" / "d.jsonl").exists()


def tap_calib(tmp_path):
    from tkt.vision import hand_calib
    calib = replace(hand_calib.DEFAULT, tap_ref={f: 1.0 for f in FINGERS})
    path = tmp_path / "cal.json"
    hand_calib.save(calib, path)
    return path


def test_from_raw_continues_without_lift_when_estimator_not_implemented(tmp_path, capsys,
                                                                         monkeypatch):
    import tkt.vision.lift as lift_mod

    class NotReady:
        def __init__(self, calib, cfg, *, aspect=1.0):
            raise NotImplementedError

    monkeypatch.setattr(lift_mod, "LiftEstimator", NotReady)
    raw, samples = make_raw(tmp_path)
    cfg = write_config(tmp_path, tap_calib(tmp_path))
    out = tmp_path / "ad.jsonl"
    assert rec.rebuild_from_raw(raw, out, cfg) == len(samples)
    assert "uyarı" in capsys.readouterr().err
    assert all(f.fingers["index"].lift is None for f in iter_frames(out) if f.present)


def test_from_raw_uses_lift_when_tap_ref_and_estimator_available(tmp_path, monkeypatch):
    import tkt.vision.lift as lift_mod

    resets = []

    class Fake:
        def __init__(self, calib, cfg, *, aspect=1.0):
            pass

        def update(self, hand, t, curls):
            return {f: 0.25 for f in FINGERS}

        def reset(self):
            resets.append(1)

    monkeypatch.setattr(lift_mod, "LiftEstimator", Fake)
    raw, samples = make_raw(tmp_path)
    cfg = write_config(tmp_path, tap_calib(tmp_path))
    out = tmp_path / "ad.jsonl"
    rec.rebuild_from_raw(raw, out, cfg)
    frames = list(iter_frames(out))
    assert all(f.fingers["index"].lift == pytest.approx(0.25) for f in frames if f.present)
    assert len(resets) == 2    # iki el-yok kare


def test_from_raw_rejects_non_raw_file(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps(info_to_header(INFO)) + "\n", encoding="utf-8")
    with pytest.raises(SchemaError):
        rec.rebuild_from_raw(p, tmp_path / "o.jsonl", write_config(tmp_path, tmp_path / "y"))


def test_cli_from_raw_and_prompt_conflict(tmp_path):
    with pytest.raises(SystemExit):
        rec.main(["--out", "x.jsonl", "--from-raw", "a.raw.jsonl", "--prompt", "3"])


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


# ------------------------------------------------------------------ hata taraması düzeltmeleri


def test_from_raw_rejects_same_input_and_output(tmp_path):
    raw, samples = make_raw(tmp_path)
    before = raw.read_bytes()
    cfg = write_config(tmp_path, tmp_path / "yok.json")
    with pytest.raises(rec.OutputIsInputError):
        rec.rebuild_from_raw(raw, raw, cfg)
    with pytest.raises(rec.OutputIsInputError):     # aynı dosyaya farklı yazım
        rec.rebuild_from_raw(raw, tmp_path / "." / "ad.raw.jsonl", cfg)
    assert raw.read_bytes() == before


def test_from_raw_rejects_hard_link_and_symlink_output(tmp_path):
    import os

    raw, _ = make_raw(tmp_path)
    before = raw.read_bytes()
    cfg = write_config(tmp_path, tmp_path / "yok.json")
    link = tmp_path / "hard.jsonl"
    os.link(raw, link)
    with pytest.raises(rec.OutputIsInputError):
        rec.rebuild_from_raw(raw, link, cfg)
    sym = tmp_path / "sym.jsonl"
    try:
        sym.symlink_to(raw)
    except (OSError, NotImplementedError):
        sym = None
    if sym is not None:
        with pytest.raises(rec.OutputIsInputError):
            rec.rebuild_from_raw(raw, sym, cfg)
    assert raw.read_bytes() == before


def test_cli_from_raw_same_file_is_clean_error(tmp_path, capsys):
    raw, _ = make_raw(tmp_path)
    before = raw.read_bytes()
    with pytest.raises(SystemExit):
        rec.main(["--from-raw", str(raw), "--out", str(raw),
                  "--config", str(write_config(tmp_path, tmp_path / "yok.json"))])
    assert raw.read_bytes() == before
    assert "aynı dosya" in capsys.readouterr().err


def test_from_raw_failure_keeps_old_output_and_leaves_no_temp(tmp_path):
    raw, _ = make_raw(tmp_path)
    with raw.open("a", encoding="utf-8") as fh:
        fh.write("bozuk satır\n")
    out = tmp_path / "ad.jsonl"
    out.write_text("eski içerik\n", encoding="utf-8")
    cfg = write_config(tmp_path, tmp_path / "yok.json")
    with pytest.raises(rec.RecordFormatError):
        rec.rebuild_from_raw(raw, out, cfg)
    assert out.read_text(encoding="utf-8") == "eski içerik\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        ["ad.raw.jsonl", "ad.jsonl", "vision.toml"])


def _interrupt_after(monkeypatch, cls, n):
    """`cls.write` n'inci çağrıdan sonra (satır yazıldıktan hemen sonra) KeyboardInterrupt atar."""
    orig = cls.write
    calls = []

    def write(self, *a, **kw):
        orig(self, *a, **kw)
        calls.append(1)
        if len(calls) == n:
            self.close()
            raise KeyboardInterrupt

    monkeypatch.setattr(cls, "write", write)


@pytest.mark.parametrize(("cls_name", "common"), [("RecordingWriter", 3), ("RawWriter", 2)])
def test_record_interrupt_keeps_both_files_consistent(tmp_path, monkeypatch, cls_name, common):
    _interrupt_after(monkeypatch, getattr(rec, cls_name), 3)
    out = tmp_path / "a.jsonl"
    n = record(FakeSource(10), out, raw=True)
    frames = list(iter_frames(out))
    raws = read_raw(tmp_path / "a.raw.jsonl")[1]
    ids = list(range(1, common + 1))
    assert n == len(frames) == len(raws) == common
    assert [f.frame_id for f in frames] == [r.frame_id for r in raws] == ids


def test_record_removes_stale_prompts_sidecar_without_prompt(tmp_path, capsys):
    out = tmp_path / "a.jsonl"
    write_prompts(prompts_path(out), [Prompt(1.0, "tap1")])
    record(FakeSource(3), out)
    assert not prompts_path(out).exists()
    assert "prompts" in capsys.readouterr().err


def test_record_replaces_stale_prompts_sidecar_with_prompt(tmp_path):
    clock = FakeClock()
    out = tmp_path / "a.jsonl"
    write_prompts(prompts_path(out), [Prompt(1.0, "tap1")] * 5)
    record(ClockedSource(clock), out, prompt=2, clock=clock, beep=lambda: None)
    assert len(read_prompts(prompts_path(out))) == 2


class SteppedSource:
    """Her karede sahte saati `steps[i]` kadar ilerleten sahte kaynak."""

    def __init__(self, clock, steps):
        self.info = INFO
        self._clock = clock
        self._steps = steps

    def frames_raw(self):
        for i, step in enumerate(self._steps, start=1):
            self._clock.now += step
            yield frame(i), hand(i), np.zeros((4, 4, 3), np.uint8)


def test_record_prompt_late_beeps_are_not_bunched(tmp_path):
    clock = FakeClock()
    out = tmp_path / "g.jsonl"
    # ilk bip 103.5'te; sonraki kare 8 sn sonra (111.5); ardından 0,5 sn adımlar
    steps = [0.5, 3.0, 8.0] + [0.5] * 12
    record(SteppedSource(clock, steps), out, prompt=3, clock=clock, beep=lambda: None)
    ts = [p.t for p in read_prompts(prompts_path(out))]
    assert ts[0] == pytest.approx(103.5)
    assert ts[1] == pytest.approx(111.5)           # gecikmiş bip, boşluktan sonraki ilk kare
    assert ts[2] - ts[1] == pytest.approx(rec.PROMPT_INTERVAL_S, abs=0.5)
    assert ts[2] - ts[1] >= rec.PROMPT_INTERVAL_S  # üst üste (20 ms arayla) değil


def test_from_raw_passes_aspect_from_raw_header(tmp_path, monkeypatch):
    import tkt.vision.lift as lift_mod

    seen = {}

    class Fake:
        def __init__(self, calib, cfg, *, aspect=1.0):
            seen["aspect"] = aspect

        def update(self, hand, t, curls):
            return {f: 0.1 for f in FINGERS}

        def reset(self):
            pass

    monkeypatch.setattr(lift_mod, "LiftEstimator", Fake)
    raw, _ = make_raw(tmp_path)        # başlık: width=512, height=640
    cfg = write_config(tmp_path, tap_calib(tmp_path))
    rec.rebuild_from_raw(raw, tmp_path / "ad.jsonl", cfg)
    assert seen["aspect"] == pytest.approx(512 / 640)


def test_from_raw_real_estimator_gets_aspect(tmp_path, monkeypatch):
    import tkt.vision.lift as lift_mod

    made = []
    real = lift_mod.LiftEstimator

    def spy(*a, **kw):
        made.append(real(*a, **kw))
        return made[-1]

    monkeypatch.setattr(lift_mod, "LiftEstimator", spy)
    raw, _ = make_raw(tmp_path)
    cfg = write_config(tmp_path, tap_calib(tmp_path))
    cfg.write_text(cfg.read_text(encoding="utf-8").replace(
        "w_angle = 1.0\n", "w_angle = 1.0\nw_length = 1.0\nbaseline_s = 1.0\n"
        "curl_rate_suppress = 3.0\n"), encoding="utf-8")
    rec.rebuild_from_raw(raw, tmp_path / "ad.jsonl", cfg)
    assert made and made[0].aspect == pytest.approx(512 / 640)


# ------------------------------------------------------------------ önizleme


def preview_source(n=60, absent=()):
    src = FakeSource(n, absent=absent)
    src.items = [(f, h, np.zeros((120, 160, 3), np.uint8)) for f, h, _ in src.items]
    return src


def test_preview_lead_frames_shown_but_not_recorded(tmp_path):
    shown = []
    stop = threading.Event()
    # kareler 1/60 sn arayla; 0,09 sn hazırlık -> ilk 6 kare (t farkı < 0,09) kayda girmez
    ps = rec.PreviewSource(preview_source(60, absent={3}), "deneme", 0.055, stop, lead=0.09,
                           show=lambda name, img: shown.append((name, img.shape)),
                           poll_key=lambda: 255)
    n = record(ps, tmp_path / "a.jsonl", duration=0.055)
    frames = list(iter_frames(tmp_path / "a.jsonl"))
    assert n == len(frames) == 5
    assert frames[0].frame_id == 7
    assert len(shown) == 6 + 5
    assert all(name == rec.PREVIEW_WINDOW and shape == (120, 160, 3) for name, shape in shown)


def test_preview_quit_key_stops_recording(tmp_path):
    keys = iter([255, 255, ord("q")])
    stop = threading.Event()
    ps = rec.PreviewSource(preview_source(30), "x", None, stop, lead=0.0,
                           show=lambda name, img: None, poll_key=lambda: next(keys, 255))
    assert record(ps, tmp_path / "a.jsonl", stop=stop) == 3


def test_preview_quit_during_lead_records_nothing(tmp_path):
    stop = threading.Event()
    ps = rec.PreviewSource(preview_source(30), "x", None, stop, lead=1.0,
                           show=lambda name, img: None, poll_key=lambda: 27)
    assert record(ps, tmp_path / "a.jsonl", stop=stop) == 0


def test_cli_preview_wraps_source(tmp_path, monkeypatch):
    import tkt.vision.live_source as ls

    seen = {}

    class Src(FakeSource):
        def close(self):
            pass

    def fake_record(source, out, note, duration, raw, video, stop, **kw):
        seen["source"] = source
        return 0

    monkeypatch.setattr(ls, "open_live_source", lambda cfg: Src(1))
    monkeypatch.setattr(rec, "record", fake_record)
    rec.main(["--out", str(tmp_path / "x.jsonl"), "--duration", "10", "--preview"])
    assert isinstance(seen["source"], rec.PreviewSource)
    rec.main(["--out", str(tmp_path / "x.jsonl"), "--duration", "10"])
    assert isinstance(seen["source"], Src)
