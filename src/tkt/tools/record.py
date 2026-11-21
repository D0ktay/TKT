"""Kayıt aracı (V1). PART2 2.4.

    python -m tkt.tools.record --out data/samples/ad.jsonl --note "..." --duration 30 --raw
    python -m tkt.tools.record --out ad.jsonl --prompt 50 --kind tap1 --raw --video
    python -m tkt.tools.record --from-raw ad.raw.jsonl --out ad.jsonl

`--prompt N --kind tap1|tap2`: kayıt sırasında bip verir; her bipin zamanı `HandFrame.t` ile aynı
saatte (`time.monotonic`) `ad.prompts.json` dosyasına yazılır (`core/io.write_prompts`).
`--from-raw`: ham kayıttan güncel `features` ve (varsa) `lift` ile HandFrame kaydını yeniden üretir.

Ham kayıt (`ad.raw.jsonl`):
    1. satır: `core/io.info_to_header` alanları + {"raw": true}
    sonraki her satır:
        {"t", "frame_id", "present", "image_pts" (21x2), "world_pts" (21x3), "score"}
    El yokken `image_pts` ve `world_pts` null, `score` 0'dır.
Bu biçimin okuma-yazma yardımcıları burada: `RawWriter`, `write_raw`, `iter_raw`, `read_raw`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType
from typing import IO, Any

import numpy as np

from tkt.core.io import (
    Prompt,
    RecordFormatError,
    RecordingWriter,
    SchemaError,
    header_to_info,
    info_to_header,
    prompts_path,
    write_prompts,
)
from tkt.core.types import SourceInfo
from tkt.vision.tracker import RawHand

N_POINTS = 21
_T_DIGITS = 6
_IMG_DIGITS = 5
_WORLD_DIGITS = 6

# Bip zamanlaması (config anahtarı değil, modül sabiti).
PROMPT_LEAD_S = 3.0       # kayıt başladıktan ilk bipe kadar hazırlık
PROMPT_INTERVAL_S = 2.0   # bipler arası
PROMPT_TAIL_S = 2.0       # son bipten sonra kayıt devam eder
# Önizleme penceresi (config anahtarı değil, modül sabiti).
PREVIEW_LEAD_S = 3.0      # pencere açıldıktan kayıt başlayana kadar hazırlık (kaydedilmez)
PREVIEW_WINDOW = "TKT kayit"
_PREVIEW_QUIT = (ord("q"), 27)
_MAC_BEEP_SOUND = "/System/Library/Sounds/Tink.aiff"
_WIN_BEEP_HZ = 880
_WIN_BEEP_MS = 150


# --------------------------------------------------------------------------- ham kayıt


@dataclass(frozen=True, eq=False)
class RawSample:
    """Ham kaydın bir satırı: zaman, kare numarası ve (varsa) ham el."""

    t: float
    frame_id: int
    hand: RawHand | None

    @property
    def present(self) -> bool:
        return self.hand is not None


def raw_header(info: SourceInfo) -> dict[str, Any]:
    """Ham kayıt başlığı: `info_to_header` alanları ve "raw": true."""
    return {**info_to_header(info), "raw": True}


def raw_to_dict(sample: RawSample) -> dict[str, Any]:
    h = sample.hand
    return {
        "t": round(float(sample.t), _T_DIGITS),
        "frame_id": int(sample.frame_id),
        "present": h is not None,
        "image_pts": None if h is None else np.round(h.image_pts, _IMG_DIGITS).tolist(),
        "world_pts": None if h is None else np.round(h.world_pts, _WORLD_DIGITS).tolist(),
        "score": 0.0 if h is None else round(float(h.score), _IMG_DIGITS),
    }


def raw_from_dict(d: dict[str, Any]) -> RawSample:
    t = float(d["t"])
    frame_id = int(d["frame_id"])
    if not d["present"]:
        return RawSample(t, frame_id, None)
    image_pts = np.asarray(d["image_pts"], dtype=np.float64)
    world_pts = np.asarray(d["world_pts"], dtype=np.float64)
    if image_pts.shape != (N_POINTS, 2) or world_pts.shape != (N_POINTS, 3):
        raise ValueError(
            f"nokta boyutu hatalı: image_pts {image_pts.shape}, world_pts {world_pts.shape}")
    return RawSample(t, frame_id, RawHand(image_pts, world_pts, float(d["score"])))


class RawWriter:
    """Ham kaydı satır satır yazar; bağlam yöneticisidir."""

    def __init__(self, path: str | Path, info: SourceInfo, flush_every: int = 60) -> None:
        self.path = Path(path)
        self.info = info
        self.count = 0
        self._flush_every = max(1, flush_every)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] | None = self.path.open("w", encoding="utf-8", newline="\n")
        self._write_line(raw_header(info))

    def _write_line(self, obj: dict[str, Any]) -> None:
        assert self._fh is not None, "yazıcı kapalı"
        self._fh.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def write(self, t: float, frame_id: int, hand: RawHand | None) -> None:
        self._write_line(raw_to_dict(RawSample(t, frame_id, hand)))
        self.count += 1
        if self.count % self._flush_every == 0:
            assert self._fh is not None
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> RawWriter:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()


def write_raw(path: str | Path, info: SourceInfo, samples: Iterable[RawSample]) -> int:
    """Tüm ham örnekleri yazar, yazılan satır sayısını döndürür."""
    with RawWriter(path, info) as w:
        for s in samples:
            w.write(s.t, s.frame_id, s.hand)
        return w.count


def _parse_raw_header(first: str, source: str) -> SourceInfo:
    if not first.strip():
        raise SchemaError(f"{source}: dosya boş")
    try:
        d = json.loads(first)
    except json.JSONDecodeError as e:
        raise SchemaError(f"{source}: başlık JSON değil: {e}") from e
    info = header_to_info(d, source)
    if d.get("raw") is not True:
        raise SchemaError(f"{source}: başlıkta \"raw\": true yok; bu bir ham kayıt değil")
    return info


def read_raw_header(path: str | Path) -> SourceInfo:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        return _parse_raw_header(fh.readline(), str(p))


def iter_raw(path: str | Path) -> Iterator[RawSample]:
    """Ham örnekleri sırayla üretir. Başlığı doğrular, boş satırları atlar."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        _parse_raw_header(fh.readline(), str(p))
        for lineno, line in enumerate(fh, start=2):
            if not line.strip():
                continue
            try:
                yield raw_from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                raise RecordFormatError(f"{p}:{lineno}: ham satır okunamadı: {e}") from e


def read_raw(path: str | Path) -> tuple[SourceInfo, list[RawSample]]:
    """Başlığı ve tüm ham örnekleri okur."""
    return read_raw_header(path), list(iter_raw(path))


# --------------------------------------------------------------------------- yollar


def sibling_paths(out: str | Path) -> tuple[Path, Path, Path]:
    """`ad.jsonl` için (ad.jsonl, ad.raw.jsonl, ad.mp4)."""
    p = Path(out)
    name = p.name[: -len(".jsonl")] if p.name.endswith(".jsonl") else p.name
    return p, p.with_name(name + ".raw.jsonl"), p.with_name(name + ".mp4")


# --------------------------------------------------------------------------- bip


_mac_procs: list[subprocess.Popen[bytes]] = []


def _beep_terminal() -> None:
    sys.stderr.write("\a")
    sys.stderr.flush()


def _beep_mac() -> None:
    # Biten süreçleri topla (zombi kalmasın), sonra bloklamadan yenisini başlat.
    _mac_procs[:] = [p for p in _mac_procs if p.poll() is None]
    try:
        _mac_procs.append(subprocess.Popen(
            ["afplay", _MAC_BEEP_SOUND],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    except OSError:
        _beep_terminal()


def _beep_windows() -> None:
    import winsound

    def run() -> None:
        try:
            winsound.Beep(_WIN_BEEP_HZ, _WIN_BEEP_MS)
        except RuntimeError:
            winsound.MessageBeep()

    threading.Thread(target=run, name="tkt-record-beep", daemon=True).start()


def play_beep() -> None:
    """Platforma göre bloklamadan bip çalar."""
    if sys.platform == "darwin":
        _beep_mac()
    elif sys.platform == "win32":
        _beep_windows()
    else:
        _beep_terminal()


# --------------------------------------------------------------------------- kayıt


def record(source: Any, out: str | Path, note: str = "", duration: float | None = None,
           raw: bool = False, video: bool = False,
           stop: threading.Event | None = None,
           prompt: int | None = None, kind: str = "tap1",
           clock: Callable[[], float] = time.monotonic,
           beep: Callable[[], None] = play_beep) -> int:
    """`source.frames_raw()` akışını kaydeder; yazılan kare sayısını döndürür.

    `source`: `info` ve `frames_raw()` olan nesne (LiveSource). Süre, kare `t`
    değerlerinden ölçülür. Ctrl+C veya `stop` ayarlanınca temiz kapanır.

    `prompt=N`: ilk kareden `PROMPT_LEAD_S` sonra, `PROMPT_INTERVAL_S` arayla N kez `beep()`
    çağrılır; her çağrının zamanı `clock()` ile (kare `t`'siyle aynı saat) alınır ve kayıt
    bitince `prompts_path(out)` dosyasına yazılır. `duration` yoksa son bipten
    `PROMPT_TAIL_S` sonra kayıt durur. Bip zamanları kare akışı sırasında denetlenir.
    """
    import cv2

    if prompt is not None and prompt < 1:
        raise ValueError("--prompt en az 1 olmalı")
    if kind not in ("tap1", "tap2"):
        raise ValueError(f"bilinmeyen bip türü: {kind!r}")

    out_path, raw_path, video_path = sibling_paths(out)
    info = replace(source.info, note=note)
    stop = stop or threading.Event()
    count = 0
    t_start: float | None = None
    vwriter = None
    prompts: list[Prompt] = []
    clock_start: float | None = None

    # Önceki kayıttan kalan bip yan dosyası bu kayda ait değildir (bip zamanları eski kaydın).
    stale_prompts = prompts_path(out_path)
    if stale_prompts.exists():
        stale_prompts.unlink()
        print(f"uyarı: eski bip dosyası silindi: {stale_prompts}", file=sys.stderr)

    writer = RecordingWriter(out_path, info)
    rwriter = RawWriter(raw_path, info) if raw else None
    interrupted = False
    stream = source.frames_raw()
    try:
        for frame, hand, image in stream:
            if t_start is None:
                t_start = frame.t
            # Önce ham, sonra işlenmiş: kesintide fazla satır `_reconcile` ile atılır.
            if rwriter is not None:
                rwriter.write(frame.t, frame.frame_id, hand)
            writer.write(frame)
            count += 1
            if video:
                if vwriter is None:
                    h, w = image.shape[:2]
                    vwriter = cv2.VideoWriter(
                        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (w, h))
                    if not vwriter.isOpened():
                        raise RuntimeError(f"video dosyası açılamadı: {video_path}")
                vwriter.write(image)
            if stop.is_set() or (duration is not None and frame.t - t_start >= duration):
                break
            if prompt is not None:
                now = clock()
                if clock_start is None:
                    clock_start = now
                if len(prompts) < prompt:
                    # Sonraki bip, son bipin verildiği andan hesaplanır; kare boşluğundan sonra
                    # geciken bipler ardı ardına yığılmaz.
                    due = (clock_start + PROMPT_LEAD_S if not prompts
                           else prompts[-1].t + PROMPT_INTERVAL_S)
                    if now >= due:
                        beep()
                        prompts.append(Prompt(t=now, kind=kind))
                elif duration is None and now - prompts[-1].t >= PROMPT_TAIL_S:
                    break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        stream.close()
        writer.close()
        if rwriter is not None:
            rwriter.close()
        if vwriter is not None:
            vwriter.release()
        if prompt is not None:
            write_prompts(prompts_path(out_path), prompts)
    if interrupted and raw:
        count = _reconcile(out_path, raw_path)
    return count


class PreviewSource:
    """`frames_raw()` akışını kayda geçirirken her kareyi pencerede gösterir.

    Önce `lead` saniye hazırlık evresi vardır: kareler gösterilir ama kayda verilmez.
    Sonra kareler kayda verilir ve ekranda kalan süre yazar. Pencerede `q`/ESC `stop`'u
    ayarlar. `show(name, img)` ve `poll_key()` testte değiştirilebilir; çizim
    `debug_view.draw_overlay` ile yapılır. macOS'ta ana thread'de çalışmalıdır.
    """

    def __init__(self, source: Any, title: str, duration: float | None,
                 stop: threading.Event, lead: float = PREVIEW_LEAD_S,
                 show: Callable[[str, np.ndarray], None] | None = None,
                 poll_key: Callable[[], int] | None = None) -> None:
        import cv2

        self.info = source.info
        self._source = source
        self._title = title
        self._duration = duration
        self._stop = stop
        self._lead = lead
        self._own_window = show is None
        self._show = show or cv2.imshow
        self._poll_key = poll_key or (lambda: cv2.waitKey(1) & 0xFF)

    def _display(self, frame: Any, hand: RawHand | None, image: np.ndarray, text: str) -> None:
        from tkt.vision import debug_view as dv

        src = self._source
        stats = dv.Stats(src.capture_fps(), src.process_fps(), src.last_process_ms) \
            if hasattr(src, "capture_fps") else dv.Stats(0.0, 0.0, 0.0)
        out = dv.draw_overlay(image, frame, hand if frame.present else None, stats,
                              dv.CalibState(None, 0.0, 0.0, 0, text))
        self._show(PREVIEW_WINDOW, out)
        if self._poll_key() in _PREVIEW_QUIT:
            self._stop.set()

    def frames_raw(self) -> Iterator[tuple[Any, RawHand | None, np.ndarray]]:
        stream = self._source.frames_raw()
        t_lead: float | None = None
        t_rec: float | None = None
        try:
            for frame, hand, image in stream:
                if t_lead is None:
                    t_lead = frame.t
                left = self._lead - (frame.t - t_lead)
                if t_rec is None and left > 0:
                    self._display(frame, hand, image,
                                  f"HAZIRLAN: {self._title} - {math.ceil(left)}")
                    if self._stop.is_set():
                        return
                    continue
                if t_rec is None:
                    t_rec = frame.t
                if self._duration is not None:
                    rem = max(0.0, self._duration - (frame.t - t_rec))
                    text = f"KAYIT: {self._title} - kalan {math.ceil(rem)} sn"
                else:
                    text = f"KAYIT: {self._title} - durdurmak icin q"
                self._display(frame, hand, image, text)
                yield frame, hand, image
        finally:
            stream.close()
            self._close_window()

    def _close_window(self) -> None:
        if self._own_window:
            import cv2

            cv2.destroyWindow(PREVIEW_WINDOW)
            cv2.waitKey(1)


def _complete_lines(path: Path) -> int:
    """Dosyadaki tamamlanmış (\\n ile biten) satır sayısı, başlık dahil."""
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.endswith(b"\n"))


def _truncate_lines(path: Path, keep: int) -> None:
    """Başlık dahil ilk `keep` tam satırı tutar, gerisini (yarım satır dahil) atar."""
    with path.open("rb") as fh:
        head = [line for _, line in zip(range(keep), fh, strict=False)]
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_bytes(b"".join(head))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _reconcile(out_path: Path, raw_path: Path) -> int:
    """Kesintiden sonra işlenmiş ve ham dosyayı ortak tamamlanmış kareye indirir; kare sayısı."""
    n = min(_complete_lines(out_path), _complete_lines(raw_path)) - 1   # başlık hariç
    for p in (out_path, raw_path):
        if _complete_lines(p) != n + 1:
            _truncate_lines(p, n + 1)
    return max(n, 0)


# --------------------------------------------------------------------------- ham kayıttan üretim


class OutputIsInputError(ValueError):
    """`--from-raw` çıktısı girdi dosyasıyla aynı dosya."""


def _same_file(a: Path, b: Path) -> bool:
    if a.resolve() == b.resolve():
        return True
    try:
        return os.path.samefile(a, b)     # sabit bağ (hard link) ve sembolik bağ
    except OSError:                       # biri yoksa aynı dosya olamaz
        return False


def rebuild_from_raw(raw_path: str | Path, out: str | Path,
                     config_path: str | Path = "config/vision.toml") -> int:
    """Ham kayıttan güncel `features.build_frame` ve `lift` ile HandFrame kaydı üretir.

    Kalibrasyon config'teki `hand.calibration` yolundan okunur (yoksa `DEFAULT`). `tap_ref`
    varsa ve `LiftEstimator` kurulabiliyorsa `lift` hesaplanır; yoksa `lift` None kalır.
    Ham kaydın `prompts` yan dosyası varsa `out`'un yan dosyasına kopyalanır. Yazılan kare
    sayısını döndürür. Çıktı girdiyle aynı dosyaysa `OutputIsInputError`; çıktı önce geçici
    dosyaya yazılır, dönüşüm bitince `os.replace` ile yerine konur (hata olursa eski çıktı kalır).
    """
    from tkt.vision import features, hand_calib, live_source
    from tkt.vision.lift import LiftEstimator
    from tkt.vision.live_source import load_vision_config

    if _same_file(Path(raw_path), Path(out)):
        raise OutputIsInputError(
            f"çıktı girdiyle aynı dosya: {out}; ham kaydı üzerine yazmamak için farklı --out verin")
    info = read_raw_header(raw_path)

    cfg = load_vision_config(config_path)
    calib_path = Path(cfg["hand"]["calibration"]).expanduser()
    calib = hand_calib.load(calib_path) or hand_calib.DEFAULT
    lift_off = live_source._lift_off(cfg["hand"])

    lift = None
    if calib.tap_ref is not None:
        try:
            lift = LiftEstimator(calib, cfg["lift"], aspect=info.width / info.height)
        except NotImplementedError:
            print("uyarı: LiftEstimator henüz uygulanmamış; lift olmadan devam ediliyor",
                  file=sys.stderr)

    out_path = Path(out)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp-{os.getpid()}")
    prev_size: float | None = None
    count = 0
    try:
        with RecordingWriter(tmp_path, info) as writer:
            for s in iter_raw(raw_path):
                if s.hand is None and lift is not None:
                    lift.reset()
                frame = features.build_frame(
                    s.hand, s.t, s.frame_id, calib, info, lift=None, prev_hand_size=prev_size,
                    lift_off=lift_off)
                if s.hand is not None and lift is not None and frame.present:
                    curls = {name: f.curl for name, f in frame.fingers.items()}
                    frame = features.build_frame(
                        s.hand, s.t, s.frame_id, calib, info,
                        lift=lift.update(s.hand, s.t, curls), prev_hand_size=prev_size,
                        lift_off=lift_off)
                prev_size = frame.hand_size if frame.present else None
                writer.write(frame)
                count += 1
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    src_prompts, dst_prompts = prompts_path(raw_path), prompts_path(out)
    if src_prompts.exists() and src_prompts.resolve() != dst_prompts.resolve():
        shutil.copyfile(src_prompts, dst_prompts)
    return count


def _watch_stdin_for_quit(stop: threading.Event) -> None:
    """Terminalde `q` + Enter yazılınca kaydı durdurur."""
    def run() -> None:
        for line in sys.stdin:
            if line.strip().lower() == "q":
                stop.set()
                return

    threading.Thread(target=run, name="tkt-record-quit", daemon=True).start()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tkt.tools.record")
    p.add_argument("--out", required=True, help="HandFrame kaydı (.jsonl)")
    p.add_argument("--note", default="", help="başlıktaki not")
    p.add_argument("--duration", type=float, default=None,
                   help="saniye; yoksa q+Enter veya Ctrl+C")
    p.add_argument("--raw", action="store_true", help="21 noktayı ad.raw.jsonl'e de yaz")
    p.add_argument("--video", action="store_true", help="görüntüyü ad.mp4'e de yaz")
    p.add_argument("--prompt", type=int, default=None, metavar="N",
                   help="2 sn arayla N bip ver, zamanları ad.prompts.json'a yaz")
    p.add_argument("--kind", choices=("tap1", "tap2"), default="tap1", help="bip türü")
    p.add_argument("--from-raw", default=None, metavar="RAW",
                   help="ham kayıttan güncel features ve lift ile HandFrame kaydı üret")
    p.add_argument("--preview", action="store_true",
                   help=f"kaydı pencerede göster; önce {PREVIEW_LEAD_S:g} sn hazırlık, q durdurur")
    p.add_argument("--config", default="config/vision.toml")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.from_raw is not None:
        if args.prompt is not None:
            build_parser().error("--from-raw ile --prompt birlikte kullanılamaz")
        try:
            n = rebuild_from_raw(args.from_raw, args.out, args.config)
        except OutputIsInputError as e:
            build_parser().error(str(e))
        print(f"{n} kare yazıldı: {args.out}")
        return
    if args.prompt is not None and args.prompt < 1:
        build_parser().error("--prompt en az 1 olmalı")

    from tkt.vision.live_source import open_live_source

    source = open_live_source(args.config)
    stop = threading.Event()
    rec_source: Any = source
    if args.preview:
        title = args.note or Path(args.out).stem
        rec_source = PreviewSource(source, title, args.duration, stop)
    if args.duration is None and sys.stdin is not None and sys.stdin.isatty():
        _watch_stdin_for_quit(stop)
        print("Kayıt başladı; durdurmak için q + Enter veya Ctrl+C.", file=sys.stderr)
    try:
        n = record(rec_source, args.out, args.note, args.duration, args.raw, args.video, stop,
                   prompt=args.prompt, kind=args.kind)
    finally:
        source.close()
    print(f"{n} kare yazıldı: {args.out}")


if __name__ == "__main__":
    main()
