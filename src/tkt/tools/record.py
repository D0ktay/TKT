"""Kayıt aracı (V1). PART2 2.4.

    python -m tkt.tools.record --out data/samples/ad.jsonl --note "..." --duration 30 --raw
    python -m tkt.tools.record --out ad.jsonl --prompt 50 --kind tap1 --raw --video
    python -m tkt.tools.record --from-raw ad.raw.jsonl --out ad.jsonl

V1-1 kapsamı: `--out --note --duration --raw --video`. `--prompt` ve `--from-raw` V1-2'dedir.

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
import sys
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType
from typing import IO, Any

import numpy as np

from tkt.core.io import (
    RecordFormatError,
    RecordingWriter,
    SchemaError,
    header_to_info,
    info_to_header,
)
from tkt.core.types import SourceInfo
from tkt.vision.tracker import RawHand

N_POINTS = 21
_T_DIGITS = 6
_IMG_DIGITS = 5
_WORLD_DIGITS = 6


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


# --------------------------------------------------------------------------- kayıt


def record(source: Any, out: str | Path, note: str = "", duration: float | None = None,
           raw: bool = False, video: bool = False,
           stop: threading.Event | None = None) -> int:
    """`source.frames_raw()` akışını kaydeder; yazılan kare sayısını döndürür.

    `source`: `info` ve `frames_raw()` olan nesne (LiveSource). Süre, kare `t`
    değerlerinden ölçülür. Ctrl+C veya `stop` ayarlanınca temiz kapanır.
    """
    import cv2

    out_path, raw_path, video_path = sibling_paths(out)
    info = replace(source.info, note=note)
    stop = stop or threading.Event()
    count = 0
    t_start: float | None = None
    vwriter = None

    writer = RecordingWriter(out_path, info)
    rwriter = RawWriter(raw_path, info) if raw else None
    stream = source.frames_raw()
    try:
        for frame, hand, image in stream:
            if t_start is None:
                t_start = frame.t
            writer.write(frame)
            count += 1
            if rwriter is not None:
                rwriter.write(frame.t, frame.frame_id, hand)
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
    except KeyboardInterrupt:
        pass
    finally:
        stream.close()
        writer.close()
        if rwriter is not None:
            rwriter.close()
        if vwriter is not None:
            vwriter.release()
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
                   help="2 sn arayla N bip ver, zamanları ad.prompts.json'a yaz (V1-2)")
    p.add_argument("--kind", choices=("tap1", "tap2"), default="tap1", help="bip türü")
    p.add_argument("--from-raw", default=None, metavar="RAW",
                   help="ham kayıttan güncel features ve lift ile HandFrame kaydı üret (V1-2)")
    p.add_argument("--config", default="config/vision.toml")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.prompt is not None:
        raise NotImplementedError("--prompt henüz yok (görev V1-2)")
    if args.from_raw is not None:
        raise NotImplementedError("--from-raw henüz yok (görev V1-2)")

    from tkt.vision.live_source import open_live_source

    source = open_live_source(args.config)
    stop = threading.Event()
    if args.duration is None and sys.stdin is not None and sys.stdin.isatty():
        _watch_stdin_for_quit(stop)
        print("Kayıt başladı; durdurmak için q + Enter veya Ctrl+C.", file=sys.stderr)
    try:
        n = record(source, args.out, args.note, args.duration, args.raw, args.video, stop)
    finally:
        source.close()
    print(f"{n} kare yazıldı: {args.out}")


if __name__ == "__main__":
    main()
