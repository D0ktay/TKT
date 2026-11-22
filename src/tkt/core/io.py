"""Kayıt (JSONL) ve bip yan dosyasının tek okuma-yazma yeri (PART1, 1.3).

Kayıt biçimi:
    1. satır: {"type": "header", "schema": 2, "width": ..., "height": ..., "fps": ...,
               "hand": ..., "note": ...}
    sonraki her satır bir `HandFrame`. `pinch` anahtarı isteğe bağlıdır; yalnızca
    `HandFrame.pinch` None değilse yazılır.

Bip yan dosyası (`ad.prompts.json`): [{"t": 12.30, "kind": "tap1"}, ...]

Şema sürümü `READABLE_SCHEMAS` içinde olmayan kayıtlar `SchemaError` ile reddedilir.
Şema 1 kayıtları dönüştürülerek okunur: okunan `SourceInfo.schema` `SCHEMA_VERSION` olur,
kareler `pinch=None` taşır. Yazma yalnızca `SCHEMA_VERSION` ile yapılır.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from tkt.core.types import (
    FINGERS,
    READABLE_SCHEMAS,
    SCHEMA_VERSION,
    Finger,
    HandFrame,
    SourceInfo,
)

PROMPT_KINDS = ("tap1", "tap2")

# Zaman damgaları mikrosaniye çözünürlüğünde yazılır.
_T_DIGITS = 6


class SchemaError(ValueError):
    """Kayıt şema sürümü desteklenmiyor veya başlık bozuk."""


class RecordFormatError(ValueError):
    """Kayıttaki bir satır okunamadı."""


@dataclass(frozen=True)
class Prompt:
    t: float     # HandFrame.t ile aynı saat
    kind: str    # "tap1" | "tap2"


# --------------------------------------------------------------------------- dönüşümler


def _finite_float(value: Any, field: str) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field}: sonlu sayı bekleniyordu")
    return result


def _pair(v: Any, field: str) -> tuple[float, float]:
    a, b = v
    return (_finite_float(a, field), _finite_float(b, field))


def info_to_header(info: SourceInfo) -> dict[str, Any]:
    """`SourceInfo`'yu başlık satırı sözlüğüne çevirir."""
    return {
        "type": "header",
        "schema": info.schema,
        "width": info.width,
        "height": info.height,
        "fps": info.fps,
        "hand": info.hand,
        "note": info.note,
    }


def header_to_info(d: dict[str, Any], source: str = "<kayıt>") -> SourceInfo:
    """Başlık sözlüğünü `SourceInfo`'ya çevirir; şema okunamıyorsa `SchemaError`.

    Eski şemalı (`READABLE_SCHEMAS`) başlıklar güncel sürüme dönüştürülür: dönen
    `schema` her zaman `SCHEMA_VERSION`'dır.
    """
    if not isinstance(d, dict) or d.get("type") != "header":
        raise SchemaError(f"{source}: ilk satır başlık değil (type='header' bekleniyordu)")
    schema = d.get("schema")
    if type(schema) is not int or schema not in READABLE_SCHEMAS:
        supported = ", ".join(str(v) for v in READABLE_SCHEMAS)
        raise SchemaError(
            f"{source}: şema sürümü {schema!r}, desteklenen {supported}. "
            "Kaydı güncel araçla yeniden üretin."
        )
    try:
        for field in ("width", "height"):
            value = d[field]
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field}: pozitif tamsayı bekleniyordu")
        fps = d["fps"]
        if type(fps) not in (int, float):
            raise ValueError("fps: pozitif ve sonlu sayı bekleniyordu")
        fps = _finite_float(fps, "fps")
        if fps <= 0:
            raise ValueError("fps: pozitif sayı bekleniyordu")
        hand = d["hand"]
        if hand not in ("right", "left"):
            raise ValueError("hand: 'right' veya 'left' bekleniyordu")
        return SourceInfo(
            schema=SCHEMA_VERSION,
            width=d["width"],
            height=d["height"],
            fps=fps,
            hand=hand,
            note=str(d.get("note", "")),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as e:
        raise SchemaError(f"{source}: başlık eksik veya bozuk: {e}") from e


def frame_to_dict(frame: HandFrame) -> dict[str, Any]:
    """`HandFrame`'i kayıt satırı sözlüğüne çevirir."""
    # JSON'un float gösterimi, jest eşiklerine yakın değerleri de aynen korur.
    fingers: dict[str, Any] = {}
    for name, f in frame.fingers.items():
        fingers[name] = {
            "img": [float(f.img[0]), float(f.img[1])],
            "surface": (
                None if f.surface is None else [float(f.surface[0]), float(f.surface[1])]
            ),
            "curl": float(f.curl),
            "lift": None if f.lift is None else float(f.lift),
        }
    row: dict[str, Any] = {
        "t": round(float(frame.t), _T_DIGITS),
        "frame_id": int(frame.frame_id),
        "present": bool(frame.present),
        "fingers": fingers,
        "thumb_out": float(frame.thumb_out),
        "palm": [float(frame.palm[0]), float(frame.palm[1])],
        "hand_size": float(frame.hand_size),
        "confidence": float(frame.confidence),
    }
    # pinch açık değilse satır şema 1 satırıyla aynı kalır.
    if frame.pinch is not None:
        row["pinch"] = {name: float(v) for name, v in frame.pinch.items()}
    return row


def _pinch_from(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("pinch: nesne veya null bekleniyordu")
    unknown = [name for name in value if name not in FINGERS]
    if unknown:
        raise ValueError(f"pinch: bilinmeyen parmak adı: {unknown[0]!r}")
    missing = [name for name in FINGERS if name not in value]
    if missing:
        raise ValueError(f"pinch: eksik parmaklar: {', '.join(missing)}")
    pinch: dict[str, float] = {}
    for name in FINGERS:
        v = value[name]
        if type(v) not in (int, float):
            raise ValueError(f"pinch.{name}: sayı bekleniyordu")
        v = _finite_float(v, f"pinch.{name}")
        if v < 0.0:
            raise ValueError(f"pinch.{name}: negatif olamaz")
        pinch[name] = v
    return pinch


def frame_from_dict(d: dict[str, Any]) -> HandFrame:
    """Kayıt satırı sözlüğünü `HandFrame`'e çevirir.

    `present=False` satırlarında eksik alanlar sıfırla doldurulur; bu alanlar okunmaz.
    `pinch` yoksa veya null ise None olur (şema 1 satırları); `present=False` ise yok sayılır.
    """
    present = bool(d["present"])
    finger_rows = d.get("fingers")
    if finger_rows is None:
        finger_rows = {}
    if not isinstance(finger_rows, dict):
        raise ValueError("fingers: nesne bekleniyordu")
    fingers: dict[str, Finger] = {}
    for name, f in finger_rows.items():
        if name not in FINGERS:
            raise ValueError(f"bilinmeyen parmak adı: {name!r}")
        if not isinstance(f, dict):
            raise ValueError(f"fingers.{name}: nesne bekleniyordu")
        surface = f.get("surface")
        lift = f.get("lift")
        fingers[name] = Finger(
            img=_pair(f["img"], f"{name}.img"),
            surface=None if surface is None else _pair(surface, f"{name}.surface"),
            curl=_finite_float(f["curl"], f"{name}.curl"),
            lift=None if lift is None else _finite_float(lift, f"{name}.lift"),
        )
    if present:
        missing = [name for name in FINGERS if name not in fingers]
        if missing:
            raise ValueError(f"eksik parmaklar: {', '.join(missing)}")

    pinch = _pinch_from(d.get("pinch")) if present else None

    confidence = _finite_float(d.get("confidence", 0.0), "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence: 0–1 arasında olmalı")
    return HandFrame(
        t=_finite_float(d["t"], "t"),
        frame_id=int(d["frame_id"]),
        present=present,
        fingers=fingers if present else {},
        thumb_out=_finite_float(d.get("thumb_out", 0.0), "thumb_out"),
        palm=_pair(d.get("palm", (0.0, 0.0)), "palm"),
        hand_size=_finite_float(d.get("hand_size", 0.0), "hand_size"),
        confidence=confidence,
        pinch=pinch,
    )


# --------------------------------------------------------------------------- kayıt yazma


class RecordingWriter:
    """Kaydı satır satır yazar. Bağlam yöneticisi olarak kullanılır.

        with RecordingWriter(path, info) as w:
            for frame in source.frames():
                w.write(frame)
    """

    def __init__(self, path: str | Path, info: SourceInfo, flush_every: int = 60) -> None:
        if info.schema != SCHEMA_VERSION:
            raise SchemaError(
                f"şema {info.schema} ile yazılamaz, güncel sürüm {SCHEMA_VERSION}"
            )
        self.path = Path(path)
        self.info = info
        self.count = 0
        self._flush_every = max(1, flush_every)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] | None = self.path.open("w", encoding="utf-8", newline="\n")
        self._write_line(info_to_header(info))

    def _write_line(self, obj: dict[str, Any]) -> None:
        assert self._fh is not None, "yazıcı kapalı"
        self._fh.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")

    def write(self, frame: HandFrame) -> None:
        self._write_line(frame_to_dict(frame))
        self.count += 1
        if self.count % self._flush_every == 0:
            assert self._fh is not None
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> RecordingWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def write_recording(path: str | Path, info: SourceInfo, frames: Iterable[HandFrame]) -> int:
    """Tüm kareleri yazar, yazılan kare sayısını döndürür."""
    with RecordingWriter(path, info) as w:
        for frame in frames:
            w.write(frame)
        return w.count


# --------------------------------------------------------------------------- kayıt okuma


def _parse_recording_header(first: str, path: Path) -> SourceInfo:
    if not first.strip():
        raise SchemaError(f"{path}: dosya boş")
    try:
        d = json.loads(first)
    except json.JSONDecodeError as e:
        raise SchemaError(f"{path}: başlık JSON değil: {e}") from e
    # header_to_info ham kayıt dönüştürücüsü tarafından da kullanılır.
    if isinstance(d, dict) and d.get("raw") is True:
        raise SchemaError(f"{path}: ham kayıt; --from-raw ile dönüştürün")
    return header_to_info(d, str(path))


def read_header(path: str | Path) -> SourceInfo:
    """Yalnızca başlığı okur."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        return _parse_recording_header(fh.readline(), p)


def iter_frames(path: str | Path) -> Iterator[HandFrame]:
    """Kareleri sırayla üretir. Başlığı doğrular, boş satırları atlar."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        _parse_recording_header(fh.readline(), p)
        for lineno, line in enumerate(fh, start=2):
            if not line.strip():
                continue
            try:
                yield frame_from_dict(json.loads(line))
            except (KeyError, TypeError, ValueError, OverflowError) as e:
                raise RecordFormatError(f"{p}:{lineno}: kare okunamadı: {e}") from e


def read_recording(path: str | Path) -> tuple[SourceInfo, list[HandFrame]]:
    """Başlığı ve tüm kareleri okur."""
    return read_header(path), list(iter_frames(path))


# --------------------------------------------------------------------------- bip yan dosyası


def prompts_path(recording_path: str | Path) -> Path:
    """Kayıt yolundan yan dosya yolunu türetir.

    `ad.jsonl` ve `ad.raw.jsonl` için `ad.prompts.json` döner.
    """
    p = Path(recording_path)
    name = p.name
    for suffix in (".raw.jsonl", ".jsonl"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return p.with_name(name + ".prompts.json")


def write_prompts(path: str | Path, prompts: Iterable[Prompt]) -> None:
    items = []
    for pr in prompts:
        if pr.kind not in PROMPT_KINDS:
            raise ValueError(f"bilinmeyen bip türü: {pr.kind!r}")
        items.append({"t": round(float(pr.t), _T_DIGITS), "kind": pr.kind})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")


def read_prompts(path: str | Path) -> list[Prompt]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RecordFormatError(f"{p}: liste bekleniyordu")
    out: list[Prompt] = []
    for i, item in enumerate(data):
        try:
            kind = str(item["kind"])
            t = float(item["t"])
        except (KeyError, TypeError, ValueError) as e:
            raise RecordFormatError(f"{p}[{i}]: bozuk öğe: {e}") from e
        if kind not in PROMPT_KINDS:
            raise RecordFormatError(f"{p}[{i}]: bilinmeyen bip türü: {kind!r}")
        out.append(Prompt(t=t, kind=kind))
    return out
