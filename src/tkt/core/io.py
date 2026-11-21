"""Kayıt (JSONL) ve bip yan dosyasının tek okuma-yazma yeri (PART1, 1.3).

Kayıt biçimi:
    1. satır: {"type": "header", "schema": 1, "width": ..., "height": ..., "fps": ...,
               "hand": ..., "note": ...}
    sonraki her satır bir `HandFrame`.

Bip yan dosyası (`ad.prompts.json`): [{"t": 12.30, "kind": "tap1"}, ...]

Şema sürümü `SCHEMA_VERSION` ile tutmayan kayıtlar `SchemaError` ile reddedilir.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from tkt.core.types import FINGERS, SCHEMA_VERSION, Finger, HandFrame, SourceInfo

PROMPT_KINDS = ("tap1", "tap2")

# Dosya boyutunu küçük tutmak için yuvarlama. 1e-5 görüntü birimi 1280 pikselde 0,013 pikseldir.
_T_DIGITS = 6
_VAL_DIGITS = 5


class SchemaError(ValueError):
    """Kayıt şema sürümü desteklenmiyor veya başlık bozuk."""


class RecordFormatError(ValueError):
    """Kayıttaki bir satır okunamadı."""


@dataclass(frozen=True)
class Prompt:
    t: float     # HandFrame.t ile aynı saat
    kind: str    # "tap1" | "tap2"


# --------------------------------------------------------------------------- dönüşümler


def _r(x: float) -> float:
    return round(float(x), _VAL_DIGITS)


def _pair(v: Any) -> tuple[float, float]:
    a, b = v
    return (float(a), float(b))


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
    """Başlık sözlüğünü `SourceInfo`'ya çevirir; şema tutmuyorsa `SchemaError`."""
    if not isinstance(d, dict) or d.get("type") != "header":
        raise SchemaError(f"{source}: ilk satır başlık değil (type='header' bekleniyordu)")
    schema = d.get("schema")
    if schema != SCHEMA_VERSION:
        raise SchemaError(
            f"{source}: şema sürümü {schema!r}, desteklenen {SCHEMA_VERSION}. "
            "Kaydı güncel araçla yeniden üretin."
        )
    try:
        return SourceInfo(
            schema=int(schema),
            width=int(d["width"]),
            height=int(d["height"]),
            fps=float(d["fps"]),
            hand=str(d["hand"]),
            note=str(d.get("note", "")),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise SchemaError(f"{source}: başlık eksik veya bozuk: {e}") from e


def frame_to_dict(frame: HandFrame) -> dict[str, Any]:
    """`HandFrame`'i kayıt satırı sözlüğüne çevirir."""
    fingers: dict[str, Any] = {}
    for name, f in frame.fingers.items():
        fingers[name] = {
            "img": [_r(f.img[0]), _r(f.img[1])],
            "surface": None if f.surface is None else [_r(f.surface[0]), _r(f.surface[1])],
            "curl": _r(f.curl),
            "lift": None if f.lift is None else _r(f.lift),
        }
    return {
        "t": round(float(frame.t), _T_DIGITS),
        "frame_id": int(frame.frame_id),
        "present": bool(frame.present),
        "fingers": fingers,
        "thumb_out": _r(frame.thumb_out),
        "palm": [_r(frame.palm[0]), _r(frame.palm[1])],
        "hand_size": _r(frame.hand_size),
        "confidence": _r(frame.confidence),
    }


def frame_from_dict(d: dict[str, Any]) -> HandFrame:
    """Kayıt satırı sözlüğünü `HandFrame`'e çevirir.

    `present=False` satırlarında eksik alanlar sıfırla doldurulur; bu alanlar okunmaz.
    """
    present = bool(d["present"])
    fingers: dict[str, Finger] = {}
    for name, f in (d.get("fingers") or {}).items():
        if name not in FINGERS:
            raise ValueError(f"bilinmeyen parmak adı: {name!r}")
        surface = f.get("surface")
        lift = f.get("lift")
        fingers[name] = Finger(
            img=_pair(f["img"]),
            surface=None if surface is None else _pair(surface),
            curl=float(f["curl"]),
            lift=None if lift is None else float(lift),
        )
    return HandFrame(
        t=float(d["t"]),
        frame_id=int(d["frame_id"]),
        present=present,
        fingers=fingers if present else {},
        thumb_out=float(d.get("thumb_out", 0.0)),
        palm=_pair(d.get("palm", (0.0, 0.0))),
        hand_size=float(d.get("hand_size", 0.0)),
        confidence=float(d.get("confidence", 0.0)),
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


def read_header(path: str | Path) -> SourceInfo:
    """Yalnızca başlığı okur."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        first = fh.readline()
    if not first.strip():
        raise SchemaError(f"{p}: dosya boş")
    try:
        d = json.loads(first)
    except json.JSONDecodeError as e:
        raise SchemaError(f"{p}: başlık JSON değil: {e}") from e
    return header_to_info(d, str(p))


def iter_frames(path: str | Path) -> Iterator[HandFrame]:
    """Kareleri sırayla üretir. Başlığı doğrular, boş satırları atlar."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        first = fh.readline()
        if not first.strip():
            raise SchemaError(f"{p}: dosya boş")
        try:
            header_to_info(json.loads(first), str(p))
        except json.JSONDecodeError as e:
            raise SchemaError(f"{p}: başlık JSON değil: {e}") from e
        for lineno, line in enumerate(fh, start=2):
            if not line.strip():
                continue
            try:
                yield frame_from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
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
