"""Ham noktalardan ölçümler (V2). PART2 2.2, 2.4.

Eşik ve zaman filtresi uygulanmaz. Güven değeri yalnızca azaltılır; "el yok" kararı
control'ün güven kapısındadır.

Nokta numaraları: bilek 0; başparmak 1–4; işaret 5-6-7-8 (kök, orta eklem, uç eklem, uç);
orta 9–12; yüzük 13–16; serçe 17–20.
"""

from __future__ import annotations

import math

import numpy as np

from tkt.core.types import FINGERS, Finger, HandFrame, SourceInfo
from tkt.vision.hand_calib import HandCalibration
from tkt.vision.tracker import RawHand

WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
MIDDLE_MCP = 9

# Parmak adı → (kök, orta eklem, uç eklem, uç) nokta numaraları.
FINGER_JOINTS: dict[str, tuple[int, int, int, int]] = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
PALM_POINTS = (0, 5, 9, 13, 17)

# Güven düşürme. Bunlar karar eşiği değildir, yalnızca güvenin ne kadar azalacağını belirler.
# Kenar payı, görüntünün kısa kenarının oranıdır (piksel cinsinden her iki eksende aynı pay).
EDGE_MARGIN = 0.03
# Bir nokta kadraj kenarına değdiğinde (veya dışına taştığında) güven bu katsayıyla çarpılır;
# pay içinde kenara uzaklıkla doğrusal olarak 1'e çıkar.
EDGE_MIN_FACTOR = 0.5
# hand_size bir önceki kareye göre bu orandan fazla değişirse güven JUMP_FACTOR ile çarpılır.
HAND_SIZE_JUMP = 0.20
JUMP_FACTOR = 0.5

_EPS = 1e-9


def _flexion(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """`b` eklemindeki bükülme açısı (radyan); düz parmakta ~0."""
    u = b - a
    v = c - b
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < _EPS or nv < _EPS:
        return 0.0
    cos = float(np.dot(u, v)) / (nu * nv)
    return math.acos(min(1.0, max(-1.0, cos)))


def _joints(finger: str) -> tuple[int, int, int, int]:
    try:
        return FINGER_JOINTS[finger]
    except KeyError:
        raise ValueError(f"bilinmeyen parmak adı: {finger!r}") from None


def curl_raw(hand: RawHand, finger: str) -> float:
    """Orta ve uç eklem bükülmesinin toplamı (radyan, dünya noktalarıyla).

    Kök eklemi dışarıdadır; kökten hareket `lift`'e aittir.
    """
    mcp, pip, dip, tip = _joints(finger)
    p = np.asarray(hand.world_pts, dtype=float)
    return _flexion(p[mcp], p[pip], p[dip]) + _flexion(p[pip], p[dip], p[tip])


def _world_hand_size(hand: RawHand) -> float:
    p = np.asarray(hand.world_pts, dtype=float)
    return float(np.linalg.norm(p[MIDDLE_MCP] - p[WRIST]))


def thumb_raw(hand: RawHand) -> float:
    """Başparmak ucu ile işaret kökü arası 3D mesafe, el boyuna oranlı.

    El boyu: dünya noktalarında bilek (0) ile orta parmak kökü (9) arası mesafe.
    """
    p = np.asarray(hand.world_pts, dtype=float)
    size = _world_hand_size(hand)
    if size < _EPS:
        return 0.0
    return float(np.linalg.norm(p[THUMB_TIP] - p[INDEX_MCP])) / size


def _normalize(raw: float, lo: float, hi: float) -> float:
    """`lo`→0, `hi`→1 doğrusal eşleme, 0–1'e kırpılır."""
    span = hi - lo
    if abs(span) < _EPS:
        return 0.0 if raw <= lo else 1.0
    return min(1.0, max(0.0, (raw - lo) / span))


def _hand_size_img(pts: np.ndarray, info: SourceInfo) -> float:
    """0–9 arası piksel mesafesi / görüntü genişliği."""
    w = float(info.width)
    h = float(info.height)
    if w <= 0:
        return 0.0
    dx = (pts[MIDDLE_MCP, 0] - pts[WRIST, 0]) * w
    dy = (pts[MIDDLE_MCP, 1] - pts[WRIST, 1]) * h
    return math.hypot(dx, dy) / w


def _edge_factor(pts: np.ndarray, info: SourceInfo) -> float:
    """Kenara en yakın noktaya göre güven katsayısı (EDGE_MIN_FACTOR..1)."""
    w = float(info.width)
    h = float(info.height)
    if w <= 0 or h <= 0:
        return 1.0
    margin_px = EDGE_MARGIN * min(w, h)
    if margin_px <= 0:
        return 1.0
    xs = pts[:, 0] * w
    ys = pts[:, 1] * h
    # Kenara en kısa piksel uzaklığı; kadraj dışındaki nokta için negatif.
    d = float(min(xs.min(), (w - xs).min(), ys.min(), (h - ys).min()))
    k = min(1.0, max(0.0, d / margin_px))
    return EDGE_MIN_FACTOR + (1.0 - EDGE_MIN_FACTOR) * k


def _empty_frame(t: float, frame_id: int) -> HandFrame:
    return HandFrame(
        t=t,
        frame_id=frame_id,
        present=False,
        fingers={},
        thumb_out=0.0,
        palm=(0.0, 0.0),
        hand_size=0.0,
        confidence=0.0,
    )


def build_frame(hand: RawHand | None, t: float, frame_id: int, calib: HandCalibration,
                info: SourceInfo, lift: dict[str, float] | None = None,
                prev_hand_size: float | None = None) -> HandFrame:
    """Tek karenin `HandFrame`'ini üretir; el yoksa `present=False`."""
    if hand is None:
        return _empty_frame(t, frame_id)

    img = np.asarray(hand.image_pts, dtype=float)

    fingers: dict[str, Finger] = {}
    for name in FINGERS:
        tip = FINGER_JOINTS[name][3]
        curl = _normalize(curl_raw(hand, name), calib.open_ref[name], calib.fist_ref[name])
        lift_v = None
        if lift is not None and name in lift and lift[name] is not None:
            lift_v = float(lift[name])
        fingers[name] = Finger(
            img=(float(img[tip, 0]), float(img[tip, 1])),
            surface=None,
            curl=curl,
            lift=lift_v,
        )

    thumb_out = _normalize(thumb_raw(hand), calib.thumb_closed, calib.thumb_open)
    palm_xy = img[list(PALM_POINTS)].mean(axis=0)
    hand_size = _hand_size_img(img, info)

    confidence = min(1.0, max(0.0, float(hand.score)))
    confidence *= _edge_factor(img, info)
    if prev_hand_size is not None and prev_hand_size > _EPS:
        if abs(hand_size / prev_hand_size - 1.0) > HAND_SIZE_JUMP:
            confidence *= JUMP_FACTOR

    return HandFrame(
        t=t,
        frame_id=frame_id,
        present=True,
        fingers=fingers,
        thumb_out=thumb_out,
        palm=(float(palm_xy[0]), float(palm_xy[1])),
        hand_size=hand_size,
        confidence=confidence,
    )
