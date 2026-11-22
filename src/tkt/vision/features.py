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
from tkt.vision.hand_calib import PINCH_CHANNELS, HandCalibration
from tkt.vision.tracker import RawHand

WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
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
# El masadan kalkık (masa modelinden sapma eşiği aşıldı): güven bu katsayıyla çarpılır.
# 0: control eli "yok" sayar, imleç durur (J10 davranışı).
LIFT_OFF_FACTOR = 0.0

_EPS = 1e-9


def _joints(finger: str) -> tuple[int, int, int, int]:
    try:
        return FINGER_JOINTS[finger]
    except KeyError:
        raise ValueError(f"bilinmeyen parmak adı: {finger!r}") from None


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


def curl_raw(hand: RawHand, finger: str, aspect: float = 1.0) -> float:
    """Orta ve uç eklem bükülmesinin toplamı (radyan), görüntü (2B) noktalarıyla.

    Kök eklemi dışarıdadır; kökten hareket `lift`'e aittir. Düz parmak kökten kalksa da
    görüntüde düz kalır, açı değişmez. `aspect` = görüntü genişliği / yüksekliği; x ekseni
    buna göre ölçeklenir, açılar piksel uzayında ölçülür.

    Dünya (3B) noktaları kullanılmaz: tek parmak açıkken MediaPipe'ın derinlik tahmini o
    parmağı kıvrık gösteriyor, görüntü noktaları ise doğru (kayıt ölçümü, 2026-11-21:
    açık ~0,1, kıvrık ~2,6–2,9 rad).
    """
    mcp, pip, dip, tip = _joints(finger)
    p = np.asarray(hand.image_pts, dtype=float) * (float(aspect), 1.0)
    return _flexion(p[mcp], p[pip], p[dip]) + _flexion(p[pip], p[dip], p[tip])


def thumb_raw(hand: RawHand, aspect: float = 1.0) -> float:
    """Görüntüde başparmak ucu ile işaret orta eklemi arası mesafe, el boyuna oranlı (2B).

    El boyu: görüntüde bilek (0) ile orta parmak kökü (9) arası. `aspect` = görüntü
    genişliği / yüksekliği; mesafeler piksel uzayında ölçülür.

    Dünya (3B) noktaları ve işaret kökü yerine görüntü noktaları ve işaret orta eklemi
    kullanılır: kayıtlarda bu ölçünün açık-kapalı aralığı kare-kare titremesine göre ~1,6 kat
    daha geniş (2026-11-21).
    """
    p = np.asarray(hand.image_pts, dtype=float) * (float(aspect), 1.0)
    size = float(np.linalg.norm(p[MIDDLE_MCP] - p[WRIST]))
    if size < _EPS:
        return 0.0
    return float(np.linalg.norm(p[THUMB_TIP] - p[INDEX_PIP])) / size


def pinch_raw(hand: RawHand, finger: str, aspect: float = 1.0) -> dict[str, float]:
    """Başparmak ucu (4) ile parmak ucu arası mesafe, iki kanalda el boyuna oranlı.

    - `"img"`: görüntüde (x `aspect` ile ölçeklenmiş piksel uzayı), bilek (0)–orta kök (9)
      görüntü mesafesine bölünür.
    - `"world"`: dünya noktalarıyla (3B), bilek–orta kök dünya mesafesine bölünür.

    Ölçülemeyen kanal (sonlu olmayan nokta, sıfır el boyu, eksik nokta) NaN döner; bu
    durumda `pinch_values` o kanalı "açık" sayar.
    """
    _mcp, _pip, _dip, tip = _joints(finger)
    return {
        "img": _tip_ratio(hand.image_pts, tip, (float(aspect), 1.0)),
        "world": _tip_ratio(hand.world_pts, tip, None),
    }


def _tip_ratio(pts: object, tip: int, scale: tuple[float, float] | None) -> float:
    """|p4 - p_tip| / |p9 - p0|; geçersizse NaN."""
    try:
        p = np.asarray(pts, dtype=float)
        if p.ndim != 2 or p.shape[0] < 21:
            return math.nan
        if scale is not None:
            p = p[:, :2] * scale
        size = float(np.linalg.norm(p[MIDDLE_MCP] - p[WRIST]))
        dist = float(np.linalg.norm(p[THUMB_TIP] - p[tip]))
    except (TypeError, ValueError):
        return math.nan
    if not (math.isfinite(size) and math.isfinite(dist)) or size < _EPS:
        return math.nan
    return dist / size


def pinch_norm(raw: float, touch: float, open_: float) -> float:
    """Tek kanal: `touch`→0, `open_`→1, 0–1'e kırpılır. Geçersiz ölçüm veya referans 1.0.

    Geçersiz değer "açık" sayılır (fail-safe): yanlışlıkla "değiyor" üretmez.
    """
    if not (math.isfinite(raw) and math.isfinite(touch) and math.isfinite(open_)):
        return 1.0
    span = open_ - touch
    if not span > _EPS:
        return 1.0
    return min(1.0, max(0.0, (raw - touch) / span))


def pinch_values(hand: RawHand, ref: dict[str, dict[str, float]],
                 aspect: float = 1.0) -> dict[str, float]:
    """Parmak başına pinch (0 uçlar değiyor, 1 açık): kanalların büyüğü (temkinli).

    Her kanal kendi açık/değme referansıyla 0–1'e oturtulur; "değiyor" için iki kanal da
    küçük olmalıdır. Referansı eksik parmak veya kanal 1.0 (açık) verir.
    """
    out: dict[str, float] = {}
    for name in FINGERS:
        r = ref.get(name)
        if not r:
            out[name] = 1.0
            continue
        raw = pinch_raw(hand, name, aspect)
        out[name] = max(
            pinch_norm(raw[ch], r.get(f"{ch}_touch", math.nan), r.get(f"{ch}_open", math.nan))
            for ch in PINCH_CHANNELS
        )
    return out


def palm_shape(hand: RawHand, aspect: float = 1.0) -> tuple[float, float, list[float]]:
    """Masa modeli girdisi: (log avuç uzunluğu, log avuç genişliği, konum terimleri).

    Uzunluk bilek (0)–orta kök (9), genişlik işaret kökü (5)–serçe kökü (17); görüntüde,
    x `aspect` ile ölçeklenmiş birimde. Konum: avuç noktalarının ortalaması (0–1);
    terimler `[1, x, y, x², y², xy]`.
    """
    img = np.asarray(hand.image_pts, dtype=float)
    p = img * (float(aspect), 1.0)
    length = max(float(np.linalg.norm(p[MIDDLE_MCP] - p[WRIST])), _EPS)
    width = max(float(np.linalg.norm(p[17] - p[INDEX_MCP])), _EPS)
    x, y = (float(v) for v in img[list(PALM_POINTS)].mean(axis=0))
    return math.log(length), math.log(width), [1.0, x, y, x * x, y * y, x * y]


def table_deviation(hand: RawHand, table: dict[str, list[float]],
                    aspect: float = 1.0) -> float:
    """Avuç şeklinin masa modelinden sapması: uzunluk ve genişlik log farklarının büyüğü.

    El masadayken ~0 (kayıtlarda p99 ≈ 0,05–0,2); masadan kalkınca belirgin büyür.
    """
    log_len, log_wid, terms = palm_shape(hand, aspect)
    pred_len = float(np.dot(table["length"], terms))
    pred_wid = float(np.dot(table["width"], terms))
    return max(abs(log_len - pred_len), abs(log_wid - pred_wid))


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
                prev_hand_size: float | None = None,
                lift_off: float | None = None) -> HandFrame:
    """Tek karenin `HandFrame`'ini üretir; el yoksa `present=False`.

    `lift_off`: masa modelinden sapma eşiği (config `hand.lift_off`). Verilir ve
    `calib.table` doluysa, sapma eşiği aşan karede güven `LIFT_OFF_FACTOR` ile çarpılır.

    `pinch`: `calib.pinch_ref` doluysa `pinch_values` (0 değiyor, 1 açık); yoksa None.
    """
    if hand is None:
        return _empty_frame(t, frame_id)

    img = np.asarray(hand.image_pts, dtype=float)
    aspect = info.width / info.height if info.height else 1.0

    fingers: dict[str, Finger] = {}
    for name in FINGERS:
        tip = FINGER_JOINTS[name][3]
        curl = _normalize(curl_raw(hand, name, aspect), calib.open_ref[name],
                          calib.fist_ref[name])
        lift_v = None
        if lift is not None and name in lift and lift[name] is not None:
            lift_v = float(lift[name])
        fingers[name] = Finger(
            img=(float(img[tip, 0]), float(img[tip, 1])),
            surface=None,
            curl=curl,
            lift=lift_v,
        )

    thumb_out = _normalize(thumb_raw(hand, aspect), calib.thumb_closed, calib.thumb_open)
    palm_xy = img[list(PALM_POINTS)].mean(axis=0)
    hand_size = _hand_size_img(img, info)

    # Güven: el bulunduysa taban 1.0; yalnızca geometrik cezalar uygulanır (kenar,
    # hand_size sıçraması). `hand.score` kullanılmaz: MediaPipe el varlığı skoru vermez ve
    # sağ/sol skoru aynı geometride 0.5–1.0 arasında oynar (bkz. RawHand).
    confidence = 1.0
    confidence *= _edge_factor(img, info)
    if prev_hand_size is not None and prev_hand_size > _EPS:
        if abs(hand_size / prev_hand_size - 1.0) > HAND_SIZE_JUMP:
            confidence *= JUMP_FACTOR
    if lift_off is not None and calib.table is not None:
        if table_deviation(hand, calib.table, aspect) > lift_off:
            confidence *= LIFT_OFF_FACTOR

    # Pinch (Sürüm C) yalnızca pinch kalibrasyonu varsa; yoksa None (Sürüm A/B değişmez).
    pinch = None
    if calib.pinch_ref is not None:
        pinch = pinch_values(hand, calib.pinch_ref, aspect)

    return HandFrame(
        t=t,
        frame_id=frame_id,
        present=True,
        fingers=fingers,
        thumb_out=thumb_out,
        palm=(float(palm_xy[0]), float(palm_xy[1])),
        hand_size=hand_size,
        confidence=confidence,
        pinch=pinch,
    )
