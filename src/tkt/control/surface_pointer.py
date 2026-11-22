"""Aktif alan (yüzey) imleci (deney): `Finger.surface` (u, v) -> imleç hareketi.

Alan koordinatı 0–1'dir; yatay ölçek için `[surface_pointer] aspect` (genişlik/yükseklik)
kullanılır ve vision `[surface] aspect` ile aynı olmalıdır. Konum (u·aspect, v) "alan
yüksekliği" biriminde One Euro ile süzülür; hız ivmesi (smoothstep) PointerMotion'daki gibi.

`SurfacePointer`, `PointerMotion`'ın arayüzünü (rebase/reset/rollback/rollback_at) taşır ve
GestureEngine'de onun yerine geçer: parmağın `surface` değeri varsa `SurfaceMotion`, yoksa
eski `PointerMotion` kullanılır. Tık/sürükleme/dondurma/geri alma mantığı değişmez.
"""

from collections import deque
from collections.abc import Mapping
from math import hypot, isfinite
from typing import Any

from tkt.control.filters import OneEuro
from tkt.control.pointer import HISTORY_MS, PointerMotion
from tkt.core.types import Finger

Move = tuple[int, int]


def _positive(cfg: Mapping[str, Any], key: str) -> float:
    value = float(cfg[key])
    if not isfinite(value) or value <= 0:
        raise ValueError(f"surface_pointer.{key} pozitif ve sonlu olmalı: {value!r}")
    return value


class SurfaceMotion:
    """Alan (u, v) konumunu göreli ekran pikseline çevirir.

    Göreli modda (varsayılan) kazanç "ekran pikseli / alan yüksekliği", hız eşikleri
    "alan yüksekliği / sn" birimindedir. Parmak ucu alanın `edge_margin` ötesindeyse hareket
    üretilmez ve içeri girişte ilk kare referans alınır (zıplama yok).

    Mutlak modda (`absolute = true`, yalnız karşılaştırma için) alan ekrana (`screen_w` ×
    `screen_h`) eşlenir. Backend'de mutlak hareket olmadığı için son gönderilen sanal konum
    tutulur ve hedefle farkı göreli hareket olarak gönderilir. İlk harekette imleç önce ekran
    kenarlarına itilerek (backend kırpar) bilinen konuma getirilir.
    """

    def __init__(self, cfg: Mapping[str, Any], pointer_cfg: Mapping[str, Any]) -> None:
        """`[surface_pointer]` ayarlarını ve zamanlama için `[pointer]` ayarlarını al."""
        self._aspect = _positive(cfg, "aspect")
        self._gain_min = float(cfg["gain_min"])
        self._gain_range = float(cfg["gain_max"]) - self._gain_min
        if not isfinite(self._gain_min) or not isfinite(self._gain_range) or self._gain_range < 0:
            raise ValueError("gain_max ve gain_min sonlu olmalı ve gain_max >= gain_min olmalı")
        self._speed_lo = float(cfg["speed_lo"])
        self._speed_range = float(cfg["speed_hi"]) - self._speed_lo
        if not isfinite(self._speed_lo) or not isfinite(self._speed_range) \
                or self._speed_range <= 0:
            raise ValueError("speed_hi ve speed_lo sonlu olmalı ve speed_hi > speed_lo olmalı")
        self._margin = float(cfg["edge_margin"])
        if not isfinite(self._margin) or self._margin < 0:
            raise ValueError("edge_margin negatif olamaz")
        self._absolute = bool(cfg.get("absolute", False))
        self._screen = (_positive(cfg, "screen_w"), _positive(cfg, "screen_h")) \
            if self._absolute else (1.0, 1.0)
        filter_options = {"d_cutoff": cfg["d_cutoff"]} if "d_cutoff" in cfg else {}
        self._filter = OneEuro(cfg["min_cutoff"], cfg["beta"], **filter_options)
        self._gap_reset_s = pointer_cfg["gap_reset_ms"] / 1000.0
        self._history_s = max(HISTORY_MS, pointer_cfg["rollback_ms"]) / 1000.0
        self._history: deque[tuple[float, int, int]] = deque()
        # Mutlak mod: son gönderilen sanal imleç konumu (None: henüz bilinmiyor).
        self._virtual: tuple[int, int] | None = None
        self.reset()

    def inside(self, uv: tuple[float, float]) -> bool:
        """(u, v) alan içinde mi (`edge_margin` kadar dışarısı dahil)?"""
        lo, hi = -self._margin, 1.0 + self._margin
        return lo <= uv[0] <= hi and lo <= uv[1] <= hi

    def update(self, uv: tuple[float, float], t: float) -> list[Move]:
        """Yeni alan konumundan gönderilecek hareketler (çoğunlukla 0 veya 1 öğe)."""
        previous_t = self._t
        if previous_t is not None:
            if t < previous_t or (t == previous_t and self._pos is not None):
                return []
            if t > previous_t + self._gap_reset_s:
                self.rebase()
        self._t = t
        while self._history and self._history[0][0] + self._history_s < t:
            self._history.popleft()
        if not (isfinite(uv[0]) and isfinite(uv[1])) or not self.inside(uv):
            self.rebase()       # dışarıda hareket yok; içeri girişte referans yeniden alınır
            return []

        u = min(1.0, max(0.0, uv[0])) if self._absolute else uv[0]
        v = min(1.0, max(0.0, uv[1])) if self._absolute else uv[1]
        previous_pos = self._pos
        x, y = self._filter((u * self._aspect, v), t)
        self._pos = x, y
        if self._absolute:
            return self._absolute_moves(x / self._aspect, y, t)
        if previous_t is None or previous_pos is None or t <= previous_t:
            return []

        dx, dy = x - previous_pos[0], y - previous_pos[1]
        speed = hypot(dx, dy) / (t - previous_t)
        fraction = min(1.0, max(0.0, (speed - self._speed_lo) / self._speed_range))
        gain = self._gain_min + self._gain_range * fraction**2 * (3.0 - 2.0 * fraction)
        out_x = dx * gain + self._remainder[0]
        out_y = dy * gain + self._remainder[1]
        send_x, send_y = round(out_x), round(out_y)
        self._remainder = out_x - send_x, out_y - send_y
        if not (send_x or send_y):
            return []
        self._history.append((t, send_x, send_y))
        return [(send_x, send_y)]

    def _absolute_moves(self, u: float, v: float, t: float) -> list[Move]:
        w, h = self._screen
        tx = round(min(1.0, max(0.0, u)) * (w - 1))
        ty = round(min(1.0, max(0.0, v)) * (h - 1))
        moves: list[Move] = []
        if self._virtual is None:
            # Konum bilinmiyor: sola it (x=0), hedef x'e git ve yukarı it (y=0), sonra hedef y.
            # Köşeye değmeden kenarlarda kırpılır (sıcak köşeler tetiklenmesin).
            far_x, far_y = -4 * round(w), -4 * round(h)
            moves += [(far_x, 0), (tx, far_y)]
            self._virtual = (tx, 0)
            self._history.clear()
        dx, dy = tx - self._virtual[0], ty - self._virtual[1]
        if dx or dy:
            self._virtual = (tx, ty)
            self._history.append((t, dx, dy))
            moves.append((dx, dy))
        return moves

    def rebase(self) -> None:
        """Sonraki kareyi hareket üretmeden referans al."""
        self._filter.reset()
        self._pos: tuple[float, float] | None = None
        self._remainder = (0.0, 0.0)

    def rollback_at(self, ms: float, t: float) -> Move:
        """`t` zamanından geriye `ms` içindeki hareketi geri al."""
        window_s = ms / 1000.0
        undo_x = undo_y = 0
        while self._history and self._history[-1][0] + window_s >= t:
            _, dx, dy = self._history.pop()
            undo_x -= dx
            undo_y -= dy
        if self._virtual is not None and (undo_x or undo_y):
            self._virtual = (self._virtual[0] + undo_x, self._virtual[1] + undo_y)
        return undo_x, undo_y

    def reset(self) -> None:
        """Filtreyi ve hareket geçmişini sıfırla (mutlak modda sanal konum korunur)."""
        self.rebase()
        self._t: float | None = None
        self._history.clear()


class SurfacePointer:
    """PointerMotion yerine geçer: `surface` varsa SurfaceMotion, yoksa PointerMotion.

    Kaynak değişince diğerinin referansı sıfırlanır (zıplama yok). Geri alma iki kaynağın
    son hareketlerini birlikte geri alır.
    """

    def __init__(self, pointer: PointerMotion, surface: SurfaceMotion) -> None:
        self._pointer = pointer
        self._surface = surface
        self._using_surface: bool | None = None
        self._t: float | None = None

    def update_finger(self, finger: Finger, hand_size: float, t: float) -> list[Move]:
        """İşaret parmağından gönderilecek hareketler."""
        use_surface = finger.surface is not None
        if self._using_surface is not None and use_surface != self._using_surface:
            self._pointer.rebase()
            self._surface.rebase()
        self._using_surface = use_surface
        if self._t is None or t > self._t:
            self._t = t
        if use_surface:
            return self._surface.update(finger.surface, t)
        move = self._pointer.update(finger.img, hand_size, t)
        return [move] if move != (0, 0) else []

    def rebase(self) -> None:
        self._pointer.rebase()
        self._surface.rebase()

    def rollback(self, ms: float) -> Move:
        if self._t is None:
            return 0, 0
        return self.rollback_at(ms, self._t)

    def rollback_at(self, ms: float, t: float) -> Move:
        px, py = self._pointer.rollback_at(ms, t)
        sx, sy = self._surface.rollback_at(ms, t)
        return px + sx, py + sy

    def reset(self) -> None:
        self._pointer.reset()
        self._surface.reset()
        self._using_surface = None
        self._t = None
