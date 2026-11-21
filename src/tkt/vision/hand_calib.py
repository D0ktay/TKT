"""Kişisel el kalibrasyonu (V2). PART2 2.2, 2.4.

`curl`, `thumb_out` ve `lift` ham değerlerini 0–1 aralığına oturtan referanslar.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from tkt.vision.tracker import RawHand


@dataclass(frozen=True)
class HandCalibration:
    open_ref: dict[str, float]         # parmak adı → curl_raw (açık el)
    fist_ref: dict[str, float]         # parmak adı → curl_raw (yumruk)
    thumb_closed: float
    thumb_open: float
    tap_ref: dict[str, float] | None   # parmak adı → tap tepe değeri (Faz 2)
    angle_sign: float = 1.0            # kalkış yönü, tap kalibrasyonundan
    # Masa modeli: el masadayken avuç uzunluğu ve genişliğinin (log) avuç konumuna göre
    # ikinci derece katsayıları ("length", "width" → 6 sayı). None: kalkış algılama kapalı.
    table: dict[str, list[float]] | None = None


# Masa modeli: anahtarlar ve terim sayısı (1, x, y, x², y², xy).
TABLE_KEYS: tuple[str, ...] = ("length", "width")
TABLE_TERMS = 6
# fit_table: en az bu kadar örnek (kamera ~30 fps'de ~2 sn).
TABLE_MIN_SAMPLES = 60


# Kalibrasyon yokken kullanılır. Kaba değerlerdir; ilk kayıtlar bununla alınır, `fit_hand`
# kişiye göre yeniler.
# - curl_raw (orta + uç eklem bükülmesi, radyan, görüntü noktaları): açık elde eklemler
#   neredeyse düz (~0,2); yumrukta ~3,0 (kayıtta kıvrık parmak 2,6–2,9 ölçüldü).
# - thumb_raw (görüntüde başparmak ucu–işaret orta eklemi / bilek–orta kök): kayıtlarda
#   ele yapışık başparmak ~0,35–0,5, açık ~0,9–1,4.
DEFAULT: HandCalibration = HandCalibration(
    open_ref={"index": 0.2, "middle": 0.2, "ring": 0.2, "pinky": 0.2},
    fist_ref={"index": 3.0, "middle": 3.0, "ring": 3.0, "pinky": 3.0},
    thumb_closed=0.45,
    thumb_open=1.1,
    tap_ref=None,
)


def _floats(d: object, key: str) -> dict[str, float]:
    if not isinstance(d, dict):
        raise ValueError(f"kalibrasyon: {key!r} sözlük olmalı")
    return {str(k): float(v) for k, v in d.items()}


def _from_dict(d: dict) -> HandCalibration:
    tap_ref = d.get("tap_ref")
    table = d.get("table")
    return HandCalibration(
        open_ref=_floats(d["open_ref"], "open_ref"),
        fist_ref=_floats(d["fist_ref"], "fist_ref"),
        thumb_closed=float(d["thumb_closed"]),
        thumb_open=float(d["thumb_open"]),
        tap_ref=None if tap_ref is None else _floats(tap_ref, "tap_ref"),
        angle_sign=float(d.get("angle_sign", 1.0)),
        table=None if table is None else _table(table),
    )


def _table(d: object) -> dict[str, list[float]]:
    if not isinstance(d, dict):
        raise ValueError("kalibrasyon: 'table' sözlük olmalı")
    out: dict[str, list[float]] = {}
    for key in TABLE_KEYS:
        v = d.get(key)
        if not isinstance(v, list) or len(v) != TABLE_TERMS:
            raise ValueError(f"kalibrasyon: table[{key!r}] {TABLE_TERMS} sayılık liste olmalı")
        out[key] = [float(x) for x in v]
        if not all(math.isfinite(x) for x in out[key]):
            raise ValueError(f"kalibrasyon: table[{key!r}] sonlu olmalı")
    return out


def load(path: Path) -> HandCalibration | None:
    """Dosyadan okur; dosya yoksa None. `~` genişletilir.

    Dosya var ama bozuksa `ValueError` yükseltir.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("kök nesne sözlük olmalı")
        return _from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{p}: kalibrasyon okunamadı: {e}") from e


def save(calib: HandCalibration, path: Path) -> None:
    """JSON olarak yazar; dizini oluşturur, `~` genişletilir."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(asdict(calib), ensure_ascii=False, indent=2) + "\n"
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def fit_hand(open_s: list[RawHand], fist_s: list[RawHand],
             thumb_closed_s: list[RawHand], aspect: float = 1.0) -> HandCalibration:
    """Açık el, yumruk ve kapalı başparmak örneklerinin medyanından kalibrasyon.

    - `open_ref[f]`: açık el örneklerinde `curl_raw` medyanı; `fist_ref[f]`: yumrukta.
    - `thumb_open`: açık el örneklerinde `thumb_raw` medyanı; `thumb_closed`: başparmak
      kapalı örneklerinde.

    Medyan, tek tük bozuk karelerin sonucu kaydırmasını önler. Boş liste veya yumruk
    değeri açık elden büyük olmayan parmak (ya da açık başparmak değeri kapalıdan büyük
    değilse) `ValueError` yükseltir. `tap_ref` None, `angle_sign` 1.0 kalır. `aspect`:
    görüntü genişliği / yüksekliği, `curl_raw`'a geçer (`build_frame` ile aynı olmalı).
    """
    # features bu modülden HandCalibration'ı import eder; döngüyü önlemek için burada.
    from tkt.vision.features import FINGER_JOINTS, curl_raw, thumb_raw

    for name, samples in (("open_s", open_s), ("fist_s", fist_s),
                          ("thumb_closed_s", thumb_closed_s)):
        if not samples:
            raise ValueError(f"fit_hand: {name} boş")

    open_ref: dict[str, float] = {}
    fist_ref: dict[str, float] = {}
    for finger in FINGER_JOINTS:
        o = _median([curl_raw(h, finger, aspect) for h in open_s])
        f = _median([curl_raw(h, finger, aspect) for h in fist_s])
        if not f > o:
            raise ValueError(
                f"fit_hand: {finger} için yumruk ({f:.3f}) açık elden ({o:.3f}) büyük değil"
            )
        open_ref[finger] = o
        fist_ref[finger] = f

    thumb_open = _median([thumb_raw(h, aspect) for h in open_s])
    thumb_closed = _median([thumb_raw(h, aspect) for h in thumb_closed_s])
    if not thumb_open > thumb_closed:
        raise ValueError(
            f"fit_hand: açık başparmak ({thumb_open:.3f}) kapalıdan ({thumb_closed:.3f}) "
            "büyük değil"
        )

    return HandCalibration(
        open_ref=open_ref,
        fist_ref=fist_ref,
        thumb_closed=thumb_closed,
        thumb_open=thumb_open,
        tap_ref=None,
    )


def fit_table(samples: list[RawHand], aspect: float = 1.0) -> dict[str, list[float]]:
    """El masada gezinirken toplanan örneklerden masa modeli (en küçük kareler).

    Her örnekte `features.palm_shape` ile avuç uzunluğu ve genişliğinin logu ve konum
    terimleri alınır; ikisi için de konuma göre ikinci derece yüzey uydurulur. Az örnek veya
    el yeterince geniş bir alanda gezinmediyse (terimler doğrusal bağımlı) `ValueError`.
    """
    from tkt.vision.features import palm_shape

    if len(samples) < TABLE_MIN_SAMPLES:
        raise ValueError(f"fit_table: en az {TABLE_MIN_SAMPLES} örnek gerekli ({len(samples)})")
    rows = [palm_shape(h, aspect) for h in samples]
    terms = np.array([r[2] for r in rows])
    if np.linalg.matrix_rank(terms) < TABLE_TERMS:
        raise ValueError("fit_table: eli mousepad'in daha geniş bir alanında gezdirin")
    out: dict[str, list[float]] = {}
    for k, key in enumerate(TABLE_KEYS):
        target = np.array([r[k] for r in rows])
        coef, *_ = np.linalg.lstsq(terms, target, rcond=None)
        out[key] = [float(c) for c in coef]
    return out


# Kalibrasyon adımları sırayla: açık el, yumruk, başparmak kapalı.
STEPS: tuple[str, ...] = ("open", "fist", "thumb_closed")
# Her adımın varsayılan süresi (saniye).
STEP_SECONDS = 2.0


class CalibrationSession:
    """Kalibrasyon adımlarını zamana göre yürütür ve örnekleri toplar (kamera/GUI yok).

    Zaman ilk `add` çağrısındaki `t` ile başlar. Her adım `step_s` saniye sürer; bir
    örnek, zamanına düşen adıma yazılır. `None` (el yok) örnekler atlanır ama zaman
    yine ilerler. Tüm adımlar bitince `done` olur ve sonraki örnekler yok sayılır.
    """

    def __init__(self, step_s: float = STEP_SECONDS) -> None:
        if not step_s > 0:
            raise ValueError("step_s pozitif olmalı")
        self.step_s = float(step_s)
        self.samples: dict[str, list[RawHand]] = {s: [] for s in STEPS}
        self._t0: float | None = None
        self._elapsed = 0.0

    @property
    def _index(self) -> int:
        return min(len(STEPS), int(self._elapsed // self.step_s))

    @property
    def done(self) -> bool:
        return self._index >= len(STEPS)

    @property
    def current_step(self) -> str | None:
        """Geçerli adımın adı; bittiyse None."""
        i = self._index
        return None if i >= len(STEPS) else STEPS[i]

    @property
    def progress(self) -> float:
        """Geçerli adımın ilerlemesi (0–1); bittiyse 1."""
        if self.done:
            return 1.0
        return (self._elapsed - self._index * self.step_s) / self.step_s

    @property
    def total_progress(self) -> float:
        """Tüm oturumun ilerlemesi (0–1)."""
        return min(1.0, self._elapsed / (self.step_s * len(STEPS)))

    def add(self, hand: RawHand | None, t: float) -> None:
        """`t` anındaki örneği geçerli adıma ekler; `hand` None ise yalnızca zaman ilerler."""
        if self.done:
            return
        if self._t0 is None:
            self._t0 = float(t)
        # Geri giden zaman damgası ilerlemeyi geri almaz.
        self._elapsed = max(self._elapsed, float(t) - self._t0)
        step = self.current_step
        if step is None or hand is None:
            return
        self.samples[step].append(hand)

    def result(self, aspect: float = 1.0) -> HandCalibration:
        """Toplanan örneklerden `fit_hand`. Oturum bitmediyse veya veri bozuksa `ValueError`."""
        if not self.done:
            raise ValueError("kalibrasyon oturumu bitmedi")
        return fit_hand(self.samples["open"], self.samples["fist"],
                        self.samples["thumb_closed"], aspect)


# fit_taps: örnekler arasında bundan uzun boşluk yeni bir bip penceresi başlatır (sn).
TAP_GAP_S = 0.25
# Pencerenin ilk bu kadar saniyesi taban çizgisidir (bipten sonra tepki ~150 ms'den önce).
TAP_BASE_S = 0.1
# Bir parmağın penceredeki tepesi, o penceredeki en büyük parmak tepesinin en az bu oranıysa
# "bu parmak kalktı" sayılır (ikili tapta işaret ve orta, tek tapta yalnızca işaret).
TAP_PEAK_REL = 0.5
# Parmak pencerelerin en az bu oranında kalkmışsa kendi tap_ref'i bulunur; yoksa bulunan
# parmakların medyanıyla doldurulur.
TAP_MIN_FOUND = 0.25


def _tap_windows(samples: list[tuple[float, RawHand]],
                 bounds: list[tuple[float, float]] | None = None,
                 ) -> list[list[tuple[float, RawHand]]]:
    ordered = sorted(samples, key=lambda s: s[0])
    if bounds is not None:
        # Verilen sınırlar: [başlangıç, bitiş); boş pencereler atılır.
        out = [[(t, h) for t, h in ordered if float(a) <= t < float(b)]
               for a, b in sorted(bounds)]
        return [w for w in out if w]
    windows: list[list[tuple[float, RawHand]]] = []
    for t, hand in ordered:
        if not windows or t - windows[-1][-1][0] > TAP_GAP_S:
            windows.append([])
        windows[-1].append((t, hand))
    return windows


def fit_taps(calib: HandCalibration, samples: list[tuple[float, RawHand]], *,
             aspect: float = 1.0,
             windows: list[tuple[float, float]] | None = None) -> HandCalibration:
    """Tap örneklerinden `tap_ref` ve `angle_sign` (Faz 2).

    `samples`: bip pencerelerinden (bip anından ~1,5 sn) toplanmış `(t, el)` kareleri.
    `windows` verilirse pencere sınırları bunlardır (`[başlangıç, bitiş)`; dışındaki örnekler
    yok sayılır, boş pencereler atılır). Verilmezse pencereler zaman boşluğundan (`TAP_GAP_S`)
    ayrılır; bu, pencereler arasında örnek toplanmadığında çalışır. Her pencerenin ilk
    `TAP_BASE_S` saniyesinin medyanı o pencerenin taban çizgisidir.

    `aspect`: görüntü genişliği / yüksekliği; `LiftEstimator` ile aynı değer verilmelidir
    (görüntü boyu piksel oranıyla ölçülür, bkz. `lift.raw_signals`).

    - `angle_sign`: her pencerede açı değişimi en büyük olan parmağın, en büyük değişim
      anındaki işareti; pencerelerin çoğunluğu. Eşitlikte `calib.angle_sign` korunur.
    - `tap_ref[f]`: `lift_raw` (lift.FIT_WEIGHTS ağırlıklarıyla, `angle_sign` uygulanmış)
      pencere tepelerinin medyanı; yalnızca parmağın kalktığı pencerelerden (`TAP_PEAK_REL`).
      Kalkışı bulunamayan parmaklar (ör. tek tapta orta, her zaman yüzük ve serçe) bulunan
      parmakların medyanıyla doldurulur.

    Örnek yoksa, `aspect` geçersizse veya hiçbir parmakta pozitif tepe yoksa `ValueError`.
    """
    # lift bu modülden HandCalibration'ı import eder; döngüyü önlemek için burada.
    from tkt.vision.lift import FIT_WEIGHTS, raw_signals

    aspect = float(aspect)
    if not (math.isfinite(aspect) and aspect > 0):
        raise ValueError(f"fit_taps: aspect sonlu ve pozitif olmalı: {aspect!r}")
    wins = _tap_windows(samples, windows)
    if not wins:
        raise ValueError("fit_taps: örnek yok")

    # Her pencere: parmak → (açı farkı dizisi, boy farkı dizisi), taban çizgisine göre.
    deltas: list[dict[str, tuple[list[float], list[float]]]] = []
    for win in wins:
        sig = [raw_signals(h, aspect) for _t, h in win]
        t0 = win[0][0]
        n_base = max(1, sum(1 for t, _h in win if t - t0 < TAP_BASE_S))
        d: dict[str, tuple[list[float], list[float]]] = {}
        for f in sig[0]:
            a0 = _median([s[f][0] for s in sig[:n_base]])
            l0 = _median([s[f][1] for s in sig[:n_base]])
            d[f] = ([s[f][0] - a0 for s in sig], [l0 - s[f][1] for s in sig])
        deltas.append(d)

    votes = 0
    for d in deltas:
        best = max((da for da, _dl in d.values()), key=lambda xs: max(abs(x) for x in xs))
        peak = max(best, key=abs)
        if abs(peak) > 0:
            votes += 1 if peak > 0 else -1
    angle_sign = calib.angle_sign if votes == 0 else (1.0 if votes > 0 else -1.0)

    wa, wl = FIT_WEIGHTS["w_angle"], FIT_WEIGHTS["w_length"]
    found: dict[str, list[float]] = {f: [] for f in deltas[0]}
    for d in deltas:
        peaks = {f: max(wa * angle_sign * a + wl * ln for a, ln in zip(da, dl, strict=True))
                 for f, (da, dl) in d.items()}
        top = max(peaks.values())
        if not top > 0:
            continue
        for f, p in peaks.items():
            if p >= TAP_PEAK_REL * top:
                found[f].append(p)

    n = len(deltas)
    tap_ref = {f: _median(v) for f, v in found.items() if v and len(v) >= TAP_MIN_FOUND * n}
    if not tap_ref:
        raise ValueError("fit_taps: hiçbir parmakta kalkış tepesi bulunamadı")
    fill = _median(list(tap_ref.values()))
    tap_ref = {f: tap_ref.get(f, fill) for f in found}
    return replace(calib, tap_ref=tap_ref, angle_sign=angle_sign)
