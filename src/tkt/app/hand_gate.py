"""El faresini barış işaretiyle açıp kapatan kapı (C modu; platformdan bağımsız).

Kamera kullanıcıya bakarken görünen her el imleci oynatır (yüze dokunmak, bardak almak…).
Kapı kapalıyken el izlenir ama jest motoruna verilmez. Barış işareti (işaret ve orta parmak
açık, yüzük ve serçe kıvrık) `hold_ms` tutulunca kapı açılır; yeniden yapılınca kapanır.
İşaret bırakılmadan ikinci kez saymaz.

İşaret öğretilebilir: `python -m tkt.app.sign_calib` adım adım kalibrasyon penceresi açar ve
her parmak için kabul aralığını `file`a yazar. Öğretilmiş işaret varsa sabit eşikler
(`curl_open` / `curl_closed`) yerine o aralıklar kullanılır; kullanıcının barış işaretinde
yüzük parmağı yarım kıvrık kalabiliyor (canlı deneme, 2026-11-22: 0,34–0,53).

A modunda (masa) kullanılmaz: orada aynı el şekli iki parmakla kaydırmadır.
"""
import json
import math
import statistics
from pathlib import Path

from tkt.core.types import FINGERS

OPEN_FINGERS = ("index", "middle")
CLOSED_FINGERS = ("ring", "pinky")


def learn_ranges(samples, margin):
    """Kıvrım örneklerinden ({parmak: kıvrım} listesi) parmak başına kabul aralığı.

    Parmaklar işaretin kendi ortasına göre "düz" ve "kıvrık" diye ayrılır (orta: en düz ve en
    kıvrık parmağın ortancalarının ortası). Düz parmak için yalnız üst sınır (%90 + `margin`),
    kıvrık parmak için yalnız alt sınır (%10 − `margin`) konur: daha düz ya da daha kıvrık
    tutmak işareti bozmaz. İki yanlı aralık canlıda dardı: kalibrasyonda serçe 0,56 ölçüldü
    (üst sınır 0,73), aynı kullanıcının önceki işaretlerinde 0,72–0,94'tü (2026-11-22).
    """
    mids = {n: statistics.median(s[n] for s in samples) for n in FINGERS}
    middle = (min(mids.values()) + max(mids.values())) / 2
    out = {}
    for name in FINGERS:
        deciles = statistics.quantiles(sorted(s[name] for s in samples), n=10)
        if mids[name] >= middle:
            out[name] = (deciles[0] - margin, math.inf)
        else:
            out[name] = (-math.inf, deciles[-1] + margin)
    return out


def load_ranges(path):
    """Öğretilmiş işaret: {parmak: (alt, üst)}; dosya yoksa ya da bozuksa None."""
    try:
        data = json.loads(Path(path).expanduser().read_text())["ranges"]
        return {n: (-math.inf if data[n][0] is None else float(data[n][0]),
                    math.inf if data[n][1] is None else float(data[n][1]))
                for n in FINGERS}
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return None


def save_ranges(path, ranges):
    data = {n: [None if math.isinf(lo) else round(lo, 3),
                None if math.isinf(hi) else round(hi, 3)]
            for n, (lo, hi) in ranges.items()}
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ranges": data}, indent=1))


def in_ranges(ranges, curls):
    return all(lo <= curls[n] <= hi for n, (lo, hi) in ranges.items())


class HandGate:
    def __init__(self, cfg, confidence_min=0.0, path=None):
        self.enabled = cfg["enabled"]
        self.active = cfg["start_active"] or not self.enabled
        self._open = cfg["curl_open"]
        self._closed = cfg["curl_closed"]
        self._hold = cfg["hold_ms"] / 1000
        self._grace = cfg["grace_ms"] / 1000
        self._conf = confidence_min
        self._since = None          # işaretin başladığı an
        self._seen = None           # işaretin son görüldüğü an
        self._fired = False         # bu işaret için kapı zaten çevrildi
        # Öğretilmiş işaret: {parmak: (alt, üst)} ya da None (sabit eşikler).
        self.ranges = load_ranges(path if path is not None else cfg["file"])

    # --- kapı ---

    def is_sign(self, f):
        if not f.present or f.confidence < self._conf or not f.fingers:
            return False
        if self.ranges is not None:
            return in_ranges(self.ranges, {n: f.fingers[n].curl for n in FINGERS})
        return (all(f.fingers[n].curl < self._open for n in OPEN_FINGERS)
                and all(f.fingers[n].curl > self._closed for n in CLOSED_FINGERS))

    @property
    def progress(self):
        """İşaretin tutulma oranı 0–1 (ekranda göstermek için); işaret yoksa 0."""
        if self._since is None or self._seen is None or self._fired:
            return 0.0
        return min(1.0, (self._seen - self._since) / self._hold) if self._hold > 0 else 1.0

    def update(self, f):
        """Kareyi işler; kapı çevrildiyse yeni durumu (True açık), yoksa None döner."""
        if not self.enabled:
            return None
        if self.is_sign(f):
            if self._since is None or f.t - self._seen > self._grace:
                # Yeni işaret (araya uzun bir kopukluk girdiyse de yeni sayılır).
                self._since, self._fired = f.t, False
            self._seen = f.t
            if not self._fired and f.t - self._since >= self._hold:
                self._fired = True
                self.active = not self.active
                return self.active
        elif self._seen is not None and f.t - self._seen > self._grace:
            # Kısa takip kopmaları (grace) işareti bozmaz; daha uzunu bırakılmış sayılır.
            self._since = self._seen = None
            self._fired = False
        return None
