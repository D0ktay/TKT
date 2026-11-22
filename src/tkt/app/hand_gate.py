"""El faresini barış işaretiyle açıp kapatan kapı (C modu; platformdan bağımsız).

Kamera kullanıcıya bakarken görünen her el imleci oynatır (yüze dokunmak, bardak almak…).
Kapı kapalıyken el izlenir ama jest motoruna verilmez. Barış işareti (işaret ve orta parmak
açık, yüzük ve serçe kıvrık) `hold_ms` tutulunca kapı açılır; yeniden yapılınca kapanır.
İşaret bırakılmadan ikinci kez saymaz.

İşaret öğretilebilir (`start_teach`): `teach_prep_ms` hazırlık, ardından `teach_record_ms`
boyunca tutulan elin dört parmak kıvrımı kaydedilir; her parmak için kabul aralığı çıkarılıp
`file`a yazılır. Öğretilmiş işaret varsa sabit eşikler (`curl_open` / `curl_closed`) yerine o
aralıklar kullanılır; kullanıcının barış işaretinde yüzük parmağı yarım kıvrık kalabiliyor
(canlı deneme, 2026-11-22: 0,34–0,53).

A modunda (masa) kullanılmaz: orada aynı el şekli iki parmakla kaydırmadır.
"""
import json
import math
import statistics
from pathlib import Path

from tkt.core.types import FINGERS

MIN_TEACH_SAMPLES = 30      # kayıtta elin göründüğü en az kare
EDGE = 0.15                 # öğretilen aralık uca bu kadar yakınsa o yanı sınırsız sayılır


OPEN_FINGERS = ("index", "middle")
CLOSED_FINGERS = ("ring", "pinky")


def learn_ranges(samples, margin):
    """Kıvrım örneklerinden ({parmak: kıvrım} listesi) parmak başına kabul aralığı.

    Aralık, örneklerin %10–%90 dilimi ± `margin`dir. Dilim 0'a ya da 1'e `EDGE` kadar
    yakınsa o yan sınırsızdır (dümdüz parmak daha da düz, sıkı yumruk daha da sıkı olabilir).
    """
    out = {}
    for name in FINGERS:
        values = sorted(s[name] for s in samples)
        deciles = statistics.quantiles(values, n=10)
        lo, hi = deciles[0], deciles[-1]
        out[name] = (-math.inf if lo <= EDGE else lo - margin,
                     math.inf if hi >= 1 - EDGE else hi + margin)
    return out


class HandGate:
    def __init__(self, cfg, confidence_min=0.0, path=None):
        self.enabled = cfg["enabled"]
        self.active = cfg["start_active"] or not self.enabled
        self.teach_hotkey = cfg["teach_hotkey"]
        self._open = cfg["curl_open"]
        self._closed = cfg["curl_closed"]
        self._hold = cfg["hold_ms"] / 1000
        self._grace = cfg["grace_ms"] / 1000
        self._prep = cfg["teach_prep_ms"] / 1000
        self._record = cfg["teach_record_ms"] / 1000
        self._margin = cfg["teach_margin"]
        self._path = Path(path if path is not None else cfg["file"]).expanduser()
        self._conf = confidence_min
        self._since = None          # işaretin başladığı an
        self._seen = None           # işaretin son görüldüğü an
        self._fired = False         # bu işaret için kapı zaten çevrildi
        self._teach = None          # öğretme sürüyorsa [başlangıç anı ya da None, örnekler]
        self._events = []           # kullanıcıya gösterilecek kısa metinler
        self.prompt = None          # öğretme sırasında ekranda gösterilen yönerge
        self.ranges = self._load()  # öğretilmiş işaret: {parmak: (alt, üst)} ya da None

    # --- öğretilmiş işaret ---

    def _load(self):
        try:
            data = json.loads(self._path.read_text())["ranges"]
            return {n: (-math.inf if data[n][0] is None else float(data[n][0]),
                        math.inf if data[n][1] is None else float(data[n][1]))
                    for n in FINGERS}
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            return None             # dosya yok ya da bozuk: sabit eşikler

    def _save(self):
        data = {n: [None if math.isinf(lo) else round(lo, 3),
                    None if math.isinf(hi) else round(hi, 3)]
                for n, (lo, hi) in self.ranges.items()}
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps({"ranges": data}, indent=1))
        except OSError:
            pass

    def start_teach(self):
        """İşaret öğretmeyi başlatır; süre, C modundaki ilk karede işlemeye başlar."""
        self._teach = [None, []]
        self.prompt = "OGRETME: hazirlan"

    def drain(self):
        """Birikmiş kısa bildirimler (kayda ve eylem listesine yazmak için)."""
        events, self._events = self._events, []
        return events

    def _teach_step(self, f):
        if self._teach[0] is None:
            self._teach[0] = f.t
        elapsed = f.t - self._teach[0]
        if elapsed < self._prep:
            left = math.ceil(self._prep - elapsed)
            self.prompt = f"OGRETME: {left} sn sonra isaretini yap ve tut"
            return
        if elapsed < self._prep + self._record:
            if f.present and f.fingers and f.confidence >= self._conf:
                self._teach[1].append({n: f.fingers[n].curl for n in FINGERS})
            left = self._prep + self._record - elapsed
            self.prompt = f"OGRETME: isareti TUT ({math.ceil(left)} sn)"
            return
        samples = self._teach[1]
        self._teach, self.prompt = None, None
        # Öğretilen işaret hâlâ tutuluyor olabilir: bırakılmadan kapıyı çevirmesin.
        self._since = self._seen = f.t
        self._fired = True
        if len(samples) < MIN_TEACH_SAMPLES:
            self._events.append("ISARET ogretilemedi: el gorunmedi, yeniden dene")
            return
        self.ranges = learn_ranges(samples, self._margin)
        self._save()
        med = {n: statistics.median(s[n] for s in samples) for n in FINGERS}
        self._events.append("ISARET ogretildi: " + " ".join(
            f"{n[0]}{med[n]:.2f}" for n in FINGERS))

    # --- kapı ---

    def is_sign(self, f):
        if not f.present or f.confidence < self._conf or not f.fingers:
            return False
        if self.ranges is not None:
            return all(lo <= f.fingers[n].curl <= hi for n, (lo, hi) in self.ranges.items())
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
        if self._teach is not None:
            self._teach_step(f)
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
