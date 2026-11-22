"""El faresini barış işaretiyle açıp kapatan kapı (C modu; platformdan bağımsız).

Kamera kullanıcıya bakarken görünen her el imleci oynatır (yüze dokunmak, bardak almak…).
Kapı kapalıyken el izlenir ama jest motoruna verilmez. Barış işareti (işaret ve orta parmak
açık, yüzük ve serçe kıvrık) `hold_ms` tutulunca kapı açılır; yeniden yapılınca kapanır.
İşaret bırakılmadan ikinci kez saymaz.

A modunda (masa) kullanılmaz: orada aynı el şekli iki parmakla kaydırmadır.
"""

OPEN_FINGERS = ("index", "middle")
CLOSED_FINGERS = ("ring", "pinky")


class HandGate:
    def __init__(self, cfg, confidence_min=0.0):
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

    def is_sign(self, f):
        if not f.present or f.confidence < self._conf or not f.fingers:
            return False
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
            if self._since is None:
                self._since = f.t
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
