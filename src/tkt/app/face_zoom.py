"""Yüz mesafesine göre yakınlaştırma seviyesi (platformdan bağımsız karar mantığı).

Kamera kullanıcıya bakarken (C modu) yüzün görüntüdeki boyutu ölçülür. Kalibrasyonda alınan
"normal mesafe" boyutuna oranı uzaklığı verir: oran = normal boyut / şimdiki boyut
(1,0 = normal, 1,3 = %30 daha uzak, 0,8 = daha yakın). Uzaklaşınca seviye artar (büyüt),
yaklaşınca azalır (küçült).

Seviyenin ikide bir oynamaması için üç önlem vardır (hepsi `[face_zoom]` ayarı):
- ölçümlerin medyanı alınır (`window`),
- her sınırın iki yanında ölü bant vardır (`hysteresis`): sınırı geçmek için biraz ötesine
  gitmek gerekir, geri dönmek için de biraz berisine,
- yeni aralıkta `dwell_ms` kesintisiz kalınmadan seviye değişmez.

Yüz görünmezse seviye olduğu gibi kalır. Seviyeyi uygulamak (menüden Büyüt / Küçült) platform
arayüzünün işidir; burası yalnız `target` seviyesini söyler.
"""
import json
import statistics
import threading
from collections import deque
from pathlib import Path

GAP_RESET_S = 2.0           # ölçümler bu kadar kesilirse medyan penceresi baştan dolar
MIN_SAMPLES = 3             # medyan ve kalibrasyon için en az ölçüm


def parse_hotkey(text):
    """"<ctrl>+<alt>+z" -> ({"ctrl", "alt"}, "z"). Tuş tek karakterdir."""
    mods, key = set(), None
    for part in text.lower().split("+"):
        part = part.strip()
        if part.startswith("<") and part.endswith(">"):
            mods.add(part[1:-1])
        elif len(part) == 1 and key is None:
            key = part
        else:
            raise ValueError(f"kısayol anlaşılamadı: {text!r}")
    if key is None or not mods <= {"ctrl", "alt", "cmd", "shift"}:
        raise ValueError(f"kısayol anlaşılamadı: {text!r}")
    return frozenset(mods), key


class FaceZoom:
    """Yüz boyutu ölçümlerinden yakınlaştırma seviyesi üretir.

    `update(t, size)`: `size` yüzün görüntüye oranı (yüz yoksa None). Bir şey değiştiyse
    (seviye, kalibrasyon bitti) kullanıcıya gösterilecek kısa metni, yoksa None döner.
    """

    def __init__(self, cfg, path=None):
        near, far = sorted(cfg["near"]), sorted(cfg["far"])
        if any(v >= 1.0 for v in near) or any(v <= 1.0 for v in far):
            raise ValueError("[face_zoom] near değerleri 1'den küçük, far değerleri büyük olmalı")
        self._bounds = near + far
        self._n_near = len(near)
        self._half = cfg["hysteresis"] / 2
        self._dwell = cfg["dwell_ms"] / 1000
        self._cal_s = cfg["calibrate_ms"] / 1000
        self.period = cfg["sample_ms"] / 1000
        self._window = deque(maxlen=cfg["window"])
        self._path = Path(path if path is not None else cfg["file"]).expanduser()
        self._lock = threading.Lock()
        self.enabled = False
        self.baseline = None
        self.level = 0
        self.ratio = None           # son medyan oran (ekranda göstermek için)
        self._cand = None           # geçilmek istenen seviye ve o aralığa girilen an
        self._cand_t = 0.0
        self._last_t = None
        self._cal = None            # kalibrasyon ölçümleri (sürüyorsa liste)
        self._cal_t = None
        self._load()

    # --- kalıcı durum ---

    def _load(self):
        try:
            data = json.loads(self._path.read_text())
            base = data.get("baseline")
            self.baseline = float(base) if base else None
            self.enabled = bool(data.get("enabled", False))
        except (OSError, ValueError, TypeError, AttributeError):
            pass                    # dosya yok ya da bozuk: kapalı ve kalibrasyonsuz başla

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(
                {"baseline": self.baseline, "enabled": self.enabled}, indent=1))
        except OSError:
            pass

    # --- kullanıcı komutları ---

    def toggle(self):
        """Aç / kapat. Kalibrasyon yoksa açarken kendiliğinden başlar. Yeni durumu döner."""
        with self._lock:
            self.enabled = not self.enabled
            self._cand = None
            if self.enabled and self.baseline is None:
                self._start_calibration()
            if not self.enabled:
                self.level = 0
                self._cal = None
            self._save()
            return self.enabled

    def calibrate(self):
        """Şimdiki mesafeyi normal say: ölçüm `calibrate_ms` sürer, seviye sıfırlanır."""
        with self._lock:
            self._start_calibration()

    def _start_calibration(self):
        self._cal, self._cal_t = [], None
        self.level, self._cand = 0, None
        self._window.clear()

    # --- durum ---

    @property
    def calibrating(self):
        return self._cal is not None

    @property
    def wants_samples(self):
        return self.enabled or self.calibrating

    @property
    def target(self):
        """Uygulanması istenen seviye (kapalıyken ve kalibrasyon sürerken 0)."""
        return self.level if self.enabled and not self.calibrating else 0

    def text(self):
        if self.calibrating:
            return "YAKINLASTIRMA: kalibrasyon, normal mesafende dur"
        if not self.enabled:
            return "YAKINLASTIRMA: kapali"
        r = "" if self.ratio is None else f" (uzaklik {self.ratio:.2f}x)"
        return f"YAKINLASTIRMA: seviye {self.level:+d}{r}"

    # --- ölçüm ---

    def update(self, t, size):
        with self._lock:
            if self._last_t is not None and t - self._last_t > GAP_RESET_S:
                self._window.clear()
                self._cand = None
            self._last_t = t
            if size is None or size <= 0:
                self._window.clear()
                self._cand = None
                return None
            if self._cal is not None:
                return self._calibration_sample(t, size)
            if not self.enabled or self.baseline is None:
                return None
            self._window.append(size)
            if len(self._window) < min(MIN_SAMPLES, self._window.maxlen):
                return None
            self.ratio = self.baseline / statistics.median(self._window)
            zone = self._zone(self.ratio, self.level)
            if zone == self.level:
                self._cand = None
            elif zone != self._cand:
                self._cand, self._cand_t = zone, t
            elif t - self._cand_t >= self._dwell:
                self.level, self._cand = zone, None
                return f"YAKINLASTIRMA seviye {zone:+d} (uzaklik {self.ratio:.2f}x)"
            return None

    def _calibration_sample(self, t, size):
        if self._cal_t is None:
            self._cal_t = t
        self._cal.append(size)
        if t - self._cal_t < self._cal_s or len(self._cal) < MIN_SAMPLES:
            return None
        self.baseline = statistics.median(self._cal)
        self._cal = None
        self.ratio = None
        self._save()
        return "YAKINLASTIRMA kalibrasyonu tamam"

    def _zone(self, ratio, level):
        """Ölü bantlı aralık: `level`den yalnız sınırın yarım bant ötesine geçilince çıkılır."""
        i = level + self._n_near        # altında kalan sınır sayısı
        while i < len(self._bounds) and ratio >= self._bounds[i] + self._half:
            i += 1
        while i > 0 and ratio < self._bounds[i - 1] - self._half:
            i -= 1
        return i - self._n_near


def _norm(title):
    return "".join(c for c in (title or "") if c.isprintable()).strip().casefold()


def pick_zoom_items(menus, in_titles, out_titles):
    """Menülerden Büyüt / Küçült çiftini seçer.

    `menus`: her menü için [(başlık, öğe)] listesi. Yalnız ikisi de aynı menüde olan çiftler
    sayılır; birden çok menüde varsa büyüt başlığı `in_titles`te en önde olan kazanır
    (ör. "Zoom In", "Bigger"den önce). Döner: (büyüt öğesi, küçült öğesi) ya da None
    (uygulama desteklenmiyor sayılır).
    """
    ins, outs = [_norm(t) for t in in_titles], [_norm(t) for t in out_titles]
    best = None
    for items in menus:
        titled = {}
        for title, item in items:
            titled.setdefault(_norm(title), item)
        rank = next((i for i, t in enumerate(ins) if t in titled), None)
        zout = next((titled[t] for t in outs if t in titled), None)
        if rank is not None and zout is not None and (best is None or rank < best[0]):
            best = (rank, titled[ins[rank]], zout)
    return None if best is None else best[1:]


def skip_menu(title, skip_titles):
    """Üst menü aramadan dışlanır mı (ör. Biçim: oradaki Büyüt belgedeki yazıyı değiştirir)."""
    return _norm(title) in {_norm(t) for t in skip_titles}
