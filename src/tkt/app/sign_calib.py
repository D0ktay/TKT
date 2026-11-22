"""El kapısı işaretinin (barış işareti) kalibrasyonu: python -m tkt.app.sign_calib

Pencerede kamera görüntüsü (ayna gibi), iskelet, dört parmağın canlı kıvrım çubukları ve
yönerge vardır. Otomatik mod kapalıyken çalıştırılır (kamerayı o kullanır).

1. İşaret yapılır, hazır olunca BOŞLUK'a basılır; `teach_sign_ms` boyunca kaydedilir. El
   kayıt sırasında oynarsa (`teach_still`) kayıt baştan başlar.
2. Deneme: işaret bırakılıp yeniden yapılır; tanınınca sayaç artar. Enter kaydeder.

Kaydı kullanıcı başlatır: zamanlı kayıt da, "el sabitlenince kaydet" de yanlış anı kaydetti
(canlı, 2026-11-22: biri geçiş halindeki eli, öteki işaretten önceki açık eli). Kaydedilen
elde en düz ve en kıvrık parmak arasında `teach_min_gap` kadar fark yoksa (açık el, yumruk)
kalibrasyon reddedilir.

Tuşlar (pencere öndeyken): boşluk kaydı başlat, Enter kaydet, R baştan, Esc / q çık.
"""
import statistics
import tomllib
from pathlib import Path

from tkt.app.hand_gate import HandGate, in_ranges, learn_ranges, save_ranges
from tkt.core.types import FINGERS

REPO = Path(__file__).resolve().parents[3]
WINDOW = "TKT isaret kalibrasyonu"
VIEW_H = 540                # kamera görüntüsünün penceredeki yüksekliği
PANEL_W = 380               # sağdaki parmak çubukları paneli
SAVED_SHOW_S = 1.5          # "kaydedildi" yazısının kaldığı süre
FINGER_LABELS = {"index": "isaret", "middle": "orta", "ring": "yuzuk", "pinky": "serce"}


class PoseRecorder:
    """Sabit tutulan bir el pozunu kaydeder.

    Kayıtta yalnız birbirine yakın kareler tutulur: bir parmağın kıvrımı tutulan karelerde
    `still`den fazla oynarsa en eskiler atılır. Kalan kesit önce `still_s` (sabitlenme), sonra
    `duration` (kayıt) kadar uzayınca poz tamamdır.
    """

    def __init__(self, duration, still, still_s):
        self._duration, self._still, self._still_s = duration, still, still_s
        self._buf = []              # (t, {parmak: kıvrım})

    def reset(self):
        self._buf = []

    def update(self, t, curls):
        """`curls` (el yoksa None) ile ilerler; kayıt oranını 0–1 döner."""
        if curls is None:
            self._buf = []
            return 0.0
        self._buf.append((t, curls))
        while len(self._buf) > 1 and any(
                max(c[n] for _, c in self._buf) - min(c[n] for _, c in self._buf) > self._still
                for n in FINGERS):
            self._buf.pop(0)
        return self.progress

    @property
    def held(self):
        return self._buf[-1][0] - self._buf[0][0] if self._buf else 0.0

    @property
    def settled(self):
        """El sabitlendi, kayıt sürüyor."""
        return self.held >= self._still_s

    @property
    def progress(self):
        if self._duration <= 0:
            return 1.0 if self.settled else 0.0
        return min(1.0, max(0.0, (self.held - self._still_s) / self._duration))

    @property
    def done(self):
        return self.held >= self._still_s + self._duration

    @property
    def samples(self):
        return [c for _, c in self._buf]


def medians(samples):
    return {n: statistics.median(s[n] for s in samples) for n in FINGERS}


def fit_sign(samples, margin, min_gap):
    """İşaret örneklerinden kabul aralıkları; el ayırt edici değilse None.

    Barış işaretinde iki parmak düz, ikisi kıvrıktır. En düz ve en kıvrık parmak arasında
    `min_gap` kadar fark yoksa (açık el, yumruk) kayıt işaret sayılmaz.
    """
    mid = medians(samples)
    if max(mid.values()) - min(mid.values()) < min_gap:
        return None
    return learn_ranges(samples, margin)


class SignCalibration:
    """Kalibrasyon akışı.

    `state`: "ready" (işaret bekleniyor, boşluk kaydı başlatır), "record", "problem"
    (kaydedilen el işarete benzemiyor), "test" (deneme).
    """

    def __init__(self, cfg, confidence_min=0.0):
        self._cfg = cfg
        self._conf = confidence_min
        self.restart()

    def restart(self):
        self.state = "ready"
        self.samples = []
        self.ranges = None
        self.hits = 0               # denemede işaretin tanınma sayısı
        self.recognized = False
        self.curls = None           # son karedeki kıvrımlar (el yoksa None)
        self._gate = None
        self._recorder = None

    def start(self):
        """Kaydı başlatır (boşluk tuşu); el görünmüyorsa bir şey yapmaz."""
        if self.state != "ready" or self.curls is None:
            return False
        self.state = "record"
        self._recorder = PoseRecorder(self._cfg["teach_sign_ms"] / 1000,
                                      self._cfg["teach_still"], 0.0)
        return True

    def _curls(self, f):
        if not f.present or not f.fingers or f.confidence < self._conf:
            return None
        return {n: f.fingers[n].curl for n in FINGERS}

    def update(self, f):
        self.curls = self._curls(f)
        if self.state == "record":
            self._recorder.update(f.t, self.curls)
            if self._recorder.done:
                self._finish()
        elif self.state == "test":
            self.recognized = self.curls is not None and in_ranges(self.ranges, self.curls)
            if self._gate.update(f) is not None:
                self.hits += 1

    def _finish(self):
        self.samples = self._recorder.samples
        self.ranges = fit_sign(self.samples, self._cfg["teach_margin"],
                               self._cfg["teach_min_gap"])
        if self.ranges is None:
            self.state = "problem"
            return
        self.state = "test"
        self._gate = HandGate({**self._cfg, "enabled": True, "start_active": False},
                              self._conf, path="")
        self._gate.ranges = self.ranges

    # --- ekrana yazılacaklar ---

    @property
    def progress(self):
        if self.state == "record":
            return self._recorder.progress
        return self._gate.progress if self.state == "test" else 0.0

    @property
    def title(self):
        if self.state == "problem":
            return "OLMADI"
        if self.state == "test":
            return "2/2: DENE"
        return "1/2: BARIS ISARETI" + (" - KAYIT" if self.state == "record" else "")

    @property
    def instruction(self):
        if self.state == "problem":
            return "Bu el isarete benzemiyor (acik el ya da yumruk gibi). R: bastan"
        if self.state == "test":
            return (f"Elini birak, isareti yeniden yap. Taninma: {self.hits}   "
                    "ENTER: kaydet   R: bastan")
        if self.curls is None:
            return "El gorunmuyor: elini kameraya goster."
        if self.state == "record":
            return "Kaydediliyor, isareti sabit tut..."
        return "Baris isaretini yap, hazir olunca BOSLUK tusuna bas."

    def save(self, path=None):
        save_ranges(path if path is not None else self._cfg["file"], self.ranges)


def render(image, hand, cal, saved=False):
    """Pencere görüntüsü: aynalı kamera + iskelet, üstte yönerge, sağda parmak çubukları."""
    import cv2
    import numpy as np

    from tkt.vision.debug_view import draw_skeleton, to_bgr

    out = to_bgr(image).copy()
    if hand is not None:
        draw_skeleton(out, hand)
    h, w = out.shape[:2]
    view = cv2.flip(cv2.resize(out, (round(w * VIEW_H / h), VIEW_H)), 1)   # ayna gibi
    vw = view.shape[1]
    white, gray, green, yellow, red = ((255, 255, 255), (150, 150, 150), (80, 220, 80),
                                       (60, 220, 240), (60, 60, 240))
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(view, (0, 0), (vw, 84), (0, 0, 0), -1)
    title_color = red if cal.state in ("problem", "record") else yellow
    cv2.putText(view, "KAYDEDILDI" if saved else cal.title, (14, 36), font, 1.0,
                green if saved else title_color, 2, cv2.LINE_AA)
    cv2.putText(view, "" if saved else cal.instruction, (14, 68), font, 0.55, white, 1,
                cv2.LINE_AA)
    cv2.rectangle(view, (0, 84), (vw, 96), (40, 40, 40), -1)
    cv2.rectangle(view, (0, 84), (round(vw * cal.progress), 96), green, -1)
    if cal.state == "test":
        label = "TANINDI" if cal.recognized else "isaret yok"
        cv2.putText(view, label, (14, VIEW_H - 20), font, 1.2,
                    green if cal.recognized else gray, 3, cv2.LINE_AA)

    panel = np.zeros((VIEW_H, PANEL_W, 3), np.uint8)
    cv2.putText(panel, "parmak kivrimi (sol: duz, sag: kivrik)", (16, 30), font, 0.48, gray, 1,
                cv2.LINE_AA)
    x0, bar_w = 16, PANEL_W - 32
    for i, name in enumerate(FINGERS):
        y = 80 + i * 100
        cv2.putText(panel, FINGER_LABELS[name], (x0, y - 14), font, 0.6, white, 1, cv2.LINE_AA)
        cv2.rectangle(panel, (x0, y), (x0 + bar_w, y + 22), (60, 60, 60), -1)
        if cal.ranges is not None:
            lo, hi = cal.ranges[name]
            a, b = max(0.0, min(1.0, lo)), max(0.0, min(1.0, hi))
            cv2.rectangle(panel, (x0 + round(a * bar_w), y), (x0 + round(b * bar_w), y + 22),
                          (40, 110, 40), -1)
        if cal.curls is not None:
            v = max(0.0, min(1.0, cal.curls[name]))
            x = x0 + round(v * bar_w)
            cv2.rectangle(panel, (x - 3, y - 4), (x + 3, y + 26), white, -1)
            cv2.putText(panel, f"{cal.curls[name]:.2f}", (x0 + bar_w - 48, y - 14), font, 0.5,
                        gray, 1, cv2.LINE_AA)
    foot = "yesil bant: isaretin kabul araligi" if cal.ranges is not None else ""
    cv2.putText(panel, foot, (16, VIEW_H - 44), font, 0.48, gray, 1, cv2.LINE_AA)
    cv2.putText(panel, "BOSLUK: kayit   R: bastan   Esc: cik", (16, VIEW_H - 18), font, 0.5,
                gray, 1, cv2.LINE_AA)
    return np.hstack([view, panel])


def main():
    import cv2

    from tkt.vision.live_source import open_live_source

    cfg = tomllib.loads((REPO / "config/control.toml").read_text())
    cal = SignCalibration(cfg["hand_gate"], cfg["general"]["confidence_min"])
    source = open_live_source(str(REPO / "config/vision-front.toml"))
    saved_until = None
    try:
        cv2.namedWindow(WINDOW)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
        for f, hand, img in source.frames_raw():
            if saved_until is None:
                cal.update(f)
            cv2.imshow(WINDOW, render(img, hand, cal, saved=saved_until is not None))
            key = cv2.waitKey(1) & 0xFF
            if saved_until is not None:
                if f.t >= saved_until:
                    break
            elif key in (27, ord("q")):
                print("vazgecildi, kaydedilmedi", flush=True)
                break
            elif key in (ord("r"), ord("R")):
                cal.restart()
            elif key == 32 and cal.state == "ready":
                cal.start()
            elif key in (13, 10) and cal.state == "test":
                cal.save()
                print(f"kaydedildi: {cal.ranges}", flush=True)
                saved_until = f.t + SAVED_SHOW_S
    finally:
        source.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    if cal.samples:
        print("isaret", {n: round(v, 2) for n, v in medians(cal.samples).items()}, flush=True)


if __name__ == "__main__":
    main()
