"""El kapısı işaretinin kalibrasyonu (C modu kamerasıyla): python -m tkt.app.sign_calib

Pencerede kamera görüntüsü, iskelet, dört parmağın canlı kıvrım çubukları ve adım adım
yönerge vardır. Otomatik mod kapalıyken çalıştırılır (kamerayı o kullanır).

Adımlar:
1. İşaret (barış işareti ya da kullanıcının seçtiği başka bir el şekli).
2–4. İşaretle karışmaması gereken pozlar: rahat açık el (imleç gezdirirken), tık (başparmak
   + işaret birleşik), yumruk.
5. Deneme: işaret bırakılıp yeniden yapılır; tanınınca sayaç artar. Enter kaydeder.

Her poz, el `teach_still_ms` boyunca sabit kaldıktan sonra kaydedilir; el oynarsa kayıt baştan
başlar. Böylece geçiş anındaki yarım kıvrık el kaydedilmez (ilk denemede zamanlı kayıt dört
parmağı da 0,25–0,35 kaydetmişti; canlı, 2026-11-22). Öteki pozlardan biri işaretin kabul
aralığına düşerse aralık, en çok ayrışan parmakta ikisinin ortasına çekilir; hiçbir parmakta
`teach_min_gap` kadar fark yoksa kalibrasyon reddedilir.

Tuşlar (pencere öndeyken): Enter / boşluk kaydet (deneme adımında), R baştan, Esc / q çık.
"""
import statistics
import tomllib
from pathlib import Path

from tkt.app.hand_gate import HandGate, in_ranges, learn_ranges, save_ranges
from tkt.core.types import FINGERS

REPO = Path(__file__).resolve().parents[3]
WINDOW = "TKT isaret kalibrasyonu"
PAUSE_S = 1.5               # iki poz arasında "tamam" yazısının kaldığı süre
VIEW_H = 540                # kamera görüntüsünün penceredeki yüksekliği
PANEL_W = 380               # sağdaki parmak çubukları paneli
SAVED_SHOW_S = 1.5          # "kaydedildi" yazısının kaldığı süre

# (anahtar, başlık, yönerge) — yazılar ASCII: OpenCV yazı tipi Türkçe harf çizemez.
POSES = (
    ("sign", "BARIS ISARETI", "Isaret ve orta parmak acik, yuzuk ve serce kivrik. Sabit tut."),
    ("relaxed", "RAHAT EL", "Elini imlec gezdirir gibi rahatca ac. Sabit tut."),
    ("pinch", "TIK", "Basparmak ile isaret parmagini birlestir. Sabit tut."),
    ("fist", "YUMRUK", "Yumruk yap. Sabit tut."),
)
STEP_COUNT = len(POSES) + 1
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


def fit_sign(sign, others, margin, min_gap):
    """İşaret örneklerinden kabul aralıkları; öteki pozlar dışarıda kalacak biçimde daraltılır.

    `others`: {poz anahtarı: örnekler}. Döner: (aralıklar, None) ya da işaret bir poza
    `min_gap`ten yakınsa (None, o pozun anahtarı).
    """
    ranges = learn_ranges(sign, margin)
    mine = medians(sign)
    for key, samples in others.items():
        other = medians(samples)
        if not in_ranges(ranges, other):
            continue
        finger = max(FINGERS, key=lambda n: abs(other[n] - mine[n]))
        if abs(other[finger] - mine[finger]) < min_gap:
            return None, key
        mid = (other[finger] + mine[finger]) / 2
        lo, hi = ranges[finger]
        ranges[finger] = (max(lo, mid), hi) if other[finger] < mine[finger] else (lo, min(hi, mid))
    return ranges, None


class SignCalibration:
    """Kalibrasyon akışı: pozları sırayla kaydeder, aralıkları çıkarır, denemeyi yürütür.

    `state`: "record" (poz bekleniyor / kaydediliyor), "pause" (poz tamam), "problem"
    (işaret bir poza çok benziyor), "test" (deneme).
    """

    def __init__(self, cfg, confidence_min=0.0):
        self._cfg = cfg
        self._conf = confidence_min
        self.restart()

    def restart(self):
        self.state = "record"
        self.step = 0               # POSES içindeki sıra; denemede len(POSES)
        self.samples = {}
        self.ranges = None
        self.problem = None         # işaretin çok benzediği pozun anahtarı
        self.hits = 0               # denemede işaretin tanınma sayısı
        self.recognized = False
        self.curls = None           # son karedeki kıvrımlar (el yoksa None)
        self._pause_until = 0.0
        self._gate = None
        self._recorder = self._new_recorder()

    def _new_recorder(self):
        ms = self._cfg["teach_sign_ms"] if self.step == 0 else self._cfg["teach_pose_ms"]
        return PoseRecorder(ms / 1000, self._cfg["teach_still"],
                            self._cfg["teach_still_ms"] / 1000)

    def _curls(self, f):
        if not f.present or not f.fingers or f.confidence < self._conf:
            return None
        return {n: f.fingers[n].curl for n in FINGERS}

    def update(self, f):
        self.curls = self._curls(f)
        if self.state == "pause":
            if f.t >= self._pause_until:
                self.state = "record"
                self._recorder = self._new_recorder()
        elif self.state == "record":
            self._recorder.update(f.t, self.curls)
            if self._recorder.done:
                self._finish_pose(f.t)
        elif self.state == "test":
            self.recognized = self.curls is not None and in_ranges(self.ranges, self.curls)
            if self._gate.update(f) is not None:
                self.hits += 1

    def _finish_pose(self, t):
        self.samples[POSES[self.step][0]] = self._recorder.samples
        self.step += 1
        if self.step < len(POSES):
            self.state, self._pause_until = "pause", t + PAUSE_S
            return
        others = {k: v for k, v in self.samples.items() if k != "sign"}
        self.ranges, self.problem = fit_sign(
            self.samples["sign"], others, self._cfg["teach_margin"], self._cfg["teach_min_gap"])
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
        if self.state == "test":
            return self._gate.progress
        return 1.0 if self.state == "pause" else 0.0

    @property
    def title(self):
        if self.state == "problem":
            return "OLMADI"
        if self.state == "test":
            return f"ADIM {STEP_COUNT}/{STEP_COUNT}: DENE"
        if self.state == "pause":
            return f"TAMAM - siradaki: {POSES[self.step][1]}"
        return f"ADIM {self.step + 1}/{STEP_COUNT}: {POSES[self.step][1]}"

    @property
    def instruction(self):
        if self.state == "problem":
            name = next(t for k, t, _ in POSES if k == self.problem)
            return f"Isaretin {name} pozuna cok benziyor. R: bastan, daha belirgin yap."
        if self.state == "test":
            return (f"Elini birak, sonra isareti yeniden yap. Taninma: {self.hits}   "
                    "ENTER: kaydet   R: bastan")
        if self.state == "pause":
            return "Elini degistirebilirsin."
        if self.curls is None:
            return "El gorunmuyor: elini kameraya goster."
        if not self._recorder.settled:
            return POSES[self.step][2]
        return "Kaydediliyor, kipirdama..."

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
    title_color = red if cal.state == "problem" else green if cal.state == "pause" else yellow
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
    cv2.putText(panel, "R: bastan    Esc: cik", (16, VIEW_H - 18), font, 0.5, gray, 1,
                cv2.LINE_AA)
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
            elif key in (13, 10, 32) and cal.state == "test":
                cal.save()
                print(f"kaydedildi: {cal.ranges}", flush=True)
                saved_until = f.t + SAVED_SHOW_S
    finally:
        source.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    for key, samples in cal.samples.items():
        print(key, {n: round(v, 2) for n, v in medians(samples).items()}, flush=True)


if __name__ == "__main__":
    main()
