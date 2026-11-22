"""Otomatik mod: kamera kullanıcıya bakıyorsa (yüz görünüyor) C, masaya bakıyorsa A.

Repo kökünden: python -m tkt.app.auto_mode   (macOS ve Windows)

- Her saniye görüntüde yüz aranır; art arda 3 karar aynıysa mod değişir. Geçişte basılı
  tuşlar bırakılır, dikte kapatılır, kamera yeni modun ayarıyla yeniden açılır.
- A (masa): `config/vision.toml`, `[pointer]`, başparmakla tık (`click_method = "thumb"`).
- C (pinch): `config/vision-front.toml`, `[pointer]` üzerine `[pointer_front]`, `index_pinch`,
  yatay ayna. Kısa pinch = tık, pinch kapalı + el kayar = sürükleme, 2 sn sabit pinch = sağ
  tık, yumruk = imleç durur, başparmak-orta parmak tutulurken el hareketi = kaydırma.
- Jest motoru DryBackend'e yazar; olaylar her karede gerçek backend'e aktarılır
  (macOS Quartz, Windows SendInput, diğerleri pynput).
- HUD: kamera görüntüsü, iskelet, o anki hareket, son eylemler; A'da aktif alan
  (`[surface].file`) yeşil/kırmızı çizilir.
- Mesafeyle yakınlaştırma (`[face_zoom]`, yalnız macOS ve C modu; `face_zoom` modülü): yüz
  normal mesafeden uzaklaşınca ön plandaki uygulama büyütülür, yaklaşınca küçültülür.
  Ctrl+M açar / kapatır, Ctrl+Shift+M şimdiki mesafeyi normal sayar.
- El kapısı (`[hand_gate]`, yalnız C; `hand_gate` modülü): C'de el faresi kapalı başlar;
  barış işareti (işaret + orta açık, yüzük + serçe kıvrık) tutulunca açılır, yeniden
  yapılınca kapanır. Kapalıyken el jest motoruna verilmez, basılı tuşlar bırakılır.
  İşaret `python -m tkt.app.sign_calib` ile öğretilir (`~/.tkt/hand-gate.json`).

Platform farkları:
- macOS (`auto_mode_mac`): HUD bir Cocoa penceresinde (kapatmak gizler); ekranın sol
  üstünde mod, imlecin yanında durum yazar; C'de hedef yardımı (Erişilebilirlik API'si):
  pinch kapanırken yeşil çerçeveli düğme seçiliyse ve bırakırken hâlâ aynıysa düğmenin
  ortasına tıklanır. Çıkış: her yerden Esc.
- Windows ve diğerleri: HUD bir OpenCV penceresi (mod ve durum onun içinde); hedef yardımı
  yok. Çıkış: Windows'ta her yerden Esc; HUD odaktayken Esc/q; HUD penceresini kapatmak.
"""
import dataclasses
import sys
import threading
import time
import tomllib
from collections import deque
from pathlib import Path

import cv2

from tkt.app.face_zoom import FaceZoom
from tkt.app.hand_gate import HandGate
from tkt.app.hud import ActionTap
from tkt.app.hud import draw as hud_draw
from tkt.app.hud import status as hud_status
from tkt.control.gestures import GestureEngine
from tkt.control.os_input.dry_backend import DryBackend
from tkt.vision import calibration
from tkt.vision.debug_view import draw_surface
from tkt.vision.live_source import open_live_source

CLICK_DEBOUNCE_S = 0.15     # bundan kısa arayla gelen ikinci tık pinch titremesidir
HUD_EVERY_S = 1 / 15        # HUD görüntüsünün yenilenme aralığı
HUD_H = 400                 # HUD'daki kamera görüntüsünün yüksekliği (piksel)
HUD_PANEL_W = 320           # sağdaki eylem listesi genişliği
HUD_HISTORY = 12            # listede tutulan eylem sayısı
HUD_WINDOW = "TKT HUD"
VK_ESCAPE = 0x1B

stop = threading.Event()
selected = None             # macOS: ((x, y, w, h), öğe) — hedef bulucu thread günceller
present = False             # el görünüyor mu (hedef bulucu yalnız el varken arar)
state_txt = [""]            # imlecin yanında / HUD'da gösterilen motor durumu
mode_txt = [""]             # ekranda gösterilen mod
state_since = [0.0]
ax_max = [0.0]              # hedef aramanın 2 sn içindeki en uzun süresi
hud_img = [None, 0]         # [son HUD görüntüsü (BGR), sıra no] — iz thread'i yazar, arayüz okur
history = deque(maxlen=HUD_HISTORY)   # (zaman, metin)
backend = None              # gerçek girdi backend'i; main() kurar
target_click = None         # macOS: (out, (x, y, w, h)) -> None, hedefin ortasına tıklar
zoom = None                 # FaceZoom; yalnız seviyeyi uygulayabilen arayüz (macOS) kurar
zoom_txt = [""]             # ekranda gösterilen yakınlaştırma durumu
gate = None                 # HandGate; main() kurar (yoksa el faresi hep açık)
GATE_OFF_TXT = "EL KAPALI (baris isareti ile ac)"
GATE_ON_TXT = "el acik (baris isareti kapatir)"


def log(msg):
    print(msg, flush=True)


def act(msg):
    """Eylemi hem kayda yazar hem HUD'daki listeye ekler."""
    log(msg)
    history.append((time.monotonic(), msg))


REPO = Path(__file__).resolve().parents[3]
FACE_MODEL = str(REPO / "models/blaze_face_short_range.tflite")
FACE_SCORE = 0.7            # bu güvenin üstünde yüz = kamera kullanıcıya bakıyor
FACE_EVERY_S = 1.0          # yüz kontrolü aralığı
SWITCH_AFTER = 3            # art arda bu kadar aynı karar gelince mod değişir
FACE_DETECT_W = 640         # yüz araması en çok bu genişlikte yapılır
FACE_ROI_SCALE = 3.0        # yüz bulununca aranan çevre: yüz boyutunun bu katı kenarlı kare
FACE_ROI_MIN_PX = 16        # bundan küçük çevre aranmaz


def load_surface(vcfg_path):
    """Vision config'deki aktif alan dosyası (yalnız HUD'da çizmek için; `enabled`a bakmaz)."""
    surface_cfg = tomllib.loads(Path(vcfg_path).read_text()).get("surface", {})
    path = Path(surface_cfg.get("file", calibration.DEFAULT_PATH)).expanduser()
    try:
        return calibration.load(path)
    except Exception as e:     # bozuk dosya HUD'u durdurmasın
        log(f"aktif alan okunamadi ({path}): {e}")
        return None


def profile(name):
    """(vision config yolu, control cfg, yatay ayna, hedef yardımı).

    Hedef yardımı yalnız platform arayüzü hedef bulabiliyorsa (`target_click` kuruluysa) açıktır.
    """
    cfg = tomllib.loads((REPO / "config/control.toml").read_text())
    if name == "A":
        cfg["general"]["click_method"] = "thumb"
        return str(REPO / "config/vision.toml"), cfg, False, False
    cfg["general"]["click_method"] = "index_pinch"
    cfg["pointer"] = {**cfg["pointer"], **cfg.get("pointer_front", {})}
    return str(REPO / "config/vision-front.toml"), cfg, True, target_click is not None


def camera_rotated(vcfg_path):
    """Görüntü 180° döndürülerek mi işleniyor (`[camera].rotate_180`, varsayılan açık)."""
    cam = tomllib.loads(Path(vcfg_path).read_text()).get("camera", {})
    return bool(cam.get("rotate_180", True))


FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                  "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")


def make_face_detector():
    import urllib.request

    import mediapipe as mp
    if not Path(FACE_MODEL).exists():
        log(f"yuz modeli indiriliyor: {FACE_MODEL}")
        Path(FACE_MODEL).parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL)
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision as mpv
    det = mpv.FaceDetector.create_from_options(mpv.FaceDetectorOptions(
        base_options=mpt.BaseOptions(model_asset_path=FACE_MODEL), min_detection_confidence=0.3))

    def detect(img):
        """BGR / gri görüntüdeki yüzler: [(güven, x, y, genişlik, yükseklik)] piksel."""
        h, w = img.shape[:2]
        k = FACE_DETECT_W / w if w > FACE_DETECT_W else 1.0
        small = cv2.resize(img, (FACE_DETECT_W, max(1, round(h * k)))) if k < 1.0 else img
        if small.ndim == 2:
            small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        return [(d.categories[0].score, d.bounding_box.origin_x / k, d.bounding_box.origin_y / k,
                 d.bounding_box.width / k, d.bounding_box.height / k) for d in r.detections]
    return FaceFinder(detect)


class FaceFinder:
    """Yüzü önce son görüldüğü yerin çevresinde, bulamazsa bütün görüntüde arar.

    Yüz bulucu (BlazeFace kısa menzil) görüntüyü 128 piksele küçültür; kullanıcı geriye
    yaslanınca yüz birkaç piksele iner, güven 0,4–0,5'e düşer ve mod yanlışlıkla A'ya geçerdi
    (canlı deneme, 2026-11-22). Yüzün çevresi (`FACE_ROI_SCALE` × yüz boyutu) kırpılıp
    aranınca yüz modelin girdisinde büyük kalır: güven düşmez, kutu boyutu da daha hassas olur.
    Çağrı `(en yüksek güven, yüz boyutu ya da None)` döner (`face_measure`).
    """

    def __init__(self, detect):
        self._detect = detect       # görüntü -> [(güven, x, y, genişlik, yükseklik)]
        self._roi = None            # (x0, y0, x1, y1) son yüzün çevresi
        self.last = (0.0, None)     # son sonuç (kayıt için)

    def __call__(self, img):
        h, w = img.shape[:2]
        faces = []
        if self._roi is not None:
            x0, y0, x1, y1 = self._roi
            x1, y1 = min(x1, w), min(y1, h)
            if x1 - x0 >= FACE_ROI_MIN_PX and y1 - y0 >= FACE_ROI_MIN_PX:
                faces = [(s, x + x0, y + y0, fw, fh)
                         for s, x, y, fw, fh in self._detect(img[y0:y1, x0:x1])
                         if s >= FACE_SCORE]
        if not faces:
            faces = self._detect(img)
        good = [f for f in faces if f[0] >= FACE_SCORE and f[3] > 0 and f[4] > 0]
        if good:
            _, x, y, fw, fh = max(good, key=lambda f: f[3] * f[4])
            half = FACE_ROI_SCALE * max(fw, fh) / 2
            cx, cy = x + fw / 2, y + fh / 2
            self._roi = (max(0, int(cx - half)), max(0, int(cy - half)),
                         int(cx + half), int(cy + half))
        else:
            self._roi = None
        self.last = face_measure([(s, fw, fh) for s, _, _, fw, fh in faces], w, h)
        return self.last


def face_measure(faces, w, h, min_score=None):
    """`faces`: [(güven, genişlik, yükseklik)] piksel. (en yüksek güven, yüz boyutu) döner.

    Boyut, güveni `min_score`u (varsayılan `FACE_SCORE`) geçen en büyük yüzün alanının
    görüntü alanına oranının kareköküdür (uzaklıkla ters orantılı); öyle yüz yoksa None.
    En büyük yüz alınır ki arkadan geçen biri ölçümü bozmasın.
    """
    min_score = FACE_SCORE if min_score is None else min_score
    best = max((s for s, _, _ in faces), default=0.0)
    areas = [fw * fh for s, fw, fh in faces if s >= min_score and fw > 0 and fh > 0]
    size = (max(areas) / (w * h)) ** 0.5 if areas else None
    return best, size


def track():
    global present
    face_score = make_face_detector()
    mode_name = "A"
    while not stop.is_set():
        vcfg, cfg, mirror, assist = profile(mode_name)
        mode_txt[0] = (
            "MOD A: masa (basparmak tik)" if mode_name == "A"
            else "MOD C: pinch (kamera sana bakiyor)")
        surface = load_surface(vcfg) if mode_name == "A" else None
        act(f"MOD {mode_name}: {Path(vcfg).name}"
            + ("" if mode_name == "C" else ", alan var" if surface else ", alan yok"))
        s = open_live_source(vcfg)
        switch_to = run_mode(s, cfg, mirror, assist, face_score, mode_name, surface,
                             rotated=camera_rotated(vcfg))
        s.close()
        if switch_to is None:
            break
        act(f"mod degisiyor: {mode_name} -> {switch_to}")
        mode_name = switch_to
    backend.release_all()


def run_mode(s, cfg, mirror, assist, face_score, mode_name, surface=None, rotated=False):
    """Bir modu çalıştırır; mod değişecekse yeni modun adını, çıkışta None döner.

    `s`: `frames_raw()` veren kaynak (canlı kamera; testte sahte). Yüz araması döndürülmüş
    görüntüde (`rotated`) önce düz hale getirilir.
    """
    global present
    sx = -1 if mirror else 1
    dry = DryBackend()
    eng = GestureEngine(cfg, dry, s.info)
    out = ActionTap(backend, time.monotonic)  # gerçek imleç + HUD için son eylem
    conf_min = cfg["general"]["confidence_min"]
    hud_t = 0.0
    scroll_sum = [0.0, 0.0]
    prev_state = ""
    pin = cfg.get("index_pinch", {})
    locked = None
    was_closed = False
    last_click = 0.0
    stat_t, ages = time.monotonic(), []
    face_t, votes = 0.0, []
    zoom_t = 0.0
    zoom_min = cfg.get("face_zoom", {}).get("face_score", FACE_SCORE)
    face_last, face_ms = (0.0, None), 0.0
    gated = gate is not None and gate.enabled and mode_name == "C"
    mode_base = mode_txt[0]
    try:
        for f, hand, img in s.frames_raw():
            if stop.is_set():
                return None
            now = time.monotonic()
            vote_due = now - face_t >= FACE_EVERY_S
            # Mesafe ölçümü yalnız C'de (kamera kullanıcıya bakarken) ve özellik açıkken sıklaşır.
            zoom_due = (zoom is not None and mode_name == "C" and zoom.wants_samples
                        and now - zoom_t >= zoom.period)
            if vote_due or zoom_due:
                got = face_score(cv2.rotate(img, cv2.ROTATE_180) if rotated else img)
                sc, size = face_last = got if isinstance(got, tuple) else (got, None)
                face_ms = max(face_ms, 1000 * (time.monotonic() - now))
                if zoom_due:
                    zoom_t = now
                    msg = zoom.update(now, size if sc >= zoom_min else None)
                    if msg:
                        act(msg)
            if zoom is not None:
                zoom_txt[0] = zoom.text() if mode_name == "C" or not zoom.enabled else (
                    f"YAKINLASTIRMA: seviye {zoom.level:+d} (A modunda olcum yok)")
            if vote_due:
                face_t = now
                want = "C" if sc >= FACE_SCORE else "A"
                votes = (votes + [want])[-SWITCH_AFTER:]
                if len(votes) == SWITCH_AFTER and all(v != mode_name for v in votes):
                    log(f"yuz guveni {sc:.2f} -> {want}")
                    return want
            ages.append(now - f.t)
            if now - stat_t >= 2.0:
                log(f"olcum [{mode_name}]: isleme {s.process_fps():.1f} fps, "
                    f"takip {s.last_process_ms:.0f} ms, "
                    f"gecikme ort {1000*sum(ages)/len(ages):.0f} / maks {1000*max(ages):.0f} ms, "
                    f"hedef arama maks {1000*ax_max[0]:.0f} ms, "
                    f"yuz guven {face_last[0]:.2f} boyut "
                    + ("yok" if face_last[1] is None else f"{face_last[1]:.3f}")
                    + f" arama maks {face_ms:.0f} ms"
                    + ("" if zoom is None or zoom.ratio is None
                       else f", uzaklik {zoom.ratio:.2f}x seviye {zoom.level:+d}")
                    + ("" if not (gated and f.present and f.fingers) else
                       ", kivrim " + " ".join(f"{n[0]}{f.fingers[n].curl:.2f}"
                                              for n in ("index", "middle", "ring", "pinky"))
                       + f" kapi {'acik' if gate.active else 'kapali'}"))
                stat_t, ages, ax_max[0], face_ms = now, [], 0.0, 0.0
            dry.events.clear()
            dry.set_time(f.t)
            if gated:
                flip = gate.update(f)
                if flip is True:
                    eng = GestureEngine(cfg, dry, s.info)     # temiz durumla başla
                    act("EL FARESI ACIK")
                elif flip is False:
                    release_engine(eng, dry)
                    act("EL FARESI KAPALI")
            active = not gated or gate.active
            if active:
                eng.process(f)
            present = f.present and active
            update_state(eng, f)
            if gated:       # kapı durumu sol üstte, mod yazısının yanında; imleç yanı boş kalır
                mode_txt[0] = f"{mode_base} - {GATE_ON_TXT if active else GATE_OFF_TXT}"
                if not active:
                    state_txt[0] = ""
            if state_txt[0] != prev_state:
                if prev_state == "KAYDIRMA":
                    act(f"KAYDIRMA bitti (dikey {scroll_sum[1]:+.0f}, yatay {scroll_sum[0]:+.0f})")
                elif state_txt[0] == "KAYDIRMA":
                    scroll_sum = [0.0, 0.0]
                    act("KAYDIRMA basladi")
                prev_state = state_txt[0]
            dist = (f.pinch or {}).get("index") if (f.present and pin) else None
            closed = dist is not None and dist <= pin["closed"]
            if closed and not was_closed:
                locked = selected if assist else None
            was_closed = closed if dist is not None else False
            evs = list(dry.events)
            i = 0
            while i < len(evs):
                v = evs[i]
                if v.kind == "move":
                    dx, dy = v.args
                    out.move_rel(sx * dx, dy)
                elif v.kind == "button":
                    name, down = v.args
                    nxt = evs[i + 1] if i + 1 < len(evs) else None
                    if (name == "left" and down and nxt is not None and nxt.kind == "button"
                            and nxt.args == ("left", False)):
                        t = time.monotonic()
                        if t - last_click < CLICK_DEBOUNCE_S:
                            act("tik yok sayildi (titreme)")
                        elif locked is not None and selected == locked:
                            target_click(out, locked[0])
                            act("SOL TIK (hedef)")
                        else:
                            out.button("left", True)
                            out.button("left", False)
                            act("SOL TIK")
                        last_click = t
                        locked = None
                        i += 2
                        continue
                    out.button(name, down)
                    if down:
                        log(f"tus {name} bas")
                    else:     # ActionTap kısa basışı TIK, uzunu SURUKLEME/BIRAKILDI sayar
                        act(out.action.replace("SOL TUS BIRAKILDI", "SURUKLEME bitti"))
                elif v.kind == "scroll":
                    sdx, sdy = v.args
                    out.scroll(sx * sdx, sdy)
                    scroll_sum[0] += sx * sdx
                    scroll_sum[1] += sdy
                elif v.kind == "system":
                    out.system_action(*v.args)
                    act(out.action)
                i += 1
            if now - hud_t >= HUD_EVERY_S:
                hud_t = now
                st = hud_status(f, eng, out, now, conf_min, False)
                if not active:
                    st = dataclasses.replace(st, label=GATE_OFF_TXT)
                hud_img[0] = render_hud(img, hand, f, st, mode_name, surface)
                hud_img[1] += 1
    finally:
        release_engine(eng, dry)
    return None


def release_engine(eng, dry):
    """Motoru sıfırlar; bıraktığı tuşları ve açık dikteyi gerçek backend'e de uygular."""
    eng.release()
    for v in dry.events:
        if v.kind == "button" and not v.args[1]:
            backend.button(v.args[0], False)
        elif v.kind == "system" and v.args[0] == "dictation_stop":
            backend.system_action("dictation_stop")     # dikte açıkken kapandı / mod değişti
            act("DIKTE KAPANDI")
    backend.release_all()
    dry.events.clear()


def render_hud(img, hand, f, st, mode_name, surface):
    """Kamera görüntüsü + iskelet + durum (hud.draw), A'da aktif alan, sağda eylem listesi."""
    import numpy as np
    h, w = img.shape[:2]
    small = cv2.resize(img, (round(w * HUD_H / h), HUD_H), interpolation=cv2.INTER_AREA)
    view = hud_draw(small, hand, st)
    if surface is not None:
        index = f.fingers.get("index") if f.present else None
        draw_surface(view, surface, None if index is None else index.img)
    panel = np.zeros((HUD_H, HUD_PANEL_W, 3), np.uint8)
    title = "MOD A: masa" if mode_name == "A" else "MOD C: pinch (yuz)"
    cv2.putText(panel, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 150, 250), 2,
                cv2.LINE_AA)
    if mode_name == "A":
        note = "alan: yesil ic / kirmizi dis" if surface else "alan yok (debug'da s)"
        cv2.putText(panel, note, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1,
                    cv2.LINE_AA)
    if zoom_txt[0]:
        cv2.putText(panel, zoom_txt[0][:38].lower(), (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (60, 220, 240), 1, cv2.LINE_AA)
    cv2.putText(panel, "son eylemler", (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (150, 150, 150), 1, cv2.LINE_AA)
    now = time.monotonic()
    y = 114
    for t, msg in reversed(list(history)):
        age = now - t
        c = (60, 220, 240) if age < 1.0 else (255, 255, 255) if age < 10 else (150, 150, 150)
        txt = f"{age:4.0f}s  {msg}".encode("ascii", "replace").decode("ascii")
        cv2.putText(panel, txt[:34], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        y += 24
        if y > HUD_H - 8:
            break
    return np.hstack([view, panel])


def update_state(eng, f):
    m = eng.mode.name
    st = getattr(eng, "_index_state", None)
    st = st.name if st is not None else ""
    if m == "SCROLL":
        txt = "KAYDIRMA"
    elif m in ("WAIT", "NONE") and f.present:
        txt = "DUR"
    elif eng.left_down:
        txt = "SURUKLEME / BASILI"
    elif st in ("CLOSING", "HELD", "CONSUMED"):
        txt = "PINCH (kilitli)"
    elif not f.present:
        txt = "EL YOK"
    else:
        txt = ""
    if txt != state_txt[0]:
        state_txt[0], state_since[0] = txt, f.t


def zoom_config():
    return tomllib.loads((REPO / "config/control.toml").read_text())["face_zoom"]


def make_zoom():
    """`[face_zoom]` ayarlarıyla FaceZoom (kalibrasyon ve açık / kapalı durumu dosyadan)."""
    return FaceZoom(zoom_config())


def make_gate():
    cfg = tomllib.loads((REPO / "config/control.toml").read_text())
    return HandGate(cfg["hand_gate"], cfg["general"]["confidence_min"])


def make_backend(platform=None):
    """Bu işletim sisteminin gerçek girdi backend'i."""
    platform = platform or sys.platform
    step_px = tomllib.loads((REPO / "config/control.toml").read_text())["scroll"]["step_px"]
    if platform == "darwin":
        from tkt.control.os_input.macos import MacBackend
        return MacBackend()
    if platform == "win32":
        from tkt.control.os_input.windows import WindowsBackend
        return WindowsBackend(step_px=step_px, smooth=True)
    from tkt.control.os_input.pynput_backend import PynputBackend
    b = PynputBackend()
    b.step_px = step_px
    return b


def esc_down():
    """Windows: Esc şu an basılı mı (hangi pencere odakta olursa olsun). Diğerlerinde False."""
    if sys.platform != "win32":
        return False
    import ctypes
    return bool(ctypes.windll.user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000)


def run_cv_ui(show=None, poll_key=None, window_open=None, worker_target=None):
    """macOS dışı arayüz: iz ayrı thread'de, HUD ana thread'de bir OpenCV penceresinde.

    Esc (Windows'ta her yerden), HUD odaktayken q veya HUD penceresini kapatmak çıkartır.
    `show` / `poll_key` / `window_open` / `worker_target` testte sahte pencere ve iz için.
    """
    own_window = show is None
    show = show or cv2.imshow
    poll_key = poll_key or (lambda: cv2.waitKey(30) & 0xFF)
    window_open = window_open or (
        lambda: cv2.getWindowProperty(HUD_WINDOW, cv2.WND_PROP_VISIBLE) >= 1)
    worker = threading.Thread(target=worker_target or track, daemon=True)
    worker.start()
    shown, opened = -1, False
    try:
        while not stop.is_set() and worker.is_alive():
            img, seq = hud_img
            if img is not None and seq != shown:
                show(HUD_WINDOW, img)
                shown, opened = seq, True
            key = poll_key()
            if key in (27, ord("q")) or esc_down() or (opened and not window_open()):
                log("cikis")
                break
    finally:
        stop.set()
        worker.join(timeout=3.0)
        backend.release_all()
        if own_window:
            cv2.destroyAllWindows()


def main():
    global backend, gate
    import faulthandler

    from tkt.app.main import _set_dpi_awareness

    faulthandler.enable()                 # yerel çökme (abort / segfault) olursa yığın kayda düşsün

    _set_dpi_awareness()                  # Windows: ölçekli ekranda imleç kaymasın
    backend = make_backend()
    gate = make_gate()
    log("hazir: otomatik mod (yuz var -> C, yok -> A), Esc ile cikis")
    if sys.platform == "darwin":
        from tkt.app import auto_mode_mac
        auto_mode_mac.run(sys.modules[__name__])
    else:
        run_cv_ui()


if __name__ == "__main__":
    main()
