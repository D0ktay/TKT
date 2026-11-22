"""Otomatik mod (yalnız macOS; repo kökünden python -m tkt.app.auto_mode).

Kamera kullanıcıya bakıyorsa (yüz görünüyor) C, masaya bakıyorsa A.


Tüm sistemde pinch ile gerçek imleç: hareket, tık (hedef yardımlı), sürükleme, sağ tık.

- Jest motoru (index_pinch) DryBackend'e yazar; olaylar her karede gerçek MacBackend'e aktarılır.
- Kısa pinch (aynı karede left down+up) = tık. Pinch kapanırken yeşil çerçeveli düğme seçiliyse
  ve bırakırken hâlâ aynı düğme seçiliyse imleç düğmenin ortasına alınıp orada tıklanır.
- Pinch kapalı + el kayar = sürükleme (motorun left down/up olayları olduğu gibi).
- 2 sn sabit pinch = sağ tık. Yumruk = imleç durur.
- Başparmak-orta parmak tutulurken el hareketi = kaydırma (doğal yön).
- Esc: çıkış (basılı tuşlar bırakılır).
"""
import math
import threading
import time
import tomllib
from pathlib import Path

import AppKit as K
import ApplicationServices as A
import cv2
import Quartz
from Foundation import NSObject, NSTimer

from tkt.control.gestures import GestureEngine
from tkt.control.os_input.dry_backend import DryBackend
from tkt.control.os_input.macos import MacBackend
from tkt.vision.live_source import open_live_source

CLICK_DEBOUNCE_S = 0.15     # bundan kısa arayla gelen ikinci tık pinch titremesidir
SNAP_PX = 8                 # hedef yardımı: imlecin bu kadar yakınındaki tek düğme

screen = K.NSScreen.mainScreen().frame()
H = K.NSScreen.screens()[0].frame().size.height
stop = threading.Event()
selected = None             # ((x, y, w, h), AXUIElement) — ana thread günceller
present = False
state_txt = [""]
mode_txt = [""]               # sol üstte gösterilen mod
state_since = [0.0]           # imlecin yanında gösterilen motor durumu
scrolled = [0.0]           # 2 sn içindeki toplam kaydırma (piksel)
ax_max = [0.0]             # hedef aramanın 2 sn içindeki en uzun süresi
mac = MacBackend()


def log(msg):
    print(msg, flush=True)


def cursor():
    loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    return loc.x, loc.y


def click_at(cx, cy):
    Quartz.CGWarpMouseCursorPosition((cx, cy))
    Quartz.CGAssociateMouseAndMouseCursorPosition(True)
    mac.button("left", True)
    mac.button("left", False)


REPO = Path(__file__).resolve().parents[3]
FACE_MODEL = str(REPO / "models/blaze_face_short_range.tflite")
FACE_SCORE = 0.7            # bu güvenin üstünde yüz = kamera kullanıcıya bakıyor
FACE_EVERY_S = 1.0          # yüz kontrolü aralığı
SWITCH_AFTER = 3            # art arda bu kadar aynı karar gelince mod değişir


def profile(name):
    """(vision config yolu, control cfg, yatay ayna, hedef yardımı)."""
    cfg = tomllib.loads((REPO / "config/control.toml").read_text())
    if name == "A":
        cfg["general"]["click_method"] = "thumb"
        return str(REPO / "config/vision.toml"), cfg, False, False
    cfg["general"]["click_method"] = "index_pinch"
    cfg["pointer"] = {**cfg["pointer"], **cfg.get("pointer_front", {})}
    return str(REPO / "config/vision-front.toml"), cfg, True, True


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

    def face_score(img):
        h, w = img.shape[:2]
        small = cv2.resize(img, (640, int(640 * h / w))) if w > 640 else img
        if small.ndim == 2:
            small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        return max((d.categories[0].score for d in r.detections), default=0.0)
    return face_score


def track():
    global present
    face_score = make_face_detector()
    mode_name = "A"
    while not stop.is_set():
        vcfg, cfg, mirror, assist = profile(mode_name)
        mode_txt[0] = (
            "MOD A: masa (basparmak tik)" if mode_name == "A"
            else "MOD C: pinch (kamera sana bakiyor)")
        log(f"mod: {mode_name}")
        s = open_live_source(vcfg)
        switch_to = run_mode(s, cfg, mirror, assist, face_score, mode_name)
        s.close()
        if switch_to is None:
            break
        log(f"mod degisiyor: {mode_name} -> {switch_to}")
        mode_name = switch_to
    mac.release_all()


def run_mode(s, cfg, mirror, assist, face_score, mode_name):
    """Bir modu çalıştırır; mod değişecekse yeni modun adını, çıkışta None döner."""
    global present
    sx = -1 if mirror else 1
    dry = DryBackend()
    eng = GestureEngine(cfg, dry, s.info)
    pin = cfg.get("index_pinch", {})
    locked = None
    was_closed = False
    last_click = 0.0
    stat_t, ages = time.monotonic(), []
    face_t, votes = 0.0, []
    s_rotated = mode_name == "A"          # A profili görüntüyü 180° döndürüyor
    try:
        for f, _hand, img in s.frames_raw():
            if stop.is_set():
                return None
            now = time.monotonic()
            if now - face_t >= FACE_EVERY_S:
                face_t = now
                sc = face_score(cv2.rotate(img, cv2.ROTATE_180) if s_rotated else img)
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
                    f"hedef arama maks {1000*ax_max[0]:.0f} ms")
                stat_t, ages, ax_max[0] = now, [], 0.0
            dry.events.clear()
            dry.set_time(f.t)
            eng.process(f)
            present = f.present
            update_state(eng, f)
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
                    mac.move_rel(sx * dx, dy)
                elif v.kind == "button":
                    name, down = v.args
                    nxt = evs[i + 1] if i + 1 < len(evs) else None
                    if (name == "left" and down and nxt is not None and nxt.kind == "button"
                            and nxt.args == ("left", False)):
                        t = time.monotonic()
                        if t - last_click < CLICK_DEBOUNCE_S:
                            log("tik yok sayildi (titreme)")
                        elif locked is not None and selected == locked:
                            (a, b, w, h), _el = locked
                            click_at(a + w / 2, b + h / 2)
                            log("tik (hedef)")
                        else:
                            mac.button("left", True)
                            mac.button("left", False)
                            log("tik (imlec)")
                        last_click = t
                        locked = None
                        i += 2
                        continue
                    mac.button(name, down)
                    log(f"tus {name} {'bas' if down else 'birak'}")
                elif v.kind == "scroll":
                    sdx, sdy = v.args
                    mac.scroll(sx * sdx, sdy)
                elif v.kind == "system":
                    mac.system_action(*v.args)
                i += 1
    finally:
        eng.release()
        for v in dry.events:
            if v.kind == "button" and not v.args[1]:
                mac.button(v.args[0], False)
        mac.release_all()
    return None


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


def attr(el, k):
    err, v = A.AXUIElementCopyAttributeValue(el, k, None)
    return v if not err else None


def target(app, px, py):
    err, el = A.AXUIElementCopyElementAtPosition(app, px, py, None)
    if err:
        return None
    for _ in range(6):
        role = attr(el, "AXRole")
        if role in ("AXTextField", "AXTextArea", "AXComboBox"):
            return "text"
        if role in ("AXButton", "AXMenuButton", "AXPopUpButton", "AXCheckBox", "AXRadioButton"):
            if not attr(el, "AXEnabled"):
                return None
            p, s = attr(el, "AXPosition"), attr(el, "AXSize")
            if p is None or s is None:
                return None
            ok, p = A.AXValueGetValue(p, A.kAXValueCGPointType, None)
            if not ok:
                return None
            ok, s = A.AXValueGetValue(s, A.kAXValueCGSizeType, None)
            return ((p.x, p.y, s.width, s.height), el) if ok else None
        el = attr(el, "AXParent")
        if el is None:
            return None
    return None


class Canvas(K.NSView):
    def drawRect_(self, rect):
        K.NSColor.clearColor().set()
        K.NSRectFillUsingOperation(rect, K.NSCompositingOperationCopy)
        if mode_txt[0]:
            big = {K.NSFontAttributeName: K.NSFont.boldSystemFontOfSize_(20),
                   K.NSForegroundColorAttributeName: K.NSColor.systemOrangeColor()}
            K.NSString.stringWithString_(mode_txt[0]).drawAtPoint_withAttributes_(
                (20, screen.size.height - 60), big)
        if state_txt[0]:
            cx, cy = cursor()
            attrs = {K.NSFontAttributeName: K.NSFont.boldSystemFontOfSize_(16),
                     K.NSForegroundColorAttributeName: K.NSColor.systemOrangeColor()}
            K.NSString.stringWithString_(state_txt[0]).drawAtPoint_withAttributes_(
                (cx - screen.origin.x + 18, H - cy - screen.origin.y - 34), attrs)
        if selected:
            a, b, w, h = selected[0]
            K.NSColor.systemGreenColor().set()
            origin = (a - screen.origin.x, H - b - h - screen.origin.y)
            path = K.NSBezierPath.bezierPathWithRect_((origin, (w, h)))
            path.setLineWidth_(3)
            path.stroke()


def find_targets():
    global selected
    me = K.NSRunningApplication.currentApplication().processIdentifier()
    while not stop.is_set():
        t0 = time.monotonic()
        sel = None
        front = K.NSWorkspace.sharedWorkspace().frontmostApplication()
        if front and front.processIdentifier() != me and present:
            app = A.AXUIElementCreateApplication(front.processIdentifier())
            A.AXUIElementSetMessagingTimeout(app, 0.05)
            px, py = cursor()
            c = target(app, px, py)
            if isinstance(c, tuple):
                sel = c
            elif c is None:
                cands = []
                for k in range(8):
                    ang = k * math.pi / 4
                    r = target(app, px + SNAP_PX * math.cos(ang), py + SNAP_PX * math.sin(ang))
                    if isinstance(r, tuple) and r not in cands:
                        cands.append(r)
                if len(cands) == 1:
                    sel = cands[0]
        selected = sel
        ax_max[0] = max(ax_max[0], time.monotonic() - t0)
        time.sleep(0.05)


class Tick(NSObject):
    def update_(self, timer):
        if Quartz.CGEventSourceKeyState(Quartz.kCGEventSourceStateCombinedSessionState, 53):
            stop.set()
            mac.release_all()
            K.NSApp.terminate_(None)
            return
        view.setNeedsDisplay_(True)


def main():
    global view
    app = K.NSApplication.sharedApplication()
    app.setActivationPolicy_(K.NSApplicationActivationPolicyAccessory)
    win = K.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        screen, K.NSWindowStyleMaskBorderless, K.NSBackingStoreBuffered, False)
    win.setOpaque_(False)
    win.setBackgroundColor_(K.NSColor.clearColor())
    win.setIgnoresMouseEvents_(True)
    win.setHasShadow_(False)
    win.setLevel_(K.NSFloatingWindowLevel)
    view = Canvas.alloc().initWithFrame_(((0, 0), screen.size))
    win.setContentView_(view)
    win.orderFrontRegardless()
    tick = Tick.alloc().init()
    timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        0.06, tick, "update:", None, True)
    main.keep = (win, tick, timer)      # pyobjc nesneleri çöp toplamaya gitmesin
    threading.Thread(target=track, daemon=True).start()
    threading.Thread(target=find_targets, daemon=True).start()
    log("hazir: otomatik mod (yuz var -> C, yok -> A), Esc ile cikis")
    app.run()


if __name__ == "__main__":
    main()
