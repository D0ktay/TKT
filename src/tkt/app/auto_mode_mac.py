"""Otomatik modun macOS arayüzü (`auto_mode.main` macOS'ta bunu çağırır).

- Ekranın tamamını kaplayan, tıklamaları geçiren saydam katman: sol üstte mod, imlecin yanında
  motor durumu, C'de hedef yardımının seçtiği düğme yeşil çerçeveyle.
- "TKT HUD" Cocoa penceresi (macOS'ta OpenCV penceresi ana thread ister, ana thread Cocoa'da);
  kapatmak yalnız gizler.
- C için hedef yardımı: Erişilebilirlik API'si ile imlecin altındaki (yoksa `SNAP_PX`
  yakınındaki tek) düğme bulunur; `core.selected`'a yazılır, tık `core.target_click` ile
  düğmenin ortasına yapılır.
- Mesafeyle yakınlaştırma (`core.zoom`, `[face_zoom]`): istenen seviye ön plandaki uygulamaya
  kendi menüsündeki Büyüt / Küçült öğelerine basılarak uygulanır (`zoom_worker`). Tuş
  gönderilmez: Türkçe klavyede uygulamalar bu kısayolu farklı tuşlara bağlıyor (Safari ⌘:,
  Electron ⌘.), menü öğesi ise klavye düzeninden bağımsız. Her uygulamaya kaç seviye
  uygulandığı tutulur; özellik kapanınca ya da uygulama öne gelince fark kapatılır. Menüsünde
  böyle bir çift olmayan uygulamaya dokunulmaz; Biçim menüsü (`skip_menus`) aranmaz, çünkü
  oradaki Büyüt belgedeki yazıyı değiştirir. Kısayollar (`toggle_hotkey`,
  `calibrate_hotkey`) her yerden çalışır ve yutulur (öndeki uygulamaya gitmez).
- Esc (her yerden) çıkartır; basılı tuşlar bırakılır, öndeki uygulamanın yakınlaştırması
  geri alınır.

Ortak durum (`stop`, `state_txt`, `hud_img`…) çekirdek modülde tutulur; `run(core)` o modülü
alır (`python -m` ile çalışınca çekirdek `__main__` olur, yeniden import edilmez).
"""
import math
import threading
import time

import AppKit as K
import ApplicationServices as A
import cv2
import Quartz
from Foundation import NSObject, NSTimer

from tkt.app.face_zoom import parse_hotkey, pick_zoom_items, skip_menu

SNAP_PX = 8                 # hedef yardımı: imlecin bu kadar yakınındaki tek düğme
ZOOM_POLL_S = 0.3           # istenen seviye ile uygulananın karşılaştırılma aralığı
ZOOM_MENU_DEPTH = 2         # menü araması kaç alt menü derine iner
ZOOM_AX_TIMEOUT_S = 0.5
ZOOM_RETRY_S = 3.0          # basılamayan (pasif / bayat) öğe için yeniden deneme aralığı
ESC_KEYCODE = 53

core = None                 # tkt.app.auto_mode (çalışan kopya); run() kurar
screen = None
H = 0.0
view = None
hud_view = None
shown_seq = [-1]            # pencereye en son konan HUD görüntüsünün sıra no'su
zoom_cfg = {}               # [face_zoom]
zoom_applied = {}           # pid -> o uygulamaya uygulanmış seviye
zoom_items = {}             # pid -> (büyüt öğesi, küçült öğesi) ya da None (desteklenmiyor)
zoom_retry = {}             # pid -> bu andan önce yeniden denenmez
zoom_lock = threading.Lock()


def cursor():
    loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    return loc.x, loc.y


def click_target(out, rect):
    """İmleci düğmenin ortasına taşıyıp orada sol tıklar."""
    a, b, w, h = rect
    Quartz.CGWarpMouseCursorPosition((a + w / 2, b + h / 2))
    Quartz.CGAssociateMouseAndMouseCursorPosition(True)
    out.button("left", True)
    out.button("left", False)


def attr(el, k):
    err, v = A.AXUIElementCopyAttributeValue(el, k, None)
    return v if not err else None


def target(app, px, py):
    """(px, py) altındaki tıklanabilir düğme: ((x, y, w, h), öğe), metin alanıysa "text"."""
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


def find_targets():
    me = K.NSRunningApplication.currentApplication().processIdentifier()
    while not core.stop.is_set():
        t0 = time.monotonic()
        sel = None
        front = K.NSWorkspace.sharedWorkspace().frontmostApplication()
        if front and front.processIdentifier() != me and core.present:
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
        core.selected = sel
        core.ax_max[0] = max(core.ax_max[0], time.monotonic() - t0)
        time.sleep(0.05)


def menu_lists(menu, depth=0):
    """Bir menüdeki ve alt menülerindeki öğeler: her menü için [(başlık, öğe)]."""
    items, out = [], []
    for item in attr(menu, "AXChildren") or []:
        items.append((attr(item, "AXTitle"), item))
        if depth < ZOOM_MENU_DEPTH:
            for sub in attr(item, "AXChildren") or []:
                out.extend(menu_lists(sub, depth + 1))
    return [items] + out


def find_zoom_items(pid):
    """Uygulamanın menü çubuğundaki Büyüt / Küçült çifti ya da None."""
    app = A.AXUIElementCreateApplication(pid)
    A.AXUIElementSetMessagingTimeout(app, ZOOM_AX_TIMEOUT_S)
    bar = attr(app, "AXMenuBar")
    menus = []
    for top in (attr(bar, "AXChildren") or []) if bar is not None else []:
        if skip_menu(attr(top, "AXTitle"), zoom_cfg["skip_menus"]):
            continue
        for menu in attr(top, "AXChildren") or []:
            menus.extend(menu_lists(menu))
    return pick_zoom_items(menus, zoom_cfg["zoom_in_titles"], zoom_cfg["zoom_out_titles"])


def zoom_sync(pid, name, want):
    """`pid` uygulamasını `want` seviyesine getirir; menü basışı başarısızsa sonraki tura kalır."""
    with zoom_lock:
        have = zoom_applied.get(pid, 0)
        if have == want or time.monotonic() < zoom_retry.get(pid, 0.0):
            return
        if pid not in zoom_items:
            zoom_items[pid] = find_zoom_items(pid)
            if zoom_items[pid] is None:
                core.log(f"yakinlastirma: {name} menusunde buyut/kucult yok, dokunulmuyor")
        pair = zoom_items[pid]
        if pair is None:
            return
        step = 1 if want > have else -1
        item = pair[0] if step > 0 else pair[1]
        while have != want:
            for _ in range(zoom_cfg["presses_per_level"]):
                if not attr(item, "AXEnabled") or A.AXUIElementPerformAction(item, "AXPress"):
                    # Öğe pasif (sınıra gelindi, pencere yok) ya da bayat: bu turu bırak,
                    # menüyü biraz sonra yeniden ara.
                    del zoom_items[pid]
                    zoom_retry[pid] = time.monotonic() + ZOOM_RETRY_S
                    zoom_applied[pid] = have
                    return
            have += step
        zoom_applied[pid] = have
        core.act(f"{name}: yakinlastirma {have:+d}")


def front_app():
    """Öndeki uygulama (pid, ad); kendimizse ya da yoksa None."""
    front = K.NSWorkspace.sharedWorkspace().frontmostApplication()
    me = K.NSRunningApplication.currentApplication().processIdentifier()
    if front is None or front.processIdentifier() == me:
        return None
    return front.processIdentifier(), str(front.localizedName() or "uygulama")


def zoom_worker():
    while not core.stop.is_set():
        front = front_app()
        if front is not None:
            try:
                zoom_sync(*front, core.zoom.target)
            except Exception as e:      # bir uygulamanın menüsü okunamadı: özellik çökmesin
                core.log(f"yakinlastirma hatasi: {e}")
        time.sleep(ZOOM_POLL_S)


def zoom_restore():
    """Çıkışta öndeki uygulamayı normal boyuta döndürür (diğerlerine erişilemez)."""
    front = front_app()
    if front is not None and zoom_applied.get(front[0], 0):
        zoom_sync(*front, 0)
    left = [pid for pid, lvl in zoom_applied.items() if lvl]
    if left:
        core.log(f"yakinlastirma: {len(left)} uygulama buyutulmus / kucultulmus kaldi")


_MOD_FLAGS = {"ctrl": K.NSEventModifierFlagControl, "alt": K.NSEventModifierFlagOption,
              "cmd": K.NSEventModifierFlagCommand, "shift": K.NSEventModifierFlagShift}


def hotkey_matches(event, hotkey):
    mods, key = hotkey
    flags = event.modifierFlags()
    held = {name for name, flag in _MOD_FLAGS.items() if flags & flag}
    chars = event.charactersIgnoringModifiers() or ""
    return held == set(mods) and chars.lower() == key


def install_zoom_hotkeys():
    """Aç / kapat ve kalibrasyon kısayolları (her uygulamada; Erişilebilirlik izni gerekir).

    Tuş bir olay musluğuyla (CGEventTap) yakalanır ve yutulur: öndeki uygulamaya gitmez
    (Ctrl+M Terminal'de Enter demektir). Musluk kurulamazsa tuş yalnız dinlenir (yutulmaz).
    """
    toggle = parse_hotkey(zoom_cfg["toggle_hotkey"])
    calibrate = parse_hotkey(zoom_cfg["calibrate_hotkey"])

    def on_key(event):
        """Kısayolsa işler ve True döner."""
        if hotkey_matches(event, toggle):
            if not event.isARepeat():
                on = core.zoom.toggle()
                core.act("YAKINLASTIRMA ACIK" if on else "YAKINLASTIRMA KAPALI")
        elif hotkey_matches(event, calibrate):
            if not event.isARepeat():
                core.zoom.calibrate()
                core.act("YAKINLASTIRMA kalibrasyonu basladi")
        else:
            return False
        core.zoom_txt[0] = core.zoom.text()
        return True

    def on_tap(proxy, kind, event, refcon):
        try:
            if kind in (Quartz.kCGEventTapDisabledByTimeout,
                        Quartz.kCGEventTapDisabledByUserInput):
                Quartz.CGEventTapEnable(tap, True)
            elif kind == Quartz.kCGEventKeyDown and on_key(K.NSEvent.eventWithCGEvent_(event)):
                return None             # kısayol: uygulamaya gitmesin
        except Exception as e:          # hata klavyeyi kesmesin: tuşu olduğu gibi geçir
            core.log(f"kisayol hatasi: {e}")
        return event

    tap = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap, Quartz.kCGEventTapOptionDefault,
        Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown), on_tap, None)
    if tap is not None:
        source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetMain(), source,
                                  Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(tap, True)
        return tap, source, on_tap

    core.log("kisayol muslugu kurulamadi; tus yalniz dinleniyor (uygulamaya da gider)")

    def on_local(event):
        return None if on_key(event) else event

    mask = K.NSEventMaskKeyDown
    return (K.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(mask, on_key),
            K.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(mask, on_local))


class Canvas(K.NSView):
    def drawRect_(self, rect):
        K.NSColor.clearColor().set()
        K.NSRectFillUsingOperation(rect, K.NSCompositingOperationCopy)
        if core.mode_txt[0]:
            big = {K.NSFontAttributeName: K.NSFont.boldSystemFontOfSize_(20),
                   K.NSForegroundColorAttributeName: K.NSColor.systemOrangeColor()}
            K.NSString.stringWithString_(core.mode_txt[0]).drawAtPoint_withAttributes_(
                (20, screen.size.height - 60), big)
        if core.zoom_txt[0]:
            small = {K.NSFontAttributeName: K.NSFont.boldSystemFontOfSize_(14),
                     K.NSForegroundColorAttributeName: K.NSColor.systemOrangeColor()}
            K.NSString.stringWithString_(core.zoom_txt[0]).drawAtPoint_withAttributes_(
                (20, screen.size.height - 84), small)
        if core.state_txt[0]:
            cx, cy = cursor()
            attrs = {K.NSFontAttributeName: K.NSFont.boldSystemFontOfSize_(16),
                     K.NSForegroundColorAttributeName: K.NSColor.systemOrangeColor()}
            K.NSString.stringWithString_(core.state_txt[0]).drawAtPoint_withAttributes_(
                (cx - screen.origin.x + 18, H - cy - screen.origin.y - 34), attrs)
        sel = core.selected
        if sel:
            a, b, w, h = sel[0]
            K.NSColor.systemGreenColor().set()
            origin = (a - screen.origin.x, H - b - h - screen.origin.y)
            path = K.NSBezierPath.bezierPathWithRect_((origin, (w, h)))
            path.setLineWidth_(3)
            path.stroke()


class Tick(NSObject):
    def update_(self, timer):
        if Quartz.CGEventSourceKeyState(Quartz.kCGEventSourceStateCombinedSessionState,
                                        ESC_KEYCODE):
            core.stop.set()
            core.backend.release_all()
            try:
                zoom_restore()
            except Exception as e:
                core.log(f"yakinlastirma geri alinamadi: {e}")
            K.NSApp.terminate_(None)
            return
        view.setNeedsDisplay_(True)
        img, seq = core.hud_img
        if img is not None and seq != shown_seq[0]:
            shown_seq[0] = seq
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                data = K.NSData.dataWithBytes_length_(buf.tobytes(), len(buf))
                hud_view.setImage_(K.NSImage.alloc().initWithData_(data))


def make_hud_window():
    """Sağ üstte, taşınabilir ve boyutlanabilir "TKT HUD" penceresi; kapatınca gizlenir."""
    global hud_view
    w, h = 533 + core.HUD_PANEL_W, core.HUD_H
    frame = ((screen.origin.x + screen.size.width - w - 20,
              screen.origin.y + screen.size.height - h - 60), (w, h))
    style = (K.NSWindowStyleMaskTitled | K.NSWindowStyleMaskClosable
             | K.NSWindowStyleMaskMiniaturizable | K.NSWindowStyleMaskResizable)
    win = K.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        frame, style, K.NSBackingStoreBuffered, False)
    win.setTitle_(core.HUD_WINDOW)
    win.setReleasedWhenClosed_(False)
    win.setLevel_(K.NSFloatingWindowLevel)
    hud_view = K.NSImageView.alloc().initWithFrame_(((0, 0), (w, h)))
    hud_view.setImageScaling_(K.NSImageScaleProportionallyUpOrDown)
    hud_view.setAutoresizingMask_(K.NSViewWidthSizable | K.NSViewHeightSizable)
    win.setContentView_(hud_view)
    win.orderFrontRegardless()
    return win


def run(core_module):
    """Cocoa uygulamasını kurar, iz ve hedef bulucu thread'lerini başlatır; Esc'ye kadar döner."""
    global core, screen, H, view, zoom_cfg
    core = core_module
    core.target_click = click_target
    core.zoom = core.make_zoom()
    core.zoom_txt[0] = core.zoom.text()
    zoom_cfg = core.zoom_config()
    screen = K.NSScreen.mainScreen().frame()
    H = K.NSScreen.screens()[0].frame().size.height
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
    hud_win = make_hud_window()
    tick = Tick.alloc().init()
    timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        0.06, tick, "update:", None, True)
    monitors = install_zoom_hotkeys()
    run.keep = (win, hud_win, tick, timer, monitors)   # pyobjc nesneleri çöp toplamaya gitmesin
    threading.Thread(target=zoom_worker, daemon=True).start()
    threading.Thread(target=core.track, daemon=True).start()
    threading.Thread(target=find_targets, daemon=True).start()
    app.run()
