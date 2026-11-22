"""Otomatik modun macOS arayüzü (`auto_mode.main` macOS'ta bunu çağırır).

- Ekranın tamamını kaplayan, tıklamaları geçiren saydam katman: sol üstte mod, imlecin yanında
  motor durumu, C'de hedef yardımının seçtiği düğme yeşil çerçeveyle.
- "TKT HUD" Cocoa penceresi (macOS'ta OpenCV penceresi ana thread ister, ana thread Cocoa'da);
  kapatmak yalnız gizler.
- C için hedef yardımı: Erişilebilirlik API'si ile imlecin altındaki (yoksa `SNAP_PX`
  yakınındaki tek) düğme bulunur; `core.selected`'a yazılır, tık `core.target_click` ile
  düğmenin ortasına yapılır.
- Esc (her yerden) çıkartır; basılı tuşlar bırakılır.

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

SNAP_PX = 8                 # hedef yardımı: imlecin bu kadar yakınındaki tek düğme
ESC_KEYCODE = 53

core = None                 # tkt.app.auto_mode (çalışan kopya); run() kurar
screen = None
H = 0.0
view = None
hud_view = None
shown_seq = [-1]            # pencereye en son konan HUD görüntüsünün sıra no'su


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


class Canvas(K.NSView):
    def drawRect_(self, rect):
        K.NSColor.clearColor().set()
        K.NSRectFillUsingOperation(rect, K.NSCompositingOperationCopy)
        if core.mode_txt[0]:
            big = {K.NSFontAttributeName: K.NSFont.boldSystemFontOfSize_(20),
                   K.NSForegroundColorAttributeName: K.NSColor.systemOrangeColor()}
            K.NSString.stringWithString_(core.mode_txt[0]).drawAtPoint_withAttributes_(
                (20, screen.size.height - 60), big)
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
    global core, screen, H, view
    core = core_module
    core.target_click = click_target
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
    run.keep = (win, hud_win, tick, timer)      # pyobjc nesneleri çöp toplamaya gitmesin
    threading.Thread(target=core.track, daemon=True).start()
    threading.Thread(target=find_targets, daemon=True).start()
    app.run()
