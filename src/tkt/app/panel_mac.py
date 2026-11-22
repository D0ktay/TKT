"""macOS kontrol penceresi ("TKT") ve menü çubuğu simgesi.

Pencere küçük ve sabittir: durum satırı, dört ayar kutusu, kalibrasyon listesi (her satırda
kayıt durumu ve o kalibrasyonu yeniden yapan düğme), Çıkış. Kapatmak yalnız gizler;
`[panel] hotkey` (varsayılan Control+N) ve menü çubuğundaki "TKT" açıp kapatır.

İçerik `panel` modülünden gelir; kalibrasyon araçları ayrı pencerede çalışır, o sırada
izleme durur ve kamera araca bırakılır (`core.request_tool`).
"""
import AppKit as K
import objc
from Foundation import NSObject

from tkt.app import panel as model

W = 420
ACCESSIBILITY_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
PAD = 16
ROW_H = 30


def _label(text, frame, size=13, bold=False, color=None):
    field = K.NSTextField.alloc().initWithFrame_(frame)
    field.setStringValue_(text)
    field.setBezeled_(False)
    field.setDrawsBackground_(False)
    field.setEditable_(False)
    field.setSelectable_(False)
    field.setFont_(K.NSFont.boldSystemFontOfSize_(size) if bold
                   else K.NSFont.systemFontOfSize_(size))
    if color is not None:
        field.setTextColor_(color)
    return field


class Panel(NSObject):
    """Pencereyi kurar ve `core` ile bağlar. `refresh()` düzenli çağrılır (Tick)."""

    def initWithCore_onQuit_(self, core, on_quit):
        self = self.init()
        if self is None:
            return None
        self.core = core
        self.on_quit = on_quit
        self.hud_window = None
        self.rows = {}              # kalibrasyon anahtarı -> (durum etiketi, düğme)
        self.items = {}             # kalibrasyon anahtarı -> Calibration
        self._build()
        return self

    # --- kurulum ---

    @objc.python_method
    def _checkbox(self, title, y, action):
        box = K.NSButton.alloc().initWithFrame_(((PAD, y), (W - 2 * PAD, 22)))
        box.setButtonType_(K.NSButtonTypeSwitch)
        box.setTitle_(title)
        box.setTarget_(self)
        box.setAction_(action)
        self.content.addSubview_(box)
        return box

    @objc.python_method
    def _build(self):
        core = self.core
        cal = model.calibrations(core.REPO, core.control_config(), core.zoom)
        groups = list(dict.fromkeys(c.group for c in cal))
        height = (PAD + 26 + 12                      # durum
                  + 4 * 26 + 14                      # kutular
                  + len(groups) * 28 + len(cal) * ROW_H + 10
                  + 20 + 36 + PAD)                   # kısayol notu, çıkış
        style = K.NSWindowStyleMaskTitled | K.NSWindowStyleMaskClosable
        self.window = K.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            ((0, 0), (W, height)), style, K.NSBackingStoreBuffered, False)
        self.window.setTitle_("TKT")
        self.window.setReleasedWhenClosed_(False)
        self.window.setLevel_(K.NSFloatingWindowLevel)
        self.window.center()
        self.content = self.window.contentView()
        y = height - PAD - 22
        self.status = _label("", ((PAD, y), (W - 2 * PAD - 96, 22)), 14, bold=True)
        self.content.addSubview_(self.status)
        self.fix_button = K.NSButton.alloc().initWithFrame_(((W - PAD - 96, y - 4), (96, 28)))
        self.fix_button.setBezelStyle_(K.NSBezelStyleRounded)
        self.fix_button.setTitle_("İzin ver…")
        self.fix_button.setTarget_(self)
        self.fix_button.setAction_("openAccessibility:")
        self.content.addSubview_(self.fix_button)
        y -= 12 + 26
        self.hand_box = self._checkbox("El faresi açık (barış işareti de açıp kapatır)", y,
                                       "handChanged:")
        y -= 26
        self.zoom_box = self._checkbox("Geriye yaslanınca büyüt (Control+M)", y, "zoomChanged:")
        y -= 26
        self.surface_box = self._checkbox("Masa modunda imleci seçili alandan sür", y,
                                          "surfaceChanged:")
        y -= 26
        self.hud_box = self._checkbox("Kamera penceresini göster", y, "hudChanged:")
        y -= 14
        gray = K.NSColor.secondaryLabelColor()
        for group in groups:
            y -= 28
            self.content.addSubview_(_label(group.upper(), ((PAD, y), (W - 2 * PAD, 18)), 11,
                                            bold=True, color=gray))
            for c in (c for c in cal if c.group == group):
                y -= ROW_H
                self.content.addSubview_(_label(c.title, ((PAD, y + 2), (140, 20))))
                state = _label("", ((PAD + 142, y + 2), (132, 20)), 11, color=gray)
                self.content.addSubview_(state)
                button = K.NSButton.alloc().initWithFrame_(((W - PAD - 112, y - 2), (112, 28)))
                button.setBezelStyle_(K.NSBezelStyleRounded)
                button.setTitle_(c.button)
                button.setTarget_(self)
                button.setAction_("calibrate:")
                button.setIdentifier_(c.key)
                self.content.addSubview_(button)
                self.rows[c.key] = (state, button)
        y -= 10 + 20
        note = f"{core.panel_hotkey_text()}: bu pencere   ·   Esc: çıkış"
        self.content.addSubview_(_label(note, ((PAD, y), (W - 2 * PAD - 90, 18)), 11, color=gray))
        quit_button = K.NSButton.alloc().initWithFrame_(((W - PAD - 90, PAD - 4), (90, 28)))
        quit_button.setBezelStyle_(K.NSBezelStyleRounded)
        quit_button.setTitle_("Çıkış")
        quit_button.setTarget_(self)
        quit_button.setAction_("quit:")
        self.content.addSubview_(quit_button)
        self.refresh()

    def addStatusItem(self):
        """Menü çubuğuna "TKT" koyar; tıklayınca pencere açılır / kapanır."""
        item = K.NSStatusBar.systemStatusBar().statusItemWithLength_(
            K.NSVariableStatusItemLength)
        item.button().setTitle_("TKT")
        item.button().setTarget_(self)
        item.button().setAction_("toggle:")
        self.status_item = item

    # --- durum ---

    def refresh(self):
        core = self.core
        tool = core.tool_running[0] or core.pending_tool[0] is not None
        self.status.setStringValue_(
            model.status_line(core.mode_now[0], core.gate, core.zoom, tool, core.ax_ok[0]))
        self.fix_button.setHidden_(core.ax_ok[0])
        gate = core.gate
        self.hand_box.setState_(1 if gate is None or gate.active else 0)
        self.hand_box.setEnabled_(gate is not None and gate.enabled and core.mode_now[0] == "C")
        self.zoom_box.setState_(1 if core.zoom is not None and core.zoom.enabled else 0)
        self.surface_box.setState_(1 if core.prefs["use_surface"] else 0)
        self.hud_box.setState_(1 if core.prefs["hud"] else 0)
        for c in model.calibrations(core.REPO, core.control_config(), core.zoom):
            self.items[c.key] = c
            state, button = self.rows[c.key]
            state.setStringValue_(c.status)
            button.setEnabled_(not tool and (c.tool is not None or core.mode_now[0] == "C"))

    def toggle_(self, sender):
        if self.window.isVisible():
            self.window.orderOut_(None)
        else:
            self.show()

    def show(self):
        self.refresh()
        K.NSApp.activateIgnoringOtherApps_(True)
        self.window.makeKeyAndOrderFront_(None)

    # --- eylemler ---

    def handChanged_(self, sender):
        if self.core.gate is not None:
            self.core.gate.set_active(bool(sender.state()))
        self.refresh()

    def zoomChanged_(self, sender):
        if self.core.zoom is not None and self.core.zoom.enabled != bool(sender.state()):
            on = self.core.zoom.toggle()
            self.core.act("YAKINLASTIRMA ACIK" if on else "YAKINLASTIRMA KAPALI")
        self.refresh()

    def surfaceChanged_(self, sender):
        self.core.set_pref("use_surface", bool(sender.state()))
        self.core.request_reopen()          # masa modu yeni ayarla yeniden açılsın
        self.refresh()

    def hudChanged_(self, sender):
        self.core.set_pref("hud", bool(sender.state()))
        self.applyHud()

    def applyHud(self):
        if self.hud_window is None:
            return
        if self.core.prefs["hud"]:
            self.hud_window.orderFrontRegardless()
        else:
            self.hud_window.orderOut_(None)

    def calibrate_(self, sender):
        c = self.items.get(str(sender.identifier()))
        if c is None:
            return
        if c.tool is None:                  # normal mesafe: uygulama içinde ölçülür
            self.core.zoom.calibrate()
            self.core.act("YAKINLASTIRMA kalibrasyonu basladi")
        else:
            self.core.request_tool(c.tool)
            self.window.orderOut_(None)     # araç penceresinin önünde durmasın
        self.refresh()

    def openAccessibility_(self, sender):
        """Sistem Ayarları'nda Erişilebilirlik listesini açar (TKT oradan işaretlenir)."""
        K.NSWorkspace.sharedWorkspace().openURL_(K.NSURL.URLWithString_(ACCESSIBILITY_URL))

    def quit_(self, sender):
        self.on_quit()
