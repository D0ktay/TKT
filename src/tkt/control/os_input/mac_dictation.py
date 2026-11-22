"""macOS sistem diktesini ön plandaki uygulamanın menüsünden açıp kapatır.

Yazılımla gönderilen "Control'e iki kez bas" kısayolunu macOS dikte olarak kabul etmiyor
(canlı deneme, 2026-11-21). Bunun yerine System Events ile ön plandaki uygulamanın
menülerinde adı "Dikte"/"Dictation" geçen öğe tıklanır (Düzen > Dikteyi Başlat / Durdur).
Terminale Erişilebilirlik izni ve System Events için Otomasyon izni gerekir.

Canlı denemede görülenler (2026-11-22, macOS 26):
- Önce yalnız Düzen/Edit menüsüne bakılır (~0,5 sn); bütün menüleri taramak ~2,5 sn
  sürüyor ve dikte o kadar geç açılıyordu. Bulunamazsa bütün menüler taranır.
- Yerel uygulamalarda (TextEdit) öğenin adı dikte açıkken "Stop Dictation", kapanırken kısa
  bir süre "Cancel Dictation" olur. Electron uygulamalarında (T3 Code, VS Code…) ad hep
  "Start Dictation…" kalır; ikinci tıklama dikteyi kapatmıyor.
- Bu yüzden durdururken ad "Stop" değilse ve dikte mikrofonu o an kullanılıyorsa (`mic_in_use`)
  Esc gönderilir: dikte ~0,4 sn'de kapanır, o ana kadar yazılan metin kalır (TextEdit'te
  denendi). Dikte sessizlikte (~7 sn) kendiliğinden kapanmışsa mikrofon boştur ve hiçbir şey
  gönderilmez.
"""

from __future__ import annotations

import ctypes
import struct
import subprocess
from collections.abc import Callable
from typing import Any

# argv: "start" | "stop", "active" | "idle" (mikrofon kullanımda mı).
# Döner: tıklanan öğenin adı, "esc: <ad>" (Esc gönderildi), "zaten: <ad>" veya "yok".
SCRIPT = """
on run argv
    set wantStart to (item 1 of argv is "start")
    set micOn to false
    if (count of argv) > 1 then set micOn to (item 2 of argv is "active")
    tell application "System Events"
        set p to first application process whose frontmost is true
        set found to missing value
        repeat with mb in (menu bar items of menu bar 1 of p)
            set nm to name of mb
            if nm is "Edit" or nm is "Düzen" then
                try
                    set its_ to (menu items of menu 1 of mb ¬
                        whose name contains "Dictation" or name contains "Dikte")
                    if its_ is not {} then set found to item 1 of its_
                end try
                exit repeat
            end if
        end repeat
        if found is missing value then
            repeat with mb in (menu bar items of menu bar 1 of p)
                try
                    set its_ to (menu items of menu 1 of mb ¬
                        whose name contains "Dictation" or name contains "Dikte")
                    if its_ is not {} then
                        set found to item 1 of its_
                        exit repeat
                    end if
                end try
            end repeat
        end if
        if found is missing value then return "yok"
        set n to name of found
        set isStop to (n contains "Durdur" or n contains "Stop")
        set isCancel to (n contains "Cancel" or n contains "İptal" or n contains "Vazgeç")
        if wantStart then
            if isStop or isCancel then return "zaten: " & n
            click found
            return n
        end if
        if isStop then
            click found
            return n
        end if
        if micOn and not isCancel then
            key code 53
            return "esc: " & n
        end if
        return "zaten: " & n
    end tell
end run
"""


def _fourcc(s: str) -> int:
    return struct.unpack(">I", s.encode("ascii"))[0]


class _Addr(ctypes.Structure):
    _fields_ = [("sel", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("elem", ctypes.c_uint32)]


def mic_in_use() -> bool:
    """Herhangi bir giriş aygıtı (mikrofon) şu an bir süreç tarafından kullanılıyor mu.

    CoreAudio `kAudioDevicePropertyDeviceIsRunningSomewhere`; yalnız giriş akışı olan
    aygıtlara bakılır (hoparlörde çalan ses sayılmaz). macOS dışında veya hata olursa False.
    """
    try:
        ca = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        system = 1                                  # kAudioObjectSystemObject
        glob, inpt = _fourcc("glob"), _fourcc("inpt")
        size = ctypes.c_uint32(0)
        addr = _Addr(_fourcc("dev#"), glob, 0)      # kAudioHardwarePropertyDevices
        if ca.AudioObjectGetPropertyDataSize(system, ctypes.byref(addr), 0, None,
                                             ctypes.byref(size)):
            return False
        devices = (ctypes.c_uint32 * (size.value // 4))()
        if ca.AudioObjectGetPropertyData(system, ctypes.byref(addr), 0, None,
                                         ctypes.byref(size), devices):
            return False
        for dev in devices:
            streams = ctypes.c_uint32(0)
            s_addr = _Addr(_fourcc("stm#"), inpt, 0)    # kAudioDevicePropertyStreams, giriş
            if ca.AudioObjectGetPropertyDataSize(dev, ctypes.byref(s_addr), 0, None,
                                                 ctypes.byref(streams)) or not streams.value:
                continue
            running = ctypes.c_uint32(0)
            n = ctypes.c_uint32(4)
            r_addr = _Addr(_fourcc("gone"), glob, 0)     # DeviceIsRunningSomewhere
            if not ca.AudioObjectGetPropertyData(dev, ctypes.byref(r_addr), 0, None,
                                                 ctypes.byref(n), ctypes.byref(running)) \
                    and running.value:
                return True
    except Exception:
        return False
    return False


def toggle(start: bool, popen: Callable[..., Any] = subprocess.Popen,
           mic_active: Callable[[], bool] = mic_in_use) -> Any:
    """Diktenin açılması (`start`) veya kapanması için menü tıklamasını başlatır.

    Bloklamaz: `osascript` arka planda çalışır (menü araması ~0,5 sn). Durdururken
    mikrofonun kullanımda olup olmadığı betiğe verilir (Electron uygulamaları için).
    Çıktı atılır; hata olursa (izin yok, menü yok) sessizce hiçbir şey olmaz.
    """
    state = "active" if (not start and mic_active()) else "idle"
    return popen(
        ["osascript", "-e", SCRIPT, "start" if start else "stop", state],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
