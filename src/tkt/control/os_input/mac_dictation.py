"""macOS sistem diktesini ön plandaki uygulamanın menüsünden açıp kapatır.

Yazılımla gönderilen "Control'e iki kez bas" kısayolunu macOS dikte olarak kabul etmiyor
(canlı deneme, 2026-11-21). Bunun yerine System Events ile ön plandaki uygulamanın
menülerinde adı "Dikte"/"Dictation" geçen öğe tıklanır (Düzen > Dikteyi Başlat / Durdur).
Terminale Erişilebilirlik izni ve System Events için Otomasyon izni gerekir.

Durdururken menüde durdurma öğesi yoksa (dikte kendiliğinden kapanmışsa) hiçbir şey
tıklanmaz; açık olmayan dikte yeniden açılmaz.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Any

# argv: "start" | "stop". Döner: tıklanan öğenin adı, "zaten: <ad>" veya "yok".
SCRIPT = """
on run argv
    set wantStart to (item 1 of argv is "start")
    tell application "System Events"
        set p to first application process whose frontmost is true
        tell p
            repeat with mb in menu bar items of menu bar 1
                try
                    repeat with mi in menu items of menu 1 of mb
                        set n to name of mi
                        if n is not missing value then
                            if n contains "Dikte" or n contains "Dictation" then
                                set isStop to (n contains "Durdur" or n contains "Stop")
                                if isStop is not wantStart then
                                    click mi
                                    return n
                                end if
                                return "zaten: " & n
                            end if
                        end if
                    end repeat
                end try
            end repeat
        end tell
    end tell
    return "yok"
end run
"""


def toggle(start: bool, popen: Callable[..., Any] = subprocess.Popen) -> Any:
    """Diktenin açılması (`start`) veya kapanması için menü tıklamasını başlatır.

    Bloklamaz: `osascript` arka planda çalışır (menü taraması birkaç yüz ms sürebilir).
    Çıktı atılır; hata olursa (izin yok, menü yok) sessizce hiçbir şey olmaz.
    """
    return popen(
        ["osascript", "-e", SCRIPT, "start" if start else "stop"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
