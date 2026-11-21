"""Dikteyi bitirip Enter'a basma (arama çubuğunda söyleneni aratmak gibi).

Dikte kapatılınca son sözcükler metne birkaç yüz ms içinde yazılır; Enter hemen
gönderilirse dikte onu yutabilir veya metin eksik gider. Bu yüzden durdurma bittikten
sonra `SUBMIT_DELAY_S` beklenip Enter gönderilir. Bekleme ana döngüyü (kamera, imleç)
durdurmasın diye ayrı thread'de yapılır.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

# Dikte durduktan sonra Enter'a kadar bekleme (sn) ve durdurma komutunu bekleme sınırı.
SUBMIT_DELAY_S = 0.6
STOP_WAIT_S = 3.0


def _run(stop: Callable[[], Any], enter: Callable[[], None], delay: float,
         sleep: Callable[[float], None]) -> None:
    proc = stop()
    wait = getattr(proc, "wait", None)
    if callable(wait):
        try:
            wait(timeout=STOP_WAIT_S)
        except Exception:
            pass
    sleep(delay)
    enter()


def submit_later(stop: Callable[[], Any], enter: Callable[[], None],
                 delay: float = SUBMIT_DELAY_S, background: bool = True,
                 sleep: Callable[[float], None] = time.sleep) -> threading.Thread | None:
    """`stop()` ile dikteyi kapatır, `delay` sonra `enter()` çağırır.

    `stop()` bir süreç döndürürse (macOS osascript) önce onun bitmesi beklenir.
    `background=False` testte aynı thread'de çalıştırır.
    """
    if not background:
        _run(stop, enter, delay, sleep)
        return None
    t = threading.Thread(target=_run, args=(stop, enter, delay, sleep),
                         name="tkt-dictation-submit", daemon=True)
    t.start()
    return t
