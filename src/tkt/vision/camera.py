"""Kamera yakalama (V1). PART2 2.2, 2.4.

Yakalama ayrı thread'de döner ve yalnızca en son kareyi tutar. Görüntü kullanıcının bakış
açısına göre 180° döndürülür.

CLI: python -m tkt.vision.camera --probe
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

BACKENDS = ("auto", "avfoundation", "dshow", "pyav")

# measured_fps() bu kadar saniyelik pencerede sayar.
_FPS_WINDOW_S = 2.0


@dataclass(frozen=True)
class CapturedFrame:
    image: np.ndarray        # BGR, 180° döndürülmüş
    t: float                 # time.monotonic(), kare alındığı an
    frame_id: int


def resolve_backend(backend: str) -> str:
    """"auto"yu platforma göre somut bir arka uca çevirir."""
    if backend not in BACKENDS:
        raise ValueError(f"bilinmeyen backend {backend!r}; seçenekler: {', '.join(BACKENDS)}")
    if backend == "auto":
        if sys.platform == "darwin":
            return "avfoundation"
        if sys.platform == "win32":
            return "dshow"
        return "auto"
    return backend


class Camera:
    """Arka plan thread'iyle yakalama yapan kamera."""

    def __init__(self, index: int, width: int, height: int, fps: int,
                 rotate_180: bool = True, backend: str = "auto") -> None:
        resolved = resolve_backend(backend)
        if resolved == "pyav":
            raise NotImplementedError("backend='pyav' henüz yok (ayrı PR)")
        self.index = index
        self.width = width        # istenen; gerçek boyut için `frame_size`
        self.height = height
        self.fps = fps            # istenen; gerçek hız için `measured_fps()`
        self.rotate_180 = rotate_180
        self.backend = resolved

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._latest: CapturedFrame | None = None
        self._last_read_id = 0
        self._frame_id = 0
        self._frame_size: tuple[int, int] | None = None
        self._times: deque[float] = deque()

    # ------------------------------------------------------------------ açma / thread

    def _open_capture(self) -> cv2.VideoCapture:
        flags = {
            "avfoundation": cv2.CAP_AVFOUNDATION,
            "dshow": cv2.CAP_DSHOW,
            "auto": cv2.CAP_ANY,
        }
        cap = cv2.VideoCapture(self.index, flags[self.backend])
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"kamera açılamadı (index={self.index}, backend={self.backend}). "
                "İzinleri ve başka bir uygulamanın kamerayı kullanıp kullanmadığını kontrol edin."
            )
        # Sıra önemli: önce çözünürlük, sonra FPS, sonra FOURCC.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if self.backend == "dshow":
            # Windows'ta MJPG açıkça seçilmezse 60 fps gelmez.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        return cap

    def start(self) -> None:
        """Yakalama thread'ini başlatır. Zaten çalışıyorsa bir şey yapmaz."""
        if self._thread is not None:
            return
        self._cap = self._open_capture()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="tkt-camera", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        cap = self._cap
        assert cap is not None
        while not self._stop.is_set():
            ok, image = cap.read()
            t = time.monotonic()   # kare alınır alınmaz
            if not ok or image is None:
                time.sleep(0.002)
                continue
            if self.rotate_180:
                image = cv2.rotate(image, cv2.ROTATE_180)
            with self._cond:
                self._frame_id += 1
                self._latest = CapturedFrame(image=image, t=t, frame_id=self._frame_id)
                self._frame_size = (image.shape[1], image.shape[0])
                self._times.append(t)
                while self._times and t - self._times[0] > _FPS_WINDOW_S:
                    self._times.popleft()
                self._cond.notify_all()

    # ------------------------------------------------------------------ okuma

    @property
    def frame_size(self) -> tuple[int, int] | None:
        """(genişlik, yükseklik): gelen gerçek karenin (döndürme sonrası) boyutu.

        İlk kare gelene kadar None.
        """
        with self._cond:
            return self._frame_size

    def read(self, timeout: float = 1.0) -> CapturedFrame | None:
        """En son kareyi döndürür; `timeout` içinde yeni kare gelmezse None.

        Aynı kare iki kez verilmez. İşleme yavaşsa aradaki kareler atlanır
        (`frame_id` boşluk bırakır).
        """
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                latest = self._latest
                if latest is not None and latest.frame_id > self._last_read_id:
                    self._last_read_id = latest.frame_id
                    return latest
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stop.is_set() or self._thread is None:
                    return None
                self._cond.wait(remaining)

    def measured_fps(self) -> float:
        """Son birkaç saniyede sayılan gerçek yakalama hızı."""
        now = time.monotonic()
        with self._cond:
            times = [t for t in self._times if now - t <= _FPS_WINDOW_S]
        if len(times) < 2 or times[-1] <= times[0]:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def close(self) -> None:
        """Thread'i durdurur, aygıtı bırakır."""
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()


# ---------------------------------------------------------------------- probe CLI


def probe(camera: Camera, seconds: float = 5.0) -> dict[str, float | int | tuple[int, int]]:
    """`seconds` boyunca kare sayar; gerçek yakalama hızını döndürür.

    Kamera önceden başlatılmış olmalıdır.
    """
    first: CapturedFrame | None = None
    last: CapturedFrame | None = None
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        cap = camera.read(timeout=0.5)
        if cap is None:
            continue
        if first is None:
            first = cap
        last = cap
    if first is None or last is None or last.t <= first.t:
        raise RuntimeError("probe süresince kare gelmedi")
    # frame_id kamera karelerini sayar; işleme atlasa bile doğru hızı verir.
    fps = (last.frame_id - first.frame_id) / (last.t - first.t)
    return {
        "fps": fps,
        "frames": last.frame_id - first.frame_id + 1,
        "seconds": last.t - first.t,
        "size": (last.image.shape[1], last.image.shape[0]),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tkt.vision.camera")
    parser.add_argument("--probe", action="store_true", help="5 sn sayıp gerçek fps'i yazar")
    parser.add_argument("--config", default="config/vision.toml")
    parser.add_argument("--seconds", type=float, default=5.0, help="--probe süresi")
    args = parser.parse_args(argv)
    if not args.probe:
        parser.error("şimdilik yalnızca --probe var")

    from tkt.vision.live_source import load_vision_config

    cfg = load_vision_config(Path(args.config))["camera"]
    cam = Camera(cfg["index"], cfg["width"], cfg["height"], cfg["fps"],
                 cfg.get("rotate_180", True), cfg.get("backend", "auto"))
    cam.start()
    try:
        r = probe(cam, args.seconds)
    finally:
        cam.close()
    w, h = r["size"]
    print(f"backend={cam.backend} istenen={cfg['width']}x{cfg['height']}@{cfg['fps']}")
    print(f"gelen kare: {w}x{h} (döndürme sonrası)")
    print(f"{r['frames']} kare / {r['seconds']:.2f} sn -> gerçek fps = {r['fps']:.1f}")


if __name__ == "__main__":
    main()
