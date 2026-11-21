"""Kamera yakalama (V1). PART2 2.2, 2.4.

Yakalama ayrı thread'de döner ve yalnızca en son kareyi tutar. Görüntü kullanıcının bakış
açısına göre 180° döndürülür.

CLI: python -m tkt.vision.camera --probe
     python -m tkt.vision.camera --list
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# "msmf" normalde seçilmez; DirectShow açılamazsa yedek olarak devreye girer.
BACKENDS = ("auto", "avfoundation", "dshow", "msmf", "pyav")

# --list'in taradığı kamera index'leri.
LIST_INDICES = tuple(range(6))

# measured_fps() bu kadar saniyelik pencerede sayar.
_FPS_WINDOW_S = 2.0

# backend="auto" iken bir aday (DirectShow) bu süre içinde ilk kareyi vermezse kapatılır ve
# sıradaki aday (MSMF) denenir. isOpened=True dönüp hiç kare vermeyen sürücüler için.
_FIRST_FRAME_TIMEOUT_S = 2.0

# close() yakalama thread'ini en fazla bu kadar bekler. Sürücü read() içinde daha uzun
# bloklanırsa uyarı yazılır; aygıtı thread read'den dönünce kendisi bırakır.
_CLOSE_TIMEOUT_S = 2.0

# Kare zamanı saati. Windows'ta çözünürlüğü ~15 ms olduğundan iki kare aynı değeri alabilir;
# _run bu durumda t'yi önceki t + _MIN_DT_S yapar (kesin artan).
_clock = time.monotonic
_MIN_DT_S = 1e-4


@dataclass(frozen=True)
class CapturedFrame:
    image: np.ndarray        # BGR, 180° döndürülmüş
    t: float                 # time.monotonic(), kare alındığı an; kesin artan (bkz. _run)
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
        self._requested_backend = backend

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()   # çalışan thread'in durdurma bayrağı; start() yeniler
        self._stop.set()
        self._cond = threading.Condition()
        self._latest: CapturedFrame | None = None
        self._last_read_id = 0
        self._frame_id = 0
        self._last_t: float | None = None
        self._frame_size: tuple[int, int] | None = None
        self._times: deque[float] = deque()
        self._negotiated: dict[str, object] = {}

    # ------------------------------------------------------------------ açma / thread

    def _candidates(self) -> list[str]:
        # DirectShow açılamazsa (veya "auto"da kare vermezse) Media Foundation'a düşülür.
        candidates = [self.backend]
        if self.backend == "dshow":
            candidates.append("msmf")
        return candidates

    def _try_open(self, backend: str) -> cv2.VideoCapture | None:
        flags = {
            "avfoundation": cv2.CAP_AVFOUNDATION,
            "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF,
            "auto": cv2.CAP_ANY,
        }
        cap = cv2.VideoCapture(self.index, flags[backend])
        if not cap.isOpened():
            cap.release()
            return None
        self._configure(cap, backend)
        return cap

    def _configure(self, cap: cv2.VideoCapture, backend: str) -> None:
        # Sıra önemli (Logitech belgesi): önce çözünürlük, sonra FPS, en son FOURCC.
        # DirectShow'da FOURCC önce veya FPS sonra verilirse sessizce YUY2'ye düşülür.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if backend in ("dshow", "msmf"):
            # Windows'ta MJPG açıkça seçilmezse 60 fps gelmez.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    @staticmethod
    def _read_negotiated(cap: cv2.VideoCapture, backend: str) -> dict[str, object]:
        code = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        chars = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
        # Bazı arka uçlar (AVFoundation) FOURCC vermez; okunamayan değer boş döner.
        fourcc = chars if code and all(c.isalnum() or c == " " for c in chars) else ""
        return {
            "backend": backend,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "fourcc": fourcc,
        }

    def negotiated(self) -> dict[str, object]:
        """Açılışta sürücünün gerçekte verdiği ayarlar: backend, width, height, fps, fourcc.

        `fourcc` 4 harf ("MJPG", "YUY2"); arka uç bildirmiyorsa boş. Kamera başlamadan boş sözlük.
        """
        return dict(self._negotiated)

    def start(self) -> None:
        """Yakalama thread'ini başlatır. Zaten çalışıyorsa bir şey yapmaz.

        `backend="auto"` iken bir adayın başarısı ilk kullanılabilir kareyle doğrulanır:
        `_FIRST_FRAME_TIMEOUT_S` içinde kare gelmezse aday kapatılır ve sıradaki denenir.
        Son aday doğrulanmaz. Açık backend adında yalnızca `isOpened()` sonucuna bakılır.
        """
        if self._thread is not None:
            return
        candidates = self._candidates()
        verify = self._requested_backend == "auto"
        for i, backend in enumerate(candidates):
            nxt = candidates[i + 1] if i + 1 < len(candidates) else None
            cap = self._try_open(backend)
            if cap is None:
                if nxt is not None:
                    log.warning("kamera %s ile açılamadı (index=%d); %s deneniyor",
                                backend, self.index, nxt)
                continue
            self.backend = backend
            self._negotiated = self._read_negotiated(cap, backend)
            self._launch(cap)
            if not verify or nxt is None or self._wait_first_frame(_FIRST_FRAME_TIMEOUT_S):
                return
            log.warning("kamera %s ile açıldı ama %.1f sn içinde kare vermedi (index=%d); "
                        "%s deneniyor", backend, _FIRST_FRAME_TIMEOUT_S, self.index, nxt)
            self._shutdown()
        self._negotiated = {}
        raise RuntimeError(
            f"kamera açılamadı (index={self.index}, backend={', '.join(candidates)}). "
            "İzinleri ve başka bir uygulamanın kamerayı kullanıp kullanmadığını kontrol edin."
        )

    def _launch(self, cap: cv2.VideoCapture) -> None:
        """`cap`'in sahipliğini yeni bir yakalama thread'ine verir."""
        stop = threading.Event()
        with self._cond:
            self._stop = stop
            self._latest = None
        self._thread = threading.Thread(target=self._run, args=(cap, stop),
                                        name="tkt-camera", daemon=True)
        self._thread.start()

    def _wait_first_frame(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._latest is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True

    def _shutdown(self) -> None:
        """Thread'i durdurur ve en fazla `_CLOSE_TIMEOUT_S` bekler. Aygıtı thread bırakır."""
        with self._cond:
            self._stop.set()
            self._latest = None
            self._cond.notify_all()
        thread, self._thread = self._thread, None
        if thread is None:
            return
        thread.join(timeout=_CLOSE_TIMEOUT_S)
        if thread.is_alive():
            log.warning("kamera thread'i %.1f sn içinde bitmedi (sürücü read() içinde "
                        "bloklu); aygıt read dönünce bırakılacak", _CLOSE_TIMEOUT_S)

    def _run(self, cap: cv2.VideoCapture, stop: threading.Event) -> None:
        """Yakalama döngüsü. `cap` bu thread'e aittir; çıkarken burada bırakılır.

        `stop` set edildikten sonra kare yayımlanmaz (kilit altında denetlenir). Kare zamanı
        `_clock()` (time.monotonic) ile okunur; aynı veya geri giden değer önceki t +
        `_MIN_DT_S` yapılır, böylece t kesin artan olur ve monotonic saatle aynı epoch'ta kalır.
        """
        try:
            while not stop.is_set():
                ok, image = cap.read()
                t = _clock()   # kare alınır alınmaz
                if stop.is_set():
                    break
                if not ok or image is None:
                    time.sleep(0.002)
                    continue
                if self.rotate_180:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                with self._cond:
                    if stop.is_set():
                        break
                    if self._last_t is not None and t <= self._last_t:
                        t = self._last_t + _MIN_DT_S
                    self._last_t = t
                    self._frame_id += 1
                    self._latest = CapturedFrame(image=image, t=t, frame_id=self._frame_id)
                    self._frame_size = (image.shape[1], image.shape[0])
                    self._times.append(t)
                    while self._times and t - self._times[0] > _FPS_WINDOW_S:
                        self._times.popleft()
                    self._cond.notify_all()
        finally:
            cap.release()

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
                if self._stop.is_set() or self._thread is None:
                    return None
                latest = self._latest
                if latest is not None and latest.frame_id > self._last_read_id:
                    self._last_read_id = latest.frame_id
                    return latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
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
        """Thread'i durdurur; aygıtı thread çıkarken bırakır.

        Thread en fazla `_CLOSE_TIMEOUT_S` beklenir; yetişmezse uyarı yazılır. Kapanıştan
        sonra kare yayımlanmaz ve `read()` None döndürür.
        """
        self._shutdown()


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


def _is_gray(image: np.ndarray) -> bool:
    """Tek kanallı veya üç kanalı birbirine eşit görüntü gri sayılır (NIR kamera)."""
    if image.ndim == 2 or image.shape[2] == 1:
        return True
    return bool(np.array_equal(image[..., 0], image[..., 1])
                and np.array_equal(image[..., 1], image[..., 2]))


def _orientation(width: int, height: int) -> str:
    if height > width:
        return "dikey"
    return "yatay" if width > height else "kare"


def list_cameras(indices: tuple[int, ...] = LIST_INDICES, *, width: int, height: int, fps: int,
                 backend: str = "auto", seconds: float = 1.0, max_frames: int = 600,
                 camera_factory=None) -> list[dict]:
    """Her index'i kısa süre açıp özelliklerini toplar; açılmayanlar `opened=False` olur.

    fps, kare zaman damgalarından ölçülür (duvar saatine bağlı değil). `camera_factory`
    testlerde sahte kamera vermek içindir.
    """
    factory = camera_factory or Camera
    rows: list[dict] = []
    for index in indices:
        row: dict = {"index": index, "opened": False}
        cam = factory(index, width, height, fps, rotate_180=False, backend=backend)
        try:
            cam.start()
            first = last = None
            for _ in range(max_frames):
                cap = cam.read(timeout=1.0)
                if cap is None:
                    break
                if first is None:
                    first = cap
                last = cap
                if cap.t - first.t >= seconds:
                    break
            if first is not None and last is not None:
                h, w = first.image.shape[:2]
                span = last.t - first.t
                fps_m = (last.frame_id - first.frame_id) / span if span > 0 else 0.0
                row.update(opened=True, width=int(w), height=int(h), fps=fps_m,
                           gray=_is_gray(first.image), orientation=_orientation(w, h),
                           negotiated=cam.negotiated())
        except RuntimeError:
            pass   # açılamadı
        finally:
            cam.close()
        rows.append(row)
    return rows


def format_camera_table(rows: list[dict]) -> str:
    lines = [f"{'index':>5}  {'çözünürlük':<11} {'fps':>6}  {'renk':<6} {'yön':<6} fourcc"]
    for r in rows:
        if not r["opened"]:
            lines.append(f"{r['index']:>5}  (açılamadı / kare gelmedi)")
            continue
        fourcc = r.get("negotiated", {}).get("fourcc") or "-"
        lines.append(
            f"{r['index']:>5}  {str(r['width']) + 'x' + str(r['height']):<11} {r['fps']:>6.1f}  "
            f"{'gri' if r['gray'] else 'renkli':<6} {r['orientation']:<6} {fourcc}")
    return "\n".join(lines)


def format_negotiated(n: dict[str, object]) -> str:
    fourcc = n.get("fourcc") or "(bildirilmedi)"
    return (f"gerçekleşen: backend={n.get('backend')} {n.get('width')}x{n.get('height')}"
            f"@{n.get('fps'):.1f} fourcc={fourcc}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tkt.vision.camera")
    parser.add_argument("--probe", action="store_true", help="5 sn sayıp gerçek fps'i yazar")
    parser.add_argument("--list", action="store_true",
                        help="0-5 arası index'leri açıp çözünürlük/fps/renk/yön tablosu yazar")
    parser.add_argument("--config", default="config/vision.toml")
    parser.add_argument("--seconds", type=float, default=5.0, help="--probe süresi")
    args = parser.parse_args(argv)
    if not (args.probe or args.list):
        parser.error("--probe veya --list verin")

    from tkt.vision.live_source import load_vision_config

    cfg = load_vision_config(Path(args.config))["camera"]
    if args.list:
        rows = list_cameras(width=cfg["width"], height=cfg["height"], fps=cfg["fps"],
                            backend=cfg.get("backend", "auto"))
        print(format_camera_table(rows))
    if not args.probe:
        return
    cam = Camera(cfg["index"], cfg["width"], cfg["height"], cfg["fps"],
                 cfg.get("rotate_180", True), cfg.get("backend", "auto"))
    cam.start()
    try:
        r = probe(cam, args.seconds)
    finally:
        cam.close()
    w, h = r["size"]
    print(f"backend={cam.backend} istenen={cfg['width']}x{cfg['height']}@{cfg['fps']}")
    print(format_negotiated(cam.negotiated()))
    print(f"gelen kare: {w}x{h} (döndürme sonrası)")
    print(f"{r['frames']} kare / {r['seconds']:.2f} sn -> gerçek fps = {r['fps']:.1f}")


if __name__ == "__main__":
    main()
