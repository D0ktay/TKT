"""Canlı kare kaynağı (V1). PART2 2.2, 2.4.

`open_live_source`, `app/main.py`'nin vision'dan çağırdığı tek fonksiyondur.
"""

from __future__ import annotations

import logging
import threading
import time
import tomllib
from collections import deque
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np

from tkt.core.types import SCHEMA_VERSION, HandFrame, SourceInfo
from tkt.vision import features, hand_calib
from tkt.vision.camera import Camera, CapturedFrame, format_choice, resolve_camera
from tkt.vision.hand_calib import HandCalibration
from tkt.vision.lift import LiftEstimator
from tkt.vision.tracker import HandTracker, RawHand

log = logging.getLogger(__name__)

# process_fps() bu kadar saniyelik pencerede sayar.
_FPS_WINDOW_S = 2.0

# LiftEstimator'ın config'ten okuduğu anahtarlar ([lift]); doğrudan verilen tahmincinin
# ayarlarını set_calibration'da korumak için kullanılır.
_LIFT_KEYS = ("w_angle", "w_length", "baseline_s", "curl_rate_suppress")


def _lift_settings(lift: object) -> dict[str, float] | None:
    """Tahmincinin [lift] ayarlarını okur; eksikse (sahte tahminci) None."""
    if lift is None or not all(hasattr(lift, k) for k in _LIFT_KEYS):
        return None
    return {k: getattr(lift, k) for k in _LIFT_KEYS}


class LiveSource:
    """HandFrameSource: kamera + tracker + ölçümler. El yokken de kare üretir.

    Kamera yoksa veya ilk kare gelmezse `RuntimeError` verir. `info.width/height` gelen gerçek
    karenin (döndürme sonrası) boyutudur; `info.fps` istenen hızdır, gerçek hız
    `capture_fps()` ile okunur.

    `frames()` / `frames_raw()` kamera `read()` zaman aşımına uğrayınca (kare gelmiyor) veya
    kaynak kapanınca biter.
    """

    info: SourceInfo

    def __init__(self, camera: Camera, tracker: HandTracker, calib: HandCalibration,
                 hand: str, lift: LiftEstimator | None = None) -> None:
        self._camera = camera
        self._tracker = tracker
        self._calib = calib
        self._lift = lift
        # `set_calibration` lift'i yeniden kurarken bunu kullanır; open_live_source doldurur.
        self.lift_cfg: Mapping | None = None
        # lift_cfg yoksa kurulumda verilen tahmincinin ayarları kullanılır.
        self._lift_settings = _lift_settings(lift)
        self._calib_lock = threading.Lock()
        self._closed = False
        self._proc_times: deque[float] = deque()
        self.last_process_ms: float = 0.0   # son karenin takip + özellik süresi

        camera.start()
        first = camera.read()
        if first is None:
            camera.close()
            raise RuntimeError("kameradan ilk kare alınamadı")
        self._first: CapturedFrame | None = first
        h, w = first.image.shape[:2]
        self.info = SourceInfo(
            schema=SCHEMA_VERSION, width=int(w), height=int(h),
            fps=float(camera.fps), hand=hand,
        )
        self._aspect = self.info.width / self.info.height
        # Dışarıdan verilen tahminci kare boyutunu bilmeden kurulmuş olabilir (aspect=1.0):
        # ayarlarını koruyarak gerçek en-boy oranıyla yeniden kur.
        if (lift is not None and self._lift_settings is not None
                and getattr(lift, "aspect", None) != self._aspect):
            self._lift = LiftEstimator(calib, self._lift_settings, aspect=self._aspect)

    # ------------------------------------------------------------------ ölçümler

    def capture_fps(self) -> float:
        """Kameradan sayılan gerçek yakalama hızı."""
        return self._camera.measured_fps()

    def process_fps(self) -> float:
        """Son birkaç saniyede işlenen kare hızı."""
        now = time.monotonic()
        times = [t for t in self._proc_times if now - t <= _FPS_WINDOW_S]
        if len(times) < 2 or times[-1] <= times[0]:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    # ------------------------------------------------------------------ kalibrasyon

    def set_calibration(self, calib: HandCalibration) -> None:
        """Sonraki karelerden itibaren yeni kalibrasyonu kullanır (başka thread'den çağrılabilir).

        Lift varsa ve `calib.tap_ref` doluysa LiftEstimator yeni kalibrasyonla yeniden kurulur;
        `tap_ref` yoksa lift kapanır (None). Lift ayarları `lift_cfg`'den, o yoksa kurulumda
        verilen tahmincinin ayarlarından alınır; ayar bilinen kaynakta lift kapalıyken de
        `tap_ref` gelince kurulur.
        """
        cfg = self.lift_cfg if self.lift_cfg is not None else self._lift_settings
        lift = None
        if calib.tap_ref is not None and cfg is not None:
            lift = LiftEstimator(calib, cfg, aspect=self._aspect)
        elif calib.tap_ref is not None and self._lift is not None:
            raise ValueError("lift ayarları bilinmiyor: LiveSource.lift_cfg verin")
        with self._calib_lock:
            self._calib = calib
            self._lift = lift

    # ------------------------------------------------------------------ akış

    def frames(self) -> Iterator[HandFrame]:
        for frame, _hand, _image in self.frames_raw():
            yield frame

    def frames_raw(self) -> Iterator[tuple[HandFrame, RawHand | None, np.ndarray]]:
        """Debug penceresi ve `record --raw` için: kare, ham noktalar, görüntü."""
        prev_size: float | None = None
        while not self._closed:
            if self._first is not None:
                cap, self._first = self._first, None
            else:
                cap = self._camera.read()
            if cap is None:
                return

            t0 = time.perf_counter()
            with self._calib_lock:      # kare boyunca aynı kalibrasyon + lift çifti
                calib, lift_est = self._calib, self._lift
            hand = self._tracker.process(cap.image, cap.t)
            if hand is None and lift_est is not None:
                lift_est.reset()
            frame = features.build_frame(
                hand, cap.t, cap.frame_id, calib, self.info,
                lift=None, prev_hand_size=prev_size)
            if hand is not None and lift_est is not None and frame.present:
                curls = {name: f.curl for name, f in frame.fingers.items()}
                frame = features.build_frame(
                    hand, cap.t, cap.frame_id, calib, self.info,
                    lift=lift_est.update(hand, cap.t, curls), prev_hand_size=prev_size)
            done = time.perf_counter()
            self.last_process_ms = (done - t0) * 1000.0
            self._proc_times.append(time.monotonic())
            while self._proc_times and self._proc_times[-1] - self._proc_times[0] > _FPS_WINDOW_S:
                self._proc_times.popleft()

            prev_size = frame.hand_size if frame.present else None
            yield frame, hand, cap.image

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._camera.close()
        self._tracker.close()


# ---------------------------------------------------------------------- yapılandırma


def load_vision_config(path: str | Path) -> dict:
    """`vision.toml` dosyasını okur."""
    with Path(path).open("rb") as fh:
        return tomllib.load(fh)


def open_live_source(config_path: str | Path = "config/vision.toml") -> LiveSource:
    """Config'e göre kamera, tracker, kalibrasyon ve (varsa) lift ile LiveSource kurar."""
    cfg = load_vision_config(config_path)
    cam_cfg, trk_cfg, hand_cfg = cfg["camera"], cfg["tracker"], cfg["hand"]

    # Model yoksa kamerayı açmadan hata ver.
    tracker = HandTracker(trk_cfg["model"], trk_cfg["min_detection"], trk_cfg["min_tracking"])
    camera: Camera | None = None
    try:
        choice = resolve_camera(cam_cfg)
        log.info("%s", format_choice(choice))
        camera = Camera(
            choice.index, cam_cfg["width"], cam_cfg["height"], cam_cfg["fps"],
            cam_cfg.get("rotate_180", True), cam_cfg.get("backend", "auto"))

        calib_path = Path(hand_cfg["calibration"]).expanduser()
        calib = hand_calib.load(calib_path) or hand_calib.DEFAULT
        lift = LiftEstimator(calib, cfg["lift"]) if calib.tap_ref is not None else None

        source = LiveSource(camera, tracker, calib, hand_cfg["side"], lift)
        source.lift_cfg = cfg["lift"]
        return source
    except BaseException:
        # Kurulum yarıda kaldı: sahiplik henüz kimseye geçmedi, açılanları kapat.
        if camera is not None:
            camera.close()
        tracker.close()
        raise
