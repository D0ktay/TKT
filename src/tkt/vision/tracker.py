"""MediaPipe el takibi sarmalayıcısı (V1). PART2 2.2, 2.4.

HandLandmarker, VIDEO modu, num_hands=1. Sağ/sol etiketi MediaPipe'tan alınmaz.

CLI: python -m tkt.vision.tracker --download   (modeli models/ altına indirir)
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = "models/hand_landmarker.task"


@dataclass(frozen=True)
class RawHand:
    image_pts: np.ndarray    # (21, 2), 0–1, kullanıcı bakışı
    world_pts: np.ndarray    # (21, 3), metre
    score: float


class HandTracker:
    """Tek el için MediaPipe HandLandmarker."""

    def __init__(self, model_path: str, min_detection: float, min_tracking: float) -> None:
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"el modeli bulunamadı: {path}. "
                "İndirmek için: python -m tkt.vision.tracker --download"
            )
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_detection,
            min_tracking_confidence=min_tracking,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def _timestamp_ms(self, t: float) -> int:
        """Milisaniye ve kesin artan zaman damgası."""
        ts = int(t * 1000.0)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    def process(self, image: np.ndarray, t: float) -> RawHand | None:
        """BGR kareyi işler; el yoksa None. `t` saniye, kesin artan olmalı."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms(t))
        if not result.hand_landmarks:
            return None
        lm = result.hand_landmarks[0]
        wl = result.hand_world_landmarks[0]
        image_pts = np.array([[p.x, p.y] for p in lm], dtype=np.float64)
        world_pts = np.array([[p.x, p.y, p.z] for p in wl], dtype=np.float64)
        score = 1.0
        if result.handedness and result.handedness[0]:
            score = float(result.handedness[0][0].score)
        return RawHand(image_pts=image_pts, world_pts=world_pts, score=score)

    def close(self) -> None:
        landmarker, self._landmarker = self._landmarker, None
        if landmarker is not None:
            landmarker.close()


def download_model(dest: str | Path = DEFAULT_MODEL_PATH, url: str = MODEL_URL) -> Path:
    """Modeli `dest` yoluna indirir (önce .part dosyasına, sonra yeniden adlandırır)."""
    dest = Path(dest).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as fh:
            while chunk := resp.read(1 << 16):
                fh.write(chunk)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tkt.vision.tracker")
    parser.add_argument("--download", action="store_true", help="modeli models/ altına indirir")
    parser.add_argument("--config", default="config/vision.toml")
    args = parser.parse_args(argv)
    if not args.download:
        parser.error("şimdilik yalnızca --download var")

    dest = DEFAULT_MODEL_PATH
    cfg_path = Path(args.config)
    if cfg_path.is_file():
        from tkt.vision.live_source import load_vision_config

        dest = load_vision_config(cfg_path).get("tracker", {}).get("model", dest)
    path = download_model(dest)
    print(f"model indirildi: {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
