"""Parmak kalkışı (lift) tahmini (V2, Faz 2). PART2 2.2, 2.5.

    n = unit(cross(P5 - P0, P17 - P0))                 # avuç normali (dünya noktaları)
    angle = angle_sign * arcsin(dot(unit(PIP - MCP), n))
    length = norm(TIP_img - MCP_img) / hand_size       # görüntüdeki parmak boyu
    lift_raw = w_angle * (angle - angle_base) + w_length * (length_base - length)
    lift = clip(lift_raw / tap_ref, 0, 1)

`hand_size` burada görüntü noktalarında bilek (0) ile orta kök (9) arasıdır. Uzunluklar piksel
oranıyla ölçülür: görüntü noktaları 0–1 oranıdır, x ekseni `aspect` (= genişlik / yükseklik) ile
çarpılır. Böylece kare olmayan görüntüde yatay ve dikey uzunluk aynı ölçekte kalır (512x640
dikey karede yatay 0,3 ile dikey 0,3 oranı 0,8'dir, 1,0 değil). `aspect` verilmezse 1.0'dır
(eski davranış); `LiftEstimator` ve `fit_taps` aynı değerle çağrılmalıdır.

CLI: python -m tkt.vision.lift --eval TAP.raw.jsonl NORMAL.raw.jsonl
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tkt.vision.features import FINGER_JOINTS, curl_raw
from tkt.vision.hand_calib import HandCalibration
from tkt.vision.tracker import RawHand

WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17

# Normalize lift bu değerin üstündeyken parmağın taban çizgisi dondurulur (kalkış sürerken
# taban çizgisi kalkışa doğru kaymasın). Config'de anahtarı yok; modül sabiti.
LIFT_FREEZE = 0.25

# `fit_taps` config almaz (imzası sabit); ağırlıkları buradan kullanır. config/vision.toml
# [lift] varsayılanlarıyla aynıdır. Config'deki ağırlıklar değişirse tap kalibrasyonu yenilenmeli.
FIT_WEIGHTS: dict[str, float] = {"w_angle": 1.0, "w_length": 1.0}

# --eval: bipten sonraki tepe arama penceresi (sn) ve ayrışma oranı eşiği (PART2 2.5).
EVAL_WINDOW_S = 1.5
EVAL_RATIO_MIN = 1.5
EVAL_PERCENTILE = 99.0

_EPS = 1e-9


# --------------------------------------------------------------------------- ham sinyaller


def palm_normal(hand: RawHand) -> np.ndarray:
    """Avuç düzlemi normali (birim), dünya noktalarıyla. Dejenere elde sıfır vektör."""
    p = np.asarray(hand.world_pts, dtype=float)
    n = np.cross(p[INDEX_MCP] - p[WRIST], p[PINKY_MCP] - p[WRIST])
    norm = float(np.linalg.norm(n))
    if norm < _EPS:
        return np.zeros(3)
    return n / norm


def _check_aspect(aspect: float) -> float:
    a = float(aspect)
    if not (math.isfinite(a) and a > 0):
        raise ValueError(f"aspect (genişlik/yükseklik) sonlu ve pozitif olmalı: {aspect!r}")
    return a


def raw_signals(hand: RawHand, aspect: float = 1.0) -> dict[str, tuple[float, float]]:
    """Parmak adı → (işaretsiz açı, görüntü boyu).

    Açı: kök→orta eklem doğrultusunun avuç düzlemiyle yaptığı açı (radyan, `angle_sign`
    uygulanmamış). Boy: kök→uç görüntü mesafesi / bilek→orta kök görüntü mesafesi; piksel
    oranıyla (x ekseni `aspect` = genişlik / yükseklik ile çarpılır).
    """
    w = np.asarray(hand.world_pts, dtype=float)
    img = np.array(hand.image_pts, dtype=float)[:, :2]
    img[:, 0] *= aspect
    n = palm_normal(hand)
    size = float(np.linalg.norm(img[MIDDLE_MCP] - img[WRIST]))
    out: dict[str, tuple[float, float]] = {}
    for name, (mcp, pip, _dip, tip) in FINGER_JOINTS.items():
        d = w[pip] - w[mcp]
        dn = float(np.linalg.norm(d))
        angle = 0.0
        if dn > _EPS:
            angle = math.asin(min(1.0, max(-1.0, float(np.dot(d / dn, n)))))
        length = float(np.linalg.norm(img[tip] - img[mcp])) / size if size > _EPS else 0.0
        out[name] = (angle, length)
    return out


def curls_from_hand(hand: RawHand, calib: HandCalibration) -> dict[str, float]:
    """`build_frame` ile aynı 0–1 curl değerleri (görüntü boyutu gerektirmez)."""
    out: dict[str, float] = {}
    for name in FINGER_JOINTS:
        lo, hi = calib.open_ref[name], calib.fist_ref[name]
        span = hi - lo
        raw = curl_raw(hand, name)
        if abs(span) < _EPS:
            out[name] = 0.0 if raw <= lo else 1.0
        else:
            out[name] = min(1.0, max(0.0, (raw - lo) / span))
    return out


# --------------------------------------------------------------------------- tahminci


@dataclass
class _FingerState:
    angle_base: float
    length_base: float
    prev_curl: float | None = None
    frozen_since: float | None = None


class LiftEstimator:
    """Kökten kalkış açısı ve görüntüdeki parmak boyundan `lift` üretir.

    - Taban çizgisi (`angle_base`, `length_base`): üstel ortalama, zaman sabiti `baseline_s`.
      İlk karede o karenin değeridir. Parmağın `lift`'i `LIFT_FREEZE` üstündeyken dondurulur;
      dondurma en çok `baseline_s` sürer (uzun süre kalkık kalan parmak veya kalıcı duruş
      değişikliği taban çizgisini sonsuza dek kilitlemesin).
    - Kıvırma bastırması: parmağın curl hızının mutlak değeri `curl_rate_suppress` (1/sn)
      üstündeyse o parmağın `lift`'i 0'dır ve taban çizgisi o anki değere çekilir. Böylece
      kıvırma veya açma bitince geride kalan fark tap gibi görünmez.
    """

    def __init__(self, calib: HandCalibration, cfg: Mapping[str, Any], *,
                 aspect: float = 1.0) -> None:
        """`aspect`: görüntü genişliği / yüksekliği (`SourceInfo.width / height`)."""
        self.aspect = _check_aspect(aspect)
        if calib.tap_ref is None:
            raise ValueError("LiftEstimator: kalibrasyonda tap_ref yok (fit_taps gerekli)")
        missing = [f for f in FINGER_JOINTS if f not in calib.tap_ref]
        if missing:
            raise ValueError(f"LiftEstimator: tap_ref eksik parmaklar: {missing}")
        bad = [f for f in FINGER_JOINTS if not calib.tap_ref[f] > 0]
        if bad:
            raise ValueError(f"LiftEstimator: tap_ref pozitif olmalı: {bad}")
        self.w_angle = float(cfg["w_angle"])
        self.w_length = float(cfg["w_length"])
        self.baseline_s = float(cfg["baseline_s"])
        self.curl_rate_suppress = float(cfg["curl_rate_suppress"])
        if not self.baseline_s > 0:
            raise ValueError("LiftEstimator: baseline_s pozitif olmalı")
        self.angle_sign = float(calib.angle_sign)
        self.tap_ref = {f: float(calib.tap_ref[f]) for f in FINGER_JOINTS}
        self._state: dict[str, _FingerState] = {}
        self._prev_t: float | None = None

    def reset(self) -> None:
        """Taban çizgilerini sıfırlar (el kaybolunca)."""
        self._state.clear()
        self._prev_t = None

    def update(self, hand: RawHand, t: float, curls: dict[str, float]) -> dict[str, float]:
        """Parmak adı → lift (0–1)."""
        t = float(t)
        dt = 0.0 if self._prev_t is None else max(0.0, t - self._prev_t)
        self._prev_t = t if self._prev_t is None else max(self._prev_t, t)
        alpha = 1.0 - math.exp(-dt / self.baseline_s)

        out: dict[str, float] = {}
        for name, (a_raw, length) in raw_signals(hand, self.aspect).items():
            angle = self.angle_sign * a_raw
            st = self._state.get(name)
            if st is None:
                st = self._state[name] = _FingerState(angle, length)

            curl = curls.get(name)
            suppressed = False
            if curl is not None:
                curl = float(curl)
                if st.prev_curl is not None and dt > 0:
                    suppressed = abs(curl - st.prev_curl) / dt > self.curl_rate_suppress
                st.prev_curl = curl

            if suppressed:
                st.angle_base, st.length_base = angle, length
                st.frozen_since = None
                out[name] = 0.0
                continue

            lift_raw = (self.w_angle * (angle - st.angle_base)
                        + self.w_length * (st.length_base - length))
            lift = min(1.0, max(0.0, lift_raw / self.tap_ref[name]))
            out[name] = lift

            if lift > LIFT_FREEZE:
                if st.frozen_since is None:
                    st.frozen_since = t
                if t - st.frozen_since < self.baseline_s:
                    continue
            else:
                st.frozen_since = None
            st.angle_base += alpha * (angle - st.angle_base)
            st.length_base += alpha * (length - st.length_base)
        return out


# --------------------------------------------------------------------------- --eval


@dataclass
class EvalResult:
    fingers: tuple[str, ...]          # değerlendirilen parmaklar
    peaks: list[float]                # her bip penceresinde (ve parmakta) max lift
    normal_p99: float                 # NORMAL kayıtta lift'in 99. yüzdeliği
    n_prompts: int
    n_missing: int                    # elin hiç görünmediği bip pencereleri
    calib: HandCalibration
    fitted: bool                      # tap_ref bu çalıştırmada fit_taps ile mi bulundu
    notes: list[str] = field(default_factory=list)

    @property
    def peak_median(self) -> float:
        return float(np.median(self.peaks)) if self.peaks else float("nan")

    @property
    def status(self) -> str:
        """"ok" | "no_peaks" (tap penceresi yok) | "no_tap" (tap tepe medyanı 0) |
        "no_noise" (normal p99 0, tap tepesi > 0)."""
        if not self.peaks:
            return "no_peaks"
        if not self.peak_median > _EPS:
            return "no_tap"
        if not self.normal_p99 > _EPS:
            return "no_noise"
        return "ok"

    @property
    def ratio(self) -> float:
        """Tap tepe medyanı / normal p99. Pay 0 veya tap yoksa nan; payda 0 ise inf
        (raporda sayı olarak yazılmaz, bkz. `status`)."""
        st = self.status
        if st in ("no_peaks", "no_tap"):
            return float("nan")
        if st == "no_noise":
            return float("inf")
        return self.peak_median / self.normal_p99

    @property
    def passed(self) -> bool:
        st = self.status
        if st == "no_noise":
            return True
        return st == "ok" and self.ratio >= EVAL_RATIO_MIN


def _aspect_of(path: str | Path) -> float:
    """Ham kaydın başlığından genişlik / yükseklik."""
    from tkt.tools.record import read_raw_header

    info = read_raw_header(path)
    return _check_aspect(info.width / info.height)


def _run_lift(path: str | Path, calib: HandCalibration,
              cfg: Mapping[str, Any]) -> list[tuple[float, dict[str, float] | None]]:
    """Ham kayıt boyunca lift (kaydın en-boy oranıyla); el yokken None ve taban çizgisi
    sıfırlanır."""
    from tkt.tools.record import iter_raw

    est = LiftEstimator(calib, cfg, aspect=_aspect_of(path))
    out: list[tuple[float, dict[str, float] | None]] = []
    for s in iter_raw(path):
        if s.hand is None:
            est.reset()
            out.append((s.t, None))
            continue
        out.append((s.t, est.update(s.hand, s.t, curls_from_hand(s.hand, calib))))
    return out


def tap_samples(path: str | Path, prompts: list[Any],
                window_s: float = EVAL_WINDOW_S) -> list[tuple[float, RawHand]]:
    """Bip pencerelerine ([t, t + window_s)) düşen, elin göründüğü ham kareler (sıralı)."""
    from tkt.tools.record import iter_raw

    starts = sorted(float(p.t) for p in prompts)
    out: list[tuple[float, RawHand]] = []
    for s in iter_raw(path):
        if s.hand is None:
            continue
        if any(b <= s.t < b + window_s for b in starts):
            out.append((s.t, s.hand))
    return out


def evaluate(tap_path: str | Path, normal_path: str | Path, calib: HandCalibration,
             cfg: Mapping[str, Any], refit: bool = False) -> EvalResult:
    """Ayrışma oranı: tap tepe medyanı / NORMAL kayıttaki lift'in 99. yüzdeliği."""
    from tkt.core.io import prompts_path, read_prompts
    from tkt.vision.hand_calib import fit_taps

    ppath = prompts_path(tap_path)
    if not ppath.is_file():
        raise FileNotFoundError(f"bip yan dosyası yok: {ppath}")
    prompts = sorted(read_prompts(ppath), key=lambda p: p.t)
    notes: list[str] = []

    fitted = False
    if calib.tap_ref is None or refit:
        # Bip pencereleri ardışık olabilir (örnek akışı kesintisiz); sınırlar açıkça verilir.
        windows = [(float(p.t), float(p.t) + EVAL_WINDOW_S) for p in prompts]
        calib = fit_taps(calib, tap_samples(tap_path, prompts), aspect=_aspect_of(tap_path),
                         windows=windows)
        fitted = True
        cfg_w = {k: float(cfg[k]) for k in FIT_WEIGHTS}
        if cfg_w != FIT_WEIGHTS:
            notes.append(f"uyarı: fit_taps ağırlıkları {FIT_WEIGHTS}, config {cfg_w}")

    has_tap2 = any(p.kind == "tap2" for p in prompts)
    fingers = ("index", "middle") if has_tap2 else ("index",)

    tap_series = _run_lift(tap_path, calib, cfg)
    peaks: list[float] = []
    missing = 0
    for p in prompts:
        names = ("index", "middle") if p.kind == "tap2" else ("index",)
        win = [lift for t, lift in tap_series
               if lift is not None and p.t <= t < p.t + EVAL_WINDOW_S]
        if not win:
            missing += 1
            continue
        for f in names:
            peaks.append(max(lift[f] for lift in win))

    normal_vals = [lift[f] for _t, lift in _run_lift(normal_path, calib, cfg)
                   if lift is not None for f in fingers]
    if not normal_vals:
        raise ValueError(f"{normal_path}: elin göründüğü kare yok")
    p99 = float(np.percentile(normal_vals, EVAL_PERCENTILE))

    return EvalResult(fingers=fingers, peaks=peaks, normal_p99=p99, n_prompts=len(prompts),
                      n_missing=missing, calib=calib, fitted=fitted, notes=notes)


def format_result(r: EvalResult) -> str:
    lines = [f"bip: {r.n_prompts} (el görünmeyen pencere: {r.n_missing}), "
             f"parmaklar: {', '.join(r.fingers)}"]
    src = "fit_taps (TAP kaydı)" if r.fitted else "kalibrasyon dosyası"
    tap_ref = r.calib.tap_ref or {}
    lines.append(f"tap_ref [{src}]: "
                 + ", ".join(f"{k}={v:.4f}" for k, v in tap_ref.items())
                 + f"; angle_sign={r.calib.angle_sign:+.0f}")
    if r.peaks:
        q = np.percentile(r.peaks, [0, 10, 25, 50, 75, 90, 100])
        lines.append(f"tap tepeleri (n={len(r.peaks)}): min={q[0]:.3f} p10={q[1]:.3f} "
                     f"p25={q[2]:.3f} medyan={q[3]:.3f} p75={q[4]:.3f} p90={q[5]:.3f} "
                     f"max={q[6]:.3f}")
    else:
        lines.append("tap tepeleri: yok")
    lines.append(f"normal lift p{EVAL_PERCENTILE:g}: {r.normal_p99:.3f}")
    verdict = "GEÇTİ" if r.passed else "GEÇMEDİ"
    tail = f"(eşik {EVAL_RATIO_MIN:g}: {verdict})"
    st = r.status
    if st == "ok":
        lines.append(f"ayrışma oranı: {r.ratio:.2f}  {tail}")
    elif st == "no_noise":
        lines.append(f"ayrışma oranı: hesaplanmadı, normal kullanımda gürültü yok "
                     f"(p{EVAL_PERCENTILE:g} = 0, tap tepe medyanı {r.peak_median:.3f} > 0)  "
                     f"{tail}")
    elif st == "no_tap":
        lines.append(f"ayrışma oranı: ölçülemez, tap tepe medyanı 0 (kalkış görülmedi)  {tail}")
    else:
        lines.append(f"ayrışma oranı: ölçülemez, tap penceresinde el yok  {tail}")
    lines.extend(r.notes)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tkt.vision.lift")
    parser.add_argument("--eval", nargs=2, metavar=("TAP", "NORMAL"),
                        help="ham kayıtlarda ayrışma oranını yazar")
    parser.add_argument("--config", default="config/vision.toml")
    parser.add_argument("--refit", action="store_true",
                        help="kalibrasyonda tap_ref olsa da TAP kaydından fit_taps ile yeniden bul")
    args = parser.parse_args(argv)
    if args.eval is None:
        parser.error("--eval TAP NORMAL gerekli")

    from tkt.vision import hand_calib
    from tkt.vision.live_source import load_vision_config

    cfg = load_vision_config(args.config)
    calib = hand_calib.load(Path(cfg["hand"]["calibration"])) or hand_calib.DEFAULT
    try:
        result = evaluate(args.eval[0], args.eval[1], calib, cfg["lift"], refit=args.refit)
    except (FileNotFoundError, ValueError) as e:
        print(f"hata: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    print(format_result(result))


if __name__ == "__main__":
    main()
