"""Kamera yakalama (V1). PART2 2.2, 2.4.

Yakalama ayrı thread'de döner ve yalnızca en son kareyi tutar. Görüntü kullanıcının bakış
açısına göre 180° döndürülür.

Kamera seçimi: config `[camera]` içinde `index` sabittir; `name` (alt dize) doluysa aygıt
adlarından eşleşen ilk index kullanılır, `kind` (any | rgb | ir) ad bulunamazsa yoklamayla
seçer (bkz. `resolve_camera`). Ad listesi:
- Windows: DirectShow (ICreateDevEnum, ctypes + COM); sıra OpenCV DSHOW index'iyle aynıdır.
- macOS: `system_profiler SPCameraDataType -json`; adlar aygıtın benzersiz kimliğine
  (unique ID) göre sıralanır, çünkü OpenCV AVFoundation index'i bu sıradır. Ölçüm
  (2026-11-22, FaceTime + LVNext + iPhone): index 0 = LVNext (0x1000…, 1080p verdi),
  1 = FaceTime (0x8020…, en çok 720p), 2 = iPhone (1535…). Kimlik USB girişine bağlıdır:
  kamera başka girişe takılınca index değişir. system_profiler yoksa
  `ffmpeg -f avfoundation -list_devices` kullanılır; onun sırası OpenCV index'iyle aynı
  DEĞİLDİR (aynı ölçümde LVNext ffmpeg'de 1, OpenCV'de 0'dı).
- Diğer: ad listesi yok; yalnızca `kind` yoklaması ve `index`.

CLI: python -m tkt.vision.camera --probe
     python -m tkt.vision.camera --list
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# "msmf" normalde seçilmez; DirectShow açılamazsa yedek olarak devreye girer.
BACKENDS = ("auto", "avfoundation", "dshow", "msmf", "pyav")

# --list'in taradığı kamera index'leri.
LIST_INDICES = tuple(range(6))

# config `kind` değerleri: any = yoklama yok, rgb = renkli, ir = gri (NIR) ve dikey kare.
KINDS = ("any", "rgb", "ir")

# `_is_gray`: JPEG (MJPG) kod çözme kanallar arasında küçük farklar üretir. Kanallar arası
# mutlak farkın ortalaması ve %99'u bu eşiklerin altındaysa görüntü gri sayılır.
GRAY_MEAN_MAX = 2.0
GRAY_P99_MAX = 8.0

# Ad listesi alan alt süreçler (ffmpeg) bu kadar saniyeden uzun beklenmez.
_NAME_LIST_TIMEOUT_S = 10.0

# `kind` yoklamasında her index bu kadar saniye / en çok bu kadar kare okunur.
_KIND_PROBE_S = 0.3
_KIND_PROBE_FRAMES = 30

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
    """Tek kanallı veya kanalları (toleransla) eşit görüntü gri sayılır (NIR kamera).

    MJPG gri akış kod çözülünce kanallar tam eşit çıkmaz; kanallar arası mutlak farkın
    ortalaması `GRAY_MEAN_MAX`, %99'u `GRAY_P99_MAX` altındaysa gri denir.
    """
    if image.ndim == 2 or image.shape[2] == 1:
        return True
    ch = image[..., :3].astype(np.int16)
    b, g, r = ch[..., 0], ch[..., 1], ch[..., 2]
    diff = np.maximum(np.maximum(np.abs(b - g), np.abs(g - r)), np.abs(b - r))
    return bool(diff.mean() < GRAY_MEAN_MAX and np.percentile(diff, 99) < GRAY_P99_MAX)


def _orientation(width: int, height: int) -> str:
    if height > width:
        return "dikey"
    return "yatay" if width > height else "kare"


# ---------------------------------------------------------------------- aygıt adları


def parse_avfoundation_list(text: str) -> list[str]:
    """`ffmpeg -f avfoundation -list_devices true -i ""` çıktısından video aygıt adları.

    Dönen listenin sırası ffmpeg'in [n] numarasıdır (boşluklar "" ile doldurulur). Ses
    aygıtları ("audio devices") dışarıda kalır.
    """
    names: dict[int, str] = {}
    in_video = False
    for line in text.splitlines():
        low = line.lower()
        if "video devices" in low:
            in_video = True
            continue
        if "audio devices" in low:
            in_video = False
            continue
        if not in_video:
            continue
        m = re.search(r"\[(\d+)\]\s+(.+?)\s*$", line)
        if m:
            names[int(m.group(1))] = m.group(2)
    if not names:
        return []
    return [names.get(i, "") for i in range(max(names) + 1)]


def parse_system_profiler_cameras(text: str) -> list[str]:
    """`system_profiler SPCameraDataType -json` çıktısından OpenCV index sırasıyla adlar.

    OpenCV (AVFoundation) aygıtları benzersiz kimliklerine göre sıralar; aynı sıra kurulur.
    """
    try:
        items = json.loads(text).get("SPCameraDataType", [])
        pairs = sorted((str(d.get("spcamera_unique-id", "")), str(d.get("_name", "")))
                       for d in items)
    except (ValueError, AttributeError, TypeError):
        return []
    return [name for _, name in pairs]


def system_profiler_device_names(run=subprocess.run, which=shutil.which) -> list[str] | None:
    """macOS kamera adları, OpenCV index sırasıyla; alınamazsa None."""
    exe = which("system_profiler")
    if not exe:
        return None
    try:
        proc = run([exe, "SPCameraDataType", "-json"], capture_output=True, text=True,
                   timeout=_NAME_LIST_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("system_profiler kamera listesi alınamadı: %s", exc)
        return None
    return parse_system_profiler_cameras(proc.stdout or "") or None


def avfoundation_device_names(run=subprocess.run, which=shutil.which) -> list[str] | None:
    """AVFoundation video aygıt adları; alınamazsa None.

    Önce `system_profiler` (OpenCV index sırasıyla), o yoksa ffmpeg (sırası OpenCV'yle aynı
    olmayabilir; modül notuna bakın).
    """
    names = system_profiler_device_names(run=run, which=which)
    if names:
        return names
    exe = which("ffmpeg")
    if not exe:
        return None
    try:
        proc = run([exe, "-hide_banner", "-f", "avfoundation", "-list_devices", "true",
                    "-i", ""], capture_output=True, text=True, timeout=_NAME_LIST_TIMEOUT_S,
                   check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("ffmpeg aygıt listesi alınamadı: %s", exc)
        return None
    names = parse_avfoundation_list((proc.stderr or "") + "\n" + (proc.stdout or ""))
    return names or None


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16),
                ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]


def _guid(text: str) -> _GUID:
    u = uuid.UUID(text)
    return _GUID(u.time_low, u.time_mid, u.time_hi_version, (ctypes.c_ubyte * 8)(*u.bytes[8:]))


class _VARIANT(ctypes.Structure):
    # 16 bayt (x86 ve x64): vt + 3 ayrılmış + 8 baytlık birleşim (burada BSTR işaretçisi).
    _fields_ = [("vt", ctypes.c_ushort), ("r1", ctypes.c_ushort), ("r2", ctypes.c_ushort),
                ("r3", ctypes.c_ushort), ("bstr", ctypes.c_wchar_p),
                ("pad", ctypes.c_ubyte * (8 - ctypes.sizeof(ctypes.c_void_p)))]


_VT_BSTR = 8
_CLSID_SYSTEM_DEVICE_ENUM = "62BE5D10-60EB-11D0-BD3B-00A0C911CE86"
_IID_ICREATE_DEV_ENUM = "29840822-5B84-11D0-BD3B-00A0C911CE86"
_CLSID_VIDEO_INPUT_CATEGORY = "860BB310-5D01-11D0-BD3B-00A0C911CE86"
_IID_IPROPERTY_BAG = "55272A00-42CB-11CE-8135-00AA004BB851"


def _com_method(obj: ctypes.c_void_p, slot: int, *argtypes):
    """COM nesnesinin vtable `slot` numaralı yöntemini çağrılabilir yapar (HRESULT döner)."""
    functype = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    return functype(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtbl[slot])


def _com_release(obj: ctypes.c_void_p) -> None:
    if obj:
        _com_method(obj, 2)(obj)


def _friendly_name(moniker: ctypes.c_void_p, oleaut32) -> str:
    bag = ctypes.c_void_p()
    # IMoniker::BindToStorage (yuva 9) -> IPropertyBag
    hr = _com_method(moniker, 9, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                     ctypes.c_void_p)(moniker, None, None,
                                      ctypes.byref(_guid(_IID_IPROPERTY_BAG)), ctypes.byref(bag))
    if hr < 0 or not bag:
        return ""
    try:
        var = _VARIANT()
        # IPropertyBag::Read (yuva 3)
        hr = _com_method(bag, 3, ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_void_p)(
            bag, "FriendlyName", ctypes.byref(var), None)
        if hr < 0 or var.vt != _VT_BSTR:
            return ""
        name = var.bstr or ""
        oleaut32.VariantClear(ctypes.byref(var))
        return name
    finally:
        _com_release(bag)


def dshow_device_names(ole32=None, oleaut32=None) -> list[str] | None:
    """DirectShow video aygıtlarının FriendlyName listesi (Windows; ctypes + COM).

    Sıra OpenCV `CAP_DSHOW` index'iyle aynıdır (aynı ICreateDevEnum sıralaması). Adı
    okunamayan aygıt "" olur (index kaymasın diye). Windows dışında ya da COM hatasında None.
    `ole32`/`oleaut32` testlerde sahte DLL vermek içindir.
    """
    if ole32 is None:
        if sys.platform != "win32":
            return None
        ole32, oleaut32 = ctypes.windll.ole32, ctypes.windll.oleaut32  # type: ignore[attr-defined]
    try:
        # RPC_E_CHANGED_MODE (negatif): thread zaten başka modda başlatılmış; devam edilir,
        # ama CoUninitialize çağrılmaz.
        inited = ole32.CoInitializeEx(None, 0) >= 0     # COINIT_MULTITHREADED
        try:
            return _dshow_names(ole32, oleaut32)
        finally:
            if inited:
                ole32.CoUninitialize()
    except OSError as exc:
        log.warning("DirectShow aygıt listesi alınamadı: %s", exc)
        return None


def _dshow_names(ole32, oleaut32) -> list[str] | None:
    dev_enum = ctypes.c_void_p()
    hr = ole32.CoCreateInstance(ctypes.byref(_guid(_CLSID_SYSTEM_DEVICE_ENUM)), None, 1,
                                ctypes.byref(_guid(_IID_ICREATE_DEV_ENUM)),
                                ctypes.byref(dev_enum))      # CLSCTX_INPROC_SERVER = 1
    if hr < 0 or not dev_enum:
        return None
    names: list[str] = []
    enum = ctypes.c_void_p()
    try:
        # ICreateDevEnum::CreateClassEnumerator (yuva 3); S_FALSE (1) = kategori boş
        hr = _com_method(dev_enum, 3, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)(
            dev_enum, ctypes.byref(_guid(_CLSID_VIDEO_INPUT_CATEGORY)), ctypes.byref(enum), 0)
        if hr < 0:
            return None
        if hr != 0 or not enum:
            return names
        while True:
            moniker, fetched = ctypes.c_void_p(), ctypes.c_ulong(0)
            # IEnumMoniker::Next (yuva 3)
            hr = _com_method(enum, 3, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p)(
                enum, 1, ctypes.byref(moniker), ctypes.byref(fetched))
            if hr != 0 or not moniker:
                break
            try:
                names.append(_friendly_name(moniker, oleaut32))
            finally:
                _com_release(moniker)
        return names
    finally:
        _com_release(enum)
        _com_release(dev_enum)


def list_device_names(backend: str = "auto") -> list[str] | None:
    """Arka uca uygun aygıt adı listesi; index sırasıyla. Bilinmiyorsa None."""
    resolved = resolve_backend(backend)
    if resolved == "dshow":
        return dshow_device_names()
    if resolved == "avfoundation":
        return avfoundation_device_names()
    return None


def match_device_name(names: Sequence[str], needle: str) -> int | None:
    """`needle`'ı (büyük/küçük harf duyarsız alt dize) içeren ilk aygıtın index'i."""
    needle = needle.strip().lower()
    if not needle:
        return None
    for i, name in enumerate(names):
        if needle in name.lower():
            return i
    return None


# ---------------------------------------------------------------------- kamera seçimi


@dataclass(frozen=True)
class CameraChoice:
    index: int
    reason: str                 # insan okunur; log ve --list/--probe çıktısında görünür
    name: str | None = None     # biliniyorsa aygıt adı


def _kind_matches(kind: str, row: dict, strict: bool = True) -> bool:
    if not row.get("opened"):
        return False
    if kind == "rgb":
        return not row["gray"]
    # ir: gri ve dikey; strict=False iken yalnızca gri
    return bool(row["gray"] and (not strict or row["orientation"] == "dikey"))


def probe_kind(kind: str, indices: Sequence[int] = LIST_INDICES, *, width: int, height: int,
               fps: int, backend: str = "auto", camera_factory=None) -> int | None:
    """Index'leri sırayla kısa süre açıp `kind`'e uyan ilkini döndürür (yoksa None).

    rgb = renkli; ir = gri ve dikey (bulunamazsa ilk gri). Uyan index bulununca durulur.
    """
    if kind not in ("rgb", "ir"):
        return None
    gray_fallback: int | None = None
    for index in indices:
        (row,) = list_cameras((index,), width=width, height=height, fps=fps, backend=backend,
                              seconds=_KIND_PROBE_S, max_frames=_KIND_PROBE_FRAMES,
                              camera_factory=camera_factory)
        if _kind_matches(kind, row):
            return index
        if kind == "ir" and gray_fallback is None and _kind_matches(kind, row, strict=False):
            gray_fallback = index
    return gray_fallback


_UNSET = object()


def resolve_camera(cfg: Mapping, *, names: Sequence[str] | None | object = _UNSET,
                   camera_factory=None) -> CameraChoice:
    """Config `[camera]`'dan kullanılacak index'i ve nedenini belirler.

    Sıra: `name` doluysa aygıt adlarında eşleşen ilk index; ad listesi alınamazsa veya
    eşleşme yoksa `kind` (rgb | ir) yoklaması; o da yoksa `index`.
    `names`: verilirse (liste ya da None) ad listesi alınmaz, o kullanılır (testler ve
    `main`'in tek sefer listelemesi için).
    """
    kind = str(cfg.get("kind", "any") or "any").lower()
    if kind not in KINDS:
        raise ValueError(f"kamera kind {kind!r} geçersiz; seçenekler: {', '.join(KINDS)}")
    want = str(cfg.get("name", "") or "").strip()
    backend = cfg.get("backend", "auto")
    index = int(cfg.get("index", 0))
    notes: list[str] = []

    dev_names: Sequence[str] | None = None
    if want:
        dev_names = list_device_names(backend) if names is _UNSET else names  # type: ignore[assignment]
        if dev_names is None:
            notes.append(f"ad listesi alınamadı (name={want!r})")
        else:
            hit = match_device_name(dev_names, want)
            if hit is not None:
                return CameraChoice(hit, f"ad eşleşti: name={want!r} -> index {hit} "
                                         f"({dev_names[hit]!r})", dev_names[hit])
            notes.append(f"ad eşleşmedi (name={want!r}, aygıtlar: {list(dev_names)})")

    if kind != "any":
        found = probe_kind(kind, width=cfg["width"], height=cfg["height"], fps=cfg["fps"],
                           backend=backend, camera_factory=camera_factory)
        if found is not None:
            nm = dev_names[found] if dev_names and found < len(dev_names) else None
            notes.append(f"kind={kind!r} yoklaması index {found} buldu")
            return CameraChoice(found, "; ".join(notes), nm or None)
        notes.append(f"kind={kind!r} yoklaması sonuç vermedi")

    nm = dev_names[index] if dev_names and 0 <= index < len(dev_names) else None
    notes.append(f"config index={index} kullanılıyor")
    return CameraChoice(index, "; ".join(notes), nm or None)


def format_choice(choice: CameraChoice) -> str:
    name = f" ({choice.name})" if choice.name else ""
    return f"seçilen kamera: index={choice.index}{name} — {choice.reason}"


# ---------------------------------------------------------------------- --list / --probe


def list_cameras(indices: tuple[int, ...] = LIST_INDICES, *, width: int, height: int, fps: int,
                 backend: str = "auto", seconds: float = 1.0, max_frames: int = 600,
                 camera_factory=None, names: Sequence[str] | None = None) -> list[dict]:
    """Her index'i kısa süre açıp özelliklerini toplar; açılmayanlar `opened=False` olur.

    fps, kare zaman damgalarından ölçülür (duvar saatine bağlı değil). `camera_factory`
    testlerde sahte kamera vermek içindir. `names` biliniyorsa satırlara `name` eklenir.
    """
    factory = camera_factory or Camera
    rows: list[dict] = []
    for index in indices:
        name = names[index] if names and 0 <= index < len(names) and names[index] else None
        row: dict = {"index": index, "opened": False, "name": name}
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
    lines = [f"{'index':>5}  {'çözünürlük':<11} {'fps':>6}  {'renk':<6} {'yön':<6} "
             f"{'fourcc':<6} ad"]
    for r in rows:
        name = r.get("name") or "-"
        if not r["opened"]:
            lines.append(f"{r['index']:>5}  {'(açılamadı / kare gelmedi)':<40} {name}")
            continue
        fourcc = r.get("negotiated", {}).get("fourcc") or "-"
        lines.append(
            f"{r['index']:>5}  {str(r['width']) + 'x' + str(r['height']):<11} {r['fps']:>6.1f}  "
            f"{'gri' if r['gray'] else 'renkli':<6} {r['orientation']:<6} {fourcc:<6} {name}")
    return "\n".join(lines)


def format_negotiated(n: dict[str, object]) -> str:
    fourcc = n.get("fourcc") or "(bildirilmedi)"
    return (f"gerçekleşen: backend={n.get('backend')} {n.get('width')}x{n.get('height')}"
            f"@{n.get('fps'):.1f} fourcc={fourcc}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tkt.vision.camera")
    parser.add_argument("--probe", action="store_true", help="5 sn sayıp gerçek fps'i yazar")
    parser.add_argument("--list", action="store_true",
                        help="0-5 arası index'leri açıp çözünürlük/fps/renk/yön/ad tablosu yazar")
    parser.add_argument("--config", default="config/vision.toml")
    parser.add_argument("--seconds", type=float, default=5.0, help="--probe süresi")
    args = parser.parse_args(argv)
    if not (args.probe or args.list):
        parser.error("--probe veya --list verin")

    from tkt.vision.live_source import load_vision_config

    cfg = load_vision_config(Path(args.config))["camera"]
    backend = cfg.get("backend", "auto")
    names = list_device_names(backend) if (args.list or cfg.get("name")) else None
    if args.list:
        rows = list_cameras(width=cfg["width"], height=cfg["height"], fps=cfg["fps"],
                            backend=backend, names=names)
        print(format_camera_table(rows))
    choice = resolve_camera(cfg, names=names)
    print(format_choice(choice))
    if not args.probe:
        return
    cam = Camera(choice.index, cfg["width"], cfg["height"], cfg["fps"],
                 cfg.get("rotate_180", True), backend)
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
