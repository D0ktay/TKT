# PART 2: Vision (Kişi A, Mac + Brio, Claude Code)

Görev: kameradan `HandFrame` üretmek. Eşik uygulanmaz, tık veya mod kararı verilmez.
Saatler, performans rakamları ve sayısal değerler tahmindir; ölçerek doğrula.

## 2.1 Çalışma alanı

- Şeritler ve dosyalar: `PART0_ORKESTRASYON.md`, 0.5 (V1, V2, V3). Aynı anda en çok 2 ajan.
- `os_input/base.py`, Faz 0'da control imza iskeletiyle gelir (B yazar).

| Saat | B'ye teslim |
|---|---|
| 3,5 | Kaba kayıtlar (kalibrasyonsuz olabilir) |
| 6 | Kalibre kayıtlar |
| 11 | `lift` dolu ayar kayıtları ve bip yan dosyaları |
| 12,5 | `lift` dolu doğrulama kayıtları |

## 2.2 İmza iskeleti (Faz 0)

Gövdeler `NotImplementedError`. Genel imzalar sonradan değişmez.

```python
# vision/camera.py                                  (V1)
@dataclass(frozen=True)
class CapturedFrame:
    image: np.ndarray        # BGR, 180° döndürülmüş
    t: float                 # time.monotonic(), kare alındığı an
    frame_id: int

class Camera:
    def __init__(self, index: int, width: int, height: int, fps: int,
                 rotate_180: bool = True, backend: str = "auto") -> None: ...
    def start(self) -> None: ...
    def read(self, timeout: float = 1.0) -> CapturedFrame | None: ...  # en son kare
    def measured_fps(self) -> float: ...
    def close(self) -> None: ...
# CLI: python -m tkt.vision.camera --probe

# vision/tracker.py                                 (V1)
@dataclass(frozen=True)
class RawHand:
    image_pts: np.ndarray    # (21, 2), 0–1, kullanıcı bakışı
    world_pts: np.ndarray    # (21, 3), metre
    score: float

class HandTracker:
    def __init__(self, model_path: str, min_detection: float, min_tracking: float) -> None: ...
    def process(self, image: np.ndarray, t: float) -> RawHand | None: ...
    def close(self) -> None: ...
# CLI: python -m tkt.vision.tracker --download   (modeli models/ altına indirir)

# vision/hand_calib.py                              (V2)
@dataclass(frozen=True)
class HandCalibration:
    open_ref: dict[str, float]         # parmak adı → curl_raw (açık el)
    fist_ref: dict[str, float]         # parmak adı → curl_raw (yumruk)
    thumb_closed: float
    thumb_open: float
    tap_ref: dict[str, float] | None   # parmak adı → tap tepe değeri (Faz 2)
    angle_sign: float = 1.0            # kalkış yönü, tap kalibrasyonundan

DEFAULT: HandCalibration               # kalibrasyon yokken kullanılır
def load(path: Path) -> HandCalibration | None: ...
def save(calib: HandCalibration, path: Path) -> None: ...
def fit_hand(open_s: list[RawHand], fist_s: list[RawHand],
             thumb_closed_s: list[RawHand]) -> HandCalibration: ...
def fit_taps(calib: HandCalibration, samples: list[tuple[float, RawHand]]) -> HandCalibration: ...

# vision/features.py                                (V2)
def curl_raw(hand: RawHand, finger: str) -> float: ...
def thumb_raw(hand: RawHand) -> float: ...
def build_frame(hand: RawHand | None, t: float, frame_id: int, calib: HandCalibration,
                info: SourceInfo, lift: dict[str, float] | None = None,
                prev_hand_size: float | None = None) -> HandFrame: ...

# vision/lift.py                                    (V2, Faz 2)
class LiftEstimator:
    def __init__(self, calib: HandCalibration, cfg: Mapping[str, Any]) -> None: ...
    def update(self, hand: RawHand, t: float, curls: dict[str, float]) -> dict[str, float]: ...
    def reset(self) -> None: ...
# CLI: python -m tkt.vision.lift --eval TAP.raw.jsonl NORMAL.raw.jsonl

# vision/live_source.py                             (V1)
class LiveSource:                      # HandFrameSource
    info: SourceInfo
    def __init__(self, camera: Camera, tracker: HandTracker, calib: HandCalibration,
                 hand: str, lift: LiftEstimator | None = None) -> None: ...
    def frames(self) -> Iterator[HandFrame]: ...
    def frames_raw(self) -> Iterator[tuple[HandFrame, RawHand | None, np.ndarray]]: ...
    def close(self) -> None: ...

def load_vision_config(path: str | Path) -> dict: ...
def open_live_source(config_path: str | Path = "config/vision.toml") -> LiveSource: ...

# vision/debug_view.py                              (V3)
def run(source: LiveSource, calib_path: Path) -> None: ...   # ana thread'de çalışır
# CLI: python -m tkt.vision.debug_view
```

- `open_live_source`, `app/main.py`'nin vision'dan çağırdığı tek fonksiyondur.
- `frames_raw`, debug penceresi ve `record --raw` içindir. `frames` onu sarar.
- `tap_ref` yoksa `LiftEstimator` kurulmaz ve `lift` alanı `None` kalır.

## 2.3 `config/vision.toml`

```toml
[camera]
index = 0
width = 1280
height = 720
fps = 60
rotate_180 = true
backend = "auto"          # auto | avfoundation | dshow | pyav

[tracker]
model = "models/hand_landmarker.task"
min_detection = 0.5
min_tracking = 0.5

[hand]
side = "right"
calibration = "~/.tkt/calibration.json"

[lift]
w_angle = 1.0
w_length = 1.0
baseline_s = 1.0
curl_rate_suppress = 3.0  # 1/sn; curl bundan hızlı artıyorsa lift bastırılır
```

## 2.4 Faz 1 (saat 1,5–7)

Faz 0'daki fps ölçümü için ajan beklenmez; 10 satırlık geçici bir OpenCV betiği yeter.

| Görev | Saat | Model | Branch | İş |
|---|---|---|---|---|
| V1-1 | 1,5–3 | Sonnet | `vision/v1-pipeline` | `camera.py`, `tracker.py`, `live_source.py`, `record.py` (`--out --note --duration --raw --video`) |
| V2-1 | 1,5–3 | Opus | `vision/v2-features` | `features.py`, `hand_calib.py` (`DEFAULT`, `load`, `save`) |
| V3-1 | 3–4,5 | Sonnet | `vision/v3-debug` | `debug_view.py` |
| V2-2 | 3–4,5 | Opus | `vision/v2-calib` | `fit_hand`, güven değeri, ilk canlı denemede çıkan düzeltmeler |
| V1-2 | 5–7 | Sonnet | `vision/v1-prompt` | `record.py`: `--prompt N --kind tap1|tap2`, `--from-raw` |

### Modül notları

**camera.py**
- macOS'ta `cv2.CAP_AVFOUNDATION`; 1280×720 ve 60 fps istenir.
- Yakalama ayrı thread'de döner, yalnızca en son kareyi tutar. Kare biriktirilmez.
- `t` kare alınır alınmaz okunur. İşleme yetişmezse `frame_id` atlar.
- `--probe` 5 saniye sayar ve gerçek fps'i yazar.
- Risk: OpenCV 60 fps isteğini yok sayabilir. 30 fps geliyorsa `backend = "pyav"` denenir
  (ek bağımlılık, ayrı PR).

**tracker.py**
- MediaPipe Tasks, `HandLandmarker`, `VIDEO` modu, `num_hands=1`.
- Zaman damgası milisaniye ve kesin artan olmalıdır.
- Sağ/sol etiketi için MediaPipe'a güvenilmez, `hand.side` kullanılır.
- Kare başına süre ölçülür. 16 ms'yi aşarsa 60 fps tutmaz.

**features.py**

Nokta numaraları: bilek 0; işaret 5-6-7-8 (kök, orta eklem, uç eklem, uç); orta 9–12;
yüzük 13–16; serçe 17–20; başparmak ucu 4.

```python
def flexion(a, b, c):                 # b eklemindeki bükülme; düz parmakta ~0
    u, v = b - a, c - b
    return arccos(clip(dot(u, v) / (norm(u) * norm(v)), -1, 1))

curl_raw = flexion(MCP, PIP, DIP) + flexion(PIP, DIP, TIP)      # dünya noktalarıyla
curl = clip((curl_raw - open_ref) / (fist_ref - open_ref), 0, 1)
```

- `curl`: Yalnızca orta ve uç eklem. Kök eklemi dışarıdadır; kökten hareket `lift`'e aittir.
- `thumb_out`: Başparmak ucu (4) ile işaret kökü (5) arası 3D mesafe, el boyuna bölünür,
  kalibrasyonla 0–1'e oturtulur.
- `img`: Parmak uçları (8, 12, 16, 20). `palm`: 0, 5, 9, 13, 17 ortalaması.
- `hand_size`: 0–9 arası mesafe pikselde hesaplanır, görüntü genişliğine bölünür.
- `confidence`: MediaPipe skoru. El kadraj kenarındaysa veya `hand_size` bir karede %20'den
  fazla sıçradıysa düşürülür.
- Zaman filtresi uygulanmaz.

**hand_calib.py**
- `DEFAULT` kaba değerler taşır (açık ≈ 0,2 rad, yumruk ≈ 3,0 rad); ilk kayıtlar bununla alınır.
- `fit_hand` örneklerin medyanını kullanır. Sonuç `~/.tkt/calibration.json` dosyasına yazılır.

**live_source.py ve record.py**
- El yokken de `present=False` kare üretilir.
- `--raw`: 21 noktayı (görüntü ve dünya) `ad.raw.jsonl` dosyasına yazar. `--video`: `ad.mp4`.
  İkisi de repoya girmez.
- `--prompt N`: 2 saniyede bir bip verir, zamanları `ad.prompts.json` dosyasına yazar
  (biçim Part 1, 1.3).
- `--from-raw`: Ham kayıttan, güncel `features` ve `lift` ile `HandFrame` kaydını yeniden üretir.

**debug_view.py**
- İskelet, parmak başına `curl` ve `lift` çubukları, `thumb_out`, güven.
- Yakalama fps'i, işleme fps'i, kare başına ms.
- Tuşlar: `c` kalibrasyon (açık el 2 sn, yumruk 2 sn, başparmak kapalı 2 sn), `q` çıkış.
- macOS'ta `cv2.imshow` ana thread'de çalışmalıdır.

### İnsan işleri (A)

| Saat | İş |
|---|---|
| 3–3,5 | V1-1 ve V2-1 birleşince kaba kayıtları al, `data/samples/` altına gönder |
| 4,5–6 | Debug penceresiyle kalibrasyon, kabul ölçütlerini ölç, kalibre kayıtları yenile |
| 6–7 | Mac izinleri (Kamera, Erişilebilirlik, Girdi İzleme). `main.py --source live` ile ilk canlı deneme |

Kayıt listesi (30'ar saniye, hepsi `--raw` ile): imleç; imleç ve taşıma; başparmak sol tık
(20 kez); sürükleme; iki parmak scroll; başparmak sağ tık (10 kez); el girip çıkma; el sabit
(10 sn).

### Kabul ölçütleri

- Ölçülen yakalama 58 fps üstü, işleme 55 fps üstü.
- Açık elde `curl < 0,15`, yumrukta `curl > 0,85`.
- Başparmak açıkken `thumb_out > 0,8`, kapalıyken `< 0,2`.
- El sabitken parmak ucu oynaması 1,5 pikselin altında (ham).

## 2.5 Faz 2 (saat 8–13)

| Görev | Saat | Model | Branch | İş |
|---|---|---|---|---|
| V2-3 | 9–11 | Opus | `vision/v2-lift` | `lift.py`, `--eval`, `fit_taps` |
| V3-2 | 9–10 | Sonnet | `vision/v3-tapcalib` | Debug penceresi: tap kalibrasyon adımı, `lift` kayan grafiği |
| V1-3 | 10–12 | Sonnet | `vision/v1-winbackend` | `camera.py`: Windows'ta DirectShow ve MJPG. B dener |

### Açı deneyi (saat 8–9, insan işi)

İki kurulum: karşıdan 27° (mevcut) ve çapraz 30° (kamera elin sol önünde).
Her birinde `--raw --video` ile:

- 50 tek tap (`--prompt 50 --kind tap1`)
- 50 ikili tap (`--prompt 50 --kind tap2`)
- 3 dk tap içermeyen kullanım: imleç, taşıma, başparmak tıkı, scroll

### lift.py

```python
n = unit(cross(P5 - P0, P17 - P0))                 # avuç düzlemi normali (dünya noktaları)
angle = angle_sign * arcsin(dot(unit(PIP - MCP), n))
length = norm(TIP_img - MCP_img) / hand_size       # görüntüdeki parmak boyu

lift_raw = w_angle * (angle - angle_base) + w_length * (length_base - length)
lift = clip(lift_raw / tap_ref, 0, 1)
```

- Açı: Mutlak değeri gürültülüdür, ani değişimi belirgindir.
- Görüntü boyu: Karşıdan kamerada kalkan parmak ucu bileğe doğru kayar (2 cm kalkışta kaba
  hesapla 20–25 piksel).
- Çapraz kamerada kalkış parmak ucunu yana kaydırır, kıvırma kaydırmaz. Gerekirse bu üçüncü
  sinyal olarak eklenir.
- Taban çizgisi: `angle_base` ve `length_base`, son `baseline_s` saniyenin yavaş izlenen
  değeridir. `lift` yüksekken dondurulur.
- `curl` hızla artıyorsa `lift` bastırılır (kıvırma tap değildir).
- `tap_ref` ve `angle_sign`, tap örneklerinin tepe değerlerinden çıkar (`fit_taps`).

### Açı seçimi (saat 11 kontrol noktası)

`lift --eval` şunu yazar:

**Ayrışma oranı = tap tepe medyanı / normal kullanımdaki `lift`'in 99. yüzdeliği**

- En yüksek oranı veren kurulum seçilir, ölçüleri ve fotoğrafı `docs/SETUP.md` içine yazılır.
- Oran 1,5'in altındaysa tap bırakılır ve B'ye haber verilir.
- Seçilen kurulumun deney kayıtları `--from-raw` ile yeniden üretilir: bunlar ayar kayıtlarıdır.

### Doğrulama kayıtları (saat 11–12,5, insan işi)

Seçilen kurulumda tap kalibrasyonu yapılır, sonra yeni çekim: 50 tek tap, 50 ikili tap,
5 dk tap içermeyen kullanım. B eşikleri ayar kayıtlarında belirler, kapıyı bunlarda ölçer.

## 2.6 Faz 3 ve dondurma (saat 14–20)

Zorunlu (insan ağırlıklı):

1. Sağlamlık: hızlı kaydırma, loş ışık, elin yandan girmesi, yumruktan açılış. Takip kopunca
   yeniden yakalama 200 ms'nin altında olmalı.
2. Performans: işleme 55 fps üstü. Altındaysa önce model ayarları, sonra elle ROI kırpma.
3. Gerçek gecikme: telefonun ağır çekimiyle (240 fps) parmak ve ekran aynı kadrajda ölçülür.
   Yazılım içi ölçüm kameranın kendi gecikmesini içermez.
4. Saat 18–20: `docs/SETUP.md`, demo videosu, hata düzeltme.

Zaman kalırsa:

| Görev | Model | Branch | İş |
|---|---|---|---|
| V3-3 | Sonnet | `vision/v3-macos` | `os_input/macos.py`: Quartz ile hareket, tık, piksel scroll |
| V3-4 | Opus | `vision/v3-surface` | `calibration.py`: 4 köşe, homografi, `surface`. Önce şema 2 için `core` PR'ı |

`macos.py` notları:

- Tuş basılıyken hareket "dragged" olayı olarak gönderilir, yoksa sürükleme çalışmaz.
- Çift tıkta olayın tıklama sayısı alanı 2 yapılır.
- Entegrasyon 1'de `pynput` ile çift tık Mac'te çalışmıyorsa V3-3 "Hedef" önceliğine çıkar.

## 2.7 Testler (`tests/vision`)

- `test_features.py`: Yapay noktalarla düz parmak `curl ≈ 0`, bükülü `≈ 1`; kalibrasyon
  0–1'e oturtuyor.
- `test_hand_calib.py`: Kaydet-yükle aynı sonucu verir; `fit_hand` medyanı kullanır.
- `test_lift.py`: Kısa bir ham kayıtta tap tepeleri beklenen karelerde; kıvırmada `lift` düşük.
- `test_live_source.py`: Sahte kamera ve tracker ile `frame_id` artar, el yokken boş kare gelir.
- `test_record.py`: `--from-raw` çıktısı `core/io.py` ile okunur.
- Gerçek kamera isteyen testler `@pytest.mark.camera` ile işaretlenir.
