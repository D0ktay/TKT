# PART 3: Control (Kişi B, Windows, Codex)

Görev: `HandFrame` akışını imleç, tık ve scroll'a çevirmek. Görüntüye dokunulmaz; eşik,
zamanlama ve karar buradadır. Saatler ve sayısal değerler tahmindir, hepsi config'den okunur.

## 3.1 Çalışma alanı

- Şeritler ve dosyalar: `PART0_ORKESTRASYON.md`, 0.5 (C1, C2, C3). Aynı anda en çok 3 ajan.
- Kamera gerekmez. Geliştirme `SyntheticSource` ve A'nın kayıtlarıyla yapılır.
- Sorun vision kaynaklıysa kayıt adı ve `frame_id` ile issue açılır, vision koduna dokunulmaz.
- A'dan gelenler: saat 3,5 kaba kayıtlar, 6 kalibre kayıtlar, 11 ayar kayıtları (`lift` dolu),
  12,5 doğrulama kayıtları.
- A'ya borç: Faz 0'da `os_input/base.py`, saat 13'te `eval_taps.py` raporu.

## 3.2 İmza iskeleti (Faz 0)

Gövdeler `NotImplementedError`. Genel imzalar sonradan değişmez. Bu PR'da `control.toml`'un
tamamı ve `tests/control/conftest.py` içinde `make_frame(...)` yardımcısı da bulunur.

```python
# control/os_input/base.py                          (C1)
class InputBackend(Protocol):
    def move_rel(self, dx: int, dy: int) -> None: ...
    def button(self, name: str, down: bool) -> None: ...   # "left" | "right"
    def scroll(self, dx: float, dy: float) -> None: ...    # piksel
    def system_action(self, name: str) -> None: ...        # "mission_control" | "desktop_left" | "desktop_right"
    def release_all(self) -> None: ...

# control/os_input/dry_backend.py                   (C1)
@dataclass(frozen=True)
class Event:
    t: float
    kind: str                # "move" | "button" | "scroll" | "system"
    args: tuple

class DryBackend:            # InputBackend; hiçbir şey göndermez
    events: list[Event]
    def set_time(self, t: float) -> None: ...   # döngü her kareden önce frame.t verir

# control/os_input/pynput_backend.py                (C1)
class PynputBackend: ...     # InputBackend

# control/sources.py                                (C1)
class ReplaySource:          # HandFrameSource
    def __init__(self, path: str | Path, fast: bool = False, loop: bool = False) -> None: ...

class SyntheticSource:       # HandFrameSource; betik metotları zincirlenir
    def __init__(self, fps: float = 60.0, noise_px: float = 0.0, drop_rate: float = 0.0,
                 seed: int = 0, realtime: bool = False, with_lift: bool = True) -> None: ...
    def rest(self, s: float) -> Self: ...
    def open(self, n: int, s: float = 0.2) -> Self: ...            # n parmak açılır
    def move(self, dx: float, dy: float, s: float) -> Self: ...    # 0–1 görüntü birimi
    def thumb(self, closed: bool, s: float = 0.1) -> Self: ...
    def tap(self, fingers: tuple[str, ...] = ("index",), ms: float = 150) -> Self: ...
    def curl_all(self, s: float = 0.2) -> Self: ...
    def vanish(self, s: float) -> Self: ...

# control/filters.py                                (C2)
class OneEuro:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None: ...
    def __call__(self, x: tuple[float, float], t: float) -> tuple[float, float]: ...
    def reset(self) -> None: ...

# control/pointer.py                                (C2)
class PointerMotion:
    def __init__(self, cfg: Mapping[str, Any], info: SourceInfo) -> None: ...   # [pointer]
    def update(self, pos: tuple[float, float], hand_size: float, t: float) -> tuple[int, int]: ...
    def rebase(self) -> None: ...                           # sonraki kare yalnızca referans alır
    def rollback(self, ms: float) -> tuple[int, int]: ...   # son ms'de gönderilenin tersi
    def reset(self) -> None: ...

# control/scroll.py                                 (C2)
class ScrollMotion:
    def __init__(self, cfg: Mapping[str, Any], info: SourceInfo) -> None: ...   # [scroll]
    def update(self, pos: tuple[float, float], hand_size: float, t: float) -> tuple[float, float]: ...
    def release(self, t: float) -> None: ...                 # parmaklar kalktı, atalet başlar
    def tick(self, t: float) -> tuple[float, float]: ...     # atalet sürerken her karede
    def stop(self) -> None: ...
    def rebase(self) -> None: ...

# control/tap.py                                    (C3, Faz 2)
@dataclass(frozen=True)
class TapUpdate:
    started: bool            # bu karede kalkış başladı: imleç geri alınır
    busy: bool               # kalkış sürüyor: imleç donuk
    kind: str | None         # "tap1" | "tap2" | None

class TapDetector:
    def __init__(self, cfg: Mapping[str, Any]) -> None: ...   # tüm config
    def update(self, frame: HandFrame, watched: tuple[str, ...]) -> TapUpdate: ...
    def reset(self) -> None: ...

# control/gestures.py                               (C3)
class Mode(Enum):
    NONE = 0; WAIT = 1; POINTER = 2; SCROLL = 3; SWIPE = 4

class GestureEngine:
    mode: Mode
    left_down: bool
    def __init__(self, cfg: Mapping[str, Any], backend: InputBackend, info: SourceInfo) -> None: ...
    def process(self, frame: HandFrame) -> None: ...
    def release(self) -> None: ...     # tuşları bırakır, durumu sıfırlar

# app/config.py                                     (C1)
def load_control_config(path: str | Path = "config/control.toml") -> dict: ...
```

- `GestureEngine` her şeyi bağlar: parmak durumu, mod, başparmak tuşu, tap, `PointerMotion`,
  `ScrollMotion` ve backend çağrıları.
- Sınıflar config'i sözlük olarak alır. `control` modülleri `app`'i import etmez.
- C3 testlerinde `PointerMotion` ve `ScrollMotion` sahte nesnelerle değiştirilebilir.

## 3.3 `config/control.toml`

```toml
[general]
panic_hotkey = "<ctrl>+<alt>+p"
click_method = "thumb"      # thumb | tap | both
confidence_min = 0.5
lost_release_ms = 300

[mode]
lock_ms = 70
curl_open = 0.35
curl_closed = 0.55
freeze_curl_rate = 3.0      # 1/sn; işaret curl'ü bundan hızlı artarsa imleç donar

[thumb]
closed = 0.30
open = 0.50
debounce_ms = 30
drag_threshold = 0.03       # el boyu
still_ms = 150              # sağ tık için elin durgun olması gereken süre

[pointer]
min_cutoff = 1.0
beta = 0.02
gain_min = 500              # ekran pikseli / el boyu
gain_max = 2500
speed_lo = 0.3              # el boyu / sn
speed_hi = 3.0
rollback_ms = 100
gap_reset_ms = 100

[scroll]
natural = true
gain = 1500
axis_lock = 0.5             # baskın olmayan eksen bu oranın altındaysa sıfırlanır
step_px = 40                # pynput için: bir adım kaç piksel
inertia_ms = 300
inertia_min_speed = 200     # piksel / sn

[tap]
lift_start = 0.25
lift_up = 0.50
min_ms = 50
max_ms = 300
pair_ms = 80
max_palm_move = 0.03        # el boyu
cooldown_ms = 100

[swipe]
distance = 0.5              # el boyu
window_ms = 500
```

## 3.4 Faz 1 (saat 1,5–7)

| Görev | Saat | Reasoning | Branch | İş |
|---|---|---|---|---|
| C1-1 | 1,5–3,5 | `medium` | `control/c1-io` | `sources.py`, `dry_backend.py`, `pynput_backend.py`, `app/config.py`, `app/main.py` (acil durdurma, bekçi) |
| C2-1 | 1,5–3,5 | `high` | `control/c2-pointer` | `filters.py`, `pointer.py` |
| C3-1 | 1,5–4 | `high` | `control/c3-gestures` | `gestures.py`: parmak durumu, mod makinesi, taşıma, imleç bağlantısı |
| C1-2 | 3,5–5 | `medium` | `control/c1-replay` | Kayıtlarla regresyon testi (`replay.py` görselleştirici: zaman kalırsa) |
| C2-2 | 3,5–5 | `medium` | `control/c2-scroll` | `scroll.py` temel (ataletsiz) |
| C3-2 | 4–6 | `high` | `control/c3-thumb` | Başparmak tuşu, sürükleme eşiği, sağ tık, scroll bağlantısı |

Saat 6–7 insan işi: kalibre kayıtlarla eşik ayarı, kayıtların Windows'ta gerçek backend ile
oynatılması.

### Modül notları

**sources.py**
- `ReplaySource`: `core/io.py` ile okur, `t` farklarına göre gerçek hızda oynatır. `fast`
  beklemez. Şema sürümü tutmuyorsa açık hatayla durur.
- `SyntheticSource`: 1280×720 varsayar. `tap()` `lift` değerini üretir; `with_lift=False` ise
  `lift` alanı `None` olur.

**Backend'ler**
- `DryBackend` testlerin, `eval_taps.py`'nin ve ajanların kullandığı backend'dir.
- `PynputBackend`:
  - Scroll tam adımla gider (`step_px`), kesir biriktirilir. Adım büyüklüğü işletim
    sistemine göre değişir, elle ayarlanır.
  - `system_action` tuş birleşimi gönderir. Windows: Win+Tab, Win+Ctrl+Sol, Win+Ctrl+Sağ.
    macOS: Ctrl+Yukarı, Ctrl+Sol, Ctrl+Sağ (sistem ayarlarında açık olmalı).
  - macOS'ta iki ayrı bas-bırak çift tık sayılmayabilir. Entegrasyon 1'de denenir;
    çalışmıyorsa A'nın `macos.py` işi öne alınır.

**app/main.py**
- Seçenekler: `--source live|replay|synthetic`, `--file`, `--backend pynput|native|dry`,
  `--fast`, `--loop`.
- `live` seçilirse yalnızca `tkt.vision.live_source.open_live_source` çağrılır.
- Döngü: her kare için `engine.process(frame)`. Backend `DryBackend` ise önce
  `set_time(frame.t)`.
- Acil durdurma: `pynput.keyboard.GlobalHotKeys`, ayrı thread. Basınca `engine.release()` ve
  çıktı durur; tekrar basınca devam eder.
- Bekçi: 300 ms kare gelmezse `engine.release()`.
- Çıkışta ve hatada (`try/finally`) `release_all()`.
- Windows'ta açılışta süreç DPI farkında yapılır, yoksa ölçekli ekranlarda hareket çarpılır.

**pointer.py**

```python
p = one_euro((x * W, y * H), t)        # kamera pikseli
d = (p - p_prev) / hand_px             # el boyu birimi
v = norm(d) / dt                       # el boyu / sn
gain = gain_min + (gain_max - gain_min) * smoothstep(speed_lo, speed_hi, v)
out = d * gain + kalan                 # ekran pikseli
gonder = round(out); kalan = out - gonder
```

- `dt`, `t` farkından hesaplanır. Boşluk `gap_reset_ms`'yi aşarsa filtre sıfırlanır ve o
  kare hareket üretmez.
- `W` ve `H`, `info`'dan gelir. `hand_px`, `hand_size * W` değerinin son 2 saniyelik medyanıdır.
- Sert ölü bölge yoktur; titremeyi One Euro keser.
- Son 150 ms'de gönderilen hareketler bir halkada tutulur. `rollback` bunların tersini döndürür.

**gestures.py: kare başına sıra**

1. `present` değilse veya güven düşükse: `lost_release_ms` sonra `release()`, mod `NONE`.
2. Parmak durumları histerezisle güncellenir.
3. Mod:
   - `WAIT`'te parmak birleşimi `lock_ms` sabit kalınca mod seçilir: yalnızca işaret
     `POINTER`, işaret ve orta `SCROLL`, işaret-orta-yüzük (serçe serbest) `SWIPE`.
   - Başka birleşimler `WAIT`'te kalır.
   - Seçilen mod, tüm parmaklar kıvrılana kadar değişmez.
4. Tap (Faz 2): `click_method` izin veriyorsa ve `lift` doluysa `TapDetector.update`.
5. Başparmak tuşu.
6. Hareket: dondurma yoksa `PointerMotion` veya `ScrollMotion` güncellenir, sonuç backend'e
   gider.

- Taşıma: İşaret parmağının `curl` hızı `freeze_curl_rate`'i aşınca `rollback` gönderilir ve
  imleç donar. Moda yeniden girişte `rebase` çağrılır.
- Dondurma sırasında `update` çağrılmaz; çözülünce `rebase` çağrılır.

**Başparmak tuşu**
- Histerezis: `thumb_out < closed` kapalı, `> open` açık; `debounce_ms` kararlılık şartı.
- `POINTER`: Kapanınca sol tuş basılır, açılınca bırakılır.
- Sürükleme eşiği: Tuş basıldıktan sonra parmak `drag_threshold` kadar kayana dek imleç tutulur.
- `SCROLL`: Son `still_ms` içinde el durgunsa kapanma anında sağ tık gönderilir, değilse yok
  sayılır.
- Mod değişirse veya el kaybolursa tuş bırakılır.

**scroll.py**
- İki parmak ucunun ortalaması izlenir; aynı filtre ve el boyu ölçeği kullanılır.
- `natural = true` ise içerik parmağı takip eder.

**tools/replay.py**
- Kaydı boş tuvalde oynatır: parmak uçları, mod, tuş durumu, sanal imleç izi.

### Kabul ölçütleri

- "20 başparmak tıkı" kaydında tam 20 sol tık.
- "El sabit" kaydında net imleç hareketi 2 pikselin altında.
- "El girip çıkma" kaydında hiçbir tuş basılı kalmaz.
- Kare başına kontrol süresi 2 ms'nin altında.
- Acil durdurma iki bilgisayarda her durumda çalışır.

## 3.5 Faz 2 (saat 8–13)

| Görev | Saat | Reasoning | Branch | İş |
|---|---|---|---|---|
| C3-3 | 8–11 | `high` | `control/c3-tap` | `tap.py` ve motor bağlantısı (`SyntheticSource.tap` ile) |
| C1-3 | 8–10 | `medium` | `control/c1-eval` | `tools/eval_taps.py` |
| C2-3 | 8–9,5 | `medium` | `control/c2-inertia` | Ataletli scroll |
| C1-4 | 10–11 | `medium` | `control/c1-latency` | `main.py`: gecikme kaydı (p50, p95) |
| C3-4 | 11–13 | `high` | `control/c3-swipe` | Dört parmak kaydırma; tap'in gerçek kayıtlara göre düzeltilmesi |

### tap.py

İzlenen her parmak için:

```
HAZIR ──lift > lift_start──► KALKIYOR ──lift > lift_up──► HAVADA ──lift < lift_start──► TAP
(started: imleç geri alınır, donar)                          │
                                               max_ms geçti ──► İPTAL (tık yok)
```

Tap sayılması için hepsi gerekir:

- Toplam süre `min_ms` ile `max_ms` arasında.
- Avuç merkezi `max_palm_move`'dan az kaymış.
- Parmak bu sırada kıvrılmamış (`curl < curl_open`).

Kurallar:

- `POINTER` (izlenen: işaret): Tap `tap1` üretir, motor iniş anında sol tık gönderir.
- `SCROLL` (izlenen: işaret ve orta): İkisi `pair_ms` içinde kalkıp inerse `tap2`, motor sağ
  tık gönderir. Yalnızca biri kalkarsa yok sayılır.
- İptalden sonra parmak inene kadar `busy` sürer; inişte motor `rebase` çağırır.
- Her tap'ten sonra `cooldown_ms` yeni tap kabul edilmez.
- İki hızlı tap iki tık üretir. İmleç donuk olduğu için aynı noktaya düşer.

### Ataletli scroll

- Parmaklar kıvrıldığında hız `inertia_min_speed`'in üstündeyse scroll üstel sönerek sürer
  (zaman sabiti `inertia_ms`).
- Kaynak el yokken de kare ürettiği için atalet kare geldikçe `tick` ile ilerler.
- Parmak yeniden açılınca, tık gelince veya el kaybolunca hemen durur.

### Dört parmak kaydırma

- `SWIPE` modunda avuç merkezi izlenir. `window_ms` içinde `distance` kadar kayma olursa
  baskın yöne göre eylem bir kez tetiklenir.
- Yukarı `mission_control`, yanlar `desktop_left` ve `desktop_right`.
- Yeniden tetiklemek için `WAIT`'e dönmek gerekir.

### eval_taps.py

- Girdi: kayıt, `ad.prompts.json`, config. Kayıt `fast` modda `DryBackend`'e oynatılır.
- Her bipten sonraki 1,5 saniyede doğru türde tık aranır (`tap1` sol, `tap2` sağ).
- Rapor:
  - Yakalama oranı ve kaçan tap'lerin `frame_id` listesi
  - Tap içermeyen kayıttaki yanlış tık sayısı
  - Tap sırasındaki net imleç kayması
  - Tap sürelerinin dağılımı
- Eşikler ayar kayıtlarında belirlenir, kapı doğrulama kayıtlarında ölçülür (Part 1, 1.5).
- Zaman kalırsa: `lift_up`, `min_ms`, `max_palm_move` için ızgara taraması.

Kapı tutarsa `click_method = "both"` varsayılan olur.

## 3.6 Faz 3 ve dondurma (saat 14–20)

Zorunlu:

1. Eşik ve his ayarı: A canlı dener, B config'i ayarlar.
2. Windows canlı test. Laptop kamerası için A'nın V1-3 işi gerekir; Brio varsa onunla.
3. Gecikme raporu: p95, 60 ms'nin altında olmalı.
4. Saat 18–20: README (kurulum, izinler, jest tablosu, ayarlar, sorun giderme), son regresyon.

Zaman kalırsa:

| Görev | Reasoning | Branch | İş |
|---|---|---|---|
| C1-5 | `medium` | `control/c1-windows` | `os_input/windows.py`: `ctypes` ile `SendInput` |

`windows.py` notları:

- İmleç: `GetCursorPos` üzerine eklenip `SetCursorPos` ile gönderilir. `SendInput`'un göreli
  hareketi Windows'un fare ivmesinden etkilenir.
- Scroll: `MOUSEEVENTF_WHEEL` ile 120'den küçük değerler pürüzsüz kaydırma verir. Bazı
  uygulamalar yok sayabilir; "tam adım" seçeneği eklenir.

README'ye yazılacak sınırlar:

- Windows'ta yönetici olarak çalışan pencereler gönderilen girdiyi yok sayar.
- macOS'ta Kamera, Erişilebilirlik ve Girdi İzleme izinleri gerekir.

## 3.7 Testler (`tests/control`)

- `test_filters.py`: Sabit girdide çıktı sabit; gürültüde titreme azalır; rampada gecikme
  sınırlı.
- `test_pointer.py`: 1000 küçük adımın toplamı doğru; zaman boşluğunda zıplama yok; `rollback`
  son 100 ms'yi geri alır.
- `test_scroll.py`: Eksen kilidi; atalet söner ve `stop` ile kesilir.
- `test_gestures.py`: Yumruktan dört parmağa açılırken ara moda girilmez; mod kilidi tutar;
  sınırdaki `curl` değerinde titreme yok; el kaybolunca tuş bırakılır.
- `test_thumb.py`: Bas-bırak tek tık; sürükleme eşiği; `SCROLL`'da hareket varken sağ tık yok.
- `test_tap.py`: Kısa kalkış tık; uzun kalkış tık değil; kıvırma tık değil; ikili tap sağ tık.
- `test_sources.py`: `ReplaySource` yaz-oku; yanlış şemada hata.
- `test_regression.py`: A'nın örnek kayıtları `DryBackend` ile oynatılır, olay sayıları
  sabitlenir.
