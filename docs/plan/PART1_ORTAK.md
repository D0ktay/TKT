# PART 1: Ortak Temel

İki taraf için de bağlayıcıdır. Sayısal değerler başlangıç tahminidir, kayıtlarla ayarlanır.
Saat planı ve ajan düzeni `PART0_ORKESTRASYON.md` içindedir.

## 1.1 Kamera ve fiziksel kurulum

### Yerleşim

```
Yandan görünüm
                               [Brio]  mercek yüksekliği ≈ 50 cm
                              /   |
                 bakış      /     |
                 ekseni   /  27°  |
                        /         | tripod
     bilek ── el ── parmak        |
═════[====== mousepad ======]═════╧═════ masa
     ←──── 20 cm ────→ ←─ 15 cm ─→
(kullanıcı)

Üstten görünüm
                 [tripod + Brio]
                        │ 15 cm
          ┌─────────────┴─────────────┐
          │        aktif alan         │ 20 cm
          │         30 × 20 cm        │
          └───────────────────────────┘
                   ↑ el buradan girer
                   (kullanıcı)
```

- Kamera elin karşısında, parmak uçları tarafında durur ve kullanıcıya doğru aşağı bakar.
- Başlangıç açısı dikeyden 25–30°. Kesin açı Faz 2 deneyinde seçilir.
- Tripod ağır veya masaya kıskaçlı olmalı. Masaya vurunca kamera titrerse tap algılama bozulur.

### Kamera ayarları

| Ayar | Değer | Neden |
|---|---|---|
| Çözünürlük ve hız | 1280×720, 60 fps, MJPG | Tap 100–200 ms sürer; 60 fps'de 6–12 kare eder |
| Görüş açısı | 65° (Logitech uygulamasından) | Ele daha çok piksel, daha az bozulma |
| Odak | Manuel, yüzeye sabit | Otomatik odak el hareket edince oynar |
| Pozlama | Manuel, ~1/200 sn | Hareket bulanıklığını önler |
| Beyaz dengesi, HDR, RightLight | Sabit veya kapalı | Kareler arası parlaklık oynamasın |
| Bağlantı | USB 3 portu, orijinal kablo | Hub üzerinden fps düşebilir |

- Titreme: Şebeke 50 Hz olduğu için bazı lambalar 1/200'de titreme yapar. Görürsen pozlamayı
  1/100'e çek veya ışığı değiştir.
- macOS: OpenCV çoğu zaman odak ve pozlamayı ayarlayamaz. Logitech uygulamasından yapılır.
- Windows: OpenCV'de MJPG açıkça seçilmezse 60 fps gelmez.

### Işık, yüzey, el

- Işık güçlü ve yaygın olmalı, önden veya üstten gelmeli.
- Yüzey mat, koyu ve desensiz olmalı.
- Kol bileğe kadar açık olmalı.

### Görüntü yönü kuralı

Kamera karşıdan baktığı için ham görüntüde el ters görünür. Vision görüntüyü 180° döndürür.
Sözleşmedeki tüm koordinatlar kullanıcının bakış açısındadır:

- `x`: 0 sol, 1 sağ.
- `y`: 0 uzak kenar (kamera tarafı), 1 yakın kenar (kullanıcı tarafı).
- Eli ileri itince `y` azalır ve imleç yukarı gider.

Eğik bakış yüzünden ileri-geri hareket yaklaşık %10 kısa görünür. Yok sayılır; aktif alan işi
yapılırsa `surface` koordinatları düzeltir.

### Kurulum doğrulama listesi (Faz 0, A)

1. Ölçülen hız 58 fps'in üzerinde (ayarlanan değil, sayılan).
2. Hızlı parmak hareketinde parmak ucu net.
3. Görüntüde titreme veya parlaklık dalgalanması yok.
4. Masaya sertçe vurunca görüntü sallanmıyor.
5. El aktif alanın dört köşesinde de tamamen kadrajda.
6. Tripodun yeri masada bantla işaretli, fotoğrafı `docs/SETUP.md` içinde.

## 1.2 El duruşu ve jest sözlüğü

### Temel duruş

```
Üstten (kullanıcının gözünden)          Gösterim
        İşr  Ort  Yüz  Srç               █  açık, yüzeyde
         █    █    █    █                ∩  kıvrık
         █    █    █    █                ▲  kalkıp vuruyor
  Baş    █    █    █    █                ·  başparmak serbest
    \  ┌──────────────────┐              ◄  başparmak ele yapışık
     \ │       avuç       │
       └──────────────────┘
        bilek (yüzeyde dinlenir)
```

- Bilek ve avuç altı her zaman yüzeydedir.
- Açık parmak "touchpad'e değiyor", kıvrık parmak "kaldırıldı" demektir.
- Başparmak parmak sayısına girmez, ayrı bir tuştur.

```
 açık (yüzeyde)        kıvrık (taşıma)          kalkık (tap)
                           _                          __/
 ●─────────            ●──/ \                   ●────/
════════════          ════════════             ════════════
 eklemler düz          orta eklem bükülü        eklemler düz,
                                                parmak kökten kalkar
```

### Jestler

| Kod | Jest | Baş İşr Ort Yüz Srç | Hareket | Çıktı | Öncelik |
|---|---|---|---|---|---|
| J0 | Dinlenme | `· ∩ ∩ ∩ ∩` | Gevşek yumruk | Yok | Zorunlu |
| J1 | İmleç | `· █ ∩ ∩ ∩` | İşaret ucu kayar | Göreli, ivmeli imleç | Zorunlu |
| J2 | Taşıma | `· █→∩→█` | Kıvır, eli taşı, aç | İmleç durur | Zorunlu |
| J3 | Sol tık | `◄ █ ∩ ∩ ∩` | Başparmak kapat-aç | Sol tuş bas-bırak | Zorunlu |
| J4 | Sağ tık | `◄ █ █ ∩ ∩` | El durgunken başparmak kapat-aç | Sağ tık | Zorunlu |
| J5 | Sürükleme | `◄ █ ∩ ∩ ∩` | Başparmak kapalıyken kaydır | Tuş basılı hareket | Zorunlu |
| J6 | Scroll | `· █ █ ∩ ∩` | İki parmak birlikte kayar | Scroll (doğal yön) | Zorunlu (ataletli: Hedef) |
| J7 | Tap | `· ▲ ∩ ∩ ∩` | Kalk-vur, 300 ms altı | Sol tık | Hedef |
| J8 | İkili tap | `· ▲ ▲ ∩ ∩` | İkisi 80 ms içinde kalk-vur | Sağ tık | Hedef |
| J9 | Dört parmak | `· █ █ █ █` | El yana veya yukarı kayar | Masaüstü geçişi, Mission Control | Hedef |
| J10 | Duraklatma | | El kaybolur veya Ctrl+Alt+P | Çıktı durur, tuşlar bırakılır | Zorunlu |

Ayrıntılar:

- J1: Parmağı oynatmak ince ayar, eli kaydırmak uzun mesafe içindir.
- J2: Kıvrılırken ve açılırken imleç dondurulur, çünkü parmak ucu kayar.
- J3: İşaret parmağı oynamadığı için imleç kaymaz. İki hızlı kapat-aç çift tıktır.
- J4: El kayıyorsa hareket scroll sayılır, tık üretilmez.
- J5: El kaybolursa tuş 300 ms içinde kendiliğinden bırakılır.
- J7: Parmak düz kalarak kökten kalkar. Bükülürse taşıma sayılır. 300 ms'den uzun havada
  kalırsa tık üretilmez. İmleç, kalkış başlamadan 100 ms önceki konumuna geri alınır.
- J9: Üç ve dört parmak aynı sayılır. Windows'ta Win+Tab ve Win+Ctrl+ok.

### Ortak davranış kuralları

- Mod kilidi: Açık parmak sayısı 70 ms sabit kalınca mod seçilir, tüm parmaklar kıvrılana
  kadar değişmez.
- Histerezis: Açık için `curl < 0,35`, kıvrık için `curl > 0,55`. Arada önceki durum korunur.
- Dondurma: Kıvrım veya kalkış hızla değişirken imleç hareketi yok sayılır.
- Güven kapısı: `confidence < 0,5` olan kareler "el yok" sayılır.

## 1.3 Sözleşme

Faz 0'da yazılır, sonra kilitlenir.

```python
# src/tkt/core/types.py
SCHEMA_VERSION = 1
FINGERS = ("index", "middle", "ring", "pinky")

@dataclass(frozen=True)
class Finger:
    img: tuple[float, float]             # parmak ucu, 0–1 (kullanıcı bakışı)
    surface: tuple[float, float] | None  # aktif alanda 0–1 (aktif alan yoksa None)
    curl: float                          # 0 düz, 1 yumruktaki kadar kıvrık
    lift: float | None                   # 0 hizada, 1 tap yüksekliğinde (Faz 1'de None)

@dataclass(frozen=True)
class HandFrame:
    t: float                    # karenin yakalandığı an (monotonic, sn)
    frame_id: int               # her kamera karesinde 1 artar
    present: bool
    fingers: dict[str, Finger]  # anahtarlar: FINGERS
    thumb_out: float            # 0 ele yapışık, 1 açık
    palm: tuple[float, float]   # el merkezi, 0–1
    hand_size: float            # bilek–orta parmak kökü mesafesi (görüntü genişliğine oranlı)
    confidence: float           # 0–1

@dataclass(frozen=True)
class SourceInfo:
    schema: int
    width: int
    height: int
    fps: float
    hand: str                   # "right" | "left"
    note: str = ""
```

```python
# src/tkt/core/source.py
class HandFrameSource(Protocol):
    info: SourceInfo
    def frames(self) -> Iterator[HandFrame]: ...
    def close(self) -> None: ...
```

### Anlam kuralları

- Ham ölçüm vision'da, karar control'dedir. Vision eşik uygulamaz, tık veya mod üretmez.
  Control görüntüye dokunmaz.
- `curl`, `lift` ve `thumb_out` kişisel kalibrasyonla 0–1 aralığına oturtulur (vision).
- Vision zaman filtresi uygulamaz. Yumuşatma control'dedir.
- `x` ve `y` ayrı ayrı 0–1'dir. Control, ölçek için `info.width / info.height` oranını kullanır.
- `t`, karenin kameradan alındığı andır. Gecikme ölçümü buna dayanır.
- `frame_id` el olmasa da artar. Atlanan kareler buradan anlaşılır.
- `present=False` ise `fingers` boştur ve diğer alanlar okunmaz.

### Kayıt formatı (JSONL)

```
{"type":"header","schema":1,"width":1280,"height":720,"fps":60,"hand":"right","note":"aci25 tap50"}
{"t":12.3456,"frame_id":741,"present":true,"fingers":{"index":{"img":[0.512,0.431],"surface":null,"curl":0.08,"lift":null}},"thumb_out":0.91,"palm":[0.50,0.62],"hand_size":0.21,"confidence":0.97}
```

- İlk satır başlık, sonraki her satır bir karedir. Gerçek kayıtta dört parmak da bulunur.
- Dosya adı: `YYYYAAGG_konu_aci_kisi.jsonl`, örnek `20250610_tap50_aci25_a.jsonl`.
- Küçük kayıtlar `data/samples/` altına girer. Video dosyaları repoya girmez.

### Bip yan dosyası

Tap kayıtlarında `record.py --prompt` her tap için bip verir ve zamanını yazar:

```
20250610_tap50_aci25_a.prompts.json
[{"t": 12.30, "kind": "tap1"}, {"t": 14.31, "kind": "tap2"}]
```

- `t`, `HandFrame.t` ile aynı saattir. `kind`: `tap1` (tek) veya `tap2` (ikili).
- Vision yazar (`record.py`), control okur (`eval_taps.py`).

Kayıt ve yan dosyanın yazma-okuma kodu `core/io.py` içinde tek yerdedir.

### Planlı değişiklik (şema 2)

Aktif alan işi yapılırsa `SourceInfo`'ya `surface_aspect: float | None` (pad'in en-boy oranı)
eklenir. İş başlamadan önce ayrı `core` PR'ı açılır.

### Değişiklik prosedürü

1. İsteyen kişi issue açar, iki kişi onaylar.
2. Yalnızca `core/` içeren ayrı bir PR açılır, `SCHEMA_VERSION` bir artar.
3. `ReplaySource` eski sürümlü kayıtları dönüştürür veya açık bir hatayla reddeder.

## 1.4 Repo yapısı, sahiplik, kurallar

```
TKT/
├─ AGENTS.md, CLAUDE.md, README.md, pyproject.toml, .gitignore   ORTAK
├─ .github/workflows/ci.yml                                       ORTAK
├─ config/
│  ├─ vision.toml                                                A
│  └─ control.toml                                               B
├─ docs/
│  ├─ plan/        PART0, PART1, PART2, PART3                    ORTAK
│  └─ SETUP.md                                                   A
├─ models/                                                       repoya girmez
├─ src/tkt/
│  ├─ core/        types.py, source.py, io.py                    ORTAK, kilitli
│  ├─ vision/      camera, tracker, features, hand_calib, lift,
│  │               calibration, live_source, debug_view          A
│  ├─ control/     filters, pointer, scroll, gestures, tap,
│  │               sources                                       B
│  │  └─ os_input/ base.py, dry_backend.py, pynput_backend.py,
│  │               windows.py                                    B
│  │               macos.py                                      A
│  ├─ app/         main.py, config.py                            B
│  └─ tools/       record.py                                     A
│                  replay.py, eval_taps.py                       B
├─ tests/vision/                                                 A
├─ tests/control/                                                B
└─ data/samples/                                                 A yazar, B okur
```

### Git kuralları

- `main` korumalıdır: doğrudan push yok, PR ve yeşil CI şart.
- Branch adları: `vision/v1-...`, `control/c2-...`, `core/...`.
- PR'ları tek geliştirici birleştirir, CI yeşilse. `core/` değişikliği yine ayrı PR olur.
- Bağımlılıklar Faz 0'da topluca eklenir. Sonradan ekleme ayrı ve küçük bir PR'dır.

### Araçlar

- Python 3.11, `uv`, `ruff`, `pytest`.
- CI her PR'da macOS ve Windows'ta `ruff` ve `pytest` çalıştırır.
- Kamera isteyen testler `@pytest.mark.camera` ile işaretlenir ve CI'da atlanır.

## 1.5 Kapılar, başarı ölçütleri, riskler

### Entegrasyon (her blok sonunda, yaklaşık 1 saat)

1. İkiniz de `main`'i çeker, başka branch birleştirilmez.
2. A canlı kamerayla jest listesini sırayla dener, B ekran paylaşımından izler.
3. B aynı oturumun kaydını Windows'ta gerçek backend ile oynatır.
4. Sorunlar `vision` veya `control` etiketiyle issue olur. Ölçüt tutuyorsa etiket atılır.

### Tap kapısı (saat 14)

`eval_taps.py`, A'nın doğrulama kayıtları üzerinde ölçer:

| Ölçüt | Hedef |
|---|---|
| Tek tap yakalama | 50 tap'in en az 48'i |
| İkili tap yakalama | 50 tap'in en az 45'i |
| Yanlış tık (5 dk tap içermeyen kullanım) | En çok 1 |
| Tap anında imleç kayması | 3 pikselin altında |

- Tutarsa tap varsayılan tık yöntemi olur, başparmak tuşu sürükleme için kalır.
- Tutmazsa başparmak tıkı varsayılan kalır, tap config'de "deneysel" durur.
- Eşikler bir kayıtta ayarlanır, başka bir kayıtta ölçülür.

### Başarı ölçütleri (saat 18)

| Ölçüt | Hedef | Nasıl ölçülür |
|---|---|---|
| Yazılım içi gecikme | 60 ms altı (p95) | `t` ile mouse olayının gönderildiği an arası |
| İşleme hızı | 55 fps üstü | Debug penceresindeki sayaç |
| Durağan titreme | 2 piksel altı | El sabitken 10 sn imleç kaydı |
| Kesintisiz kullanım | 15 dk | Acil durdurmaya gerek kalmadan |

### Riskler ve yedekler

| Risk | Belirti | Yedek |
|---|---|---|
| Tap güvenilir algılanamaz | Kapı tutmaz | Başparmak tıkı varsayılan kalır |
| Mac'te 60 fps gelmez | Ölçülen hız ~30 fps | PyAV denenir; olmazsa 30 fps ile devam, tap riski artar |
| Pozlama Mac'te sabitlenemez | Bulanık parmak ucu | Logitech uygulaması ve daha güçlü ışık |
| Kıvrık parmakta takip bozulur | Yumrukta el kaybolur | Kamera açısı artırılır, güven kapısı ayarlanır |
| Ajan limitleri dolar | Ajan durur | Rutin işler alt modele iner, sonra ek hesap |
| Paralel ajanlar çakışır | Birleştirme çatışması | Şerit dosya listesine dönülür, iş bölünür |
