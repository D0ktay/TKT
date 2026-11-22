# TKT

TKT, masaya bakan bir kamerayla elinizi touchpad olarak kullanmanızı sağlar. El
takibi ve kalibrasyon, göreli imleç hareketi, başparmakla tıklama ve sürükleme,
iki parmakla kaydırma, üç veya dört parmakla sistem kısayolları ve kayıt
oynatma uygulanmış durumda. macOS ve Windows
hedefleniyor. Parmakla vurma (tap) deneysel ve varsayılan olarak kapalı.

## Kurulum

Python 3.11 ve `uv` ile komutları repo kökünden çalıştırın. Intel macOS için
MediaPipe sürümü `uv.lock` içinde 0.10.21'e sabitlenmiştir.

```bash
uv sync
uv run python -m tkt.vision.tracker --download
```

## Kamera ve izinler

LogiInsight kit kamerasını (LVNext) USB 3 kablosuyla doğrudan USB 3 porta
bağlayın. Kamerayı parmak uçlarının karşısına, masaya doğru bakacak şekilde
sabitleyin. macOS'ta renkli akış OpenCV üzerinden yaklaşık 30 fps verir.
IR (CPD) akışı 512×640 boyutunda, dikey ve gri tonludur. Windows'ta uygulama
DirectShow ile MJPG ister ve DirectShow açılamazsa Media Foundation'ı dener.

```bash
uv run python -m tkt.vision.camera --list
uv run python -m tkt.vision.camera --probe
```

`--list` ile kamera indeksini bulun. [vision.toml](config/vision.toml) içinde
`[camera]` bölümündeki `index`, `width` ve `height` değerlerini ayarlayın.
`--probe` seçili kameranın gerçek kare hızını ölçer.

- **macOS** için Sistem Ayarları → Gizlilik ve Güvenlik bölümünde çalıştıran
  terminale veya uygulamaya Kamera, Erişilebilirlik ve Girdi İzleme izinlerini verin.
- **Windows**'ta yönetici olarak çalışan pencereler, normal yetkiyle çalışan
  TKT'nin gönderdiği girdiyi yok sayar.

## Kullanım

```bash
uv run python -m tkt.vision.debug_view
```

Debug penceresi el iskeletini, parmak ölçümlerini, lift grafiğini ve fps'yi
gösterir.

- `c`: el kalibrasyonu (açık el 2 sn, yumruk 2 sn, başparmak kapalı 2 sn).
- `t`: tap kalibrasyonu; ekranda "TAP!" görünce işaret parmağıyla tap yapın (10 kez).
- `s` (deney): aktif alan kalibrasyonu; 4 köşede (sol-uzak, sağ-uzak, sağ-yakın, sol-yakın)
  işaret parmağı ucunu köşeye koyup boşluk; `g` köşeyi tekrarlatır. `[surface].file` yoluna yazılır.
- `q`, Esc veya pencereyi kapatmak: çıkış.

Kalibrasyon hemen uygulanır ve `[hand].calibration` yoluna yazılır.

```bash
uv run python -m tkt.app.main --source live --backend pynput
uv run python -m tkt.app.main --source live --backend native --latency
```

`pynput` macOS ve Windows'ta kullanılabilir. `native` Windows'ta `SendInput`,
macOS'ta Quartz kullanır; macOS'ta sürükleme olayları ve çift tık sayısı doğru
gönderilir (Erişilebilirlik izni gerekir). `--latency` her iki backend ile canlı
çalıştırmada kare zamanından girdi gönderimine kadar geçen süreyi ve
kontrol işlem süresini ölçer. Plan hedefi gecikmede p95 < 60 ms.
`Ctrl+Alt+P` çıktıyı duraklatıp basılı tuşları bırakır, tekrar basınca
devam eder. Kısayol `[general].panic_hotkey` ile değiştirilebilir.

### Kayıt ve değerlendirme

```bash
uv run python -m tkt.tools.record --out tap.jsonl --raw --prompt 10 --kind tap1
uv run python -m tkt.tools.record --out normal.jsonl --raw --duration 20
uv run python -m tkt.app.main --source replay --file tap.jsonl --backend dry
uv run python -m tkt.tools.eval_taps tap.jsonl normal.jsonl
uv run python -m tkt.vision.lift --eval tap.raw.jsonl normal.raw.jsonl
```

İlk komutta her bipte işaret parmağıyla bir kez vurun. `--raw` el noktalarını
`*.raw.jsonl` dosyasına, `--prompt` bip zamanlarını `*.prompts.json` dosyasına
yazar. Bipli kayıt son bipten kısa süre sonra kendiliğinden biter.
Süre belirtilmediyse kaydı `q` ardından Enter veya Ctrl+C ile de
durdurabilirsiniz. Normal kayıtta tap yapmadan eli hareket ettirin.

`dry` oynatma işletim sistemine girdi göndermez. `eval_taps`, kayıttaki
`lift` ölçümlerinden bip başına doğru tıkları ve normal kayıttaki yanlış
tıkları sayar. `lift --eval` ham kayıtlarda tap ile normal hareketin
ayrışmasını ölçer ve gerekirse değerlendirme için geçici bir `tap_ref` hesaplar.

## Jestler

Bileğinizi ve avuç altını masaya dayayın. Açık parmak yüzeyde kabul edilir.
Mod değiştirmek için önce dört parmağı da kıvırın.

| Kod | Hareket | Çıktı ve durum |
|---|---|---|
| J0 | Gevşek yumruk | Uygulandı. Dinlenme, yeni jest başlamaz. |
| J1 | Yalnız işaret parmağını açıp kaydır | Uygulandı. Göreli, ivmeli imleç hareketi. |
| J2 | İşareti kıvır, eli taşı, yeniden aç | Uygulandı. İmleci durdurup yeniden konumlanma. |
| J3 | İşaret açıkken başparmağı kapatıp aç | Uygulandı. Sol tuşa basıp bırakma. |
| J4 | İşaret ve orta açıkken eli sabit tutup başparmağı kapat | Uygulandı. Sağ tık. |
| J5 | İşaret açıkken başparmağı kapalı tutup kaydır | Uygulandı. Sol tuşla sürükleme. |
| J6 | İşaret ve orta parmağı birlikte kaydır | Uygulandı. Doğal yönlü, ataletli kaydırma. |
| J7 | İşaret parmağını kaldırıp yüzeye vur | Deneysel. Sol tık. |
| J8 | İşaret ve orta parmakla birlikte vur | Deneysel. Sağ tık. |
| J9 | Üç veya dört parmak açıkken eli yana ya da yukarı kaydır | Uygulandı. Masaüstü geçişi ve Mission Control (Windows’ta Görev Görünümü). |
| J10 | Eli kadrajdan çek veya Ctrl+Alt+P'ye bas | Uygulandı. El kaybında çıktı kesilir ve süre dolunca tuşlar bırakılır. Kısayol çıktıyı duraklatır. |

macOS’ta J9 için Mission Control ve masaüstü geçişi klavye kısayolları
etkin olmalı.

## Ayarlar

| Dosya | Başlıca anahtarlar |
|---|---|
| [vision.toml](config/vision.toml) | `[camera]` `index`, `width`, `height`, `fps`, `backend`, `rotate_180`. |
| [vision.toml](config/vision.toml) | `[tracker]` `model`, `min_detection`, `min_tracking`. `[hand]` `side`, `calibration`. |
| [control.toml](config/control.toml) | `[general]` `panic_hotkey`, `click_method`, `confidence_min`, `lost_release_ms`. |
| [control.toml](config/control.toml) | `[mode]` `lock_ms`, `curl_open`, `curl_closed`. `[thumb]` `closed`, `open`, `drag_threshold`. |
| [control.toml](config/control.toml) | `[pointer]` `gain_min`, `gain_max`. `[scroll]` `natural`, `gain`, `step_px`, `inertia_ms`. |
| [control.toml](config/control.toml) | `[tap]` `lift_start`, `lift_up`, `min_ms`, `max_ms`, `pair_ms`. `[swipe]` `distance`, `window_ms`. |
| [vision.toml](config/vision.toml) | Deney: `[surface]` `enabled`, `file`, `aspect` (aktif alan, varsayılan kapalı). |
| [control.toml](config/control.toml) | Deney: `[surface_pointer]` `enabled`, `aspect`, `gain_*`, `speed_*`, `absolute`. `aspect` iki dosyada aynı olmalı. |

## Sorun giderme

- **Düşük fps** için USB 3 portunu ve kabloyu kontrol edin. Daha düşük bir
  çözünürlük deneyip `--probe` ile tekrar ölçün.
- **Kamera açılmıyorsa** kamera izinlerini ve kameranın başka bir uygulamada
  açık olup olmadığını kontrol edin.
- **Yanlış kamera açılıyorsa** `--list` çıktısına göre `[camera].index` değerini değiştirin.
- **Ölçekli Windows ekranında hareket çarpıksa** Windows ekran ölçeğini kontrol
  edin ve uygulamayı `tkt.app.main` üzerinden başlatın. Bu giriş noktası süreçte
  DPI farkındalığını etkinleştirir.
- **Tap çalışmıyorsa** `[general].click_method` değerini `tap` veya `both`
  yapın ve `[hand].calibration` dosyasında geçerli `tap_ref` değerlerinin
  bulunduğunu kontrol edin. `c` ile yapılan el kalibrasyonu tap referansı
  üretmez. Tap değerlendirmesinde kayıtta `lift` değerleri bulunmalıdır.

## Geliştirme

```bash
uv run ruff check .
uv run pytest
```

Kamera gerektiren testler varsayılan olarak atlanır. Çalışma planları
[docs/plan/](docs/plan/) altında, ajan kuralları [AGENTS.md](AGENTS.md) içinde.
