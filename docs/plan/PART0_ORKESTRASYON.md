# PART 0: Orkestrasyon (20 saat, çoklu ajan)

Saatler tahmindir. Kontrol noktalarındaki kararlar bağlayıcıdır.

## 0.1 Kapsam

| Öncelik | İçerik |
|---|---|
| Zorunlu | J0–J6 ve J10: Mac'te canlı; Windows'ta önce kayıt oynatarak, sonra Brio ile canlı. İkisinde de `pynput` |
| Hedef | J7–J8 tap (kapıya bağlı). `pynput` Mac'te çift tık veya sürüklemede sorun çıkarırsa `macos.py` |
| Zaman kalırsa | `windows.py`, ataletli scroll, J9 dört parmak kaydırma, `replay.py` görselleştirici, aktif alan |
| Kapsam dışı | Tray, paketleme, kurulum sihirbazı, zoom |

## 0.2 Ajanlar ve modeller

| Taraf | Ajan | Aynı anda |
|---|---|---|
| vision (V1–V3) | Claude Code (`ysf`), Mac'te | En çok 2 |
| control (C1–C3) | Codex (`emre`), Mac'te | En çok 2 |
| Windows'a özel düzeltmeler | İkisinden biri, Windows makinede | 1, yalnızca Windows testleri sırasında |

| İş türü | Claude | Codex |
|---|---|---|
| İskelet, `core`, imza iskeletleri, `features`, `lift`, `gestures`, `tap`, `pointer` | En yeni Opus | En güçlü model, reasoning `high` |
| Diğer işler | En yeni Sonnet | Bir alt model veya reasoning `medium` |

Control ajanları sentetik veriyle gözetimsiz çalışır. Dikkatin vision'da olur;
control PR'larını blok aralarında toplu birleştirirsin.

## 0.3 Saat planı

| Saat | Blok | Claude (vision) | Codex (control) | Sen |
|---|---|---|---|---|
| 0–1,5 | Faz 0 | İskelet ve `core`, sonra vision imza iskeleti | Control imza iskeleti | Kamera kurulumu, fps ölçümü, birleştirme. Windows'ta repoyu klonla, `uv sync` ve `pytest` çalıştır |
| 1,5–4 | Faz 1a | V1-1, V2-1 | C1-1, C3-1; biri bitince C2-1 | Saat 3: kaba kayıtlar |
| 4–7 | Faz 1b | V3-1, V2-2, sonra V1-2 | C3-2, C2-2 | Kalibrasyon, kalibre kayıtlar, canlı deneme, eşik ayarı |
| 7–8 | MVP kontrolü | | | Mac'te J0–J6 ve J10 canlı. Windows testi 1: kayıtları gerçek backend ile oynat. `v0.1` |
| 8–13 | Faz 2 | V2-3, V3-2, V1-3 | C3-3, C1-3, sonra C1-4 | Açı deneyi, açı seçimi, doğrulama kayıtları, canlı tap denemesi |
| 13–14 | Tap kapısı | | | `eval_taps.py` raporu, karar, `v0.2` |
| 14–18 | Faz 3 | Zaman kalırsa V3-3 | Zaman kalırsa C1-5, C2-3, C3-4 | Sağlamlık, his ayarı, gecikme ölçümü. Windows testi 2: Brio Windows'ta, canlı (~1 saat) |
| 18–20 | Dondurma | Hata düzeltme | README | `SETUP.md`, demo videosu, `v1.0` |

Windows testi 1 (kamera gerekmez):
- `main.py --source replay --backend pynput` ile kayıtlar oynatılır, imleç gerçekten
  hareket eder.
- Bakılacaklar: tık, sürükleme, scroll adımı, acil durdurma, ölçekli ekranda hareket.

Windows testi 2 (Brio Windows'a takılır):
- Logitech uygulamasından odak ve pozlama sabitlenir; `camera --probe` ile fps ölçülür.
- El kalibrasyonu yeniden yapılır (kalibrasyon dosyası makineye özeldir).
- J0–J6 ve J10 canlı denenir; tap kapısı tuttuysa J7–J8 de.

## 0.4 Kontrol noktaları

| Saat | Soru | Hayırsa |
|---|---|---|
| 1,5 | Windows'ta `uv sync` ve `pytest` yeşil mi? | Kurulum sorunu hemen çözülür, sona bırakılmaz |
| 2 | Mac'te ölçülen hız 58 fps üstü mü? | 1 saat PyAV denenir. Olmazsa 30 fps ile devam, tap "deneysel" olur |
| 3,5 | Kaba kayıtlar repoda mı? | Diğer işler bırakılır, önce kayıt alınır |
| 7 | Zorunlu jestler Mac'te canlı çalışıyor mu? | Faz 2 başlamaz, süre MVP'ye harcanır |
| 8 | Windows'ta kayıt oynatma sorunsuz mu? | Sorunlar issue olur, Faz 2'de C1 şeridi düzeltir |
| 11 | Ayrışma oranı 1,5'in üstünde mi? | Tap bırakılır, başparmak tıkı varsayılan kalır |
| 14 | Tap kapısı tuttu mu? | Tap config'de "deneysel" kalır |
| 16 | Windows'ta canlı 58 fps geliyor mu? | V1-3 Windows makinede düzeltilir. Olmazsa Windows'ta 30 fps ve başparmak tıkı |
| 18 | | Yeni özellik eklenmez |

## 0.5 Ajan şeritleri

Bir şerit, bir ajan, bir dosya kümesi. Aynı dosyada aynı anda iki ajan çalışmaz.

| Şerit | Dosyalar | Faz 1 | Faz 2 ve sonrası |
|---|---|---|---|
| V1 boru hattı | `vision/camera.py`, `tracker.py`, `live_source.py`, `tools/record.py` | Yakalama, MediaPipe, kayıt (`--raw`, `--video`) | `--prompt`, Windows yakalama backend'i |
| V2 ölçüm | `vision/features.py`, `hand_calib.py`, `lift.py` | `curl`, `thumb_out`, el kalibrasyonu | `lift`, tap kalibrasyonu |
| V3 araçlar | `vision/debug_view.py`, `docs/SETUP.md`, `calibration.py`, `os_input/macos.py` | Debug penceresi | Zaman kalırsa: `macos.py`, aktif alan |
| C1 giriş-çıkış | `control/sources.py`, `os_input/` (`macos.py` hariç), `app/`, `tools/replay.py`, `tools/eval_taps.py` | Kaynaklar, `DryBackend`, `pynput`, acil durdurma, `main` | `eval_taps.py`. Zaman kalırsa: `windows.py` |
| C2 hareket | `control/filters.py`, `pointer.py`, `scroll.py` | One Euro, ivme, geri alma, temel scroll | Ataletli scroll |
| C3 jestler | `control/gestures.py`, `tap.py` | Durum makinesi, taşıma, başparmak tuşu | Tap dedektörü, dört parmak kaydırma |

Her şerit kendi dosyalarının testini de yazar.

## 0.6 Faz 0 adımları

1. A, tek ajan, en güçlü model: repo iskeleti.
   - `pyproject.toml` (tüm bağımlılıklar baştan), `ruff`, `pytest`, CI, `.gitignore`
   - `AGENTS.md`, `CLAUDE.md`, `docs/plan/` dosyaları
   - `core/types.py`, `core/source.py`, `core/io.py` ve yaz-oku testi
2. PR'ı kendin birleştirirsin. `main` koruması açılır.
3. Her taraf kendi klasöründe imza iskeleti PR'ı açar: tüm modüller, sınıf ve fonksiyon
   imzaları, gövdeler `NotImplementedError`, config dosyasının tamamı. İmzalar Part 2 ve
   Part 3'tedir.
4. Paralel ajanlar ancak imza iskeleti birleşince başlar.
5. A, ajan çalışırken: kamera kurulumu ve doğrulama listesi (Part 1, 1.1).

Bağımlılıklar: `opencv-python`, `mediapipe`, `numpy`, `pynput`,
`pyobjc-framework-Quartz` (yalnızca macOS), geliştirme için `pytest` ve `ruff`.

## 0.7 Ajanla çalışma kuralları

- Her ajan thread'i ayrı branch ve ayrı worktree'de açılır.
- Branch adı şeridi taşır: `vision/v2-features`, `control/c3-gestures`.
- Ajana "planı uygula" denmez, tek iş verilir.
- PR küçük tutulur. Kendi klasöründeki PR'ı sahibi, CI yeşilse kendisi birleştirir.
- Birleştirmeden sonra açık ajanlar `main`'i çeker.
- İnsan işleri: kamera kurulumu, kayıt, canlı deneme, his ayarı, birleştirme.
- Ajan kayıt oynatırken gerçek backend kullanmaz, imleç kontrolden çıkar.

Görev şablonu:

```
Oku: AGENTS.md, docs/plan/PART1_ORTAK.md, docs/plan/PART3_CONTROL.md (3.2, pointer bölümü).
Şerit: C2. Branch: control/c2-pointer.
Dosyalar: src/tkt/control/filters.py, src/tkt/control/pointer.py, tests/control/test_filters.py,
tests/control/test_pointer.py.
İş: One Euro filtresi ve göreli imleç hesabı (ivme eğrisi, kesir biriktirme, geri alma).
Bitti: uv run ruff check . ve uv run pytest tests/control/test_filters.py tests/control/test_pointer.py yeşil.
Liste dışında değişiklik gerekiyorsa yapma, dur ve nedenini yaz.
```
