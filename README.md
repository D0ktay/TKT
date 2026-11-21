# TKT

Masaya bakan bir kamerayla elden touchpad: imleç, tık, sürükleme ve scroll.
Kamera görüntüsü `vision` tarafında `HandFrame` akışına çevrilir, `control` tarafı bu akışı
işletim sistemi girdisine dönüştürür.

Durum: Faz 0 (iskelet). Plan `docs/plan/` altında, ajan kuralları `AGENTS.md` içinde.

## Kurulum

```bash
uv sync
uv run ruff check .
uv run pytest
```

- Python 3.11 ve `uv` gerekir.
- Hedef platformlar macOS (Intel ve Apple Silicon) ve Windows.
- Kamera isteyen testler `camera` işaretlidir ve varsayılan olarak atlanır.

## Yapı

| Klasör | İçerik |
|---|---|
| `src/tkt/core/` | Ortak sözleşme: `HandFrame`, kaynak protokolü, kayıt okuma-yazma |
| `src/tkt/vision/` | Kamera, el takibi, ölçümler |
| `src/tkt/control/` | Filtreler, imleç, scroll, jestler, giriş backend'leri |
| `src/tkt/app/` | Uygulama girişi ve config |
| `src/tkt/tools/` | Kayıt, oynatma, değerlendirme araçları |
| `config/` | `vision.toml`, `control.toml` |
| `data/samples/` | Küçük örnek kayıtlar |
