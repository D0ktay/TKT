# TKT ajan kuralları

Projede tek geliştirici var. Planlardaki "Kişi A" ve "Kişi B" aynı kişidir;
"B'ye teslim" ifadesi "repoya gönder" demektir. Geliştirme macOS'ta yapılır.
Windows da hedef platformdur ve elle test edilir; kod iki işletim sisteminde
çalışır kalmalıdır.

İşe başlamadan önce oku: `docs/plan/PART0_ORKESTRASYON.md`, `docs/plan/PART1_ORTAK.md`
ve kendi tarafının dosyası (`PART2_VISION.md` veya `PART3_CONTROL.md`).

## Sınırlar
- Sana verilen görevde bir şerit (V1–V3 veya C1–C3), bir branch ve bir dosya listesi vardır.
  Yalnızca o dosyaları ve onların testlerini değiştir.
- Liste dışında değişiklik gerekiyorsa yapma; dur ve nedenini yaz.
- `src/tkt/core/` altını asla kendiliğinden değiştirme.
- `vision` ve `control` birbirini import edemez; ikisi de yalnızca `core`'u import eder.
  İkisini yalnızca `app` bir araya getirir.
- `pyproject.toml`'a bağımlılık, config dosyalarına yeni anahtar ekleme; gerekiyorsa dur ve sor.
- İmza iskeletindeki genel (public) imzaları değiştirme; gerekiyorsa dur ve sor.
- Eşik ve zamanlama değerlerini koda gömme, config'den oku.

## Çalışma şekli
- `main`'e push yapma, başka branch'leri birleştirme. İş bitince PR aç.
- Test dosyası modül adını taşır: `tests/control/test_pointer.py`.
- Kamerayı, ekranı ve gerçek imleci göremezsin. Testlerde `SyntheticSource`, `DryBackend`
  ve kayıt dosyalarını kullan; gerçek giriş backend'ini çalıştırma.
- Bitti sayılması için: `uv run ruff check .` ve kendi testlerin yeşil.
- PR açıklamasına, donanım olmadan doğrulayamadıklarını madde madde yaz.
