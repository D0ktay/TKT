# Claude API Chat

Anthropic API'yi (Fable 5.2, Opus 5.5, Sonnet 5.5, Haiku 5.5) tarayıcıdan kullanmak için tek dosyalık sohbet arayüzü. Kurulum, sunucu veya `npm install` gerekmez.

## Kurulum (herkes kendi bilgisayarında)

1. Repoyu indir:
   ```bash
   git clone https://github.com/D0ktay/TKT.git
   ```
   ya da GitHub'da **Code → Download ZIP**.
2. `claude-chat/claude-chat.html` dosyasına çift tıkla, tarayıcıda açılır.
3. Açılan **⚙ Ayarlar** penceresine **kendi** API anahtarını yapıştır (`sk-ant-...`).
   Anahtarı buradan alırsın: https://platform.claude.com → API Keys.
4. **Anahtarı test et** → **Kaydet** → yazmaya başla.

## Özellikler

- Model seçimi, effort, max output token ve temperature ayarı
- Cevaplar akışla gelir, istersen **Durdur** ile kesebilirsin
- Resim, PDF ve kod/metin dosyası eklenebilir (📎, sürükle-bırak veya yapıştır)
- Mesaj, sohbet ve toplam bazında tahmini maliyet sayacı (fiyatlar Ayarlar'dan güncellenebilir)
- Prompt caching otomatik açık, uzun sohbetlerde geçmiş daha ucuza gider
- Sohbet geçmişi, sohbete özel system prompt, mesaj düzenleme, yeniden üretme, `.md` olarak dışa aktarma

## Güvenlik ⚠️

- API anahtarı **sadece senin tarayıcının localStorage'ında** tutulur ve yalnızca `api.anthropic.com`'a gönderilir. Repoya veya başka bir yere gitmez.
- **Anahtarını asla repoya commit'leme, gruba yapıştırma veya dosyaya yazma.** Herkes kendi anahtarını kullansın. Anahtarı paylaşmak, faturayı da paylaşmak demek.
- Ortak bilgisayar kullanıyorsan işin bitince Ayarlar'dan anahtarı sil.

## Maliyet ipuçları

| Model | Giriş $/1M token | Çıkış $/1M token |
|---|---|---|
| Fable 5.2 | 8 | 40 |
| Opus 5.5 | 4 | 20 |
| Sonnet 5.5 | 2 | 10 |
| Haiku 5.5 | 0.10 | 0.50 |

*(Kasım 2026 liste fiyatları)*

- Günlük işler için Sonnet ya da Opus kullan, Fable'ı zor problemlere sakla.
- Effort'u `low` veya `medium` yapmak çıkış tokenını, dolayısıyla maliyeti en çok düşüren ayar.
- Konu değişince yeni sohbet aç. Her mesajda tüm geçmiş tekrar gönderilir.
- Gerçek harcama için: Console → **Spending & usage → Billing**.

## Sorun giderme

- **401 / authentication_error:** Anahtar yanlış ya da silinmiş.
- **Effort seçince hata:** Effort'u "varsayılan"da bırak.
- **Model bulunamadı:** "Anahtarı test et" ile hangi modellere erişimin olduğuna bak.
- Markdown ve kod renklendirme CDN'den yüklenir. İnternet yoksa cevaplar düz metin olarak görünür.
