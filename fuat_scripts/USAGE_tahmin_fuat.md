**tahmin_fuat.py Kullanim Ozeti**

python fuat_scripts/tahmin_fuat.py \
  --mode live \
  --input_source mic \
  --start_time 0 \
  --end_time 60 \
  --window_size 1.0 \
  --step_size 0.02

Kısa açıklama
- `tahmin_fuat.py`: VAP/VAD modelinden cikarim yapmak icin kullanilan script. Hem toplu (`batch`) hem de canli (`live`) modlari destekler. Mikrofon ile canli calisma icin `sounddevice` gereklidir.

Parametre açıklamaları (önemli olanlar)
- `--mode`: `batch` veya `live`.
- `--input_source`: `wav` veya `mic`.
- `--audio_path`: Girdi WAV dosyası (stereo bekler). `batch` mod için gereklidir.
- `--start_time`, `--end_time`: `live` modda kullanılacak zaman aralığı (saniye).
- `--window_size`, `--step_size`: kaydırmalı pencere boyutu ve hop (saniye). 0.02 = 20 ms adım.
- `--print_only_events`: True ise sadece `shift` veya `backchannel` kararları yazdırılır.
- `--model_weights`, `--config_path`: model ağırlıkları ve yapılandırma yolları (varsayılan script içi yolları kullanılır).

Notlar ve ipuçları
- Script modelin girişinin stereo olduğunu varsayar. Mono mikrofon kullandığınızda script otomatik olarak ikinci kanalı sıfırlar (mikrofon → sol kanal, sağ kanal sıfır).
- `batch` çıktısı `processed_data/<basename>.pkl` olarak kaydedilir ve validator ile uyumludur.
- M1 mac için MPS kullanılmaya çalışılır; yoksa CPU ile çalışır.
- Mikrofon izinleri ve `sounddevice` bağımlılığı macOS tarafında kullanıcı ayarlarına bağlıdır.

Scriptler
- Tum modlar tek dosyada: `fuat_scripts/run_fuat.sh`.
- Geriye donuk uyumluluk icin `run_batch.sh`, `run_live_wav.sh`, `run_live_mic.sh`, `run_zero_stereo.sh` dosyalari `run_fuat.sh` scriptine yonlendirilir.

Sorun yaşarsanız kısa notlar
- Model veya config bulunamıyorsa `--model_weights` ve `--config_path` parametrelerini tam yolla verin.
- `end_time` ≤ `start_time` olursa hata verir; aralığı kontrol edin.

---
Ozet: Bu dosya temel kullanim orneklerini ve dikkat edilmesi gereken parametreleri icerir. Daha detayli ihtiyac varsa, ornekleri dogrudan `fuat_scripts/tahmin_fuat.py` icindeki argumanlari kullanarak genisletebilirim.
