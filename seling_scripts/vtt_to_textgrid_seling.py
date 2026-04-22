import re
import textgrid
import librosa
import os
import argparse

def time_to_sec(time_str):
    """00:04:31.561 formatındaki VTT zamanını saniyeye çevirir."""
    h, m, s = time_str.split(':')
    s, ms = s.split('.')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def convert_vtt_to_word_textgrid(vtt_path, audio_path, output_path, speaker_map):
    print("1. Ses dosyasından toplam süre alınıyor...")
    # Sadece süreyi öğrenmek için okuyoruz
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    max_time = len(y) / sr

    print(f"2. VTT dosyası okunuyor: {vtt_path}")
    with open(vtt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # VTT bloklarını yakalayan Regex
    pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\n<v\s+([^>]+)>(.*?)</v>", re.DOTALL)
    matches = pattern.findall(content)

    # TextGrid Ana Objesi
    tg = textgrid.TextGrid(minTime=0.0, maxTime=max_time)
    
    # Hoparlörlerin zaman çizelgelerini tutacağımız sözlük
    speaker_data = {0: [], 1: []}
    
    print("3. Metinler kelimelere bölünüp zaman damgaları hesaplanıyor...")
    for start_str, end_str, speaker_name, text in matches:
        speaker_name = speaker_name.strip()
        
        # Eğer VTT'deki isim haritalamamızda yoksa atla
        if speaker_name not in speaker_map:
            continue 
            
        ch_id = speaker_map[speaker_name]
        start_sec = time_to_sec(start_str)
        end_sec = time_to_sec(end_str)
        
        # Metni temizle (yeni satırları ve fazla boşlukları sil)
        clean_text = " ".join(text.replace('\n', ' ').split())
        words = clean_text.split()
        
        if not words:
            continue
            
        # Bloğun süresini kelime sayısına bölerek her kelimeye eşit süre veriyoruz
        word_duration = (end_sec - start_sec) / len(words)
        
        current_word_start = start_sec
        for word in words:
            speaker_data[ch_id].append({
                "start": current_word_start,
                "end": current_word_start + word_duration,
                "word": word
            })
            current_word_start += word_duration

    print("4. TextGrid katmanları (Tiers) oluşturuluyor...")
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    for ch_id in [0, 1]:
        # İsimlendirme modelin beklediği gibi olmalı
        tier = textgrid.IntervalTier(name=f"{base_name}_ch{ch_id}", minTime=0.0, maxTime=max_time)
        
        current_time = 0.0
        # Olayları zamana göre kronolojik sırala
        events = sorted(speaker_data[ch_id], key=lambda x: x["start"])
        
        for event in events:
            start = event["start"]
            end = event["end"]
            word = event["word"]
            
            # Üst üste binmeleri (overlap) önlemek için basit düzeltme
            if start < current_time:
                start = current_time 
            if end <= start:
                continue
                
            # ÖNEMLİ: İki kelime arasında veya cümlenin başında boşluk varsa ekle
            if start > current_time:
                tier.add(current_time, start, "")
                
            # Kelimeyi ekle
            tier.add(start, end, word)
            current_time = end
            
        # Son kelimeden sesin sonuna kadar olan süreyi sessizlik olarak doldur
        if current_time < max_time:
            tier.add(current_time, max_time, "")
            
        tg.tiers.append(tier)

    # Dosyayı Kaydet
    tg.write(output_path)
    print(f"✅ İşlem tamam! Kusursuz boşlukları olan VTT tabanlı TextGrid oluşturuldu: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VTT dosyasını kelime düzeyinde TextGrid formatına çevirir.")
    parser.add_argument("--vtt", type=str, required=True, help="Girdi VTT dosyasının yolu")
    parser.add_argument("--audio", type=str, required=True, help="Referans WAV dosyasının yolu (süre hesaplaması için)")
    parser.add_argument("--out", type=str, required=True, help="Çıktı TextGrid dosyasının yolu")
    parser.add_argument("--spk0", type=str, required=True, help="Kanal 0 (ch0) için VTT'deki kişi adı (örn: 'SELİN COŞKUN')")
    parser.add_argument("--spk1", type=str, required=True, help="Kanal 1 (ch1) için VTT'deki kişi adı (örn: 'Ahmet Tuğrul Bayrak')")

    args = parser.parse_args()

    # Terminalden gelen isimleri doğrudan 0 ve 1 kanallarına mapliyoruz
    speaker_mapping = {
        args.spk0: 0,
        args.spk1: 1
    }
    
    convert_vtt_to_word_textgrid(args.vtt, args.audio, args.out, speaker_mapping)