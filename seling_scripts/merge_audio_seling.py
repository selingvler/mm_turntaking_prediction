import librosa
import soundfile as sf
import numpy as np
import os
import argparse

def create_stereo_wav(ch0_path, ch1_path, output_path):
    print("1. Ses dosyaları 16kHz formatında okunuyor...")
    # Sesleri okurken otomatik olarak 16kHz'e dönüştürürüz
    y0, sr0 = librosa.load(ch0_path, sr=16000)
    y1, sr1 = librosa.load(ch1_path, sr=16000)

    print("2. Ses uzunlukları eşitleniyor...")
    # Dosyalardan biri diğerinden birkaç milisaniye uzun olabilir, eşitliyoruz
    max_len = max(len(y0), len(y1))
    y0_padded = np.pad(y0, (0, max_len - len(y0)))
    y1_padded = np.pad(y1, (0, max_len - len(y1)))

    print("3. Stereo matris oluşturuluyor ve kaydediliyor...")
    # İki kanalı üst üste koyup Stereo (2 kanal) yapıyoruz
    stereo_audio = np.vstack((y0_padded, y1_padded))
    
    # Soundfile ile kaydederken matrisin transpozunu (.T) almamız gerekir
    sf.write(output_path, stereo_audio.T, 16000)
    print(f"✅ İşlem tamam! Stereo dosya oluşturuldu: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="İki mono ses dosyasını birleştirerek stereo WAV oluşturur.")
    parser.add_argument("--ch0", type=str, required=True, help="Kanal 0 (Sol) için ses dosyası yolu")
    parser.add_argument("--ch1", type=str, required=True, help="Kanal 1 (Sağ) için ses dosyası yolu")
    parser.add_argument("--out", type=str, required=True, help="Çıktı stereo WAV dosyasının yolu")

    args = parser.parse_args()
    
    create_stereo_wav(args.ch0, args.ch1, args.out)