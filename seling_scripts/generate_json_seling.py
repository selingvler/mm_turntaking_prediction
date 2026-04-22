import os
import sys
import json
import argparse # Terminal argümanları için eklendi

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Projenin kendi olay çıkarıcı fonksiyonunu içe aktarıyoruz
from dataset_management.dataset_manager.scripts.get_tt_events import get_events_from_tg

def generate_sample_json(file_id):
    # Dosya yollarımızı file_id parametresine göre dinamik oluşturuyoruz
    textgrid_file = f"processed_data/{file_id}.TextGrid"
    output_dir = "my_outputs/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Çıktı dosyasının adını TextGrid ile aynı yapıyoruz (.json uzantılı)
    output_file = os.path.join(output_dir, f"{file_id}.json")

    print(f"'{textgrid_file}' dosyası işleniyor...")

    try:
        # 1. TextGrid dosyasından konuşma olaylarını (shifts, holds vb.) çıkar
        events = get_events_from_tg(textgrid_file)

        # 2. Sonuçları JSON formatında my_outputs klasörüne kaydet
        with open(output_file, 'w') as f:
            json.dump(events, f, indent=4)
            
        print(f"✅ Harika! Cevap anahtarı (JSON) başarıyla oluşturuldu: {output_file}")
        
    except Exception as e:
        print(f"❌ Bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TextGrid dosyasından turn-taking olaylarını çıkarıp JSON oluşturur.")
    parser.add_argument("file_id", type=str, help="İşlenecek dosyanın ID'si (örneğin: bau_veri_toplama_03)")
    
    args = parser.parse_args()
    
    generate_sample_json(args.file_id)