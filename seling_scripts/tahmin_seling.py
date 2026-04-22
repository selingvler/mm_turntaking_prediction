import os
import sys
import torch
import yaml
import argparse # <-- Terminalden argüman almak için eklendi

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- CUDA YAMASI ---
if not torch.cuda.is_available():
    torch.cuda.set_device = lambda device: None
    torch.cuda.is_available = lambda: False
# -------------------

from turn_taking.analysis.validation.validation import run_all
from turn_taking.model.model import StereoTransformerModel

# Config dosyasının içindeki gizli "cuda" ayarlarını Mac'e uyarlayan yardımcı fonksiyon
def fix_config_device(d, target_device):
    if isinstance(d, dict):
        for k, v in d.items():
            if v == "cuda" or v == "cuda:0":
                d[k] = str(target_device)
            elif isinstance(v, (dict, list)):
                fix_config_device(v, target_device)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            if v == "cuda" or v == "cuda:0":
                d[i] = str(target_device)
            elif isinstance(v, (dict, list)):
                fix_config_device(v, target_device)

def test_model_on_sample(file_id): # <-- Fonksiyona parametre eklendi
    # 1. Dosya Yolları
    sample_data_dir = "processed_data/"
    output_predictions_dir = "my_outputs/"
    os.makedirs(output_predictions_dir, exist_ok=True)

    # DİKKAT: Projedeki eğitilmiş modelin yolunu buraya yazmalısın
    model_weights_path = "acl_sample_data_models/sample_trained_models/VAP_candor/20240822_105427_fold_0_epoch_10" 

    # 2. Mac M1 (MPS) Kontrolü
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Mac M1 (MPS) hızlandırıcısı kullanılıyor.")
    else:
        device = torch.device("cpu")
        print("Uyarı: MPS bulunamadı, CPU kullanılacak.")

    # 3. Modeli Yükleme
    print("Model mimarisi kuruluyor...")
    cfg_path = "acl_sample_data_models/sample_trained_models/VAP_candor/20240822_105427_params.yaml"
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config dosyası bulunamadı: {cfg_path}")
        
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Config içindeki olası 'cuda' parametrelerini temizle
    fix_config_device(cfg, device)
        
    model = StereoTransformerModel(cfg=cfg)
    
    print("Eğitilmiş ağırlıklar yükleniyor...")
    checkpoint = torch.load(model_weights_path, map_location=device) 
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()

    # 4. Tahminleri Üretme
    print(f"Ses dosyası ({file_id}) işleniyor ve tahminler üretiliyor...")
    with torch.no_grad():
        run_all(
            model=model,
            ids=[file_id], # <-- Dinamik parametre kullanıldı
            wav_dir=sample_data_dir, 
            transcript_dir=sample_data_dir,
            output_dir=output_predictions_dir,
            mode='VAP' 
        )
        
    print(f"✅ İşlem başarıyla tamamlandı! Sonuçlar '{output_predictions_dir}' klasörüne kaydedildi.")

if __name__ == "__main__":
    # <-- Dosya doğrudan çalıştırıldığında terminal argümanlarını okuyan bölüm
    parser = argparse.ArgumentParser(description="MM-VAP Tahmin Betiği")
    parser.add_argument("file_id", type=str, help="İşlenecek ses dosyasının ID'si (örneğin: bau_veri_toplama_03)")
    args = parser.parse_args()

    # Alınan ID'yi fonksiyona gönderiyoruz
    test_model_on_sample(args.file_id)