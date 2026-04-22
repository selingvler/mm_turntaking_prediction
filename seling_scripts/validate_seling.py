import os
import sys
import torch
import copy
import argparse # <-- Terminal argümanları için eklendi
from sklearn import metrics

if not torch.cuda.is_available():
    torch.cuda.set_device = lambda device: None
    torch.cuda.is_available = lambda: False

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Orijinal dosyandaki tüm skorlama fonksiyonlarını içe aktarıyoruz
from turn_taking.analysis.validation.validation import (
    get_preds, 
    optimal_thresholds, 
    apply_thresholds, 
    apply_score, 
    f1_score_func,
    f1_score_func_detailed
)

def evaluate_final_detailed(file_id): # <-- Fonksiyona parametre eklendi
    output_dir = "my_outputs/"

    print(f"Tahminler (.pkl) ve Cevap Anahtarı (.json) eşleştiriliyor... (Dosya: {file_id})")

    try:
        # 1. Tahminleri ve gerçek etiketleri al
        y_true, y_score = get_preds([file_id], output_dir, output_dir)

        if not y_true:
            print("Eşleşme sağlanamadı. Dosyaların my_outputs klasöründe olduğundan emin ol.")
            return

        print("Olasılıklar için en iyi eşik değerleri (threshold) hesaplanıyor...")
        
        # ---------------------------------------------------------
        # AŞAMA 1: F1 Skoruna Göre Optimizasyon ve Hesaplama
        # ---------------------------------------------------------
        y_score_f1 = copy.deepcopy(y_score)
        thresholds_f1 = optimal_thresholds(y_true, y_score_f1, func_to_optimise=f1_score_func)
        y_pred_f1 = apply_thresholds(y_score_f1, thresholds_f1)
        f1_results = apply_score(y_true, y_pred_f1, f1_score_func_detailed)

        # ---------------------------------------------------------
        # AŞAMA 2: Dengeli Doğruluğa (Balanced Acc) Göre Optimizasyon
        # ---------------------------------------------------------
        y_score_bal = copy.deepcopy(y_score)
        thresholds_bal = optimal_thresholds(y_true, y_score_bal, func_to_optimise=metrics.balanced_accuracy_score)
        y_pred_bal = apply_thresholds(y_score_bal, thresholds_bal)
        bal_acc_results = apply_score(y_true, y_pred_bal, metrics.balanced_accuracy_score)


        print("\n" + "="*60)
        print(f"🎉 MODEL BAŞARI SONUÇLARI ({file_id}) 🎉")
        print("="*60)

        # Orijinal dosyada üretilen TÜM ana sınıflar ve alt sınıflar üzerinde döner
        for event_class in y_true.keys():
            print(f"\n[{event_class.upper()}] Kategorisi:")
            
            for sub_event in y_true[event_class].keys():
                # Eğer bu alt olay için veri varsa yazdır
                if len(y_true[event_class][sub_event]) > 0:
                    
                    # f1_results detaylı fonksiyonu 3 değer döndürür: (weighted, hold_f1, shift_f1)
                    f1_vals = f1_results[event_class][sub_event][0]
                    # bal_acc_results tek bir değer döndürür
                    bal_acc = bal_acc_results[event_class][sub_event][0]

                    print(f"  -> {sub_event}:")
                    print(f"     * Ağırlıklı F1 (Weighted) : {f1_vals[0]:.4f}")
                    print(f"     * Sırada Kalma F1 (Hold)  : {f1_vals[1]:.4f}")
                    print(f"     * Sıra Geçişi F1 (Shift)  : {f1_vals[2]:.4f}")
                    print(f"     * Dengeli Doğruluk (Bal.) : {bal_acc:.4f}")

    except Exception as e:
        print(f"❌ Bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    # <-- Dosya doğrudan çalıştırıldığında terminal argümanlarını okuyan bölüm
    parser = argparse.ArgumentParser(description="Tahminleri ve cevap anahtarını karşılaştırıp detaylı F1 skorlarını hesaplar.")
    parser.add_argument("file_id", type=str, help="Değerlendirilecek dosyanın ID'si (örneğin: bau_veri_toplama_03)")
    
    args = parser.parse_args()
    
    evaluate_final_detailed(args.file_id)