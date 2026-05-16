import os
import pickle

# --- KENDİ 5 KAYDININ BENZERSİZ İSİMLERİNİ (ID'LERİNİ) BURAYA YAZ ---
my_ids = [
    "aeeee708", 
    "b6fe2761", 
    "ccf63032", # Kendi gerçek ID'lerinle değiştir
    "d8363f9d",
    "f08c05d1"
]

# Fold'ların kaydedileceği ana klasör
output_dir = "turkce_veriseti_klasoru/folds"
os.makedirs(output_dir, exist_ok=True)

# 5-Fold Cross Validation İçin Dosyaları Bölme
for i in range(len(my_ids)):
    fold_dir = os.path.join(output_dir, f"fold_{i}")
    os.makedirs(fold_dir, exist_ok=True)
    
    # 1 tanesi Validation (Doğrulama), diğer 4'ü Train (Eğitim)
    val_id = my_ids[i]
    train_ids = [vid for vid in my_ids if vid != val_id]
    
    # Projenin beklediği Sözlük (Dictionary) Listesi formatı
    train_data = [{'session': tid} for tid in train_ids]
    val_data = [{'session': val_id}]
    
    # .pkl olarak kaydet
    with open(os.path.join(fold_dir, "train.pkl"), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(fold_dir, "val.pkl"), 'wb') as f:
        pickle.dump(val_data, f)
        
    print(f"Fold {i} başarıyla oluşturuldu. Train: {len(train_ids)} kayıt, Val: 1 kayıt ({val_id})")