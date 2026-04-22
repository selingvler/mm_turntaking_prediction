python seling_scripts/merge_audio_seling.py --ch0 "raw_data/aeeee708_synced/synced_selin_coskun.webm" --ch1 "raw_data/aeeee708_synced/synced_tugrul.webm" --out "processed_data/aeeee708.wav"
python seling_scripts/vtt_to_textgrid_seling.py \
    --vtt "raw_data/aeeee708_synced/synced_Multimodal - Bau Veri Toplama 03.vtt" \
    --audio "processed_data/aeeee708.wav" \
    --out "processed_data/aeeee708.TextGrid" \
    --spk0 "SELİN COŞKUN" \
    --spk1 "Ahmet Tuğrul Bayrak"
python seling_scripts/tahmin_seling.py bau_veri_toplama_03

python seling_scripts/generate_json_seling.py bau_veri_toplama_03

python seling_scripts/validate_seling.py bau_veri_toplama_03
