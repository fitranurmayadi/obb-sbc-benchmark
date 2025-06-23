import os
import time
import csv
from ultralytics import YOLO

# ======================
# Konfigurasi
# ======================
versions = ['n', 's', 'm']
resolutions = [320, 640, 1280]
formats = ['pt']
devices = ['0', 'cpu']  # jalankan GPU dulu, baru CPU
base_model_path = "../trained_models"
data_path = "../datasets/obb_mini_things-3/data.yaml"
csv_output_file = "val_results_yolo11obb.csv"

# ======================
# Mulai Validasi
# ======================
results = []
start_time = time.time()

for device in devices:  # loop device di luar
    for fmt in formats:
        for v in versions:
            for res in resolutions:
                model_file = f"yolo11{v}-obb-{res}.{fmt}"
                model_path = os.path.join(base_model_path, model_file)

                if not os.path.exists(model_path):
                    print(f"[SKIP] Model tidak ditemukan: {model_path}")
                    continue

                print(f"\nValidating {model_file} @ {res} on device: {device}")
                try:
                    model = YOLO(model_path)
                    val_result = model.val(
                        data=data_path,
                        imgsz=res,
                        device=device,
                        verbose=False
                    )

                    results.append({
                        "model": model_file,
                        "format": fmt,
                        "imgsz": res,
                        "device": device,
                        "mAP50": val_result.box.map50(),
                        "mAP50-95": val_result.box.map(),
                        "precision": val_result.box.mp(),
                        "recall": val_result.box.mr(),
                        "inference_time_ms": val_result.speed['inference'],
                        "fps": 1000 / val_result.speed['inference'] if val_result.speed['inference'] else 0
                    })

                except Exception as e:
                    print(f"[ERROR] Gagal validasi {model_file} di {device}: {e}")

total_time = time.time() - start_time
print(f"\n[INFO] Total waktu validasi: {total_time:.2f} detik")

# ======================
# Simpan ke CSV
# ======================
if results:
    with open(csv_output_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] Hasil validasi disimpan ke {csv_output_file}")
else:
    print("[WARNING] Tidak ada hasil untuk disimpan.")
