import os
import cv2
import time
from ultralytics import YOLO

# === Konfigurasi ===
MODEL_PATH = "C:/Users/usudl/Fitra/OBB-YOLO/datasets/runs/obb/train7/weights/best.pt"   # Ganti dengan path model Anda
CONFIDENCE_THRESHOLD = 0.5           # Tampilkan deteksi dengan confidence > 50%
SOURCE = 0                           # 0 untuk kamera USB; bisa juga path video/gambar

# === Load Model ===
model = YOLO(MODEL_PATH, task="obb")

# === Buka kamera atau file ===
cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    print(f"❌ Tidak bisa membuka sumber: {SOURCE}")
    exit(1)

# === Loop inferensi ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membaca frame")
        break

    start = time.perf_counter()
    results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, stream=False)
    end = time.perf_counter()

    # Hitung FPS secara aman
    duration = end - start
    fps = 1 / duration if duration > 0 else 0

    # Ambil hasil prediksi dari satu frame
    annotated_frame = results[0].plot()

    # Tampilkan kecepatan inferensi
    fps = 1 / (end - start)
    cv2.putText(annotated_frame, f"Inference: {fps:.2f} FPS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Tampilkan hasil
    cv2.imshow("YOLOv11-OBB Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
