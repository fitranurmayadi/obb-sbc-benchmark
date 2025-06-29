import cv2
from ultralytics import YOLO
import time
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Inisialisasi model YOLO (RKNN)
model = YOLO("obb-yolov11n-320_rknn_model", task="obb")  # Ganti dengan path model Anda jika berbeda

# Buka kamera USB (biasanya device 0 atau 1)
cap = cv2.VideoCapture("/dev/video60")  # Ganti angka jika USB cam Anda bukan di /dev/video0
if not cap.isOpened():
    print("❌ Gagal membuka kamera.")
    exit()

print("✅ Kamera terbuka. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membaca frame dari kamera.")
        break

    # Inference
    t1 = time.time()
    results = model.predict(task='obb', conf=0.5, source=frame, imgsz=320)  # img langsung dari frame
    t2 = time.time()
    
    # Ambil hasil
    annotated_frame = results[0].plot()
    fps = 1 / (t2 - t1)
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan
    cv2.imshow("RKNN-YOLOV11N-OBB-320", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()
