import cv2
import os
import psutil
import sensors
import csv
import time
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# Konfigurasi
os.environ["QT_QPA_PLATFORM"] = "xcb"
sbc_name = "khadasedge2"
precision = "fp16"
video_src = "/dev/video60"
output_dir = "01_benchmark_results"
os.makedirs(output_dir, exist_ok=True)
sensors.init()

# Daftar model dan resolusi input
models = [
    ("yolo11n-obb-320_ncnn_model", 320, "ncnn"),
    ("yolo11s-obb-320_ncnn_model", 320, "ncnn"),
    ("yolo11m-obb-320_ncnn_model", 320, "ncnn"),
    ("yolo11n-obb-640_ncnn_model", 640, "ncnn"),
    ("yolo11s-obb-640_ncnn_model", 640, "ncnn"),
    ("yolo11m-obb-640_ncnn_model", 640, "ncnn"),
    ("yolo11n-obb-1280_ncnn_model", 1280, "ncnn"),
    ("yolo11s-obb-1280_ncnn_model", 1280, "ncnn"),
    ("yolo11m-obb-1280_ncnn_model", 1280, "ncnn"),
    ("yolo11n-obb-320_rknn_model", 320, "rknn"),
    ("yolo11s-obb-320_rknn_model", 320, "rknn"),
    ("yolo11m-obb-320_rknn_model", 320, "rknn"),
    ("yolo11n-obb-640_rknn_model", 640, "rknn"),
    ("yolo11s-obb-640_rknn_model", 640, "rknn"),
    ("yolo11m-obb-640_rknn_model", 640, "rknn"),
    ("yolo11n-obb-1280_rknn_model", 1280, "rknn"),
    ("yolo11s-obb-1280_rknn_model", 1280, "rknn"),
    ("yolo11m-obb-1280_rknn_model", 1280, "rknn"),
]

# Fungsi monitoring
get_cpu_load = lambda: psutil.cpu_percent(interval=None)
get_ram_usage = lambda: psutil.virtual_memory().percent
def get_gpu_load():
    try:
        with open("/sys/class/devfreq/fb000000.gpu/load", "r") as f:
            return int(f.read().strip().split("@")[0])
    except:
        return -1

def get_temperatures():
    temps = {}
    for chip in sensors.iter_detected_chips():
        for feature in chip:
            if "temp" in feature.label.lower():
                temps[f"{chip}-{feature.label}"] = feature.get_value()
    return temps

def get_temp_by_hint(temps, hint):
    for label, value in temps.items():
        if hint in label:
            return value
    return -1

# File log CSV
logfile = os.path.join(output_dir, f"benchmark_realtime_{sbc_name}_{precision}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
with open(logfile, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "format", "imgsz", "fps", "cpu_usage", "gpu_usage", "ram_usage",
                     "cpu_temp", "gpu_temp", "npu_temp", "soc_temp"])

# Looping model
for model_dir, imgsz, fmt in models:
    model_id = f"{model_dir.replace('-', '_')}_{sbc_name}_{fmt}_{precision}"
    input(f"\n⚠️ Siapkan pengukur daya untuk {model_id}. Tekan ENTER untuk lanjut...")

    # Countdown sebelum mulai
    info_frame = 255 * np.ones((240, 640, 3), dtype=np.uint8)
    for sec in range(10, 0, -1):
        frame_copy = info_frame.copy()
        cv2.putText(frame_copy, f"Model: {model_id}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(frame_copy, f"Mulai dalam {sec} detik...", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow("Monitoring", frame_copy)
        cv2.waitKey(1000)

    print(f"🚀 Mulai inferensi: {model_id}")
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print("❌ Kamera gagal dibuka")
        continue

    model = YOLO(model_dir, task="obb")

    # Inisialisasi
    total_fps = total_cpu = total_gpu = total_ram = 0
    total_cpu_temp = total_gpu_temp = total_npu_temp = total_soc_temp = 0
    frame_count = 0
    t1 = time.time()

    while frame_count < 1000:
        ret, frame = cap.read()
        if not ret:
            print("❌ Gagal membaca frame")
            break

        result = model.predict(source=frame, imgsz=imgsz, verbose=False)
        annotated_frame = result[0].plot()

        t2 = time.time()
        fps = 1 / (t2 - t1)
        t1 = time.time()

        cpu = get_cpu_load()
        gpu = get_gpu_load()
        ram = get_ram_usage()
        temps = get_temperatures()

        cpu_temp = get_temp_by_hint(temps, "bigcore")
        gpu_temp = get_temp_by_hint(temps, "gpu_thermal")
        npu_temp = get_temp_by_hint(temps, "npu_thermal")
        soc_temp = get_temp_by_hint(temps, "soc_thermal")

        total_fps += fps
        total_cpu += cpu
        total_gpu += gpu
        total_ram += ram
        total_cpu_temp += cpu_temp
        total_gpu_temp += gpu_temp
        total_npu_temp += npu_temp
        total_soc_temp += soc_temp
        frame_count += 1

        # Overlay
        cv2.putText(annotated_frame, f"Model: {model_id}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(annotated_frame, f"CPU: {cpu:.1f}% {cpu_temp:.1f}C", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(annotated_frame, f"GPU: {gpu:.1f}% {gpu_temp:.1f}C", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(annotated_frame, f"NPU Temp: {npu_temp:.1f}C", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 0), 1)
        cv2.putText(annotated_frame, f"SoC Temp: {soc_temp:.1f}C", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(annotated_frame, f"RAM: {ram:.1f}%", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        cv2.imshow("Monitoring", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⏹️ Dihentikan oleh pengguna.")
            break

    if frame_count > 0:
        avg_data = [model_id, fmt.upper(), imgsz,
                    f"{total_fps / frame_count:.2f}", f"{total_cpu / frame_count:.1f}",
                    f"{total_gpu / frame_count:.1f}", f"{total_ram / frame_count:.1f}",
                    f"{total_cpu_temp / frame_count:.1f}", f"{total_gpu_temp / frame_count:.1f}",
                    f"{total_npu_temp / frame_count:.1f}", f"{total_soc_temp / frame_count:.1f}"]
        with open(logfile, "a") as f:
            writer = csv.writer(f)
            writer.writerow(avg_data)

    cap.release()

cv2.destroyAllWindows()
sensors.cleanup()
print("✅ Semua model selesai dievaluasi dan data disimpan.")

