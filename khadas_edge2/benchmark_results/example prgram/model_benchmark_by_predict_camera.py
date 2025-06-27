import cv2
from ultralytics import YOLO
import time
import os
import psutil
import sensors
import csv
from datetime import datetime

os.environ["QT_QPA_PLATFORM"] = "xcb"

# Daftar model dan konfigurasi
models = [
    ("yolo11n-obb-320_ncnn_model", 320),
    ("yolo11n-obb-320.mnn", 320 ),
    ("yolo11n-obb-320_rknn_model", 320 ),
    ("yolo11s-obb-320_ncnn_model", 320 ),
    ("yolo11s-obb-320_rknn_model", 320 ),
    ("yolo11s-obb-320.mnn", 320 ),
    ("yolo11n-obb-320.onnx", 320 ),
    ("yolo11n-obb-640_ncnn_model", 640 ),
    ("yolo11n-obb-320_openvino_model", 320 ),
    ("yolo11n-obb-640.mnn", 640 ),
    ("yolo11n-obb-320_float32.tflite", 320 ),
    ("yolo11n-obb-640_rknn_model", 640 ),
    ("yolo11m-obb-320_rknn_model", 320 ),
    ("yolo11n-obb-320_saved_model", 320 ),
    ("yolo11m-obb-320_ncnn_model", 320 ),
    ("yolo11s-obb-320_openvino_model", 320 ),
]


video_src = "/dev/video60"
output_dir = "benchmark_results"
os.makedirs(output_dir, exist_ok=True)
sensors.init()

# Fungsi pengambilan data sensor
def get_cpu_load(): return psutil.cpu_percent(interval=None)
def get_ram_usage(): return psutil.virtual_memory().percent

def get_gpu_load():
    try:
        with open("/sys/class/devfreq/fb000000.gpu/load", "r") as f:
            return int(f.read().strip().split("@")[0])
    except: return -1

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

logfile = os.path.join(output_dir, f"predict_multi_model_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

with open(logfile, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "imgsz", "fps", "cpu_usage", "gpu_usage", "ram_usage",
                     "cpu_temp", "gpu_temp", "npu_temp", "soc_temp"])

# Iterasi per model
for model_path, imgsz in models:
    print(f"\n🚀 Mulai evaluasi: {model_path} | {imgsz}x{imgsz}")
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print("❌ Kamera gagal dibuka")
        continue

    model = YOLO(model_path, task="obb")
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
        _ = result[0].plot()

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

    # Rata-rata
    avg_data = [
        model_path, imgsz,
        f"{total_fps/frame_count:.2f}", f"{total_cpu/frame_count:.1f}", f"{total_gpu/frame_count:.1f}",
        f"{total_ram/frame_count:.1f}", f"{total_cpu_temp/frame_count:.1f}", f"{total_gpu_temp/frame_count:.1f}",
        f"{total_npu_temp/frame_count:.1f}", f"{total_soc_temp/frame_count:.1f}"
    ]

    with open(logfile, "a") as f:
        writer = csv.writer(f)
        writer.writerow(avg_data)

    cap.release()

sensors.cleanup()
print("✅ Semua model selesai dievaluasi.")

