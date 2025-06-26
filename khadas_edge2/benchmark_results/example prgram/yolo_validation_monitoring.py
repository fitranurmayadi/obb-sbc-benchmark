import os
import cv2
import time
import psutil
import sensors
import csv
import datetime
from ultralytics import YOLO

# === Inisialisasi Global ===
models = [
    ("yolo11n-obb-320_rknn_model", 320),
    ("yolo11s-obb-320_rknn_model", 320),
    ("yolo11m-obb-320_rknn_model", 320),
    ("yolo11n-obb-640_rknn_model", 640),
    ("yolo11s-obb-640_rknn_model", 640),
    ("yolo11m-obb-640_rknn_model", 640),
    ("yolo11n-obb-1280_rknn_model", 1280),
    ("yolo11s-obb-1280_rknn_model", 1280),
    ("yolo11m-obb-1280_rknn_model", 1280)
]
data_yaml = "../datasets/obb_mini_things-3/data.yaml"

def get_temperatures():
    temps = {}
    for chip in sensors.iter_detected_chips():
        chip_name = str(chip)
        for feature in chip:
            if "temp" in feature.label.lower():
                label = f"{chip_name}-{feature.label}"
                temps[label] = feature.get_value()
    return temps

def get_temp_by_hint(temps, hint):
    for label, value in temps.items():
        if hint in label:
            return value
    return -1

def get_cpu_load():
    return psutil.cpu_percent(interval=None)

def get_ram_usage():
    return psutil.virtual_memory().percent

def get_gpu_load():
    try:
        with open("/sys/class/devfreq/fb000000.gpu/load", "r") as f:
            return int(f.read().strip().split("@")[0])
    except:
        return -1

def run_monitoring_for_model(model_path, imgsz):
    model = YOLO(model_path, task="obb")
    cap = cv2.VideoCapture("/dev/video60")
    if not cap.isOpened():
        print(f"❌ Gagal membuka kamera untuk {model_path}")
        return {k: -1 for k in ["fps", "cpu_usage", "gpu_usage", "ram_usage", "cpu_temp", "gpu_temp", "npu_temp", "soc_temp"]}

    total_cpu, total_gpu, total_ram, total_fps = 0, 0, 0, 0
    total_cpu_temp, total_gpu_temp, total_npu_temp, total_soc_temp = 0, 0, 0, 0
    frame_count = 0

    t1 = time.time()
    while frame_count < 1000:
        ret, frame = cap.read()
        if not ret:
            break

        _ = model.predict(task='obb', source=frame, imgsz=imgsz, verbose=False)

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

        total_cpu += cpu
        total_gpu += gpu
        total_ram += ram
        total_fps += fps
        total_cpu_temp += cpu_temp
        total_gpu_temp += gpu_temp
        total_npu_temp += npu_temp
        total_soc_temp += soc_temp

        frame_count += 1

    cap.release()
    return {
        "fps": total_fps / frame_count,
        "cpu_usage": total_cpu / frame_count,
        "gpu_usage": total_gpu / frame_count,
        "ram_usage": total_ram / frame_count,
        "cpu_temp": total_cpu_temp / frame_count,
        "gpu_temp": total_gpu_temp / frame_count,
        "npu_temp": total_npu_temp / frame_count,
        "soc_temp": total_soc_temp / frame_count
    }

# === Main Execution ===
sensors.init()
os.makedirs("benchmark_results", exist_ok=True)
csv_path = f"benchmark_results/final_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "model", "imgsz",
        "precision", "recall", "mAP50", "mAP50_95",
        "preprocess_ms", "inference_ms", "postprocess_ms",
        "fps", "cpu_usage", "gpu_usage", "ram_usage",
        "cpu_temp", "gpu_temp", "npu_temp", "soc_temp"
    ])

    for model_path, imgsz in models:
        print(f"🚀 Validasi + Monitoring: {model_path} @ {imgsz}x{imgsz}")
        model = YOLO(model_path, task="obb")
        metrics = model.val(
            data=data_yaml,
            task='obb',
            imgsz=imgsz,
            batch=16,
            conf=0.25,
            iou=0.6,
            plots=False,
            verbose=False,
        )

        precision = metrics.box.mp
        recall = metrics.box.mr
        map50 = metrics.box.map50
        map50_95 = metrics.box.map

        preprocess = metrics.speed['preprocess']
        inference = metrics.speed['inference']
        postprocess = metrics.speed['postprocess']

        avg_data = run_monitoring_for_model(model_path, imgsz)

        writer.writerow([
            model_path, imgsz,
            precision, recall, map50, map50_95,
            preprocess, inference, postprocess,
            avg_data["fps"], avg_data["cpu_usage"], avg_data["gpu_usage"], avg_data["ram_usage"],
            avg_data["cpu_temp"], avg_data["gpu_temp"], avg_data["npu_temp"], avg_data["soc_temp"]
        ])

sensors.cleanup()
print(f"✅ Semua model selesai. Hasil disimpan di {csv_path}")

