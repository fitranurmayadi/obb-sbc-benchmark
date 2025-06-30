#!/usr/bin/env python3
import os
import cv2
import csv
import time
import psutil
import glob
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# ─────── CONFIG ───────
os.environ["QT_QPA_PLATFORM"] = "xcb"
sbc_name   = "jetsonorin"
precision  = "fp16"
video_src  = "/dev/video0"
output_dir = "01_benchmark_results"
os.makedirs(output_dir, exist_ok=True)

# Daftar model (nama file, resolusi, tag format)
models = [
    ("yolo11n-obb-320.engine",  320, "engine"),
    ("yolo11s-obb-320.engine",  320, "engine"),
    ("yolo11m-obb-320.engine",  320, "engine"),
    ("yolo11n-obb-640.engine",  640, "engine"),
    ("yolo11s-obb-640.engine",  640, "engine"),
    ("yolo11m-obb-640.engine",  640, "engine"),
    ("yolo11n-obb-1280.engine",1280, "engine"),
    ("yolo11s-obb-1280.engine",1280, "engine"),
    ("yolo11m-obb-1280.engine",1280, "engine"),
    #("yolo11n-obb-320_ncnn_model", 320, "ncnn"),
    #("yolo11s-obb-320_ncnn_model", 320, "ncnn"),
    #("yolo11m-obb-320_ncnn_model", 320, "ncnn"),
    #("yolo11n-obb-640_ncnn_model", 640, "ncnn"),
    #("yolo11s-obb-640_ncnn_model", 640, "ncnn"),
    #("yolo11m-obb-640_ncnn_model", 640, "ncnn"),
    #("yolo11n-obb-1280_ncnn_model",1280, "ncnn"),
    #("yolo11s-obb-1280_ncnn_model",1280, "ncnn"),
    #("yolo11m-obb-1280_ncnn_model",1280, "ncnn"),
]

# ─────── Helpers ───────
def get_cpu_load():
    return psutil.cpu_percent(interval=None)

def get_ram_usage():
    return psutil.virtual_memory().percent

def get_gpu_load():
    """
    Baca GPU load Jetson (dalam /sys/devices/gpu.0/load, satuan %*10).
    """
    try:
        with open("/sys/devices/gpu.0/load", "r") as f:
            return int(f.read().strip()) / 10.0
    except:
        return -1

def read_temp_zones():
    """
    Baca semua thermal_zone*/temp & type, kembalikan dict {label:celcius}.
    """
    temps = {}
    for temp_path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            # nilai aslinya dalam millidegree
            raw = int(open(temp_path).read().strip())
            c = raw / 1000.0
            # label
            type_path = temp_path.replace("temp", "type")
            label = open(type_path).read().strip()
            temps[label] = c
        except:
            continue
    return temps

def extract_temps(temps: dict):
    """
    Pilih label untuk CPU, GPU, SoC. Kalau tidak ada, kembalikan -1.
    """
    cpu_temp = temps.get("CPU-therm", temps.get("tj-therm", -1))
    gpu_temp = temps.get("GPU-therm", -1)
    # ambil max SoC jika ada multi-zone
    soc_vals = [v for k,v in temps.items() if k.startswith("SOC")]
    soc_temp = max(soc_vals) if soc_vals else -1
    # Jetson Orin Nano tidak expose NPU/DLA sensor standar
    npu_temp = -1
    return cpu_temp, gpu_temp, npu_temp, soc_temp

# ─────── Setup CSV ───────
logfile = os.path.join(
    output_dir,
    f"benchmark_realtime_{sbc_name}_{precision}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)
with open(logfile, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "model","format","imgsz","fps",
        "cpu_usage","gpu_usage","ram_usage",
        "cpu_temp","gpu_temp","npu_temp","soc_temp"
    ])

# ─────── Main Loop ───────
for model_file, imgsz, fmt in models:
    model_id = f"{os.path.splitext(model_file)[0]}_{sbc_name}_{fmt}_{precision}"
    input(f"\n⚠️ Siapkan pengukur daya untuk `{model_id}` dan tekan ENTER…")

    # Countdown 3 detik
    info = 255 * np.ones((240,640,3),dtype=np.uint8)
    for s in range(3,0,-1):
        img = info.copy()
        cv2.putText(img, f"Model: {model_id}", (30,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0),2)
        cv2.putText(img, f"Mulai dalam {s} detik…", (30,150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255),2)
        cv2.imshow("Monitoring", img)
        cv2.waitKey(1000)

    # Load camera & model
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f"❌ Kamera `{video_src}` gagal dibuka, skip `{model_id}`")
        continue
    model = YOLO(model_file, task="obb")
    device_id = 0 if fmt == "engine" else "cpu"
    print(f"🧠 Model `{model_file}` dijalankan di device `{device_id}`")
    # Inisialisasi akumulasi
    total_fps = total_cpu = total_gpu = total_ram = 0.0
    total_cpu_temp = total_gpu_temp = total_npu_temp = total_soc_temp = 0.0
    frame_count = 0
    t_prev = time.time()

    print(f"🚀 Mulai inferensi `{model_id}` …")
    while frame_count < 1000:
        ret, frame = cap.read()
        if not ret:
            print("❌ Gagal baca frame, break loop.")
            break

        # infer & plot (non-blocking)
        res = model.predict(source=frame, imgsz=imgsz, device=device_id, verbose=False)
        annotated = res[0].plot()

        # hitung FPS
        t_cur = time.time()
        fps = 1.0 / (t_cur - t_prev) if (t_cur - t_prev)>0 else 0
        t_prev = t_cur

        # baca utilisasi & suhu
        cpu = get_cpu_load()
        gpu = get_gpu_load()
        ram = get_ram_usage()
        temps = read_temp_zones()
        cpu_t, gpu_t, npu_t, soc_t = extract_temps(temps)

        # akumulasi
        total_fps      += fps
        total_cpu      += cpu
        total_gpu      += gpu
        total_ram      += ram
        total_cpu_temp += cpu_t
        total_gpu_temp += gpu_t
        total_npu_temp += npu_t
        total_soc_temp += soc_t
        frame_count   += 1

        # overlay teks
        cv2.putText(annotated, f"FPS: {fps:.1f}",   (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
        cv2.putText(annotated, f"CPU:{cpu:.1f}% {cpu_t:.1f}C", (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0),1)
        cv2.putText(annotated, f"GPU:{gpu:.1f}% {gpu_t:.1f}C", (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255),1)
        cv2.putText(annotated, f"RAM:{ram:.1f}%",             (10,85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255),1)
        cv2.imshow("Monitoring", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⏹️ Dihentikan user.")
            break

    # tulis rata-rata ke CSV
    if frame_count>0:
        with open(logfile, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                model_id, fmt.upper(), imgsz,
                f"{(total_fps/frame_count):.1f}",
                f"{(total_cpu/frame_count):.1f}",
                f"{(total_gpu/frame_count):.1f}",
                f"{(total_ram/frame_count):.1f}",
                f"{(total_cpu_temp/frame_count):.1f}",
                f"{(total_gpu_temp/frame_count):.1f}",
                f"{(total_npu_temp/frame_count):.1f}",
                f"{(total_soc_temp/frame_count):.1f}"
            ])

    cap.release()

cv2.destroyAllWindows()
print("✅ Semua model selesai, hasil disimpan di:", logfile)

