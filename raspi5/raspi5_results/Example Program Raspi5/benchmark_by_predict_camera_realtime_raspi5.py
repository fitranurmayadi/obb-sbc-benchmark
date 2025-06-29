import cv2
import os
import psutil
import csv
import time
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# Konfigurasi umum
os.environ["QT_QPA_PLATFORM"] = "xcb"
sbc_name = "raspberrypi5"
precision = "fp16"
video_src = "/dev/video0"
output_dir = "01_benchmark_results"
os.makedirs(output_dir, exist_ok=True)

# Daftar model dan resolusi
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
]
# Utility functions
def get_cpu_load():
    """Mengembalikan persentase penggunaan CPU saat ini"""
    return psutil.cpu_percent(interval=None)

def get_ram_usage():
    """Mengembalikan persentase penggunaan RAM saat ini"""
    return psutil.virtual_memory().percent

def read_cpu_temp():
    """Membaca suhu CPU dari thermal_zone0"""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read().strip()) / 1000.0
    except Exception as e:
        print(f"[WARN] Gagal membaca suhu CPU: {e}")
        return -1.0

def read_soc_temp():
    try:
        out = subprocess.check_output(["sensors"]).decode()
        for line in out.splitlines():
            if "rp1_adc" in line.lower():
                continue
            if "temp1" in line.lower():
                temp = line.split()[-2].replace("+", "").replace("°C", "")
                return float(temp)
    except:
        pass
    return -1.0

# Prepare CSV log
logfile = os.path.join(
    output_dir,
    f"benchmark_realtime_{sbc_name}_{precision}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)
with open(logfile, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "model","format","imgsz","fps","cpu_usage","gpu_usage","ram_usage",
        "cpu_temp","gpu_temp","npu_temp","soc_temp"
    ])

# Loop tiap model
for model_path, imgsz, fmt in models:
    model_id = f"{os.path.splitext(model_path)[0]}_{sbc_name}_{fmt}_{precision}"
    input(f"\n⚠️ Siapkan pengukur daya untuk {model_id}. Tekan ENTER untuk mulai...")

    # Countdown
    info = np.full((240,640,3),255,dtype=np.uint8)
    for sec in range(5,0,-1):
        tmp = info.copy()
        cv2.putText(tmp, f"Model: {model_id}", (30,80),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
        cv2.putText(tmp, f"Mulai dalam {sec}s", (30,150),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        cv2.imshow("Start", tmp); cv2.waitKey(1000)
    cv2.destroyWindow("Start")

    print(f"🚀 Mulai inferensi: {model_id}")
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print("❌ Kamera gagal dibuka. Lewati model ini.")
        continue

    model = YOLO(model_path, task="obb")
    # Inisialisasi akumulasi
    sum_fps = sum_cpu = sum_ram = 0.0
    sum_cpu_t = sum_soc_t = 0.0
    frames = 0
    t_prev = time.time()

    while frames < 1000:
        ret, frame = cap.read()
        if not ret: break

        # Predict & overlay (tanpa plot)
        result = model.predict(source=frame, imgsz=imgsz, verbose=False,
                               device="cpu", half=True)
        annotated = result[0].plot()

        # Hitung FPS
        t_cur = time.time()
        fps = 1.0 / (t_cur - t_prev) if (t_cur - t_prev)>0 else 0
        t_prev = t_cur

        # Ambil metrik system
        cpu = get_cpu_load()
        ram = get_ram_usage()
        cpu_t = read_cpu_temp()
        soc_t = read_soc_temp()
        # GPU & NPU tidak tersedia di Pi5
        gpu = -1.0
        gpu_t = -1.0
        npu_t = -1.0

        # Akumulasi
        sum_fps += fps
        sum_cpu += cpu
        sum_ram += ram
        sum_cpu_t += cpu_t
        sum_soc_t += soc_t
        frames += 1

        # Overlay kecil
        cv2.putText(annotated,f"FPS:{fps:.1f}",(10,20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.imshow("Monitor", annotated)
        if cv2.waitKey(1)&0xFF==ord('q'):
            print("⏹️ Dihentikan.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if frames>0:
        avg = [
            model_id, fmt.upper(), imgsz,
            f"{sum_fps/frames:.2f}",
            f"{sum_cpu/frames:.1f}",
            f"{gpu:.1f}",               # GPU usage
            f"{sum_ram/frames:.1f}",
            f"{sum_cpu_t/frames:.1f}",
            f"{gpu_t:.1f}",             # GPU temp
            f"{npu_t:.1f}",             # NPU temp
            f"{sum_soc_t/frames:.1f}"
        ]
        with open(logfile,"a",newline="") as f:
            csv.writer(f).writerow(avg)

print("✅ Semua model selesai dan data tersimpan di", logfile)
