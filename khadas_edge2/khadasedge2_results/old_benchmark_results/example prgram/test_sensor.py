
import cv2
from ultralytics import YOLO
import time
import os
import psutil
import sensors
import csv

os.environ["QT_QPA_PLATFORM"] = "xcb"
model = YOLO("yolo11n-obb-320_rknn_model", task="obb")
cap = cv2.VideoCapture("/dev/video60")
if not cap.isOpened():
    print("❌ Gagal membuka kamera.")
    exit()
print("✅ Kamera terbuka.")

sensors.init()

def get_temperatures():
    temps = {}
    for chip in sensors.iter_detected_chips():
        chip_name = str(chip)
        for feature in chip:
            if "temp" in feature.label.lower():
                label = f"{chip_name}-{feature.label}"
                temps[label] = feature.get_value()
    return temps

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

def get_temp_by_hint(temps, hint):
    for label, value in temps.items():
        if hint in label:
            return value
    return -1

print("✅ Mulai")


total_cpu, total_gpu, total_ram, total_fps = 0, 0, 0, 0
total_cpu_temp, total_gpu_temp, total_npu_temp, total_soc_temp = 0, 0, 0, 0
frame_count = 0

initial_temps = get_temperatures()
initial_data = {
	"fps": "-",
    "cpu_usage": get_cpu_load(),
    "gpu_usage": get_gpu_load(),
    "ram_usage": get_ram_usage(),
    "cpu_temp": get_temp_by_hint(initial_temps, "bigcore"),
    "gpu_temp": get_temp_by_hint(initial_temps, "gpu_thermal"),
    "npu_temp": get_temp_by_hint(initial_temps, "npu_thermal"),
    "soc_temp": get_temp_by_hint(initial_temps, "soc_thermal")
}

print("✅ Data awal:", initial_data)

t1 = time.time()
while frame_count < 1000:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membaca frame.")
        break

    results = model.predict(task='obb', source=frame, imgsz=320, verbose=False)
    annotated_frame = results[0].plot()

    t2 = time.time()
    fps = 1 / (t2 - t1)
    t1 = time.time()

    cpu_usage = get_cpu_load()
    ram_usage = get_ram_usage()
    gpu_usage = get_gpu_load()
    temps = get_temperatures()
    cpu_temp = get_temp_by_hint(temps, "bigcore")
    gpu_temp = get_temp_by_hint(temps, "gpu_thermal")
    npu_temp = get_temp_by_hint(temps, "npu_thermal")
    soc_temp = get_temp_by_hint(temps, "soc_thermal")

    total_cpu += cpu_usage
    total_gpu += gpu_usage
    total_ram += ram_usage
    total_fps += fps
    total_cpu_temp += cpu_temp
    total_gpu_temp += gpu_temp
    total_npu_temp += npu_temp
    total_soc_temp += soc_temp
    frame_count += 1

    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(annotated_frame, f"CPU: {cpu_usage:.1f}% {cpu_temp:.1f}C", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
    cv2.putText(annotated_frame, f"GPU: {gpu_usage:.1f}% {gpu_temp:.1f}C", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
    cv2.putText(annotated_frame, f"NPU Temp: {npu_temp:.1f}C", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,128,0), 1)
    cv2.putText(annotated_frame, f"SoC Temp: {soc_temp:.1f}C", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    cv2.putText(annotated_frame, f"RAM: {ram_usage:.1f}%", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1)

    cv2.imshow("YOLOv11 OBB Monitoring", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

avg_data = {
    "fps": total_fps / frame_count,
    "cpu_usage": total_cpu / frame_count,
    "gpu_usage": total_gpu / frame_count,
    "ram_usage": total_ram / frame_count,
    "cpu_temp": total_cpu_temp / frame_count,
    "gpu_temp": total_gpu_temp / frame_count,
    "npu_temp": total_npu_temp / frame_count,
    "soc_temp": total_soc_temp / frame_count
}

print("✅ Rata-rata data selama 1000 frame:", avg_data)

output_dir = "benchmark_results"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "yolo_monitoring_summary.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Initial"] + list(initial_data.keys()))
    writer.writerow([""] + list(initial_data.values()))
    writer.writerow(["Average"] + list(avg_data.keys()))
    writer.writerow([""] + list(avg_data.values()))

sensors.cleanup()
cap.release()
cv2.destroyAllWindows()
