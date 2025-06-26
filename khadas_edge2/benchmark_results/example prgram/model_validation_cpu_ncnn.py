import os
from ultralytics import YOLO
import datetime

# Daftar model dan konfigurasi
models = [
    ("yolo11n-obb-320_ncnn_model", 320),
    ("yolo11s-obb-320_ncnn_model", 320),
    ("yolo11m-obb-320_ncnn_model", 320),
    ("yolo11n-obb-640_ncnn_model", 640),
    ("yolo11s-obb-640_ncnn_model", 640),
    ("yolo11m-obb-640_ncnn_model", 640),
    ("yolo11n-obb-1280_ncnn_model", 1280),
    ("yolo11s-obb-1280_ncnn_model", 1280),
    ("yolo11m-obb-1280_ncnn_model", 1280),
]

# Konfigurasi validasi
data_yaml = "../datasets/obb_mini_things-3/data.yaml"
batch = 1
conf = 0.25
iou = 0.6
device = 'cpu'  # Bisa ganti ke "cpu" atau "cuda:0"

# Folder hasil validasi
os.makedirs("validation_results", exist_ok=True)
log_file = f"validation_results/validation_rk3588_cpu_ncnn_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Header file log
with open(log_file, "w") as f:
    f.write("model,imgsz,precision,recall,mAP50,mAP50_95,preprocess_ms,inference_ms,postprocess_ms\n")

# Loop semua model
for model_path, imgsz in models:
    print(f"🚀 Memvalidasi model: {model_path} dengan resolusi {imgsz}x{imgsz}")

    # Load model
    model = YOLO(model_path, task="obb")

    # Validasi
    metrics = model.val(
        data=data_yaml,
        task='obb',
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        plots=True,
        verbose=False,
        device=device,
    )

    # Ambil metrik yang relevan
    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map50_95 = metrics.box.map

    preprocess_ms = metrics.speed['preprocess']
    inference_ms = metrics.speed['inference']
    postprocess_ms = metrics.speed['postprocess']

    # Tulis ke file log
    with open(log_file, "a") as f:
        f.write(f"{model_path},{imgsz},{precision:.4f},{recall:.4f},{map50:.4f},{map50_95:.4f},{preprocess_ms:.1f},{inference_ms:.1f},{postprocess_ms:.1f}\n")
