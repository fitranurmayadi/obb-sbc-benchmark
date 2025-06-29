import os
from ultralytics import YOLO
import datetime

# Daftar model dan konfigurasi
models = [
    ("yolo11n-obb-320_ncnn_model", 320, "cpu"),
    ("yolo11n-obb-320.mnn", 320, "cpu"),
    ("yolo11n-obb-320_rknn_model", 320, "cpu"),
    ("yolo11s-obb-320_ncnn_model", 320, "cpu"),
    ("yolo11s-obb-320_rknn_model", 320, "cpu"),
    ("yolo11s-obb-320.mnn", 320, "cpu"),
    ("yolo11n-obb-320.onnx", 320, "cpu"),
    ("yolo11n-obb-640_ncnn_model", 640, "cpu"),
    ("yolo11n-obb-320_openvino_model", 320, "cpu"),
    ("yolo11n-obb-640.mnn", 640, "cpu"),
    ("yolo11n-obb-320_float32.tflite", 320, "cpu"),
    ("yolo11n-obb-640_rknn_model", 640, "cpu"),
    ("yolo11m-obb-320_rknn_model", 320, "cpu"),
    ("yolo11n-obb-320_saved_model", 320, "cpu"),
    ("yolo11m-obb-320_ncnn_model", 320, "cpu"),
    ("yolo11s-obb-320_openvino_model", 320, "cpu"),
]

# Konfigurasi validasi
data_yaml = "../datasets/obb_mini_things-3/data.yaml"
batch = 16
conf = 0.25
iou = 0.6

# Folder hasil validasi
os.makedirs("benchmark_results", exist_ok=True)
log_file = f"benchmark_results/benchmarks_validation_rk3588_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Header file log
with open(log_file, "w") as f:
    f.write("model,imgsz,format,device,precision,recall,mAP50,mAP50_95,preprocess_ms,inference_ms,postprocess_ms,total_ms,FPS\n")

# Loop semua model
for model_path, imgsz, device in models:
    format_hint = "RKNN" if "rknn" in model_path else (
        "MNN" if ".mnn" in model_path else (
        "NCNN" if "ncnn" in model_path else (
        "ONNX" if ".onnx" in model_path else (
        "OpenVINO" if "openvino" in model_path else (
        "TFLite" if ".tflite" in model_path else (
        "SavedModel" if "saved_model" in model_path else "Other"))))))

    print(f"\n🚀 Validasi model: {model_path} | Format: {format_hint} | Resolusi: {imgsz} | Device: {device}")

    try:
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
            plots=False,
            device=device,
            verbose=False,
        )

        # Ambil metrik
        precision = metrics.box.mp
        recall = metrics.box.mr
        map50 = metrics.box.map50
        map50_95 = metrics.box.map
        preprocess_ms = metrics.speed['preprocess']
        inference_ms = metrics.speed['inference']
        postprocess_ms = metrics.speed['postprocess']
        total_ms = preprocess_ms + inference_ms + postprocess_ms
        fps = 1000 / total_ms if total_ms > 0 else 0

        # Tulis ke file
        with open(log_file, "a") as f:
            f.write(f"{model_path},{imgsz},{format_hint},{device},{precision:.4f},{recall:.4f},{map50:.4f},{map50_95:.4f},"
                    f"{preprocess_ms:.1f},{inference_ms:.1f},{postprocess_ms:.1f},{total_ms:.1f},{fps:.1f}\n")

    except Exception as e:
        print(f"❌ Gagal memproses model {model_path}: {e}")

