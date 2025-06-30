import os
import datetime
import torch
import gc
import time
from ultralytics import YOLO

# Konfigurasi dasar
architectures = ["n", "s", "m"]
#resolutions = [320, 640, 1280]
#formats = ["openvino", "ncnn", "mnn", "onnx", "pytorch", "torchscript"]
formats = ["ncnn"]

resolutions = [320, 640, 1280]


sbc_name = "jetsonorinnano"
data_yaml = "../../datasets/obb_mini_things-3/data.yaml"
batch = 1
conf = 0.25
iou = 0.6

# Folder hasil log
os.makedirs("01_benchmark_results", exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f"01_benchmark_results/benchmarks_by_validation_{sbc_name}_via_cpu_{timestamp}.csv"

# Header CSV
with open(log_file, "w") as f:
    f.write("model\tformat\tSize(MB)\tprecision\trecall\tmAP50\tmAP50_95\tpreprocess_ms\tinference_ms\tpostprocess_ms\ttotal_ms\tFPS\n")

# Loop validasi semua model
for arch in architectures:
    for res in resolutions:
        model_base = f"yolo11{arch}-obb-{res}"

        for fmt in formats:
            if fmt == "openvino":
                model_name = f"yolo11{arch}_obb_{res}_{sbc_name}_openvino_fp16"
                model_dir = f"{model_base}_openvino_model"
                model_file_ext = ".bin"
            elif fmt == "ncnn":
                model_name = f"yolo11{arch}_obb_{res}_{sbc_name}_ncnn_fp16"
                model_dir = f"{model_base}_ncnn_model"
                model_file_ext = ".ncnn.bin"
            elif fmt == "mnn":
                model_name = f"yolo11{arch}_obb_{res}_{sbc_name}_mnn_fp16"
                model_dir = "."
                model_file_ext = f"{model_base}.mnn"
            elif fmt == "onnx":
                model_name = f"yolo11{arch}_obb_{res}_{sbc_name}_onnx_fp16"
                model_dir = "."
                model_file_ext = f"{model_base}.onnx"
            elif fmt == "pytorch":
                model_name = f"yolo11{arch}_obb_{res}_{sbc_name}_pytorch_fp16"
                model_dir = "."
                model_file_ext = f"{model_base}.pt"
            elif fmt == "torchscript":
                model_name = f"yolo11{arch}_obb_{res}_{sbc_name}_torchscript_fp16"
                model_dir = "."
                model_file_ext = f"{model_base}.torchscript"
            elif fmt == "tflite":
                model_name = f"yolo11{arch}_obb_{res}_{sbc_name}_tflite_fp16"
                model_dir = "."
                model_file_ext = f"{model_base}.tflite"
            else:
                continue  # skip unsupported

            try:
                # Cari file model
                if isinstance(model_file_ext, str) and model_file_ext.startswith("."):
                    # Folder berisi banyak file dengan ekstensi
                    model_files = [f for f in os.listdir(model_dir) if f.endswith(model_file_ext)]
                    if not model_files:
                        raise FileNotFoundError(f"Tidak ada file dengan ekstensi {model_file_ext} di folder: {model_dir}")
                    model_path = os.path.join(model_dir, model_files[0])
                else:
                    # Model langsung sebagai 1 file
                    model_path = os.path.join(model_dir, model_file_ext)
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")
                
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"\n🚀 Validasi model: {model_name} | Format: {fmt.upper()} | Resolusi: {res}")

                model = YOLO(model_path if fmt in ["mnn", "onnx", "pytorch", "torchscript", "tflite"] else model_dir, task="obb")

                metrics = model.val(
                    data=data_yaml,
                    task='obb',
                    imgsz=res,
                    batch=batch,
                    conf=conf,
                    iou=iou,
                    plots=False,
                    device="cpu",
                    verbose=False,
                    half=True,
                )

                precision = metrics.box.mp
                recall = metrics.box.mr
                map50 = metrics.box.map50
                map50_95 = metrics.box.map
                preprocess_ms = metrics.speed['preprocess']
                inference_ms = metrics.speed['inference']
                postprocess_ms = metrics.speed['postprocess']
                total_ms = preprocess_ms + inference_ms + postprocess_ms
                fps = 1000 / total_ms if total_ms > 0 else 0

                with open(log_file, "a") as f:
                    f.write(f"{model_name}\t{fmt.upper()}\t{size_mb:.1f}\t{precision:.4f}\t{recall:.4f}\t{map50:.4f}\t{map50_95:.4f}"
                            f"\t{preprocess_ms:.1f}\t{inference_ms:.1f}\t{postprocess_ms:.1f}\t{total_ms:.1f}\t{fps:.1f}\n")

                del model
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(3)

            except Exception as e:
                print(f"❌ Gagal memproses model {model_name}: {e}")
