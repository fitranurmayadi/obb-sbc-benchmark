import os
import datetime
import torch
import gc
import time
from ultralytics import YOLO

# Konfigurasi
architectures = ["n", "s", "m"]
resolutions = [320, 640, 1280]
#architectures = ["n", "s", "m"]
#resolutions = [320]


formats = ["engine"]
sbc_name = "jetsonorin"
data_yaml = "../../datasets/obb_mini_things-3/data.yaml"
batch = 1
conf = 0.25
iou = 0.6

# Folder hasil log
os.makedirs("01_benchmark_results", exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f"01_benchmark_results/benchmarks_by_validation_{sbc_name}_{timestamp}.csv"

# Header
with open(log_file, "w") as f:
    f.write("model\tformat\tSize(MB)\tprecision\trecall\tmAP50\tmAP50_95\tpreprocess_ms\tinference_ms\tpostprocess_ms\ttotal_ms\tFPS\n")

# Loop validasi
for res in resolutions:
    for arch in architectures:
    
        model_base = f"yolo11{arch}-obb-{res}"

        for fmt in formats:
            if fmt == "engine":
                model_name = f"{model_base}_{sbc_name}_engine_fp16"
                model_path = f"{model_base}.engine"
                device = "0"
                model_dir = model_path  # Langsung file
            elif fmt == "ncnn":
                model_name = f"{model_base}_{sbc_name}_ncnn_fp16"
                model_folder = f"{model_base}_ncnn_model"
                model_file_ext = ".ncnn.bin"
                device = "cpu"
                # Cari file .ncnn.bin
                try:
                    model_files = [f for f in os.listdir(model_folder) if f.endswith(model_file_ext)]
                    if not model_files:
                        raise FileNotFoundError(f"Tidak ada file .ncnn.bin di {model_folder}")
                    model_dir = model_folder  # Folder ncnn untuk Ultralytics
                    model_path = os.path.join(model_folder, model_files[0])
                except Exception as e:
                    print(f"❌ {model_name}: {e}")
                    continue
            else:
                continue

            try:
                size_mb = os.path.getsize(model_path) / (1024 * 1024)

                print(f"\n🚀 Validasi model: {model_name} | Format: {fmt.upper()} | Resolusi: {res}")

                model = YOLO(model_dir, task="obb")
                metrics = model.val(
                    data=data_yaml,
                    task="obb",
                    imgsz=res,
                    batch=batch,
                    conf=conf,
                    iou=iou,
                    plots=False,
                    device=device,
                    workers = 0,
                    verbose=False,
                    half=True,
                )

                # Metrik
                precision = metrics.box.mp
                recall = metrics.box.mr
                map50 = metrics.box.map50
                map50_95 = metrics.box.map
                preprocess_ms = metrics.speed['preprocess']
                inference_ms = metrics.speed['inference']
                postprocess_ms = metrics.speed['postprocess']
                total_ms = preprocess_ms + inference_ms + postprocess_ms
                fps = 1000 / total_ms if total_ms > 0 else 0

                # Simpan log
                with open(log_file, "a") as f:
                    f.write(f"{model_name}\t{fmt.upper()}\t{size_mb:.1f}\t{precision:.4f}\t{recall:.4f}\t{map50:.4f}\t{map50_95:.4f}"
                            f"\t{preprocess_ms:.1f}\t{inference_ms:.1f}\t{postprocess_ms:.1f}\t{total_ms:.1f}\t{fps:.1f}\n")
                
                
                del model
                gc.collect()
            
                torch.cuda.empty_cache()
                time.sleep(3)
            
            except Exception as e:
                print(f"❌ Gagal memproses {model_name}: {e}")
                
            



