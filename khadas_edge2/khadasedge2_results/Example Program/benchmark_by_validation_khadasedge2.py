import os
import datetime
from ultralytics import YOLO

# Konfigurasi dasar
architectures = ["n", "s", "m"]
resolutions = [320, 640, 1280]
formats = ["rknn", "ncnn"]
sbc_name = "khadasedge2"
data_yaml = "../../datasets/obb_mini_things-3/data.yaml"
batch = 1
conf = 0.25
iou = 0.6

# Folder hasil log
os.makedirs("01_benchmark_results", exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f"01_benchmark_results/benchmarks_by_validation_{sbc_name}_{timestamp}.csv"

# Header CSV
with open(log_file, "w") as f:
    f.write("model\tformat\tSize(MB)\tprecision\trecall\tmAP50\tmAP50_95\tpreprocess_ms\tinference_ms\tpostprocess_ms\ttotal_ms\tFPS\n")

# Loop validasi semua model
for arch in architectures:
    for res in resolutions:
        model_base = f"yolo11{arch}-obb-{res}"

        for fmt in formats:
            if fmt == "rknn":
                model_name = f"yolo11{arch}_obb_{res}_{sbc_name}_rknn_fp16"
                model_dir = f"{model_base}_rknn_model"
                model_file_ext = ".rknn"
            elif fmt == "ncnn":
                model_name = f"yolo11{arch}_obb_{res}_{sbc_name}_ncnn_fp16"
                model_dir = f"{model_base}_ncnn_model"
                model_file_ext = ".ncnn.bin"
            else:
                continue  # skip unsupported formats

            try:
                # Cari file model
                model_files = [f for f in os.listdir(model_dir) if f.endswith(model_file_ext)]
                if not model_files:
                    raise FileNotFoundError(f"Tidak ada file dengan ekstensi {model_file_ext} di folder: {model_dir}")

                model_path = os.path.join(model_dir, model_files[0])
                size_mb = os.path.getsize(model_path) / (1024 * 1024)

                print(f"\n🚀 Validasi model: {model_name} | Format: {fmt.upper()} | Resolusi: {res}")

                # Load model (Ultralytics akan handle backend sesuai format)
                model = YOLO(model_dir, task="obb")

                # Validasi
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

                # Simpan ke CSV
                with open(log_file, "a") as f:
                    f.write(f"{model_name}\t{fmt.upper()}\t{size_mb:.1f}\t{precision:.4f}\t{recall:.4f}\t{map50:.4f}\t{map50_95:.4f}"
                            f"\t{preprocess_ms:.1f}\t{inference_ms:.1f}\t{postprocess_ms:.1f}\t{total_ms:.1f}\t{fps:.1f}\n")

            except Exception as e:
                print(f"❌ Gagal memproses model {model_name}: {e}")

