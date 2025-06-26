import os
from ultralytics import YOLO

# Daftar model (nama file dan resolusi)
models = [
    ("yolo11n-obb-320.pt", 320),
    ("yolo11n-obb-640.pt", 640),
    ("yolo11n-obb-1280.pt", 1280),
    ("yolo11s-obb-320.pt", 320),
    ("yolo11s-obb-640.pt", 640),
    ("yolo11s-obb-1280.pt", 1280),
    ("yolo11m-obb-320.pt", 320),
    ("yolo11m-obb-640.pt", 640),
    ("yolo11m-obb-1280.pt", 1280),
]

# Path ke dataset YAML
data_path = '../datasets/obb_mini_things-3/data.yaml'
device = 'cpu'  # GPU id, ubah ke 'cpu' jika tidak ada CUDA

# Direktori hasil
os.makedirs("benchmark_results", exist_ok=True)

# Benchmark loop
for model_file, imgsz in models:
    model_name = model_file.replace(".pt", "")
    print(f"🔧 Benchmarking {model_name} (imgsz={imgsz})...")

    model = YOLO(model_file)
    results = model.benchmark(data=data_path, imgsz=imgsz, device=device)

    # Simpan hasil ke file
    output_file = f"benchmark_results/{model_name}_benchmark.txt"
    with open(output_file, "w") as f:
        f.write(f"Benchmark result for {model_name} (imgsz={imgsz}):\n")
        f.write(str(results))

    print(f"✅ Saved: {output_file}")

