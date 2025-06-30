import os

# Arsitektur model dan resolusi input
architectures = ["n", "s", "m"]
resolutions = [320, 640, 1280]
data_yaml = "../../datasets/obb_mini_things-3/data.yaml"

# Format yang akan diekspor
#formats = ["openvino", "ncnn", "mnn", "onnx", "pytorch", "torchscript", "tflite"]

#uji untuk satu format
formats = ["ncnn"]

# Iterasi semua kombinasi model dan format
for fmt in formats:
    for res in resolutions:
        for arch in architectures:
            model_name = f"yolo11{arch}-obb-{res}.pt"
            save_name = f"yolo11{arch}_obb_{res}_{fmt}_fp16"

            # Perintah ekspor sesuai format
            command = (
                f"yolo export model={model_name} data={data_yaml} "
                f"format={fmt} half=True name={save_name} device=cpu"
            )

            print(f"[INFO] Exporting {model_name} → {fmt.upper()} (FP16)")
            os.system(command)

