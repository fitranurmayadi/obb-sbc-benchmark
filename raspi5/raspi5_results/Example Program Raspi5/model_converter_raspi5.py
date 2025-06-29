import os

# Arsitektur model dan resolusi input
architectures = ["n", "s", "m"]
resolutions = [320, 640, 1280]
data_yaml = "../../datasets/obb_mini_things-3/data.yaml"

# Format yang akan diekspor
formats = [ "ncnn"]

# Iterasi semua kombinasi model dan format
for arch in architectures:
    for res in resolutions:
        model_name = f"yolo11{arch}-obb-{res}.pt"

        for fmt in formats:
            if fmt == "ncnn":
                save_name = f"cpu_ncnn/yolo11{arch}-obb-{res}_ncnn"
                command = (
                    f"yolo export model={model_name} data={data_yaml} "
                    f"format=ncnn half=True name={save_name}"
                )
            else:
                continue

            print(f"[INFO] Exporting {model_name} → {fmt.upper()}")
            os.system(command)

